'''
standalone training for the e-prop models, per-timestep.

The counterpart to main.py's --train branch, with the batched forward/backward
replaced by eprop_online.online_train_step: each timestep's loss is backwarded as
its logits are produced, so peak memory is flat in T rather than ~830MB at
T=300 (see eprop_online.py). The gradient is unchanged --
test_eprop_streaming_equivalence.py asserts it against the batched path.

Checkpoints and logs are written in exactly main.py's layout, so the standalone
evaluation pass is main.py's and is NOT duplicated here -- that code is
model-agnostic and needs no e-prop variant:

    python main_eprop.py --config_file hgrsnn_eprop_large_config.yaml \
        --model_name hgrsnn_eprop --data_dir ... --output_dir ...

    python main.py --config_file hgrsnn_eprop_large_config.yaml \
        --model_name hgrsnn_eprop --data_dir ... --output_dir ...

The first trains and reports per-seed test metrics; the second aggregates across
seeds and writes the ROC plots and metrics csv.
'''

import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import yaml
import random
import numpy as np
import pandas as pd

from dataloader import train_test_val_split, get_dataset_statistics, get_beat_bounds
from models import create_model
from eprop_online import online_train_step
from utils.encoding import Current_Encoder
from utils.logging import log_model
import utils.metrics as metrics

# evaluation is identical to the batched path (it runs under no_grad, where the
# O(T) trace retention does not arise), so it is imported rather than reimplemented
from main import evaluate_model, get_multiclass_metrics, AAMI_CLASS_NAMES

torch.set_num_threads(1)


def train_model(
        model,
        train_loader,
        val_loader,
        encoder,
        optimizer,
        loss_fn,
        num_epochs,
        val_steps=2,
        device='cpu',
        beat_bounds=None,
        binary=True,
        logging_configs={
            'model_name': None,
            'weight_folder': 'train_weights',
            'log_folder': 'train_logs',
            'log_steps': 10
        }
    ):

    '''
    train an e-prop model with per-timestep backward.

    Mirrors main.train_model -- same checkpointing and logging, final-epoch
    weights saved -- differing only in calling online_train_step for the epoch.
    '''

    train_losses = []
    val_losses = []

    train_aucs = []
    val_aucs = []

    model_path = os.path.join(logging_configs['weight_folder'], logging_configs['model_name']) + '.pth'
    log_folder = os.path.join(logging_configs['log_folder'], logging_configs['model_name'])

    os.makedirs(logging_configs['weight_folder'], exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = online_train_step(
            model, train_loader, encoder, optimizer, loss_fn, device, beat_bounds, binary,
            epoch=epoch, log_every=100)
        train_losses.append((epoch, train_loss))

        print('Epoch {} Average Loss: {}'.format(epoch, train_loss))
        sys.stdout.flush()

        if (epoch + 1) % val_steps == 0:
            val_loss, val_auc = evaluate_model(model, val_loader, encoder, loss_fn, device, beat_bounds, binary)
            val_losses.append((epoch, val_loss))
            val_aucs.append((epoch, val_auc))

            # same forward pass over the training set, in eval mode
            _, train_auc = evaluate_model(model, train_loader, encoder, loss_fn, device, beat_bounds, binary,
                                          split_name='Train')
            train_aucs.append((epoch, train_auc))

            print('Epoch {} AUC: train {:.4f} | val {:.4f} | gap {:+.4f}'.format(
                epoch, train_auc, val_auc, train_auc - val_auc))
            sys.stdout.flush()

            log_model(train_losses, val_losses, log_folder, train_aucs, val_aucs)

    # the final epoch's weights are the ones evaluated (no early stopping)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_auc': val_auc,
    }, model_path)
    
    sys.stdout.flush()

    return train_losses, val_losses


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str) # model config (in nested_configs folder, not the full path)
    parser.add_argument("--model_name", type=str) # desired model name (s_{seed}_{model_name} will be saved)
    parser.add_argument("--data_dir", type=str) # input data directory
    parser.add_argument("--output_dir", type=str) # output directory to store model weights/metrics csv/train plots

    parser.add_argument("--num_seeds", type=int, default=5) # how many seeds to train for
    parser.add_argument("--use_best_model", action='store_true') # use the saved hyperparams from the best performing model

    args = parser.parse_args()

    config_key = 'xbest' if args.use_best_model else 'x0'
    seeds = list(range(args.num_seeds))

    for seed in seeds:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = f's_{seed}_{args.model_name}'

        with open(f'nested_configs/{args.config_file}', 'r') as file:
            model_configs = yaml.safe_load(file)

        num_epochs = model_configs['kwargs']['num_epochs']
        binary = model_configs['kwargs'].get('binary', True)

        # patient_norm: z-score each beat by its own record instead of by one
        # global constant. false reproduces the original behaviour exactly.
        patient_norm = model_configs['kwargs'].get('patient_norm', False)

        model = create_model(model_configs, config_key)
        learning_rate = model_configs[config_key]['lr']

        # --- Create Datasets/DataLoaders --- #
        dataset_configs = {
            "data_file": f"{args.data_dir}/data.hdf5",
            "metadata": f"{args.data_dir}/data.csv",
            "train_prop": 0.6,
            "val_prop": 0.2,
            "test_prop": 0.2,
            "binary": binary,
            "random_state": seed,
            "balance": True,
            "fourier": False,
            "return_stats": patient_norm,
        }

        train_batch_size = 64
        eval_batch_size = 128

        train_dataset, val_dataset, test_dataset = train_test_val_split(**dataset_configs)

        # the labeled beat's span within the window, read from the dataset itself
        beat_bounds = get_beat_bounds(train_dataset)

        train_mean, train_std = get_dataset_statistics(train_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size)

        logging_configs = {
            'model_name': model_name,
            'weight_folder': f'{args.output_dir}/model_weights',
            'log_folder': f'{args.output_dir}/train_logs',
            'log_steps': 1
        }

        # --- Train the Model --- #
        encoder = Current_Encoder(train_mean, train_std)

        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        loss_fn = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()

        val_steps = 10

        print(f'Training Model {seed} (per-timestep e-prop)')
        sys.stdout.flush()

        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            encoder,
            optimizer,
            loss_fn,
            num_epochs,
            val_steps,
            device,
            beat_bounds,
            binary,
            logging_configs,
        )

        model_path = os.path.join(logging_configs['weight_folder'], logging_configs['model_name']) + '.pth'
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

        log_dir = os.path.join(logging_configs['log_folder'], logging_configs['model_name'])

        if binary:
            threshold = metrics.get_threshold(model, val_loader, encoder, device, fix_sensitivity=True,
                                              beat_bounds=beat_bounds)

            accuracy = metrics.get_accuracy(model, test_loader, encoder, binary, threshold, device, beat_bounds=beat_bounds)
            print('\nTest Set Accuracy: {}'.format(accuracy))

            metrics.get_roc_curve(model, test_loader, encoder, threshold, log_dir, device, beat_bounds=beat_bounds)
        else:
            accuracy, macro_auc, _, per_class, y_true, y_prob = get_multiclass_metrics(
                model, test_loader, encoder, device, beat_bounds=beat_bounds)

            print('\nTest Set Accuracy: {:.4f}, Macro OVR AUC: {:.4f}'.format(accuracy, macro_auc))
            print(pd.DataFrame(per_class).T)

            metrics.get_multiclass_roc_curve(y_true, y_prob, log_dir,
                                             class_names=AAMI_CLASS_NAMES)

    print('\nTraining done. For the cross-seed metrics/ROC aggregation, run main.py '
          'with the same --config_file/--model_name/--output_dir (without --train).')
