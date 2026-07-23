'''
train/evaluate the non-spiking baselines.

this is a standalone entry point: it does not touch main.py, nested_train.py or
models/, and nothing in those imports this. what it *does* reuse, read-only, is
the data splitting and the metric helpers -- a baseline trained on different
splits or scored with a different threshold rule would not be comparable to the
SNN numbers, which is the entire point of having it.

usage (from the arrhythmia_detection directory):

    python -m baselines.train \
        --config_file=conv1d_config.yaml \
        --model_name=conv1d \
        --data_dir=../beats \
        --output_dir=local_results/baselines \
        --num_seeds=5 --train
'''

import os
import sys
import time
import yaml
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# allow both `python -m baselines.train` and `python baselines/train.py` from the
# arrhythmia_detection directory -- the imports below are relative to that dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader import train_test_val_split, get_dataset_statistics, get_beat_bounds
from utils.encoding import Current_Encoder, unpack_batch
from utils.readout import prepare_loss_inputs, compute_auc
from utils.logging import log_model
import utils.metrics as metrics

# the SNN metric definitions, reused verbatim so the two sets of numbers mean the
# same thing. main.py guards its own driver behind __main__, so this import only
# pulls in the functions.
from main import get_binary_metrics, get_multiclass_metrics, AAMI_CLASS_NAMES

from baselines.models import create_baseline

torch.set_num_threads(1)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')


def build_optimizer(model, config):
    name = config.get('optimizer', 'adamw')
    lr = config['lr']
    weight_decay = config.get('weight_decay', 0.0)

    if name == 'sgd':
        # momentum defaults off: main.py trains the SNNs on plain optim.SGD(lr),
        # so anything else would be comparing optimizers, not architectures
        return optim.SGD(model.parameters(), lr=lr, momentum=config.get('momentum', 0.0),
                         weight_decay=weight_decay)
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"optimizer must be 'sgd', 'adam' or 'adamw', got {name!r}")


def train_step(model, train_loader, encoder, optimizer, loss_fn, epoch, device,
               beat_bounds=None, binary=True, grad_clip=None):

    model.train()
    epoch_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        data, target, stats = unpack_batch(batch)

        data = encoder.encode(data, *stats).float()
        data, target = data.to(device), target.to(device)

        output, _ = model(data)

        # supervise only the labeled beat span; pre/post context is input only
        start, end = beat_bounds if beat_bounds is not None else (0, output.shape[1])
        output, target = output[:, start:end], target[:, start:end]

        output, target = prepare_loss_inputs(output, target, binary)

        loss = loss_fn(output, target)
        loss.backward()

        # unlike the SNNs (which are trained unclipped) the RNN baselines can
        # blow up over a 2s window, so clipping is available and on by default
        # in the configs
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))
            sys.stdout.flush()

        epoch_loss += loss.item() * data.shape[0]

    return epoch_loss / len(train_loader.dataset)


def evaluate_model(model, loader, encoder, loss_fn, device, beat_bounds=None, binary=True,
                   split_name='Validation'):

    model.eval()
    test_loss = 0

    y_true = []
    y_score = []

    with torch.no_grad():
        for batch in loader:
            data, target, stats = unpack_batch(batch)
            data = encoder.encode(data, *stats).float()
            data, target = data.to(device), target.to(device)

            output, _ = model(data)

            start, end = beat_bounds if beat_bounds is not None else (0, output.shape[1])
            output, target = output[:, start:end], target[:, start:end]

            # the label is constant across the window, so column 0 is the beat label
            y_true.append(target[:, 0].cpu().numpy())
            # mean, not sum, over the supervised timesteps. the two are monotonically
            # related so binary AUC is identical, but multiclass_auc softmaxes this
            # score -- and summing over the ~180-sample span scales logits by ~180,
            # saturating softmax so ~85% of rows hit probability exactly 1.0. measured
            # cost on the multiclass convnet: macro AUC 0.824 -> 0.651. the error also
            # tracks readout magnitude, which the `threshold` search dimension moves by
            # ~12 orders of magnitude, so summing biases model ranking toward
            # small-logit configs rather than better ones.
            y_score.append(output.mean(1).cpu().numpy())

            output, target = prepare_loss_inputs(output, target, binary)
            test_loss += loss_fn(output, target).item() * data.shape[0]

    test_loss /= len(loader.dataset)

    y_true = np.concatenate(y_true, axis=0)
    y_score = np.concatenate(y_score, axis=0)

    auc = compute_auc(y_true, y_score, binary)

    print('\n{} set: Average loss: {:.4f}, AUC: {:.4f}\n'.format(split_name, test_loss, auc))
    sys.stdout.flush()

    return test_loss, auc


def train_model(model, train_loader, val_loader, encoder, optimizer, loss_fn, config,
                device, beat_bounds, model_path, log_folder):

    '''
    train one baseline and save the FINAL epoch's weights to model_path.

    main.py saves the SNN's last epoch, so this does too. selecting the best
    epoch by val AUC would hand the baseline an early-stopping advantage the SNN
    does not get, which is exactly the kind of asymmetry this setup exists to
    avoid. the per-epoch val AUCs are still logged, so an early-stopped number
    can be read off the log if it is ever wanted.
    '''

    num_epochs = config['num_epochs']
    val_steps = config.get('val_steps', 2)
    binary = config.get('binary', True)
    grad_clip = config.get('grad_clip', None)

    train_losses, val_losses, train_aucs, val_aucs = [], [], [], []

    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, encoder, optimizer, loss_fn, epoch,
                                device, beat_bounds, binary, grad_clip)
        train_losses.append((epoch, train_loss))

        print('Epoch {} Average Loss: {}'.format(epoch, train_loss))
        sys.stdout.flush()

        if (epoch + 1) % val_steps != 0 and epoch != num_epochs - 1:
            continue

        val_loss, val_auc = evaluate_model(model, val_loader, encoder, loss_fn, device,
                                           beat_bounds, binary)
        # same forward pass over the training set, in eval mode -- train and val
        # AUC read together separate underfitting from a generalization gap
        _, train_auc = evaluate_model(model, train_loader, encoder, loss_fn, device,
                                      beat_bounds, binary, split_name='Train')

        val_losses.append((epoch, val_loss))
        val_aucs.append((epoch, val_auc))
        train_aucs.append((epoch, train_auc))

        print('Epoch {} AUC: train {:.4f} | val {:.4f} | gap {:+.4f}'.format(
            epoch, train_auc, val_auc, train_auc - val_auc))
        sys.stdout.flush()

        log_model(train_losses, val_losses, log_folder, train_aucs, val_aucs)

    torch.save({
        'epoch': num_epochs - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, model_path)

    return train_losses, val_losses


def make_dataset_configs(config, data_dir, binary, seed):
    dataset_configs = dict(config.get('dataset_configs', {}))

    dataset_configs.setdefault('train_prop', 0.6)
    dataset_configs.setdefault('val_prop', 0.2)
    dataset_configs.setdefault('test_prop', 0.2)
    dataset_configs.setdefault('balance', True)
    dataset_configs.setdefault('fourier', False)

    # patient_norm: z-score each beat by its own record's mean/std instead of by
    # one global constant, so patient-specific gain does not survive into the
    # input. needs a dataset built by the current create_beat_dataset.py.
    dataset_configs.setdefault('return_stats', config.get('patient_norm', False))

    # --data_dir wins over anything in the config, matching main.py
    dataset_configs['data_file'] = f'{data_dir}/data.hdf5'
    dataset_configs['metadata'] = f'{data_dir}/data.csv'
    dataset_configs['binary'] = binary
    dataset_configs['random_state'] = seed

    return dataset_configs


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)          # file in baselines/configs
    parser.add_argument('--model_name', type=str)           # weights saved as s_{seed}_{model_name}.pth
    parser.add_argument('--plot_title', type=str, default=None)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--weight_dir', type=str, default=None)

    args = parser.parse_args()

    config_path = args.config_file

    if not os.path.exists(config_path):
        config_path = os.path.join(CONFIG_DIR, args.config_file)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    binary = config.get('binary', True)
    out_dims = config.get('out_dims', 1 if binary else len(AAMI_CLASS_NAMES))
    model_kwargs = config.get('model_kwargs', {})

    train_batch_size = config.get('train_batch_size', 64)
    eval_batch_size = config.get('eval_batch_size', 128)

    seeds = list(range(args.num_seeds))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weight_folder = args.weight_dir if args.weight_dir is not None else f'{args.output_dir}/model_weights'
    log_root = f'{args.output_dir}/train_logs'
    results_folder = f'{args.output_dir}/model_results'
    plots_dir = os.path.join(results_folder, 'plots')
    metrics_dir = os.path.join(results_folder, 'metrics')

    for folder in (weight_folder, log_root, plots_dir, metrics_dir):
        os.makedirs(folder, exist_ok=True)

    if args.train:
        for seed in seeds:
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            start_time = time.time()
            model_name = f's_{seed}_{args.model_name}'

            dataset_configs = make_dataset_configs(config, args.data_dir, binary, seed)
            train_dataset, val_dataset, _ = train_test_val_split(**dataset_configs)
            beat_bounds = get_beat_bounds(train_dataset)

            model = create_baseline(config['model_type'], out_dims=out_dims,
                                    beat_bounds=beat_bounds, **model_kwargs).to(device)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'Training {model_name} ({n_params} trainable params)')

            if hasattr(model, 'receptive_field'):
                print(f'Receptive field: {model.receptive_field()} samples')

            sys.stdout.flush()

            train_mean, train_std = get_dataset_statistics(train_dataset)
            encoder = Current_Encoder(train_mean, train_std)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size)

            optimizer = build_optimizer(model, config)
            loss_fn = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()

            model_path = os.path.join(weight_folder, model_name) + '.pth'
            log_folder = os.path.join(log_root, model_name)
            os.makedirs(log_folder, exist_ok=True)

            train_model(model, train_loader, val_loader, encoder, optimizer, loss_fn,
                        config, device, beat_bounds, model_path, log_folder)

            print('Seed {} done in {:.2f} minutes'.format(seed, (time.time() - start_time) / 60))
            sys.stdout.flush()

    # --- evaluate every trained seed on the held-out test set --- #
    print('Evaluating Models')

    # only meaningful for binary: multiclass predicts by argmax, no threshold to fix
    fix_sensitivity = binary

    metrics_name = args.model_name

    if fix_sensitivity:
        metrics_name += '_fixed_sensitivity'
    elif not binary:
        metrics_name += '_multiclass'

    model_vals = []
    roc_data = []

    for seed in seeds:
        model_path = os.path.join(weight_folder, f's_{seed}_{args.model_name}') + '.pth'

        if not os.path.exists(model_path):
            continue

        dataset_configs = make_dataset_configs(config, args.data_dir, binary, seed)
        train_dataset, val_dataset, test_dataset = train_test_val_split(**dataset_configs)
        beat_bounds = get_beat_bounds(train_dataset)

        train_mean, train_std = get_dataset_statistics(train_dataset)
        encoder = Current_Encoder(train_mean, train_std)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size)

        model = create_baseline(config['model_type'], out_dims=out_dims,
                                beat_bounds=beat_bounds, **model_kwargs)
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])

        folder_name = f'{args.model_name}_seed_{seed}'

        if binary:
            # threshold is picked on validation, never on test
            threshold = metrics.get_threshold(model, val_loader, encoder,
                                              fix_sensitivity=True, beat_bounds=beat_bounds)

            auc, accuracy, sensitivity, specificity, ppv, npv, y_true, y_score = get_binary_metrics(
                model, test_loader, encoder, threshold, beat_bounds=beat_bounds)
            roc_data.append((y_true, y_score))

            if fix_sensitivity:
                folder_name += '_fixed_sensitivity'

            plot_dir = os.path.join(plots_dir, folder_name)
            os.makedirs(plot_dir, exist_ok=True)
            metrics.get_roc_curve(model, test_loader, encoder, threshold, plot_dir, 'cpu',
                                  beat_bounds=beat_bounds)

            model_vals.append({
                'auc': auc,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
            })
        else:
            accuracy, macro_auc, weighted_auc, per_class, y_true, y_prob = get_multiclass_metrics(
                model, test_loader, encoder, 'cpu', beat_bounds=beat_bounds)
            roc_data.append((y_true, y_prob))

            plot_dir = os.path.join(plots_dir, folder_name)
            os.makedirs(plot_dir, exist_ok=True)
            metrics.get_multiclass_roc_curve(y_true, y_prob, plot_dir, class_names=AAMI_CLASS_NAMES)

            row = {'accuracy': accuracy, 'macro_auc': macro_auc, 'weighted_auc': weighted_auc}

            # flatten the per-class dict into columns like 'S_recall', 'V_f1', ...
            for cls_name, cls_metrics in per_class.items():
                for metric_name, value in cls_metrics.items():
                    row[f'{cls_name}_{metric_name}'] = value

            model_vals.append(row)

    if not model_vals:
        raise SystemExit(f'no trained weights found in {weight_folder} -- run with --train first')

    model_df = pd.DataFrame(model_vals)

    # both aggregates come off the per-seed rows only -- computing std after
    # appending the mean row would fold the mean into its own spread
    mean_vals, std_vals = model_df.mean(), model_df.std()

    model_df.loc['mean'] = mean_vals
    model_df.loc['std'] = std_vals

    print(model_df)
    model_df.to_csv(os.path.join(metrics_dir, metrics_name + '.csv'), index=True)

    # aggregate ROC across seeds: mean curve with a shaded 95% CI band
    mean_roc_name = args.model_name + '_mean_roc'

    if binary:
        if fix_sensitivity:
            mean_roc_name += '_fixed_sensitivity'
        metrics.plot_mean_roc_curve(roc_data, os.path.join(plots_dir, mean_roc_name),
                                    model_name=args.plot_title or args.model_name)
    else:
        metrics.plot_mean_multiclass_roc_curve(roc_data,
                                               os.path.join(plots_dir, mean_roc_name + '_multiclass'),
                                               class_names=AAMI_CLASS_NAMES,
                                               model_name=args.plot_title or args.model_name)
