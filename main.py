import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import snntorch.surrogate as surrogate

import os
import sys
import yaml
import random
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support


from dataloader import train_test_val_split, get_dataset_statistics, get_beat_bounds
from models import create_model
from utils.encoding import Rate_Encoder, Current_Encoder, unpack_batch
from utils.readout import prepare_loss_inputs, compute_auc, multiclass_auc

from utils.logging import log_model
import utils.metrics as metrics

torch.set_num_threads(1)
# get 5-fold validation model performance

def get_binary_metrics(model, test_loader, encoder, threshold, device='cpu', beat_bounds=None):
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            data, target, stats = unpack_batch(batch)

            if encoder is not None:
                data = encoder.encode(data, *stats)

            data, target = data.to(device), target.to(device)

            output, _ = model(data)

            # the labeled beat's own span, from the dataset (see dataloader.get_beat_bounds)
            start, end = beat_bounds if beat_bounds is not None else (0, output.shape[1])
            output = output[:, start:end].mean(dim=(1,2))

            y_true.append(target[:, 0].numpy())
            y_pred.append(output.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    auc = roc_auc_score(y_true, y_pred)

    # keep the raw scores so callers can build an aggregate (mean) ROC curve
    y_score = y_pred.copy()

    y_pred = (y_pred  > threshold).astype(int)

    accuracy = (y_true == y_pred).mean()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)

    return auc, accuracy, sensitivity, specificity, ppv, npv, y_true, y_score


AAMI_CLASS_NAMES = ['N', 'S', 'V', 'F']


def get_multiclass_metrics(model, test_loader, encoder, device='cpu', beat_bounds=None,
                           class_names=AAMI_CLASS_NAMES):
    '''
    multiclass counterpart to get_binary_metrics.

    there is no single operating threshold to tune here -- the prediction is the
    argmax over classes -- so sensitivity/specificity/ppv/npv are reported
    per-class (one-vs-rest) rather than as scalars.

    returns overall accuracy, macro/weighted OVR AUC, a per-class dict, and the
    raw (y_true, y_prob) so callers can aggregate ROC curves across seeds.
    '''

    y_true = []
    y_score = []

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            data, target, stats = unpack_batch(batch)

            if encoder is not None:
                data = encoder.encode(data, *stats)

            data, target = data.to(device), target.to(device)

            output, _ = model(data)

            # the labeled beat's own span, from the dataset (see dataloader.get_beat_bounds)
            start, end = beat_bounds if beat_bounds is not None else (0, output.shape[1])
            output = output[:, start:end].mean(dim=1)

            y_true.append(target[:, 0].cpu().numpy())
            y_score.append(output.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0).astype(int)
    y_score = np.concatenate(y_score, axis=0)

    n_classes = y_score.shape[1]
    labels = np.arange(n_classes)

    y_prob = softmax(y_score, axis=1)
    y_pred = y_prob.argmax(axis=1)

    accuracy = (y_true == y_pred).mean()

    macro_auc = multiclass_auc(y_true, y_score, average='macro')
    weighted_auc = multiclass_auc(y_true, y_score, average='weighted')

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # per-class one-vs-rest specificity, read off the multiclass confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    per_class = {}

    for c in labels:
        name = class_names[c] if c < len(class_names) else str(c)

        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp

        per_class[name] = {
            'precision': precision[c],   # == PPV for this class
            'recall': recall[c],         # == sensitivity for this class
            'f1': f1[c],
            'specificity': tn / (tn + fp + 1e-8),
            'support': int(support[c]),
        }

    return accuracy, macro_auc, weighted_auc, per_class, y_true, y_prob


def train_step(model, train_loader, encoder, optimizer, loss_fn, epoch, device, beat_bounds=None, binary=True):

    model.train()
    epoch_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        data, target, stats = unpack_batch(batch)

        data = encoder.encode(data, *stats)
        data, target = data.to(device), target.to(device)

        output, _ = model(data)

        # supervise only the labeled beat span; pre/post context is input only
        start, end = beat_bounds if beat_bounds is not None else (0, output.shape[1])
        output, target = output[:, start:end], target[:, start:end]

        output, target = prepare_loss_inputs(output, target, binary)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))
        sys.stdout.flush()

        epoch_loss += loss.item() * data.shape[0]

    epoch_loss /= len(train_loader.dataset)
    
    return epoch_loss

def evaluate_model(model, test_loader, encoder, loss_fn, device, beat_bounds=None, binary=True,
                   split_name='Validation'):

    model.eval()
    test_loss = 0

    y_true = []
    y_score = []

    with torch.no_grad():
        for batch in test_loader:
            data, target, stats = unpack_batch(batch)
            data = encoder.encode(data, *stats)
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
            loss = loss_fn(output, target)

            test_loss += loss.item() * data.shape[0]

    test_loss /= len(test_loader.dataset)

    y_true = np.concatenate(y_true, axis=0)
    y_score = np.concatenate(y_score, axis=0)

    auc = compute_auc(y_true, y_score, binary)

    print('\n{} set: Average loss: {:.4f}, AUC: {:.4f}\n'.format(split_name, test_loss, auc))

    return test_loss, auc

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
    train a model
    model: model object
    train_loader: dataloader for training data
    val_loader: dataloader for validation data
    encoder: encoding strategy for spiking input
    optimizer: optimizer for training
    loss_fn: loss function for training
    num_epochs: number of training epoch
    val_steps: how often to evaluate model on validation set
    device: device to run model on
    beat_bounds: (start, end) window-local indices of the labeled beat span,
        from dataloader.get_beat_bounds. None supervises the whole window.
    binary: whether this is binary (single logit + BCE) or multiclass (class
        logits + cross entropy)
    logging_configs: information for saving weights and logging model output

    '''

    train_losses = []
    val_losses = []

    # train AUC is logged alongside val AUC so the two can be read together:
    # both low -> underfitting (budget/lr/capacity), train high + val low ->
    # generalization gap. train loss alone can't tell these apart.
    train_aucs = []
    val_aucs = []

    model_path = os.path.join(logging_configs['weight_folder'], logging_configs['model_name']) + '.pth'
    log_folder = os.path.join(logging_configs['log_folder'], logging_configs['model_name'])

    os.makedirs(logging_configs['weight_folder'], exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, encoder, optimizer, loss_fn, epoch, device, beat_bounds, binary)
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

    checkpoint = { 
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    
    torch.save(checkpoint, model_path)

    return train_losses, val_losses

if __name__ == '__main__':

    '''
        Main function to train/evaluate arrhythmia detection models. 
        Can be used to train a specific model config using 5-fold validation and/or evaluate a specific model config
    '''

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str) # model config (in nested_configs folder, not the full path) 
    parser.add_argument("--model_name", type=str) # desired model name (s_{seed}_{model_name} will be saved)
    parser.add_argument("--plot_title", type=str, default=None) # display name for the ROC plot title (defaults to --model_name)
    parser.add_argument("--data_dir", type=str) # input data directory
    parser.add_argument("--output_dir", type=str) # output directory to store model weights/metrics csv/train plots

    parser.add_argument("--num_seeds", type=int, default=5) # how many seeds to train/evaluate model for
    parser.add_argument("--train", action='store_true') # whether or not to train the model first
    parser.add_argument("--use_best_model", action='store_true') # whether or not to use the saved hyperparams from the best performing model
    parser.add_argument("--weight_dir", type=str, default=None) # folder of pretrained s_{seed}_{model_name}.pth weights to evaluate (defaults to {output_dir}/model_weights)
    
    args = parser.parse_args()

    if args.use_best_model:
        config_key = 'xbest'
    else:
        config_key = 'x0'

    seeds = list(range(args.num_seeds))

    if args.train:
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

            logging_configs={
                'model_name': model_name,
                'weight_folder': f'{args.output_dir}/model_weights',
                'log_folder': f'{args.output_dir}/train_logs',
                'log_steps': 1
            }
            
            # --- Train the Model --- #
            encoder = Current_Encoder(train_mean, train_std)

            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
            if binary:
                loss_fn = nn.BCEWithLogitsLoss() 
            else:
                loss_fn = nn.CrossEntropyLoss()

            val_steps = 10

            print(f'Training Model {seed}')
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
                threshold = metrics.get_threshold(model, val_loader, encoder, device, beat_bounds=beat_bounds)

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
    

    # get model performance metrics
    print('Evaluating Models')

    with open(f'nested_configs/{args.config_file}', 'r') as file:
        model_configs = yaml.safe_load(file)

    binary = model_configs['kwargs'].get('binary', True)
    patient_norm = model_configs['kwargs'].get('patient_norm', False)

    # --- Create Datasets/DataLoaders --- #
    dataset_configs = {
        "data_file": f"{args.data_dir}/data.hdf5",
        "metadata": f"{args.data_dir}/data.csv",
        "train_prop": 0.6,
        "val_prop": 0.2,
        "test_prop": 0.2,
        "binary": binary,
        "random_state": 42,
        "balance": True,
        "fourier": False,
        "return_stats": patient_norm,
    }

    model_weights = [f's_{seed}_{args.model_name}' for seed in seeds]
    # allow evaluating a separate set of pretrained weights (e.g. remote_weights/model_weights)
    weight_folder = args.weight_dir if args.weight_dir is not None else f'{args.output_dir}/model_weights'
    results_folder = f'{args.output_dir}/model_results'

    os.makedirs(results_folder, exist_ok=True)

    # only meaningful for binary: multiclass predicts by argmax, with no threshold to fix
    fix_sensitivity = binary

    metrics_name = args.model_name

    if fix_sensitivity:
        metrics_name += '_fixed_sensitivity'
    elif not binary:
        metrics_name += '_multiclass'

    metrics_path = os.path.join(results_folder, 'metrics', metrics_name + '.csv')

    os.makedirs(results_folder, exist_ok=True)
    plots_dir = os.path.join(results_folder, 'plots')
    metrics_dir = os.path.join(results_folder, 'metrics')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    results_path = os.path.join(results_folder, args.model_name + '.csv')

    model = create_model(model_configs, config_key)

    model_vals = []
    roc_data = []

    for seed, model_weight in zip(seeds, model_weights):
        
        train_batch_size = 64
        eval_batch_size = 128

        dataset_configs['random_state'] = seed

        train_dataset, val_dataset, test_dataset = train_test_val_split(**dataset_configs)
        beat_bounds = get_beat_bounds(train_dataset)
        train_mean, train_std = get_dataset_statistics(train_dataset)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size)
        encoder = Current_Encoder(train_mean, train_std)

        model_path = os.path.join(weight_folder, model_weight) + '.pth'

        if not os.path.exists(model_path):
            continue

        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])            

        folder_name = f'{args.model_name}_seed_{seed}'

        if binary:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size)
            threshold = metrics.get_threshold(model, val_loader, encoder, fix_sensitivity=True, beat_bounds=beat_bounds)

            # only cache the threshold back into our own weights; leave external/reference weights untouched
            if args.weight_dir is None:
                checkpoint['threshold'] = threshold
                torch.save(checkpoint, model_path)

            auc, accuracy, sensitivity, specificity, ppv, npv, y_true, y_score = get_binary_metrics(model, test_loader, encoder, threshold, beat_bounds=beat_bounds)
            roc_data.append((y_true, y_score))

            if fix_sensitivity:
                folder_name += '_fixed_sensitivity'

            plot_dir = os.path.join(plots_dir, folder_name)
            os.makedirs(plot_dir, exist_ok=True)
            metrics.get_roc_curve(model, test_loader, encoder, threshold, plot_dir, 'cpu', beat_bounds=beat_bounds)

            model_vals.append(
                {
                    'auc': auc,
                    'accuracy': accuracy,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'ppv': ppv,
                    'npv': npv
                }
            )
        else:
            # no threshold to tune: the multiclass prediction is the argmax
            accuracy, macro_auc, weighted_auc, per_class, y_true, y_prob = get_multiclass_metrics(
                model, test_loader, encoder, 'cpu', beat_bounds=beat_bounds)
            roc_data.append((y_true, y_prob))

            plot_dir = os.path.join(plots_dir, folder_name)
            os.makedirs(plot_dir, exist_ok=True)
            metrics.get_multiclass_roc_curve(y_true, y_prob, plot_dir,
                                             class_names=AAMI_CLASS_NAMES)

            row = {
                'accuracy': accuracy,
                'macro_auc': macro_auc,
                'weighted_auc': weighted_auc,
            }

            # flatten the per-class dict into columns like 'S_recall', 'V_f1', ...
            for cls_name, cls_metrics in per_class.items():
                for metric_name, value in cls_metrics.items():
                    row[f'{cls_name}_{metric_name}'] = value

            model_vals.append(row)

    model_df = pd.DataFrame(model_vals)
    mean_vals = model_df.mean()
    std_vals = model_df.std()

    model_df.loc["mean"] = mean_vals
    model_df.loc["std"] = std_vals

    print(model_df)
    model_df.to_csv(metrics_path, index=True)

    # aggregate ROC across seeds: mean curve with a shaded 95% CI band
    mean_roc_name = args.model_name + '_mean_roc'

    if binary:
        if fix_sensitivity:
            mean_roc_name += '_fixed_sensitivity'
        mean_roc_path = os.path.join(plots_dir, mean_roc_name)
        metrics.plot_mean_roc_curve(roc_data, mean_roc_path,
                                    model_name=args.plot_title or args.model_name)
    else:
        mean_roc_path = os.path.join(plots_dir, mean_roc_name + '_multiclass')
        metrics.plot_mean_multiclass_roc_curve(roc_data, mean_roc_path,
                                               class_names=AAMI_CLASS_NAMES,
                                               model_name=args.plot_title or args.model_name)