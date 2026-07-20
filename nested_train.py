import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import snntorch.surrogate as surrogate

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dataloader import train_test_val_split, get_dataset_statistics

from models import create_model as build_model
from utils.encoding import Current_Encoder
from utils.masking import center_bounds
from utils.readout import prepare_loss_inputs, compute_auc
import utils.metrics as metrics

from nested.utils import Context, param_array_to_dict
from nested.optimize_utils import update_source_contexts


context = Context()

def create_model(param_vals):
    # delegate to the shared builder so the optimization and train/eval paths can never
    # disagree about how a model_type maps onto an architecture
    model_configs = {
        'kwargs': {
            'model_type': context.model_type,
            'neuron_type': context.neuron_type,
            'out_type': context.out_type,
            'bias': context.bias,
            'recurrent_dims': context.recurrent_dims,
            # readout width: 1 for binary, the class count for multiclass
            'out_dims': getattr(context, 'out_dims', 1),
        },
        'x0': param_vals,
    }

    return build_model(model_configs, 'x0')


def train_step(model, train_loader, encoder, optimizer, loss_fn, device, center_frac=1.0, binary=True):

    model.train()
    epoch_loss = 0

    for data, target in train_loader:
        optimizer.zero_grad()

        data = encoder.encode(data)
        data, target = data.to(device), target.to(device)

        output, _ = model(data)

        # supervise only the labeled (center) beat -- the flanking context is fed
        # to the model as input but carries the wrong label, see utils.masking
        start, end = center_bounds(output.shape[1], center_frac)
        output, target = output[:, start:end], target[:, start:end]

        output, target = prepare_loss_inputs(output, target, binary)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * data.shape[0]

    epoch_loss /= len(train_loader.dataset)

    return epoch_loss

def evaluate_model(model, test_loader, encoder, loss_fn, device, center_frac=1.0, binary=True):

    model.eval()
    test_loss = 0

    y_true = []
    y_score = []

    with torch.no_grad():
        for data, target in test_loader:
            data = encoder.encode(data)
            data, target = data.to(device), target.to(device)

            output, _ = model(data)

            start, end = center_bounds(output.shape[1], center_frac)
            output, target = output[:, start:end], target[:, start:end]

            # the label is constant across the window, so column 0 is the beat label
            y_true.append(target[:, 0].cpu().numpy())
            # sum logits over the supervised timesteps to get one score per sample
            y_score.append(output.sum(1).cpu().numpy())

            output, target = prepare_loss_inputs(output, target, binary)
            loss = loss_fn(output, target)

            test_loss += loss.item() * data.shape[0]

    test_loss /= len(test_loader.dataset)

    y_true = np.concatenate(y_true, axis=0)
    y_score = np.concatenate(y_score, axis=0)

    # macro one-vs-rest when multiclass, so the objective stays comparable
    test_auc = compute_auc(y_true, y_score, binary)

    return test_loss, test_auc

def get_random_seeds():
    return [list(range(context.num_seeds))]

def train_model(parameters, seed, model_id=None, export=False, plot=False):
    start = time.time()

    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    # torch.use_deterministic_algorithms(True)

    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['OMP_NUM_THREADS'] = '1'


    # set seeds and device #
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model #
    param_vals = param_array_to_dict(parameters, context.param_names) # get param vals
    model = create_model(param_vals)
    print(param_vals)
    sys.stdout.flush()
    # create dataloaders #
    binary = getattr(context, 'binary', True)

    dataset_configs = context.dataset_configs
    dataset_configs['binary'] = binary
    dataset_configs['random_state'] = seed

    train_dataset, val_dataset, _ = train_test_val_split(**dataset_configs)
    train_mean, train_std = get_dataset_statistics(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=context.train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=context.eval_batch_size)

    # get training params 
    encoder = Current_Encoder(train_mean, train_std)
    loss_fn = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()

    # fraction of the window that is supervised/read out, centered on the beat.
    # absent from a config (e.g. the 1s beat windows) means use the whole window.
    center_frac = getattr(context, 'center_frac', 1.0)

    if context.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=param_vals['lr'])
    elif context.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=param_vals['lr'])
    elif context.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=param_vals['lr'])

    for epoch in range(context.num_epochs):
        if context.disp:
            print('Training Seed {}, Epoch {}'.format(seed, epoch))
            sys.stdout.flush()
        train_step(model, train_loader, encoder, optimizer, loss_fn, device, center_frac, binary)

    val_loss, val_auc = evaluate_model(model, val_loader, encoder, loss_fn, device, center_frac, binary)

    end = time.time()
    print('Time Elapsed: {:.2f} minutes'.format((end - start) / 60))
    print('Val Loss: {}, Val {} AUC: {}'.format(
        val_loss, 'macro OVR' if not binary else 'binary', val_auc))

    # a NaN objective would silently corrupt the optimizer's ranking, so fold an
    # undefined AUC (too few classes present to score) into the worst value
    if np.isnan(val_auc):
        print('warning: val AUC undefined for seed {}, scoring it as chance'.format(seed))
        val_auc = 0.5

    if export == True:
        model_path = os.path.join(context.output_dir, f's_{seed}_optimized_{context.model_name}.pth')
        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        # multiclass predicts by argmax, so there is no operating threshold to cache
        if binary:
            checkpoint['threshold'] = metrics.get_threshold(
                model, val_loader, encoder, device, center_frac=center_frac)

        torch.save(checkpoint, model_path)


    return {'val_loss': val_loss, 'val_auc': 1 - val_auc} # return auc as a negative value (to optimize)

def combine_losses(primitives, current_features, model_id=None, export=False, plot=False):

    avg_loss = 0
    avg_auc = 0

    for p in primitives:
        avg_loss += p['val_loss']
        avg_auc += p['val_auc']
    
    avg_loss /= len(primitives)
    avg_auc /= len(primitives)

    return {'avg_val_loss': avg_loss, 'avg_val_auc': avg_auc}

def get_objectives(features, model_id=None, export=False, plot=False):
    return features, features
