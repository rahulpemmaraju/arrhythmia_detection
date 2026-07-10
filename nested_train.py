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
        },
        'x0': param_vals,
    }

    return build_model(model_configs, 'x0')


def train_step(model, train_loader, encoder, optimizer, loss_fn, device):

    model.train()
    epoch_loss = 0

    for data, target in train_loader: 
        optimizer.zero_grad()
       
        data = encoder.encode(data)
        data, target = data.to(device), target.to(device)

        output, _ = model(data)
        output = output.squeeze(-1)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * data.shape[0]

    epoch_loss /= len(train_loader.dataset)

    return epoch_loss

def evaluate_model(model, test_loader, encoder, loss_fn, device):

    model.eval()
    test_loss = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in test_loader:
            data = encoder.encode(data)
            data, target = data.to(device), target.to(device)

            output, _ = model(data)
            output = output.squeeze(-1)
            
            loss = loss_fn(output, target)

            y_true.append(target[:, 0].cpu().numpy())
            y_pred.append(output.sum(1).cpu().numpy())

            test_loss += loss.item() * data.shape[0]

    test_loss /= len(test_loader.dataset)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    test_auc = roc_auc_score(y_true, y_pred)

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
    dataset_configs = context.dataset_configs
    dataset_configs['binary'] = True
    dataset_configs['random_state'] = seed

    train_dataset, val_dataset, _ = train_test_val_split(**dataset_configs)
    train_mean, train_std = get_dataset_statistics(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=context.train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=context.eval_batch_size)

    # get training params 
    encoder = Current_Encoder(train_mean, train_std)
    loss_fn = nn.BCEWithLogitsLoss()

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
        train_step(model, train_loader, encoder, optimizer, loss_fn, device)

    val_loss, val_auc = evaluate_model(model, val_loader, encoder, loss_fn, device)

    end = time.time()
    print('Time Elapsed: {:.2f} minutes'.format((end - start) / 60))
    print('Val Loss: {}, Val AUC: {}'.format(val_loss, val_auc))

    if export == True:
        model_path = os.path.join(context.output_dir, f's_{seed}_optimized_{context.model_name}.pth')
        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        threshold = metrics.get_threshold(model, val_loader, encoder, device)
        checkpoint['threshold'] = threshold
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
