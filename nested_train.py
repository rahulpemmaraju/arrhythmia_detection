import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import snntorch.surrogate as surrogate

import os
import random
import numpy as np
import pandas as pd

from dataloader import train_test_val_split, get_dataset_statistics

from models import ThreeLayer_HierarchicalSNN, TwoLayer_HierarchicalSNN, ThreeLayer_MultiscaleSNN, TwoLayer_MultiscaleSNN, ThreeLayer_GatedSNN, TwoLayer_GatedSNN, TwoLayer_SNN, OneLayer_SNN
from utils.encoding import Current_Encoder

from nested.utils import Context, param_array_to_dict
from nested.optimize_utils import update_source_contexts


context = Context()

def create_model(param_vals):

    if context.model_type in ['one_layer_snn']:
        neuron_options = {
            'alpha': param_vals['alpha_0'],
            'threshold': param_vals['threshold_0'],
            'spike_grad': surrogate.atan(alpha=2), 
            'linear_options':{'bias': context.bias}
        }

        if context.neuron_type == 'alif':
            neuron_options['beta'] = param_vals['beta_0']
            neuron_options['rho'] = param_vals['rho_0']
    
    elif context.model_type in ['two_layer_snn', 'two_layer_hsnn', 'two_layer_msnn', 'two_layer_gsnn']:
        neuron_options = [
            {
                'alpha': param_vals['alpha_0'],
                'threshold': param_vals['threshold_0'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': context.bias}
            },
            {
                'alpha': param_vals['alpha_1'],
                'threshold': param_vals['threshold_1'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': context.bias}
            }
        ]

        if context.neuron_type == 'alif':
            neuron_options[0]['beta'] = param_vals['beta_0']
            neuron_options[0]['rho'] = param_vals['rho_0']
            neuron_options[1]['beta'] = param_vals['beta_1']
            neuron_options[1]['rho'] = param_vals['rho_1']

    elif context.model_type in ['three_layer_hsnn', 'three_layer_msnn', 'three_layer_gsnn']:
        neuron_options = [
            {
                'alpha': param_vals['alpha_0'],
                'threshold': param_vals['threshold_0'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': context.bias}
            },
            {
                'alpha': param_vals['alpha_1'],
                'threshold': param_vals['threshold_1'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': context.bias}
            },
            {
                'alpha': param_vals['alpha_2'],
                'threshold': param_vals['threshold_2'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': context.bias}
            }

        ]

        if context.neuron_type == 'alif':
            neuron_options[0]['beta'] = param_vals['beta_0']
            neuron_options[0]['rho'] = param_vals['rho_0']
            neuron_options[1]['beta'] = param_vals['beta_1']
            neuron_options[1]['rho'] = param_vals['rho_1']
            neuron_options[2]['beta'] = param_vals['beta_2']
            neuron_options[2]['rho'] = param_vals['rho_2']


    out_options = {}

    if context.model_type not in ['three_layer_hsnn', 'three_layer_gsnn', 'three_layer_msnn']:
        if context.out_type == 'lif':
            out_options = {
                'alpha': param_vals['alpha_out'],
                'threshold': param_vals['threshold_out'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': context.bias}
            }

        elif context.out_type == 'alif':
            out_options = {
                'alpha': param_vals['alpha_out'],
                'beta': param_vals['beta_out'],
                'rho': param_vals['rho_out'],
                'threshold': param_vals['threshold_out'],
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{'bias': context.bias}
            }

    model_configs = {
            'in_dims': 1,
            'out_dims': 1,
            'recurrent_dims': context.recurrent_dims,
            'neuron_options': neuron_options,
            'neuron_type': context.neuron_type,
            'out_type': context.out_type,
            'out_options': out_options,
    }

    if context.model_type == 'one_layer_snn':
        return OneLayer_SNN(**model_configs)
    
    if context.model_type == 'two_layer_snn':
        return TwoLayer_SNN(**model_configs)
    
    if context.model_type == 'two_layer_hsnn':
        return TwoLayer_HierarchicalSNN(**model_configs)

    if context.model_type == 'two_layer_msnn':
        return TwoLayer_MultiscaleSNN(**model_configs)
    
    if context.model_type == 'two_layer_gsnn':
        return TwoLayer_GatedSNN(**model_configs)
    
    if context.model_type == 'three_layer_hsnn':
        return ThreeLayer_HierarchicalSNN(**model_configs)
    
    if context.model_type == 'three_layer_msnn':
        return ThreeLayer_MultiscaleSNN(**model_configs)
    
    if context.model_type == 'three_layer_gsnn':
        return ThreeLayer_GatedSNN(**model_configs)


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

    with torch.no_grad():
        for data, target in test_loader:
            data = encoder.encode(data)
            data, target = data.to(device), target.to(device)

            output, _ = model(data)
            output = output.squeeze(-1)
            
            loss = loss_fn(output, target)

            test_loss += loss.item() * data.shape[0]

    test_loss /= len(test_loader.dataset)

    return test_loss

def get_random_seeds():
    return [list(range(context.num_seeds))]

def train_model(parameters, seed, model_id=None, export=False, plot=False):

    # set seeds and device #
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model #
    param_vals = param_array_to_dict(parameters, context.param_names) # get param vals
    model = create_model(param_vals)

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


    # return best model loss
    min_val_loss = np.inf

    for epoch in range(context.num_epochs):
        train_loss = train_step(model, train_loader, encoder, optimizer, loss_fn, device)

        if (epoch + 1) % context.val_steps == 0:
            val_loss = evaluate_model(model, val_loader, encoder, loss_fn, device)

            if val_loss < min_val_loss:
                min_val_loss = val_loss

    return {'min_val_loss': min_val_loss}

def combine_losses(primitives, current_features, model_id=None, export=False, plot=False):
    avg_loss = sum([p['min_val_loss'] for p in primitives]) / len(primitives)
    return {'avg_min_val_loss': avg_loss}

def get_objectives(features, model_id=None, export=False, plot=False):
    return features, features