import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import snntorch.surrogate as surrogate

import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from dataloader import train_test_val_split, get_dataset_statistics
from models import SpikingNetwork, TwoLayerSNN, ThreeLayerSNN, TwoLayer_HierarchicalSNN, TwoLayer_MultiscaleSNN
from utils.encoding import Rate_Encoder, Current_Encoder

# get 5-fold validation model performance

def get_binary_metrics(model, test_loader, encoder, threshold, device='cpu'):
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:

            if encoder is not None:
                data = encoder.encode(data)

            data, target = data.to(device), target.to(device)

            output = model(data).squeeze(1)
    
            y_true.append(target.numpy())
            y_pred.append(output.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = (np.concatenate(y_pred, axis=0)  > threshold).astype(int)
    
    accuracy = (y_true == y_pred).mean()
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)

    return accuracy, sensitivity, specificity, ppv, npv


if __name__ == '__main__':

    random_seeds = [0, 1, 2, 3, 42]

    # --- Create Datasets/DataLoaders --- #
    binary = True

    dataset_configs = {
        "data_file": "/Users/rahul/Documents/G1/BrainInspiredComputing/TermProject/beat_neurokit_1.hdf5",
        "metadata": "/Users/rahul/Documents/G1/BrainInspiredComputing/TermProject/beat_neurokit_1.csv",
        "train_prop": 0.6,
        "val_prop": 0.2,
        "test_prop": 0.2,
        "binary": binary,
        "random_state": 42,
        "balance": True,
        "fourier": False,
    }

    train_batch_size = 64
    eval_batch_size = 128

    train_dataset, _, test_dataset = train_test_val_split(**dataset_configs)
    train_mean, train_std = get_dataset_statistics(train_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size)

    if binary:
        out_dims = 1
    else:
        out_dims = 5

    # --- Model Configs --- #
    model_configs = {
        'in_dims': 360,
        'out_dims': out_dims,
        'recurrent_dims': [180, 60],
        'out_act': 'none',
        'out_act_kwargs': {},
        'neuron_options': {
            'beta': 0.9,
            'threshold': 1.0,
            'spike_grad': surrogate.atan(alpha=2), 
            'linear_options':{
                'learning_rule': 'oja',
                'learning_rate': 1e-4,
                'bias': True
            }
        },
        'spike_accumulator': 'sum',
    }

    # model_name = 'two_layer_h_snn_binary_32_64_current_balanced'
    # model_weights = [f's_{seed}_{model_name}' for seed in random_seeds]
    model_name = 's_42_snn_binary_test'
    model_weights = [model_name]
    weight_folder = '../train_weights'
    results_folder = '../model_results'

    os.makedirs(results_folder, exist_ok=True)

    results_path = os.path.join(results_folder, model_name + '.csv')
    
    model = TwoLayerSNN(**model_configs)
    encoder = Current_Encoder(train_mean, train_std)
    timesteps = 20
    encoder = Rate_Encoder(timesteps)

    model_vals = []

    for model_weight in model_weights:

        model_path = os.path.join(weight_folder, model_weight) + '.pth'
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        threshold = checkpoint['threshold']

        accuracy, sensitivity, specificity, ppv, npv = get_binary_metrics(model, test_loader, encoder, threshold)

        model_vals.append(
            {
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv
            }
        )

    

    model_df = pd.DataFrame(model_vals)
    mean_vals = model_df.mean()
    std_vals = model_df.std()

    model_df.loc["mean"] = mean_vals
    model_df.loc["std"] = std_vals

    print(model_df)

    model_df.to_csv(results_path, index=True)
