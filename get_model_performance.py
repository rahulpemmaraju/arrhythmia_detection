import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import snntorch.surrogate as surrogate

import os
import yaml
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from dataloader import train_test_val_split, get_dataset_statistics
from models import create_model
from utils.encoding import Rate_Encoder, Current_Encoder
import utils.metrics as metrics


torch.set_num_threads(1)

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
             
            output, _ = model(data)
            output = output.sum(dim=(1,2))
    
            y_true.append(target[:, 0].numpy())
            y_pred.append(output.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    auc = roc_auc_score(y_true, y_pred)

    y_pred = (y_pred  > threshold).astype(int)
    
    accuracy = (y_true == y_pred).mean()
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)

    return auc, accuracy, sensitivity, specificity, ppv, npv


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--model_name", type=str)

    args = parser.parse_args()

    random_seeds = [0, 1, 2, 3, 4]

    # --- Create Datasets/DataLoaders --- #
    binary = True

    dataset_configs = {
        "data_file": "/scratch2/11354/rahulpemmaraju/arr_files/beat_neurokit_1.hdf5",
        "metadata": "/scratch2/11354/rahulpemmaraju/arr_files/beat_neurokit_1.csv",
        "train_prop": 0.6,
        "val_prop": 0.2,
        "test_prop": 0.2,
        "binary": binary,
        "random_state": 42,
        "balance": True,
        "fourier": False,
    }

    if binary:
        out_dims = 1
    else:
        out_dims = 5


    model_name = args.model_name
    with open(args.config_file, 'r') as file:
        model_configs = yaml.safe_load(file)
    model_weights = [f's_{seed}_{model_name}' for seed in random_seeds]
    weight_folder = '/scratch2/11354/rahulpemmaraju/arr_files/model_weights'
    results_folder = '/scratch2/11354/rahulpemmaraju/arr_files/model_results'

    os.makedirs(results_folder, exist_ok=True)

    results_path = os.path.join(results_folder, model_name + '.csv')
            
    model = create_model(model_configs)
    

    model_vals = []

    for seed, model_weight in zip(random_seeds, model_weights):
        train_batch_size = 64
        eval_batch_size = 128

        dataset_configs['random_state'] = seed

        train_dataset, val_dataset, test_dataset = train_test_val_split(**dataset_configs)
        train_mean, train_std = get_dataset_statistics(train_dataset)
        print(train_mean, train_std)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size)
        encoder = Current_Encoder(train_mean, train_std)
        
        if not os.path.exists(os.path.join(weight_folder, model_weight) + '.pth'):
            continue

        model_path = os.path.join(weight_folder, model_weight) + '.pth'
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])

        if 'threshold' not in checkpoint.keys():
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size)
            threshold = metrics.get_threshold(model, val_loader, encoder, 'cpu')
            checkpoint['threshold'] = threshold
            torch.save(checkpoint, model_path)

        threshold = checkpoint['threshold']
        auc, accuracy, sensitivity, specificity, ppv, npv = get_binary_metrics(model, test_loader, encoder, threshold)

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

    

    model_df = pd.DataFrame(model_vals)
    mean_vals = model_df.mean()
    std_vals = model_df.std()

    model_df.loc["mean"] = mean_vals
    model_df.loc["std"] = std_vals

    print(model_df)

    model_df.to_csv(results_path, index=True)
