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
from sklearn.metrics import confusion_matrix, roc_auc_score


from dataloader import train_test_val_split, get_dataset_statistics
from models import create_model
from utils.encoding import Rate_Encoder, Current_Encoder

from utils.logging import log_model
import utils.metrics as metrics

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


def train_step(model, train_loader, encoder, optimizer, loss_fn, epoch, device):

    model.train()
    epoch_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader): 
        optimizer.zero_grad()
       
        data = encoder.encode(data)
        data, target = data.to(device), target.to(device)

        output, _ = model(data)
        output = output.squeeze(-1)

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


    auc = roc_auc_score(y_true, y_pred)

    print('\nValidation set: Average loss: {:.4f}, AUC: {:.4f}\n'.format(test_loss, auc))

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
    logging_configs: information for saving weights and logging model output

    '''

    train_losses = []
    val_losses = []

    model_path = os.path.join(logging_configs['weight_folder'], logging_configs['model_name']) + '.pth'
    log_folder = os.path.join(logging_configs['log_folder'], logging_configs['model_name'])

    os.makedirs(logging_configs['weight_folder'], exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, encoder, optimizer, loss_fn, epoch, device)
        train_losses.append((epoch, train_loss))

        print('Epoch {} Average Loss: {}'.format(epoch, train_loss))
        sys.stdout.flush()

        if (epoch + 1) % val_steps == 0:
            val_loss, val_auc = evaluate_model(model, val_loader, encoder, loss_fn, device)
            val_losses.append((epoch, val_loss))
            log_model(train_losses, val_losses, log_folder)

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
    parser.add_argument("--data_dir", type=str) # input data directory
    parser.add_argument("--output_dir", type=str) # output directory to store model weights/metrics csv/train plots

    parser.add_argument("--num_seeds", type=int, default=5) # how many seeds to train/evaluate model for
    parser.add_argument("--train", action='store_true') # whether or not to train the model first
    parser.add_argument("--use_best_model", action='store_true') # whether or not to use the saved hyperparams from the best performing model
    
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

            binary = True

            if binary:
                out_dims = 1
            else:
                out_dims = 5

            num_epochs = 10

            model_name = f's_{seed}_{args.model_name}'
            
            with open(f'nested_configs/{args.config_file}', 'r') as file:
                model_configs = yaml.safe_load(file)
            
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
            }

            train_batch_size = 64
            eval_batch_size = 128

            train_dataset, val_dataset, test_dataset = train_test_val_split(**dataset_configs)

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
                logging_configs,
            )

            model_path = os.path.join(logging_configs['weight_folder'], logging_configs['model_name']) + '.pth'
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model'])

            threshold = None
            if binary:
                threshold = metrics.get_threshold(model, val_loader, encoder, device)

            accuracy = metrics.get_accuracy(model, test_loader, encoder, binary, threshold, device)
            print('\nTest Set Accuracy: {}'.format(accuracy))

            if binary:
                metrics.get_roc_curve(model, test_loader, encoder, threshold, os.path.join(logging_configs['log_folder'], logging_configs['model_name']), device)
    

    # get model performance metrics
    print('Evaluating Models')

    # --- Create Datasets/DataLoaders --- #
    dataset_configs = {
        "data_file": f"{args.data_dir}/data.hdf5",
        "metadata": f"{args.data_dir}/data.csv",
        "train_prop": 0.6,
        "val_prop": 0.2,
        "test_prop": 0.2,
        "binary": True,
        "random_state": 42,
        "balance": True,
        "fourier": False,
    }

    out_dims = 1


    with open(f'nested_configs/{args.config_file}', 'r') as file:
        model_configs = yaml.safe_load(file)

    model_weights = [f's_{seed}_{args.model_name}' for seed in seeds]
    weight_folder = f'{args.output_dir}/model_weights'
    results_folder = f'{args.output_dir}/model_results'

    os.makedirs(results_folder, exist_ok=True)

    fix_sensitivity = True

    metrics_path = os.path.join(results_folder, 'metrics', args.model_name + '.csv')

    if fix_sensitivity:
        metrics_path = os.path.join(results_folder, 'metrics', args.model_name + '_fixed_sensitivity.csv')

    os.makedirs(results_folder, exist_ok=True)
    plots_dir = os.path.join(results_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    results_path = os.path.join(results_folder, args.model_name + '.csv')

    model = create_model(model_configs)

    model_vals = []

    for seed, model_weight in zip(random_seeds, model_weights):
        
        train_batch_size = 64
        eval_batch_size = 128

        dataset_configs['random_state'] = seed

        train_dataset, val_dataset, test_dataset = train_test_val_split(**dataset_configs)
        train_mean, train_std = get_dataset_statistics(train_dataset)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size)
        encoder = Current_Encoder(train_mean, train_std)

        model_path = os.path.join(weight_folder, model_weight) + '.pth'

        if not os.path.exists(model_path):
            continue

        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])            

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size)
        threshold = metrics.get_threshold(model, val_loader, encoder, fix_sensitivity=True)
        checkpoint['threshold'] = threshold
        torch.save(checkpoint, model_path)

        auc, accuracy, sensitivity, specificity, ppv, npv = get_binary_metrics(model, test_loader, encoder, threshold)
        metrics.get_roc_curve(model, test_loader, encoder, threshold, plots_dir, 'cpu', model_weight)

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
    model_df.to_csv(metrics_path, index=True)