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
from sklearn.metrics import roc_auc_score

from dataloader import train_test_val_split, get_dataset_statistics
from models import create_model
from utils.encoding import Rate_Encoder, Current_Encoder

from utils.logging import log_model
import utils.metrics as metrics

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

    os.makedirs(log_folder, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, encoder, optimizer, loss_fn, epoch, device)
        train_losses.append((epoch, train_loss))

        print('Epoch {} Average Loss: {}'.format(epoch, train_loss))

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

    for seed in [0, 1, 2, 3, 4]:

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

        model_name = f's_{seed}_optimized_sgd_twolayer_snn_lif'
        
        with open('nested_configs/twolayer_snn_32_64_lif_config.yaml', 'r') as file:
            model_configs = yaml.safe_load(file)
        
        model = create_model(model_configs)
        learning_rate = model_configs['x0']['lr']

        # --- Create Datasets/DataLoaders --- #
        dataset_configs = {
            "data_file": "/Users/rahul/Documents/Milstein_lab/arrhythmia_detection/beat_neurokit_1.hdf5",
            "metadata": "/Users/rahul/Documents/Milstein_lab/arrhythmia_detection/beat_neurokit_1.csv",
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
            'weight_folder': '../train_weights',
            'log_folder': '../train_logs',
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

        print('Training Model')

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
            checkpoint['threshold'] = threshold
            torch.save(checkpoint, model_path)


        accuracy = metrics.get_accuracy(model, test_loader, encoder, binary, threshold, device)
        print('\nTest Set Accuracy: {}'.format(accuracy))

        if binary:
            metrics.get_roc_curve(model, test_loader, encoder, threshold, os.path.join(logging_configs['log_folder'], logging_configs['model_name']), device)