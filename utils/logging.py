import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay, roc_curve

import torch

import json
import os

def log_model(train_losses, val_losses, log_dir, train_aucs=None, val_aucs=None):
    # train_aucs/val_aucs are optional lists of (epoch, auc). when given, they
    # are written to losses.json and plotted to aucs.png alongside the losses.

    train_epochs, train_vals = zip(*train_losses)
    val_epochs, val_vals = zip(*val_losses)

    log = {
        'train_epochs': train_epochs,
        'train_losses': train_vals,
        'val_epochs': val_epochs,
        'val_losses': val_vals,
    }

    if train_aucs:
        auc_epochs, train_auc_vals = zip(*train_aucs)
        log['auc_epochs'] = auc_epochs
        log['train_aucs'] = train_auc_vals

    if val_aucs:
        auc_epochs, val_auc_vals = zip(*val_aucs)
        log['auc_epochs'] = auc_epochs
        log['val_aucs'] = val_auc_vals

    # save the training and validation losses
    with open(os.path.join(log_dir, 'losses.json'), 'w') as f:
        json.dump(log, f)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(train_epochs, train_vals, label="Train Loss", linewidth=2)
    ax.plot(val_epochs, val_vals, label="Validation Loss", linewidth=2)

    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14)

    # Grid, legend, and tight layout
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()

    # Save to file without showing
    fig.savefig(os.path.join(log_dir, 'losses.png'), dpi=300)  # high resolution
    plt.close(fig)  # close the figure to free memory

    if not (train_aucs or val_aucs):
        return

    # the gap between these two curves is the thing to read: both low means
    # underfitting, a wide gap means the model is fitting the training patients
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    if train_aucs:
        ax.plot(*zip(*train_aucs), label="Train AUC", linewidth=2, marker='o')
    if val_aucs:
        ax.plot(*zip(*val_aucs), label="Validation AUC", linewidth=2, marker='o')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Chance')

    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Training vs Validation AUC", fontsize=14)
    ax.set_ylim(0.4, 1.0)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()

    fig.savefig(os.path.join(log_dir, 'aucs.png'), dpi=300)
    plt.close(fig)