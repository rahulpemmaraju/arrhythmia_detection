import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay, roc_curve

import torch

import json
import os

def log_model(train_losses, val_losses, log_dir):
    
    train_epochs, train_vals = zip(*train_losses)
    val_epochs, val_vals = zip(*val_losses)

    # save the training and validation losses
    with open(os.path.join(log_dir, 'losses.json'), 'w') as f:
        json.dump(
            {
                'train_epochs': train_epochs,
                'train_losses': train_vals,
                'val_epochs': val_epochs,
                'val_losses': val_vals,
            },
        f)
    
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