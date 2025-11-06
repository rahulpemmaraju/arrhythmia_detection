import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay

import torch

import json
import os

# use validation set to compute the threshold (point closest to 0, 1) on roc curve
def get_threshold(model, val_loader, encoder, device='cpu'):
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for data, target in val_loader:

            if encoder is not None:
                data = encoder.encode(data)

            data, target = data.to(device), target.to(device)

            output = model(data).squeeze(1)
    
            y_true.append(target.numpy())
            y_pred.append(output.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    points = zip(fpr, tpr)
    distances = [x[0]**2 + (1 - x[1]) ** 2 for x in points]
    operating_point = np.argmin(distances)

    return thresholds[operating_point]

def get_accuracy(model, test_loader, encoder, binary=True, threshold=None, device='cpu'):   
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:

            if encoder is not None:
                data = encoder.encode(data)

            data, target = data.to(device), target.to(device)

            output = model(data).squeeze(1)

            if not binary:
                output = output.argmax(-1)
            else:
                output = (output > threshold).int()

            y_true.append(target.numpy())
            y_pred.append(output.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    accuracy = (y_true == y_pred).astype(int).mean()
    
    return accuracy
    


def get_roc_curve(model, test_loader, encoder, threshold, log_dir, device='cpu'):
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
    y_pred = np.concatenate(y_pred, axis=0)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    ax.plot(fpr, tpr, label='ROC Curve')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()

    fig.savefig(os.path.join(log_dir, 'test_roc.png'), dpi=300)  # high resolution
    plt.show()

    y_pred = (y_pred > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues')

    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(log_dir, 'test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
