import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from scipy import stats

import torch

import json
import os

# use validation set to compute the threshold (point closest to 0, 1) on roc curve
def get_threshold(model, val_loader, encoder, device='cpu', fix_sensitivity=False):
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for data, target in val_loader:

            if encoder is not None:
                data = encoder.encode(data)

            data, target = data.to(device), target.to(device)

            output, _ = model(data)
            output = output.sum(dim=(1,2))
    
            y_true.append(target[:, 0].numpy())
            y_pred.append(output.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    if fix_sensitivity:
        target_sens = 0.90
        valid_idxs = np.where(tpr >= target_sens)[0]
        best_idx = valid_idxs[np.argmin(fpr[valid_idxs])]
        threshold = thresholds[best_idx]
        return threshold

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

            output, _ = model(data)
            output = output.squeeze(-1).sum(dim=(1))

            if not binary:
                output = output.argmax(-1)
            else:
                output = (output > threshold).int()

            y_true.append(target[:, 0].numpy())
            y_pred.append(output.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    accuracy = (y_true == y_pred).astype(int).mean()
    
    return accuracy
    


def get_roc_curve(model, test_loader, encoder, threshold, log_dir, device='cpu', display=False):
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:

            if encoder is not None:
                data = encoder.encode(data)

            data, target = data.to(device), target.to(device)

            output, _ = model(data)
            output = output.squeeze(-1).sum(dim=(1))
    
            y_true.append(target[:, 0].numpy())
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

    if display == True:
        plt.show()
        
    y_pred = (y_pred > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues')

    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(log_dir, 'test_confusion_matrix.png'), dpi=300, bbox_inches='tight')

    if display == True:
        plt.show()


def plot_mean_roc_curve(roc_data, save_path, n_points=200, ci=0.95, display=False,
                        model_name=None):
    '''
    Plot a publication-ready mean ROC curve across seeds/folds.

    roc_data: list of (y_true, y_score) tuples, one per seed/fold
    save_path: full path (without extension) to save the figure; a .png (dpi=600,
               transparent) and a vector .pdf are written for publication use
    n_points: number of points on the common FPR grid used for interpolation
    ci: confidence level for the AUC interval quoted in the legend (default 0.95)
    model_name: used as the plot title; falls back to 'Mean ROC Curve'
    '''

    if len(roc_data) == 0:
        print('plot_mean_roc_curve: no ROC data provided, skipping plot')
        return

    # interpolate each fold's ROC onto a shared FPR grid so they can be averaged
    mean_fpr = np.linspace(0, 1, n_points)
    interp_tprs = []
    aucs = []

    for y_true, y_score in roc_data:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # force curve to start at the origin
        interp_tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    interp_tprs = np.stack(interp_tprs, axis=0)
    aucs = np.asarray(aucs)
    n_folds = interp_tprs.shape[0]

    mean_tpr = interp_tprs.mean(axis=0)
    mean_tpr[-1] = 1.0  # force curve to end at (1, 1)
    mean_auc = aucs.mean()

    # confidence interval on the mean AUC using a t-distribution (small-sample safe)
    if n_folds > 1:
        sem_auc = aucs.std(ddof=1) / np.sqrt(n_folds)
        auc_ci = stats.t.ppf(0.5 + ci / 2, df=n_folds - 1) * sem_auc
        auc_label = f'Mean ROC (AUC = {mean_auc:.3f} $\\pm$ {auc_ci:.3f})'
    else:
        auc_label = f'ROC (AUC = {mean_auc:.3f})'

    # --- publication-ready styling --- #
    with plt.rc_context({
        'font.size': 24,
        'axes.titlesize': 32,
        'axes.labelsize': 28,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 20,
        'axes.linewidth': 2.0,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'xtick.major.size': 7,
        'ytick.major.size': 7,
    }):
        fig, ax = plt.subplots(figsize=(9, 9), dpi=600)

        ax.plot(mean_fpr, mean_tpr, color='#1f4e79', lw=3.5, label=auc_label, zorder=3)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2.5,
                label='Chance', zorder=1)

        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(model_name if model_name else 'Mean ROC Curve', pad=15)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='lower right', frameon=False)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        fig.tight_layout()

        fig.savefig(save_path + '.png', dpi=600, transparent=True, bbox_inches='tight')
        fig.savefig(save_path + '.pdf', transparent=True, bbox_inches='tight')

        if display:
            plt.show()

        plt.close(fig)