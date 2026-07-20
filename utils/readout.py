import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score


def prepare_loss_inputs(output, target, binary):
    '''
    shape a model's (B, T, C) output and its (B, T) target for the loss.

    binary: BCEWithLogitsLoss wants a single logit per timestep, so C == 1 is
        squeezed away and the target stays float (B, T).
    multiclass: CrossEntropyLoss wants class logits on dim 1, i.e. (B, C, T)
        against a long target of (B, T). without the permute it would silently
        read the time axis as the class axis and train on nonsense.
    '''

    if binary:
        return output.squeeze(-1), target

    return output.permute(0, 2, 1), target.long()


def multiclass_auc(y_true, y_score, average='macro'):
    '''
    one-vs-rest AUC over class logits.

    a class with no positives in this split has no defined OVR curve. rather
    than let roc_auc_score raise and lose the whole run, those classes are
    dropped and the remaining probabilities renormalized.

    y_true: (N,) integer class labels
    y_score: (N, C) class logits (not probabilities)
    '''

    y_true = np.asarray(y_true)
    y_prob = softmax(y_score, axis=1)
    present = [c for c in range(y_prob.shape[1]) if (y_true == c).any()]

    if len(present) < 2:
        return float('nan')

    renorm = y_prob[:, present] / y_prob[:, present].sum(axis=1, keepdims=True)

    # with exactly two classes left this is a binary problem, and sklearn wants a
    # 1d score for the positive class rather than an (N, 2) probability matrix
    if len(present) == 2:
        return roc_auc_score((y_true == present[1]).astype(int), renorm[:, 1])

    return roc_auc_score(y_true, renorm, multi_class='ovr', average=average, labels=present)


def compute_auc(y_true, y_score, binary, average='macro'):
    '''
    AUC for either head, so the binary and multiclass paths report a comparable
    number. y_score is (N, 1) logits for binary, (N, C) for multiclass.
    '''

    if binary:
        return roc_auc_score(y_true, np.asarray(y_score).squeeze(-1))

    return multiclass_auc(y_true, y_score, average=average)
