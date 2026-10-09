# Utility functions for feeding the ECG to the models
import torch


def unpack_batch(batch):
    '''
    (data, target, *stats) from a loader batch.

    a dataset built with return_stats=True yields 4-tuples instead of 2-tuples,
    so every eval loop would otherwise need to know which kind it was handed.
    stats is empty for the plain case and splats straight into encoder.encode().
    '''

    return batch[0], batch[1], tuple(batch[2:])

# treat the ecg signal as a continuous current -> normalize using z-score
class Current_Encoder():
    def __init__(self, mean, std):
        # fallback statistics, used when encode() is called without any. these
        # are global (one scalar pair for the whole training set), which leaves
        # each patient's amplitude and offset in the signal.
        self.mean = mean
        self.std = std

    def encode(self, x, mean=None, std=None):
        '''
        z-score the waveform and add the channel axis.

        x: (B, T) waveform batch
        mean/std: optional per-sample statistics, shape (B,) -- pass the
            patient_mean/patient_std columns to normalize each beat by its own
            record instead of by one global constant. omitting them keeps the
            original global behaviour, so existing callers are unaffected.
        '''

        if mean is None:
            mean, std = self.mean, self.std
        elif torch.is_tensor(mean):
            # (B,) -> (B, 1) so it broadcasts across time rather than across the
            # batch, which would silently normalize each sample by another's stats
            mean = mean.reshape(-1, *([1] * (x.dim() - 1))).to(x.dtype)
            std = std.reshape(-1, *([1] * (x.dim() - 1))).to(x.dtype)

        x = (x - mean) / std # z-score normalization
        out = x.unsqueeze(-1)
        return out