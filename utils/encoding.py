# Utility functions for converting continuous input into spikes using different strategies
import torch

# basic rate encoding
class Rate_Encoder():
    def __init__(self, timesteps):
        self.timesteps = timesteps

    def encode(self, x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-8) # normalize to 0 -> 1
        out = torch.stack([torch.greater(x, torch.rand_like(x)) for _ in range(self.timesteps)]).transpose(0, 1).float()
        return out
    
# basic rate encoding plus reshaping
class Rate_Encoder_CNN():
    def __init__(self, timesteps):
        self.timesteps = timesteps

    def encode(self, x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-8) # normalize to 0 -> 1
        x = x.unsqueeze(1)
        out = torch.stack([torch.greater(x, torch.rand_like(x)) for _ in range(self.timesteps)]).transpose(0, 1).float()
        return out
    
# treat the ecg signal as a continuous current -> normalize using z-score
class Current_Encoder():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def encode(self, x):
        x = (x - self.mean) / (self.std) # z-score normalization
        out = x.unsqueeze(-1)
        return out