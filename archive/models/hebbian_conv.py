import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math
from collections.abc import Iterable


def hebb_rule(x, y, w):
    with torch.no_grad():
        delta_w = y.transpose(1, 2) @ x
    return delta_w.mean(0)

def oja_rule(x, y, w):
    with torch.no_grad():
        delta_w = y.transpose(1, 2) @ x - (torch.pow(y, 2).unsqueeze(-1) * w).sum(1)
    return delta_w.mean(0)

def sanger_rule(x, y, w):
    with torch.no_grad():
        delta_w = y.transpose(1, 2) @ x - torch.tril(y.transpose(1, 2) @ y) @ w
    return delta_w.mean(0)

LEARNING_RULES = {
    'hebb': hebb_rule,
    'oja': oja_rule,
    'sanger': sanger_rule
}

class HebbianConv2dLayer(nn.Module):

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            bias=False, 
            learning_rule='oja',
            learning_rate=1e-4,
        ):

        super(HebbianConv2dLayer, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate

        self.w = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self._init_weights()

        self.learning_rule = LEARNING_RULES[learning_rule]
        self.register_buffer('dw', torch.zeros_like(self.w))
               
    def _init_weights(self):

        k = 6 / (self.in_channels * np.prod(self.kernel_size))
        bound = math.sqrt(k)
        
        nn.init.uniform_(self.w, -bound, bound)

    def forward(self, x):

        B, _, H, W = x.shape
        
        out_spatial_size = (
            (H + 2*self.padding - self.kernel_size[0])//self.stride + 1,
            (W + 2*self.padding - self.kernel_size[1])//self.stride + 1,
        )

        x_unf = F.unfold(x, self.kernel_size, stride=self.stride, padding=self.padding)
        w_unf = self.w.view(self.w.size(0), -1).t()
        y_unf = x_unf.transpose(1, 2).matmul(w_unf).transpose(1, 2)

        y = y_unf.view(B, self.out_channels, *out_spatial_size)

        return y


    def forward_hebbian(self, x):
        # runs forward method then computes hebbian weight changes -> adds to current weight change
        # x_unf: batch x [in_channels * window_size] x num_patches
        # y_unf: batch x out_channels x num_patches

        with torch.no_grad():
            B, _, H, W = x.shape
            
            out_spatial_size = (
                (H + 2*self.padding - self.kernel_size[0])//self.stride + 1,
                (W + 2*self.padding - self.kernel_size[1])//self.stride + 1,
            )

            x_unf = F.unfold(x, self.kernel_size, stride=self.stride, padding=self.padding).transpose(1, 2)
            w_unf = self.w.view(self.w.size(0), -1)

            y_unf = x_unf.matmul(w_unf.t())

            y = y_unf.transpose(1, 2).view(B, self.out_channels, *out_spatial_size)

            delta_w = self.learning_rule(x_unf, y_unf, w_unf).reshape(*self.w.shape)
            self.dw += delta_w # accumulate weight changes

        return y
    
    def apply_weight_change(self):

        # apply weight change
        with torch.no_grad():
            self.w += self.learning_rate * self.dw

        # reset accumulator
        self.dw = torch.zeros_like(self.w)
        

if __name__ == '__main__':

    x = torch.randn(3, 16, 28, 28)
    network = HebbianConv2dLayer(16, 32, 3, padding=1, learning_rule='sanger')
    network.apply_weight_change()
    print(network.forward_hebbian(x).shape)

