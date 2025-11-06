import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math
from collections.abc import Iterable


def hebb_rule(x, y, w):
    with torch.no_grad():
        delta_w = y.T @ x / x.shape[0]
    return delta_w

def oja_rule(x, y, w):
    with torch.no_grad():
        delta_w = y.T @ x / x.shape[0]
        delta_w -= (torch.pow(y, 2).mean(0).unsqueeze(-1) * w)
    return delta_w

def sanger_rule(x, y, w):
    with torch.no_grad():
        delta_w = y.T @ x - torch.tril(y.T @ y) @ w
    return delta_w / x.shape[0]

LEARNING_RULES = {
    'hebb': hebb_rule,
    'oja': oja_rule,
    'sanger': sanger_rule,
}

class HebbianLinearLayer(nn.Module):

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            bias=False, 
            learning_rule='oja',
            learning_rate=1e-4,
        ):

        super(HebbianLinearLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate

        self.w = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = bias

        if self.bias == True:
            self.w = nn.Parameter(torch.Tensor(out_channels, in_channels + 1))

        self._init_weights()

        self.learning_rule = LEARNING_RULES[learning_rule]
        self.register_buffer('dw', torch.zeros_like(self.w))
               
    def _init_weights(self):
        
        in_dim = self.in_channels + 1 if self.bias else self.in_channels
        k = 6 / in_dim
        bound = math.sqrt(k)
        
        nn.init.uniform_(self.w, -bound, bound)

    def forward(self, x):
        if self.bias:
            x = torch.cat((x, torch.ones((x.shape[0], 1), device=x.device)), dim=1)
        
        y = x @ self.w.t()

        return y


    def forward_hebbian(self, x):
        # runs forward method then computes hebbian weight changes -> adds to current weight change
        # augment the input matrix and weight matrix to take the bias into account
        # w: out_channels x in_channels
        # x: batch x in_channels
        # y: batch x out_channels

        with torch.no_grad():

            if self.bias:
                x = torch.cat((x, torch.ones((x.shape[0], 1), device=x.device)), dim=1)
            
            y = x @ self.w.t()
                
            delta_w = self.learning_rule(x, y, self.w)
            self.dw += delta_w

        return y
    
    def apply_weight_change(self):

        # apply weight change
        with torch.no_grad():
            self.w += self.learning_rate * self.dw

        # reset accumulator
        self.dw = torch.zeros_like(self.w)
    
if __name__ == '__main__':
    x = torch.randn(3, 16)
    network = HebbianLinearLayer(16, 32, bias=True, learning_rule='sanger')
    
    print(network.forward_hebbian(x).shape)