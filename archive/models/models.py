import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from collections.abc import Iterable

from .hebbian_conv import HebbianConv2dLayer
from .hebbian_linear import HebbianLinearLayer

def get_activation_fn(act_name, **act_kwargs):
    if act_name == 'relu':
        return nn.ReLU(**act_kwargs)
    elif act_name == 'softmax':
        return nn.Softmax(**act_kwargs)
    elif act_name == 'none':
        return nn.Identity()

class HebbianConvNet(nn.Module):
    def __init__(
            self, 
            in_size: Iterable[int],
            in_channels: int, 
            n_filters: Iterable[int], 
            out_channels: int, 
            conv_options: dict = {
                'kernel_size': (3,3), 
                'stride': 1, 
                'padding': 0, 
                'bias': False, 
                'learning_rate': 1e-4
            },
            conv_act: str = 'relu',
            fc_options: dict = {
                'bias': False, 
                'learning_rate': 1e-4
            },
            fc_act: str = 'softmax',
            learning_rule: str = 'oja',
            
        ):

        '''
        initializes a basic convolutional neural network for training with hebbian learning rules
        network is structured as: conv -> conv -> ... -> maxpool2d -> linear
        in_size: size of input images
        in_channels: number of input channels
        n_filters: list where each element refers to the output channels of a conv layer
        out_channels: number of output units in a final linear layer
        conv_options: dictionary of options to pass to HebbianConv2d layer. if list, will apply separately to each conv layer
        conv_act: activation fn to use with convolution operators
        fc_options: dictionary of options to pass to HebbianLinear output layer
        fc_act: activation fn to use in output layer
        learning_rule: hebbian learning rule to use during hebbian-like learning
        '''

        super(HebbianConvNet, self).__init__()

        if isinstance(conv_options, dict):
            conv_options = [conv_options for _ in n_filters]


        self.conv_layers = nn.ModuleList([self._make_conv_layer(in_channels, n_filters[0], conv_options[0], conv_act, learning_rule)])
        out_size = self._calculate_conv_size(in_size, conv_options[0])

        for i in range(len(n_filters) - 1):
            self.conv_layers.append(
                self._make_conv_layer(n_filters[i], n_filters[i+1], conv_options[i+1], conv_act, learning_rule)
            )
            out_size = self._calculate_conv_size(out_size, conv_options[i+1])
            
        self.pool = nn.MaxPool2d(2, 2)
        out_size = self._calculate_conv_size(out_size, {'padding': 0, 'kernel_size': (2,2), 'stride': 2})
        
        self.fc = nn.ModuleList(
            [HebbianLinearLayer(math.prod(out_size) * n_filters[-1], out_channels, **fc_options),
            get_activation_fn(fc_act, dim=1)]
        )

    def _make_conv_layer(self, in_channels, out_channels, conv_options, conv_act, learning_rule):
        # creates a conv -> activation network
        return nn.ModuleList(
            [HebbianConv2dLayer(in_channels, out_channels, **conv_options, learning_rule=learning_rule),
            get_activation_fn(conv_act)]
        )
    
    def _calculate_conv_size(self, in_size, conv_options):
        # calculate the output size of a conv or conv-like operation (ex. max pool)

        padding = conv_options['padding']
        kernel_size = conv_options['kernel_size']
        stride = conv_options['stride']

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        out_size = (
            (in_size[0] + 2*padding - kernel_size[0])//stride + 1,
            (in_size[1] + 2*padding - kernel_size[1])//stride + 1,
        )

        return out_size

    def forward(self, x):
        for conv, act in self.conv_layers:
            x = conv(x)
            x = act(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.fc[0](x)
        x = self.fc[1](x)

        return x
    
    def forward_hebbian(self, x):
        for conv, act in self.conv_layers:
            x = conv.forward_hebbian(x)
            x = act(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        return x
    
    def apply_hebbian_updates(self):
        for conv, _ in self.conv_layers:
            conv.apply_weight_change()
    
if __name__ == '__main__':

    network = HebbianConvNet(
        in_size = (28, 28),
        in_channels = 1,
        n_filters = [16, 32],
        out_channels = 10,
        conv_options = {
                'kernel_size': (3,3), 
                'stride': 1, 
                'padding': 0, 
                'bias': False, 
                'learning_rate': 1e-4
        },
        conv_act = 'relu',
        fc_options = {
            'bias': False, 
            'learning_rate': 1e-4
        },
        fc_act = 'softmax',
        learning_rule = 'oja',
    )

    x = torch.randn(3, 1, 28, 28)

    y = network.forward_hebbian(x)
    network.apply_hebbian_updates()
    print(y.shape)
