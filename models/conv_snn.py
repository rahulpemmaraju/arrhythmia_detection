import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
import snntorch.surrogate as surrogate

from typing import Union, List, Dict

from .neurons import LIF, ALIF
from .hebbian_linear import HebbianLinearLayer

def get_activation_fn(act_name, act_kwargs):
    if act_name == 'relu':
        return nn.ReLU(**act_kwargs)
    elif act_name == 'softmax':
        return nn.Softmax(**act_kwargs)
    elif act_name == 'none':
        return nn.Identity()
    elif act_name == 'sigmoid':
        return nn.Sigmoid()
    
def get_neuron_models(neuron_name, **neuron_kwargs):
    if neuron_name == 'lif':
        return snn.LIF(**neuron_kwargs)


class SpikingCNN(nn.Module):
    def __init__(
            self, 
            in_size: int,
            in_dims: int, 
            out_dims: int,
            n_filters: List[int], 
            neuron_models: Union[str, List[str]] ='lif', 
            neuron_options: Union[Dict, List[Dict]] = {
                'beta': 0.9,
                'threshold': 1.0,
                'spike_grad': surrogate.atan(alpha=2),
            },
            conv_options: Union[Dict, List[Dict]] = {
                'kernel_size': 36,
                'stride': 1,
                'padding': 'same', 
                'dilation': 1,
                'groups': 1,
                'bias': True,
            },
            out_act: str = 'none',
            out_act_kwargs: dict = {},
            spike_accumulator: str = 'sum',
        ):

        '''
        initializes a basic spiking neural network for training (potentially with hebbian learning rules)
        network is structured as: linear -> spiking -> linear -> spiking ... -> output
        in_size: size of network input
        in_dims: number of channels in network input
        n_filters: list where each element refers to the output dimensions of each conv layer before spiking layer
        neuron_models: spiking model to use
        neuron_options: dictionary of options to pass to spiking layer. if list, will apply separately to each spiking layer
        linear_options: dictionary of options to pass to spiking layer. if list, will apply separately to each spiking layer
        out_act: output activation fn to apply
        spike_accumulator: either "sum" or "last" -> how to pass spikes for prediction
        '''
        
        super(SpikingCNN, self).__init__()

        if isinstance(neuron_models, str):
            neuron_models = [neuron_models for _ in n_filters]

        if isinstance(neuron_options, dict):
            neuron_options = [neuron_options for _ in n_filters]

        if isinstance(conv_options, dict):
            conv_options = [conv_options for _ in n_filters]

        self.neuron_models = neuron_models
        self.neuron_options = neuron_options
        self.linear_options = conv_options
        self.n_filters = n_filters

        self.conv_layers = nn.ModuleList([nn.Conv1d(in_dims, n_filters[0], **conv_options[0])])
        self.spiking_layers = nn.ModuleList([get_neuron_models(neuron_models[i], **neuron_options[i]) for i in range(len(neuron_models))])
        
        out_size = self._calculate_conv_size(in_size, conv_options[0])

        for i in range(len(conv_options) - 1):
            self.conv_layers.append(nn.Conv1d(n_filters[i], n_filters[i+1], **conv_options[i+1]))
            out_size = self._calculate_conv_size(out_size, conv_options[i+1])
        
        self.fc_out = nn.Sequential(
            nn.Linear(out_size * n_filters[-1], out_dims),
            get_activation_fn(out_act, out_act_kwargs)
        )

        self.accumulate = spike_accumulator

    def _calculate_conv_size(self, in_size, conv_options):
        # calculate the output size of a conv or conv-like operation (ex. max pool)

        padding = conv_options['padding']

        if padding == 'same':
            return in_size
        
        kernel_size = conv_options['kernel_size']
        stride = conv_options['stride']

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        out_size = (in_size[0] + 2*padding - kernel_size[0]) //stride + 1

        return out_size


    def reset_voltages(self, neuron_models, batch_dims, fc_dims):
        voltages = []

        for i, (neuron_name, out_dim) in enumerate(zip(neuron_models, fc_dims)):
            if neuron_name == 'lif_custom' or neuron_name == 'lif':
                voltages.append(torch.zeros(batch_dims, out_dim))
            elif neuron_name == 'alif_custom': 
                voltages.append((torch.zeros(batch_dims, out_dim), torch.zeros(batch_dims, out_dim)))

    
    def forward(self, x):
        # forward method: takes in an input [batch_size, num_timesteps, in_dims] and computes output over several timesteps

        B, T, C, L = x.shape 

        voltages = self.reset_voltages(self.neuron_models, B, [L * n for n in self.n_filters]) # initialize the voltages
        out_spikes = []
        out_volts = []

        for t in range(T):
            x_t = x[:, t, :, :]
            spike_out, volt_out, voltages = self.forward_timestep(x_t, voltages)
            spike_out = spike_out.reshape(spike_out.shape[0], -1)
            out_spikes.append(spike_out)
            out_volts.append(volt_out)

        if self.accumulate == 'sum':
            out_spikes = torch.stack(out_spikes).transpose(0, 1).mean(1)
            output = self.fc_out(out_spikes)
        else:
            output = self.fc_out(spike_out)
        
        return output

    def forward_timestep(self, x, in_voltages):
        # computes individual timestep output and returns output spiking and voltage
        out_voltages = []

        for (neuron_name, conv_layer, spike_layer, in_volt) in zip(self.neuron_models, self.conv_layers, self.spiking_layers, in_voltages):
            x = conv_layer(x)
            B, C, L = x.shape

            x = x.reshape(B, -1)
            if neuron_name == 'lif':
                x, v = spike_layer(in_volt, x)
                out_voltages.append(v)

            elif neuron_name == 'alif':
                x, v, a = spike_layer(*in_volt, x)
                out_voltages.append((v, a))

            x = x.reshape(B, C, L)

        return x, v, out_voltages

if __name__ == '__main__':
    net = SpikingCNN(64, 1, 1, [48, 64])
    x = torch.randn((5, 10, 1, 64))

    out = net(x)
    print(out.shape)
