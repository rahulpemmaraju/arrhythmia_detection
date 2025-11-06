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
        return snn.Leaky(**neuron_kwargs)


class SpikingNetwork(nn.Module):
    def __init__(
            self, 
            in_dims: int, 
            out_dims: int,
            fc_dims: List[int], 
            neuron_models: Union[str, List[str]] ='lif', 
            neuron_options: Union[Dict, List[Dict]] = {
                'beta': 0.9,
                'threshold': 1.0,
                'spike_grad': surrogate.atan(alpha=2),
            },
            linear_options: Union[Dict, List[Dict]] = {
                'bias': True, 
            },
            out_act: str = 'none',
            out_act_kwargs: dict = {},
            spike_accumulator: str = 'sum',
        ):

        '''
        initializes a basic spiking neural network for training (potentially with hebbian learning rules)
        network is structured as: linear -> spiking -> linear -> spiking ... -> output
        in_dims: size of network input
        fc_dims: list where each element refers to the output dimensions of each linear layer before spiking layer
        neuron_models: spiking model to use
        neuron_options: dictionary of options to pass to spiking layer. if list, will apply separately to each spiking layer
        linear_options: dictionary of options to pass to spiking layer. if list, will apply separately to each spiking layer
        out_act: output activation fn to apply
        spike_accumulator: either "sum" or "last" -> how to pass spikes for prediction
        '''
        
        super(SpikingNetwork, self).__init__()


        if isinstance(neuron_models, str):
            neuron_models = [neuron_models for _ in fc_dims]

        if isinstance(neuron_options, dict):
            neuron_options = [neuron_options for _ in fc_dims]

        if isinstance(linear_options, dict):
            linear_options = [linear_options for _ in fc_dims]

        self.neuron_models = neuron_models
        self.neuron_options = neuron_options
        self.linear_options = linear_options
        self.fc_dims = fc_dims

        self.linear_layers = nn.ModuleList([HebbianLinearLayer(in_dims, fc_dims[0], **linear_options[0])])
        self.spiking_layers = nn.ModuleList([get_neuron_models(neuron_models[i], **neuron_options[i]) for i in range(len(neuron_models))])
        
        for i in range(len(fc_dims) - 1):
            self.linear_layers.append(HebbianLinearLayer(fc_dims[i], fc_dims[i+1], **linear_options[i+1]))
        
        self.fc_out = nn.Sequential(
            nn.Linear(fc_dims[-1], out_dims),
            get_activation_fn(out_act, out_act_kwargs)
        )

        self.accumulate = spike_accumulator

    def reset_voltages(self, neuron_models, batch_dims, fc_dims):
        voltages = []

        for i, (neuron_name, out_dim) in enumerate(zip(neuron_models, fc_dims)):
            if neuron_name == 'lif_custom' or neuron_name == 'lif':
                voltages.append(torch.zeros(batch_dims, out_dim))
            elif neuron_name == 'alif_custom': 
                voltages.append((torch.zeros(batch_dims, out_dim), torch.zeros(batch_dims, out_dim)))
                
        return voltages
    
    def forward(self, x):
        # forward method: takes in an input [batch_size, num_timesteps, in_dims] and computes output over several timesteps

        B, T, I = x.shape

        voltages = self.reset_voltages(self.neuron_models, B, self.fc_dims) # initialize the voltages
        out_spikes = []
        out_volts = []

        for t in range(T):
            x_t = x[:, t, :]
            spike_out, volt_out, voltages = self.forward_timestep(x_t, voltages)
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

        for (neuron_name, lin_layer, spike_layer, in_volt) in zip(self.neuron_models, self.linear_layers, self.spiking_layers, in_voltages):
            x = lin_layer(x)
            if neuron_name == 'lif':
                x, v = spike_layer(in_volt, x)
                out_voltages.append(v)

            elif neuron_name == 'alif':
                x, v, a = spike_layer(*in_volt, x)
                out_voltages.append((v, a))

        return x, v, out_voltages

if __name__ == '__main__':
    net = SpikingNetwork(32, [48, 64])
    x = torch.randn((5, 10, 32))

    s, v = net(x)
