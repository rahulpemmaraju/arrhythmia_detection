import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
import snntorch.surrogate as surrogate

from typing import Union, List, Dict

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

'''
LIF Layer
v[t] = beta * v[t-1] + W_recurrent * z[t-1] + W_in * x[t] - z[t-1] * v_thresh
z[t] = (v[t] - v_thresh) > 0
'''
class SNN_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims, 
            beta = 0.9, 
            threshold = 1.0, 
            spike_grad = surrogate.atan(alpha=2), 
            linear_options = {
                'learning_rule': 'oja',
                'learning_rate': 1e-4,
                'bias': True
        }):
        
        '''
        in_dims: shape of input to layer
        out_dims: shape of output of layer
        beta: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        linear_options: options to pass to projection layers
        '''

        super(SNN_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.beta = beta
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.x_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # input to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, **linear_options) # spike to voltage

    def forward(self, x_in, v_in, z_in):
        v_out = self.beta * v_in + self.z_v(z_in) + self.x_v(x_in) - (z_in * self.threshold).detach()
        z_out = self.spike_grad(v_out - self.threshold)

        return v_out, z_out
    
    def forward_hebbian(self, x_in, v_in, z_in):
        with torch.no_grad():
            v_out = self.beta * v_in + self.z_v.forward_hebbian(z_in) + self.x_v.forward_hebbian(x_in) - (z_in * self.threshold).detach()
            z_out = self.spike_grad(v_out - self.threshold)

        return v_out, z_out
    
    def apply_weight_change(self):
        self.z_v.apply_weight_change()
        self.x_v.apply_weight_change()
    
# Basic 2-Layer SNN Using LIF Neurons
class TwoLayerSNN(nn.Module):

    def __init__(
            self,
            in_dims: int,
            out_dims: int,
            recurrent_dims: List[int],
            out_act: str = 'none',
            out_act_kwargs: dict = {},
            neuron_options: dict = {
                'beta': 0.9,
                'threshold': 1.0,
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{
                    'learning_rule': 'oja',
                    'learning_rate': 1e-4,
                    'bias': True
                }
            },
            spike_accumulator: str = 'sum',
        ):

        '''
        in_dims: input dimensions
        out_dims: output dimensions
        recurrent_dimensions: list of length two with output dimensions of recurrent layers
        out_act: activation fn for output layer
        out_act_kwargs: kwargs to pass to output layer
        neuron_options: kwargs to pass to SNN_Cell
        spike_accumulator: either "sum" or "last" -> how to pass spikes to prediction layer
        '''

        super(TwoLayerSNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.accumulate = spike_accumulator

        self.snn_1 = SNN_Cell(self.in_dims, self.recurrent_dims[0], **neuron_options)
        self.snn_2 = SNN_Cell(self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options)

        self.fc = HebbianLinearLayer(self.recurrent_dims[1], self.out_dims, **neuron_options['linear_options'])
        self.fc_act = get_activation_fn(out_act, out_act_kwargs)

    def forward(self, x):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])

        out_spikes = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0 = self.snn_1(x_t, v_0, z_0) # output spikes are inputs to next layer
            v_1, z_1 = self.snn_2(z_0, v_1, z_1)

            out_spikes.append(z_1)
        
        if self.accumulate == 'sum':
            out_spikes = torch.stack(out_spikes).transpose(0, 1).mean(1)
            output = self.fc(out_spikes)
            output = self.fc_act(output)
        else:
            output = self.fc(z_1)
            output = self.fc_act(output)

        return output
    
    def forward_hebbian(self, x):
        # x shape: B x T x IN_DIMS

        with torch.no_grad():
            B, T, _ = x.shape

            v_0 = torch.zeros(B, self.recurrent_dims[0])
            z_0 = torch.zeros(B, self.recurrent_dims[0])

            v_1 = torch.zeros(B, self.recurrent_dims[1])
            z_1 = torch.zeros(B, self.recurrent_dims[1])

            out_spikes = []

            for t in range(T):
                x_t = x[:, t, :]

                v_0, z_0 = self.snn_1.forward_hebbian(x_t, v_0, z_0) # output spikes are inputs to next layer
                v_1, z_1 = self.snn_2.forward_hebbian(z_0, v_1, z_1)

                out_spikes.append(z_1)
            
            if self.accumulate == 'sum':
                out_spikes = torch.stack(out_spikes).transpose(0, 1).mean(1)
                output = self.fc.forward_hebbian(out_spikes)
                output = self.fc_act(output)
            else:
                output = self.fc.forward_hebbian(z_1)
                output = self.fc_act(output)

            return output
        
    def apply_weight_change(self):
        self.snn_1.apply_weight_change()
        self.snn_2.apply_weight_change()
        self.fc.apply_weight_change()

    
# 3-Layer SNN Using LIF Neurons
class ThreeLayerSNN(nn.Module):

    def __init__(
            self,
            in_dims: int,
            out_dims: int,
            recurrent_dims: List[int],
            out_act: str = 'none',
            out_act_kwargs: dict = {},
            neuron_options: dict = {
                'beta': 0.9,
                'threshold': 1.0,
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{
                    'learning_rule': 'oja',
                    'learning_rate': 1e-4,
                    'bias': True
                }
            },
            spike_accumulator: str = 'sum',
        ):

        '''
        in_dims: input dimensions
        out_dims: output dimensions
        recurrent_dimensions: list of length two with output dimensions of recurrent layers
        out_act: activation fn for output layer
        out_act_kwargs: kwargs to pass to output layer
        neuron_options: kwargs to pass to SNN_Cell
        spike_accumulator: either "sum" or "last" -> how to pass spikes to prediction layer
        '''

        super(ThreeLayerSNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.accumulate = spike_accumulator

        self.snn_1 = SNN_Cell(self.in_dims, self.recurrent_dims[0], **neuron_options)
        self.snn_2 = SNN_Cell(self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options)
        self.snn_3 = SNN_Cell(self.recurrent_dims[1], self.recurrent_dims[2], **neuron_options)

        self.fc = HebbianLinearLayer(self.recurrent_dims[2], self.out_dims, **neuron_options['linear_options'])
        self.fc_act = get_activation_fn(out_act, out_act_kwargs)

    def forward(self, x):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])

        v_2 = torch.zeros(B, self.recurrent_dims[2])
        z_2 = torch.zeros(B, self.recurrent_dims[2])


        out_spikes = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0 = self.snn_1(x_t, v_0, z_0) # output spikes are inputs to next layer
            v_1, z_1 = self.snn_2(z_0, v_1, z_1)
            v_2, z_2 = self.snn_3(z_1, v_2, z_2)

            out_spikes.append(z_2)
        
        if self.accumulate == 'sum':
            out_spikes = torch.stack(out_spikes).transpose(0, 1).mean(1)
            output = self.fc(out_spikes)
            output = self.fc_act(output)
        else:
            output = self.fc(z_2)
            output = self.fc_act(output)

        return output

    def forward_hebbian(self, x):
        # x shape: B x T x IN_DIMS

        with torch.no_grad():

            B, T, _ = x.shape

            v_0 = torch.zeros(B, self.recurrent_dims[0])
            z_0 = torch.zeros(B, self.recurrent_dims[0])

            v_1 = torch.zeros(B, self.recurrent_dims[1])
            z_1 = torch.zeros(B, self.recurrent_dims[1])

            v_2 = torch.zeros(B, self.recurrent_dims[2])
            z_2 = torch.zeros(B, self.recurrent_dims[2])

            out_spikes = []

            for t in range(T):
                x_t = x[:, t, :]

                v_0, z_0 = self.snn_1.forward_hebbian(x_t, v_0, z_0) # output spikes are inputs to next layer
                v_1, z_1 = self.snn_2.forward_hebbian(z_0, v_1, z_1)
                v_2, z_2 = self.snn_3.forward_hebbian(z_1, v_2, z_2)

                out_spikes.append(z_2)
            
            if self.accumulate == 'sum':
                out_spikes = torch.stack(out_spikes).transpose(0, 1).mean(1)
                output = self.fc(out_spikes)
                output = self.fc_act(output)
            else:
                output = self.fc(z_2)
                output = self.fc_act(output)

            return output
        
    def apply_weight_change(self):
        self.snn_1.apply_weight_change()
        self.snn_2.apply_weight_change()
        self.snn_3.apply_weight_change()
        self.fc.apply_weight_change()

    
if __name__ == '__main__':
    net = TwoLayerSNN(32, 16, [48, 64])
    x = torch.randn((5, 10, 32))

    output = net(x)



