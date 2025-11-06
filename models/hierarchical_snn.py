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
Hierarchical LIF Layer: see https://arxiv.org/pdf/1609.01704 for inspiration
The idea: neurons might only share different time-scale information sparsely as needed
when a "boundary" is detected, a neuron can decide to share its state with other neurons
high-level neurons always accumulate their internal state; they can be given a bottom-up signal to integrate useful low-level abstractions
low-level neurons always integrate raw stimulus; they can be given a top-down signal to "reset" their internal state to only capture fine-grained features
middle-level neurons receive top-down and bottom-up signals to integrate both top-down and bottom-up signals

"Top Cell": v[t] = beta * v[t-1] + W_recurrent * z[t-1] + B * b_{l-1}[t] * z_{l-1}[t] - z[t-1] * v_thresh
"Bottom Cell": v[t] = (1 - b_{l}[t-1]) * (beta * v[t-1] + W_recurrent * z[t-1]) + B * x[t] + T * b_{l+1}[t-1] * z_{l+1}[t-1] - z[t-1] * v_thresh
"Middle Cell": v[t] = (1 - b_{l}[t-1]) * (beta * v[t-1] + W_recurrent * z[t-1]) + B * b_{l-1}[t] * z_{l-1}[t] + T * b_{l+1}[t-1] * z_{l+1}[t-1] - z[t-1] * v_thresh

z[t] = (v[t] - v_thresh) > 0
b[t] = Sigmoid(U * z[t]) -> use soft sigmoid
'''
class Hierarchical_Top_Cell(nn.Module):

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

        super(Hierarchical_Top_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.beta = beta
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.b_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # bottom to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, **linear_options) # spike to voltage

        # spike to boundary
        self.z_b = nn.Sequential(
            HebbianLinearLayer(out_dims, 1, **linear_options),
            nn.Sigmoid()
        )

        self.b_grad = surrogate.sigmoid()

    def forward(self, b_in, b_b, v_previous, z_previous):

        v_out = self.beta * v_previous + self.z_v(z_previous) # layer dynamics
        v_out += b_b * self.b_v(b_in) # bottom-up signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        z_out = self.spike_grad(v_out - self.threshold)

        b_out = self.z_b(z_out)
        b_out = self.b_grad(b_out - 0.5)

        return v_out, z_out, b_out
    
class Hierarchical_Middle_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims,
            next_dims,
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
        next_dims: shape of output of layer l+1
        beta: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        linear_options: options to pass to projection layers
        '''

        super(Hierarchical_Middle_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.beta = beta
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.b_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # bottom to voltage
        self.t_v = HebbianLinearLayer(next_dims, out_dims, **linear_options) # top to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, **linear_options) # spike to voltage

        # spike to boundary
        self.z_b = nn.Sequential(
            HebbianLinearLayer(out_dims, 1, **linear_options),
            nn.Sigmoid()
        )
        self.b_grad = surrogate.sigmoid()

    def forward(self, b_in, b_b, t_in, t_b, v_previous, z_previous, b_previous):

        v_out = (1 - b_previous) * (self.beta * v_previous + self.z_v(z_previous)) # layer dynamics
        v_out += b_b * self.b_v(b_in) # bottom-up signal
        v_out += t_b * self.t_v(t_in) # top_down signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        z_out = self.spike_grad(v_out - self.threshold)

        b_out = self.z_b(z_out)
        b_out = self.b_grad(b_out - 0.5)

        return v_out, z_out, b_out
    
class Hierarchical_Bottom_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims,
            next_dims,
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
        next_dims: shape of output of layer l+1
        beta: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        linear_options: options to pass to projection layers
        '''

        super(Hierarchical_Bottom_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.beta = beta
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.b_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # bottom to voltage
        self.t_v = HebbianLinearLayer(next_dims, out_dims, **linear_options) # top to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, **linear_options) # spike to voltage

        # spike to boundary
        self.z_b = nn.Sequential(
            HebbianLinearLayer(out_dims, 1, **linear_options),
            nn.Sigmoid()
        )
        self.b_grad = surrogate.sigmoid()

    def forward(self, b_in, t_in, t_b, v_previous, z_previous, b_previous):
        v_out = (1 - b_previous) * (self.beta * v_previous + self.z_v(z_previous)) # layer dynamics
        v_out += self.b_v(b_in) # bottom-up signal
        v_out += t_b * self.t_v(t_in) # top_down signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        z_out = self.spike_grad(v_out - self.threshold)

        b_out = self.z_b(z_out)
        b_out = self.b_grad(b_out - 0.5)

        return v_out, z_out, b_out

# Basic 2-Layer SNN Using LIF Neurons
class TwoLayer_HierarchicalSNN(nn.Module):

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
            logging: bool = False,
        ):

        '''
        in_dims: input dimensions
        out_dims: output dimensions
        recurrent_dimensions: list of length two with output dimensions of recurrent layers
        out_act: activation fn for output layer
        out_act_kwargs: kwargs to pass to output layer
        neuron_options: kwargs to pass to SNN_Cell
        spike_accumulator: either "sum" or "last" -> how to pass spikes to prediction layer
        logging: whether or not to return boundary values
        '''

        super(TwoLayer_HierarchicalSNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.accumulate = spike_accumulator
        self.logging = logging

        self.bottom_cell = Hierarchical_Bottom_Cell(self.in_dims, self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options)
        self.top_cell = Hierarchical_Top_Cell(self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options)

        self.fc = nn.Sequential(
            HebbianLinearLayer(self.recurrent_dims[1], self.out_dims, **neuron_options['linear_options']),
            get_activation_fn(out_act, out_act_kwargs)
        )


    def forward(self, x):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1)

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])
        b_1 = torch.zeros(B, 1)

        out_spikes = []
        b_vals = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, b_0 = self.bottom_cell(x_t, z_1, b_1, v_0, z_0, b_0) # output spikes are inputs to next layer
            v_1, z_1, b_1 = self.top_cell(z_0, b_0, v_1, z_1)

            out_spikes.append(z_1)

            if self.logging:
                b_vals.append((b_0.detach().numpy(), b_1.detach().numpy()))
        
        if self.accumulate == 'sum':
            out_spikes = torch.stack(out_spikes).transpose(0, 1).mean(1)
            output = self.fc(out_spikes)
        else:
            output = self.fc(z_1)

        if self.logging:
            return output, b_vals

        return output
    
if __name__ == '__main__':
    net = TwoLayer_HierarchicalSNN(32, 16, [48, 64])
    x = torch.randn((5, 10, 32))

    output = net(x)
    print(output)