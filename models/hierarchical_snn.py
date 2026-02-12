import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
import snntorch.surrogate as surrogate

from typing import Union, List, Dict

from .hebbian_linear import HebbianLinearLayer
from .vanilla_snn import LIF_Cell, ALIF_Cell

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

"Top Cell": v[t] = alpha * v[t-1] + W_recurrent * z[t-1] + B * b_{l-1}[t] * z_{l-1}[t] - z[t-1] * v_thresh
"Bottom Cell": v[t] = (1 - b_{l}[t-1]) * (alpha * v[t-1] + W_recurrent * z[t-1]) + B * x[t] + T * b_{l+1}[t-1] * z_{l+1}[t-1] - z[t-1] * v_thresh
"Middle Cell": v[t] = (1 - b_{l}[t-1]) * (alpha * v[t-1] + W_recurrent * z[t-1]) + B * b_{l-1}[t] * z_{l-1}[t] + T * b_{l+1}[t-1] * z_{l+1}[t-1] - z[t-1] * v_thresh

z[t] = (v[t] - v_thresh) > 0
b[t] = Sigmoid(U * z[t]) -> use soft sigmoid
'''
class LIF_Top_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims,
            alpha = 0.9, 
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
        alpha: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        linear_options: options to pass to projection layers
        '''

        super(LIF_Top_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.alpha = alpha
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.b_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # bottom to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, masked=True, **linear_options) # spike to voltage

        # spike to boundary
        self.z_b = nn.Sequential(
            HebbianLinearLayer(out_dims, 1, **linear_options),
            nn.Sigmoid()
        )

        self.b_grad = surrogate.atan(alpha=2)

    def forward(self, b_in, b_b, v_previous, z_previous):

        v_out = self.alpha * v_previous + self.z_v(z_previous) # layer dynamics
        v_out += b_b * self.b_v(b_in) # bottom-up signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        z_out = self.spike_grad(v_out - self.threshold)

        b_out = self.z_b(z_out)
        b_out = self.b_grad(b_out - 0.5)

        return v_out, z_out, b_out
    
class LIF_Middle_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims,
            next_dims,
            alpha = 0.9, 
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
        alpha: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        linear_options: options to pass to projection layers
        '''

        super(LIF_Middle_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.alpha = alpha
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.b_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # bottom to voltage
        self.t_v = HebbianLinearLayer(next_dims, out_dims, **linear_options) # top to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, masked=True, **linear_options) # spike to voltage

        # spike to boundary
        self.z_b = nn.Sequential(
            HebbianLinearLayer(out_dims, 1, **linear_options),
            nn.Sigmoid()
        )
        self.b_grad = surrogate.atan(alpha=2)

    def forward(self, b_in, b_b, t_in, t_b, v_previous, z_previous, b_previous):

        v_out = (1 - b_previous) * (self.alpha * v_previous + self.z_v(z_previous)) # layer dynamics
        v_out += b_b * self.b_v(b_in) # bottom-up signal
        v_out += t_b * self.t_v(t_in) # top_down signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        z_out = self.spike_grad(v_out - self.threshold)

        b_out = self.z_b(z_out)
        b_out = self.b_grad(b_out - 0.5)

        return v_out, z_out, b_out
    
class LIF_Bottom_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims,
            next_dims,
            alpha = 0.9, 
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
        alpha: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        linear_options: options to pass to projection layers
        '''

        super(LIF_Bottom_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.alpha = alpha
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.b_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # bottom to voltage
        self.t_v = HebbianLinearLayer(next_dims, out_dims, **linear_options) # top to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, masked=True, **linear_options) # spike to voltage

        # spike to boundary
        self.z_b = nn.Sequential(
            HebbianLinearLayer(out_dims, 1, **linear_options),
            nn.Sigmoid()
        )
        self.b_grad = surrogate.atan(alpha=2)

    def forward(self, b_in, t_in, t_b, v_previous, z_previous, b_previous):
        v_out = (1 - b_previous) * (self.alpha * v_previous + self.z_v(z_previous)) # layer dynamics
        v_out += self.b_v(b_in) # bottom-up signal
        v_out += t_b * self.t_v(t_in) # top_down signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        z_out = self.spike_grad(v_out - self.threshold)

        b_out = self.z_b(z_out)
        b_out = self.b_grad(b_out - 0.5)

        return v_out, z_out, b_out
    
class ALIF_Top_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims,
            alpha = 0.9, 
            beta = 0.1,
            rho = 0.1,
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
        alpha: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        linear_options: options to pass to projection layers
        '''

        super(ALIF_Top_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.b_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # bottom to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, masked=True, **linear_options) # spike to voltage

        # spike to boundary
        self.z_b = nn.Sequential(
            HebbianLinearLayer(out_dims, 1, **linear_options),
            nn.Sigmoid()
        )

        self.b_grad = surrogate.atan(alpha=2)

    def forward(self, b_in, b_b, v_previous, z_previous, a_previous):

        v_out = self.alpha * v_previous + self.z_v(z_previous) # layer dynamics
        v_out += b_b * self.b_v(b_in) # bottom-up signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        a_out = self.rho * a_previous + z_previous
        z_out = self.spike_grad(v_out - (self.threshold + self.beta * a_out))

        b_out = self.z_b(z_out)
        b_out = self.b_grad(b_out - 0.5)

        return v_out, z_out, b_out, a_out
    
class ALIF_Middle_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims,
            next_dims,
            alpha = 0.9, 
            beta = 0.1,
            rho = 0.1,
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
        alpha: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        linear_options: options to pass to projection layers
        '''

        super(ALIF_Middle_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.b_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # bottom to voltage
        self.t_v = HebbianLinearLayer(next_dims, out_dims, **linear_options) # top to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, masked=True, **linear_options) # spike to voltage

        # spike to boundary
        self.z_b = nn.Sequential(
            HebbianLinearLayer(out_dims, 1, **linear_options),
            nn.Sigmoid()
        )
        self.b_grad = surrogate.atan(alpha=2)

    def forward(self, b_in, b_b, t_in, t_b, v_previous, z_previous, b_previous, a_previous):

        v_out = (1 - b_previous) * (self.alpha * v_previous + self.z_v(z_previous)) # layer dynamics
        v_out += b_b * self.b_v(b_in) # bottom-up signal
        v_out += t_b * self.t_v(t_in) # top_down signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        a_out = self.rho * a_previous + z_previous
        z_out = self.spike_grad(v_out - (self.threshold + self.beta * a_out))

        b_out = self.z_b(z_out)
        b_out = self.b_grad(b_out - 0.5)

        return v_out, z_out, b_out, a_out
    
class ALIF_Bottom_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims,
            next_dims,
            alpha = 0.9, 
            beta = 0.1,
            rho = 0.1,
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
        alpha: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        linear_options: options to pass to projection layers
        '''

        super(ALIF_Bottom_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.b_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # bottom to voltage
        self.t_v = HebbianLinearLayer(next_dims, out_dims, **linear_options) # top to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, masked=True, **linear_options) # spike to voltage

        # spike to boundary
        self.z_b = nn.Sequential(
            HebbianLinearLayer(out_dims, 1, **linear_options),
            nn.Sigmoid()
        )
        self.b_grad = surrogate.atan(alpha=2)

    def forward(self, b_in, t_in, t_b, v_previous, z_previous, b_previous, a_previous):
        v_out = (1 - b_previous) * (self.alpha * v_previous + self.z_v(z_previous)) # layer dynamics
        v_out += self.b_v(b_in) # bottom-up signal
        v_out += t_b * self.t_v(t_in) # top_down signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        a_out = self.rho * a_previous + z_previous
        z_out = self.spike_grad(v_out - (self.threshold + self.beta * a_out))

        b_out = self.z_b(z_out)
        b_out = self.b_grad(b_out - 0.5)

        return v_out, z_out, b_out, a_out


# Basic 2-Layer SNN with separate spiking head
class TwoLayer_HierarchicalSNN(nn.Module):

    def __init__(
            self,
            in_dims: int,
            out_dims: int,
            recurrent_dims: List[int],
            out_act: str = 'none',
            out_act_kwargs: dict = {},
            neuron_options: Union[dict, List[dict]] = {
                'alpha': 0.9,
                'threshold': 1.0,
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{
                    'learning_rule': 'oja',
                    'learning_rate': 1e-4,
                    'bias': True
                }
            },
            neuron_type: str = 'LIF',
            out_type: str = 'linear',
            out_options: dict = {
                'bias': True,
            }
        ):

        '''
        in_dims: input dimensions
        out_dims: output dimensions
        recurrent_dimensions: list of length two with output dimensions of recurrent layers
        neuron_options: kwargs to pass to SNN_Cell
        '''

        super(TwoLayer_HierarchicalSNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.out_type = out_type

        if isinstance(neuron_options, dict):
            neuron_options = [neuron_options, neuron_options]

        if neuron_type in ['ALIF', 'alif']:
            self.neuron_type = 'alif'
            self.bottom_cell = ALIF_Bottom_Cell(self.in_dims, self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options[0])
            self.top_cell = ALIF_Top_Cell(self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options[1])
        else:
            self.neuron_type = 'lif'
            self.bottom_cell = LIF_Bottom_Cell(self.in_dims, self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options[0])
            self.top_cell = LIF_Top_Cell(self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options[1])

        if self.out_type == 'linear':
            self.fc = nn.Sequential(
                HebbianLinearLayer(self.recurrent_dims[0], self.out_dims, **neuron_options['linear_options']),
                get_activation_fn(out_act, out_act_kwargs)
            )

        elif self.out_type == 'lif':
            self.fc = LIF_Cell(self.recurrent_dims[1], self.out_dims, **out_options)
        else:
            self.fc = ALIF_Cell(self.recurrent_dims[1], self.out_dims, **out_options)


    def _forward_lif(self, x):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1)

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])
        b_1 = torch.zeros(B, 1)

        v_fc = torch.zeros(B, self.out_dims)
        z_fc = torch.zeros(B, self.out_dims)
        a_fc = torch.zeros(B, self.out_dims)


        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, b_0 = self.bottom_cell(x_t, z_1, b_1, v_0, z_0, b_0) # output spikes are inputs to next layer
            v_1, z_1, b_1 = self.top_cell(z_0, b_0, v_1, z_1)

            if self.out_type == 'linear':
                output = self.fc(z_1)
            elif self.out_type == 'lif':
                v_fc, z_fc = self.fc(z_1, v_fc, z_fc)
                output = z_fc
            else:
                v_fc, z_fc, a_fc = self.fc(z_1, v_fc, z_fc, a_fc)
                output = z_fc
                
            outputs.append(output.clone())

        outputs = torch.stack(outputs, 1)

        return outputs, output
    
    def _forward_alif(self, x):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1)
        a_0 = torch.zeros(B, self.recurrent_dims[0])

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])
        b_1 = torch.zeros(B, 1)
        a_1 = torch.zeros(B, self.recurrent_dims[1])

        v_fc = torch.zeros(B, self.out_dims)
        z_fc = torch.zeros(B, self.out_dims)
        a_fc = torch.zeros(B, self.out_dims)


        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, b_0, a_0 = self.bottom_cell(x_t, z_1, b_1, v_0, z_0, b_0, a_0) # output spikes are inputs to next layer
            v_1, z_1, b_1, a_1 = self.top_cell(z_0, b_0, v_1, z_1, a_1)
            
            if self.out_type == 'linear':
                output = self.fc(z_1)
            elif self.out_type == 'lif':
                v_fc, z_fc = self.fc(z_1, v_fc, z_fc)
                output = z_fc
            else:
                v_fc, z_fc, a_fc = self.fc(z_1, v_fc, z_fc, a_fc)
                output = z_fc
                
            outputs.append(output.clone())

        outputs = torch.stack(outputs, 1)

        return outputs, output

    
    def forward(self, x):
        if self.neuron_type == 'lif':
            return self._forward_lif(x)
        else:
            return self._forward_alif(x)
        
class ThreeLayer_HierarchicalSNN(nn.Module):

    def __init__(
            self,
            in_dims: int,
            out_dims: int,
            recurrent_dims: List[int],
            out_act: str = 'none', # JUST HERE TO KEEP THINGS CONSISTENT (NOT USED)
            out_act_kwargs: dict = {}, # JUST HERE TO KEEP THINGS CONSISTENT (NOT USED)
            neuron_options: Union[dict, List[dict]] = {
                'alpha': 0.9,
                'threshold': 1.0,
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{
                    'learning_rule': 'oja',
                    'learning_rate': 1e-4,
                    'bias': True
                }
            },
            neuron_type: str = 'LIF',
            out_type: str = 'linear', # JUST HERE TO KEEP THINGS CONSISTENT (NOT USED)
            out_options: dict = { # JUST HERE TO KEEP THINGS CONSISTENT (NOT USED)
                'bias': True,
            }
        ):

        '''
        in_dims: input dimensions
        out_dims: output dimensions
        recurrent_dimensions: list of length two with output dimensions of recurrent layers
        neuron_options: kwargs to pass to SNN_Cell
        '''

        super(ThreeLayer_HierarchicalSNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.out_type = out_type

        if isinstance(neuron_options, dict):
            neuron_options = [neuron_options, neuron_options, neuron_options]

        if neuron_type in ['ALIF', 'alif']:
            self.neuron_type = 'alif'
            self.bottom_cell = ALIF_Bottom_Cell(self.in_dims, self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options[0])
            self.middle_cell = ALIF_Middle_Cell(self.recurrent_dims[0], self.recurrent_dims[1], self.recurrent_dims[2], **neuron_options[1])
            self.top_cell = ALIF_Top_Cell(self.recurrent_dims[1], self.out_dims, **neuron_options[2])
        else:
            self.neuron_type = 'lif'
            self.bottom_cell = LIF_Bottom_Cell(self.in_dims, self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options[0])
            self.middle_cell = LIF_Middle_Cell(self.recurrent_dims[0], self.recurrent_dims[1], self.recurrent_dims[2], **neuron_options[1])
            self.top_cell = LIF_Top_Cell(self.recurrent_dims[1], self.out_dims, **neuron_options[2])

    def _forward_lif(self, x):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1)

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])
        b_1 = torch.zeros(B, 1)

        v_2 = torch.zeros(B, self.recurrent_dims[2])
        z_2 = torch.zeros(B, self.recurrent_dims[2])
        b_2 = torch.zeros(B, 1)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, b_0 = self.bottom_cell(x_t, z_1, b_1, v_0, z_0, b_0) # output spikes are inputs to next layer
            v_1, z_1, b_1 = self.middle_cell(z_0, b_0, z_2, b_2, v_1, z_1, b_1)
            v_2, z_2, b_2 = self.top_cell(z_1, b_1, v_2, z_2)

            outputs.append(z_2.clone())

        outputs = torch.stack(outputs, 1)

        return outputs, z_2
    
    def _forward_alif(self, x):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1)
        a_0 = torch.zeros(B, self.recurrent_dims[0])

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])
        b_1 = torch.zeros(B, 1)
        a_1 = torch.zeros(B, self.recurrent_dims[1])

        v_2 = torch.zeros(B, self.recurrent_dims[2])
        z_2 = torch.zeros(B, self.recurrent_dims[2])
        b_2 = torch.zeros(B, 1)
        a_2 = torch.zeros(B, self.recurrent_dims[2])

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, b_0, a_0 = self.bottom_cell(x_t, z_1, b_1, v_0, z_0, b_0, a_0) # output spikes are inputs to next layer
            v_1, z_1, b_1, a_1 = self.middle_cell(z_0, b_0, z_2, b_2, v_1, z_1, b_1, a_1)
            v_2, z_2, b_2, a_2 = self.top_cell(z_1, b_1, v_2, z_2, a_2)

            outputs.append(z_2.clone())

        outputs = torch.stack(outputs, 1)

        return outputs, z_2

    
    def forward(self, x):
        if self.neuron_type == 'lif':
            return self._forward_lif(x)
        else:
            return self._forward_alif(x)

    
if __name__ == '__main__':

    from torchviz import make_dot, make_dot_from_trace

    net = TwoLayer_HierarchicalSNN(32, 16, [48, 64], neuron_options = {
                'alpha': 0.9,
                'threshold': 1.0,
                'spike_grad': surrogate.atan(alpha=2), 
                'linear_options':{
                    'learning_rule': 'oja',
                    'learning_rate': 1e-4,
                    'bias': False,
                }
            },)
    x = torch.randn((3, 10, 32))

    output = net(x)
    dot = make_dot(output.mean(), params=dict(net.named_parameters()))
    dot.render("computational_graph_two_layer_hnn", format="png", cleanup=True)

