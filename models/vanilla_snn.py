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
    
class LIF_Cell(nn.Module):

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
        next_dims: shape of output of layer l+1
        alpha: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        linear_options: options to pass to projection layers
        '''

        super(LIF_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.alpha = alpha
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.x_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # bottom to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, masked=True, **linear_options) # spike to voltage


    def forward(self, x_in, v_previous, z_previous):
        v_out = self.alpha * v_previous + self.z_v(z_previous) # layer dynamics
        v_out += self.x_v(x_in) # bottom-up signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism
        z_out = self.spike_grad(v_out - self.threshold)

        return v_out, z_out

class ALIF_Cell(nn.Module):

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
        next_dims: shape of output of layer l+1
        alpha: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        linear_options: options to pass to projection layers
        '''

        super(ALIF_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.threshold = threshold
        self.spike_grad = spike_grad

        self.x_v = HebbianLinearLayer(in_dims, out_dims, **linear_options) # bottom to voltage
        self.z_v = HebbianLinearLayer(out_dims, out_dims, masked=True, **linear_options) # spike to voltage


    def forward(self, x_in, v_previous, z_previous, a_previous):
        v_out = self.alpha * v_previous + self.z_v(z_previous) # layer dynamics
        v_out += self.x_v(x_in) # bottom-up signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        a_out = self.rho * a_previous + z_previous
        z_out = self.spike_grad(v_out - (self.threshold + self.beta * a_out))

        return v_out, z_out, a_out
    
# Basic 2-Layer SNN
class TwoLayer_SNN(nn.Module):

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
            },

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

        super(TwoLayer_SNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.out_type = out_type

        if isinstance(neuron_options, dict):
            neuron_options = [neuron_options, neuron_options]

        if neuron_type in ['ALIF', 'alif']:
            self.neuron_type = 'alif'
            self.bottom_cell = ALIF_Cell(self.in_dims, self.recurrent_dims[0], **neuron_options[0])
            self.top_cell = ALIF_Cell(self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options[1])
        else:
            self.neuron_type = 'lif'
            self.bottom_cell = LIF_Cell(self.in_dims, self.recurrent_dims[0], **neuron_options[0])
            self.top_cell = LIF_Cell(self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options[1])

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

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])

        v_fc = torch.zeros(B, self.out_dims)
        z_fc = torch.zeros(B, self.out_dims)
        a_fc = torch.zeros(B, self.out_dims)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0 = self.bottom_cell(x_t, v_0, z_0) # output spikes are inputs to next layer
            v_1, z_1 = self.top_cell(z_0, v_1, z_1)

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
        a_0 = torch.zeros(B, self.recurrent_dims[0])

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])
        a_1 = torch.zeros(B, self.recurrent_dims[1])

        v_fc = torch.zeros(B, self.out_dims)
        z_fc = torch.zeros(B, self.out_dims)
        a_fc = torch.zeros(B, self.out_dims)


        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, a_0 = self.bottom_cell(x_t, v_0, z_0, a_0) # output spikes are inputs to next layer
            v_1, z_1, a_1 = self.top_cell(z_0, v_1, z_1, a_1)

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

# Basic 1-Layer SNN
class OneLayer_SNN(nn.Module):

    def __init__(
            self,
            in_dims: int,
            out_dims: int,
            recurrent_dims: List[int],
            out_act: str = 'none',
            out_act_kwargs: dict = {},
            neuron_options: dict = {
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
            },
        ):

        '''
        in_dims: input dimensions
        out_dims: output dimensions
        recurrent_dimensions: list of length two with output dimensions of recurrent layers
        out_act: activation fn for output layer
        out_act_kwargs: kwargs to pass to output layer
        neuron_options: kwargs to pass to SNN_Cell
        neuron_type: either "LIF" or "ALIF", indicates whether or not to have adaptive neurons
        out_type: either "linear" or "alif" or "lif", indicates whether the output layer should also be spiking
        '''

        super(OneLayer_SNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.out_type = out_type

        if neuron_type in ['ALIF', 'alif']:
            self.neuron_type = 'alif'
            self.snn = ALIF_Cell(self.in_dims, self.recurrent_dims[0], **neuron_options)
        else:
            self.neuron_type = 'lif'
            self.snn = LIF_Cell(self.in_dims, self.recurrent_dims[0], **neuron_options)

        if self.out_type == 'linear':
            self.fc = nn.Sequential(
                HebbianLinearLayer(self.recurrent_dims[0], self.out_dims, **neuron_options['linear_options']),
                get_activation_fn(out_act, out_act_kwargs)
            )
        elif self.out_type == 'lif':
            self.fc = LIF_Cell(self.recurrent_dims[0], self.out_dims, **out_options)
        else:
            self.fc = ALIF_Cell(self.recurrent_dims[0], self.out_dims, **out_options)


    def _forward_lif(self, x):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])

        v_fc = torch.zeros(B, self.out_dims)
        z_fc = torch.zeros(B, self.out_dims)
        a_fc = torch.zeros(B, self.out_dims)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0 = self.snn(x_t, v_0, z_0) # output spikes are inputs to next layer

            if self.out_type == 'linear':
                output = self.fc(z_0)
            elif self.out_type == 'lif':
                v_fc, z_fc = self.fc(z_0, v_fc, z_fc)
                output = z_fc
            else:
                v_fc, z_fc, a_fc = self.fc(z_0, v_fc, z_fc, a_fc)
                output = z_fc
                
            outputs.append(output.clone())

        outputs = torch.stack(outputs, 1)

        return outputs, output
    
    def _forward_alif(self, x):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        a_0 = torch.zeros(B, self.recurrent_dims[0])

        v_fc = torch.zeros(B, self.out_dims)
        z_fc = torch.zeros(B, self.out_dims)
        a_fc = torch.zeros(B, self.out_dims)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, a_0 = self.snn(x_t, v_0, z_0, a_0) # output spikes are inputs to next layer

            if self.out_type == 'linear':
                output = self.fc(z_0)
            elif self.out_type == 'lif':
                v_fc, z_fc = self.fc(z_0, v_fc, z_fc)
                output = z_fc
            else:
                v_fc, z_fc, a_fc = self.fc(z_0, v_fc, z_fc, a_fc)
                output = z_fc
                
            outputs.append(output.clone())

        outputs = torch.stack(outputs, 1)

        return outputs, output

    def forward(self, x):
        if self.neuron_type == 'lif':
            return self._forward_lif(x)
        else:
            return self._forward_alif(x)
        
