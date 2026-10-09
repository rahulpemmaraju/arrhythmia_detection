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
Hierarchically gated recurrent SNN (HG-RSNN), inspired by hierarchical multiscale RNNs
(https://arxiv.org/abs/1609.01704). The bottom and middle cells each emit a binary gate
b_l[t] (g_l in the paper) from their spikes. When it is on, the layer above reads this
layer's spikes (sharing), and on the next step this layer's state is reset (flushing).
The bottom cell always receives the ECG; the top cell is never flushed.

The bottom and middle cells also build a top-down projection (t_v) that is never used.
It is kept so the published checkpoints load unchanged; it does not affect the output
and is excluded from the parameter counts in the paper.

"Top Cell": v[t] = alpha * v[t-1] + W_recurrent * z[t-1] + B * b_{l-1}[t] * z_{l-1}[t] - z[t-1] * v_thresh
"Bottom Cell": v[t] = (1 - b_{l}[t-1]) * (alpha * v[t-1] + W_recurrent * z[t-1]) + B * x[t] - z[t-1] * v_thresh
"Middle Cell": v[t] = (1 - b_{l}[t-1]) * (alpha * v[t-1] + W_recurrent * z[t-1]) + B * b_{l-1}[t] * z_{l-1}[t] - z[t-1] * v_thresh

z[t] = H(v[t] - v_thresh)
b[t] = H(Sigmoid(U * z[t]) - 0.5)       (H: Heaviside step, arctan surrogate gradient)
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

    def forward(self, b_in, b_b, v_previous, z_previous):

        v_out = self.alpha * v_previous + self.z_v(z_previous) # layer dynamics
        v_out += b_b * self.b_v(b_in) # bottom-up signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        z_out = self.spike_grad(v_out - self.threshold)

        return v_out, z_out
    
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

    def forward(self, b_in, b_b, v_previous, z_previous, b_previous):

        v_out = (1 - b_previous) * (self.alpha * v_previous + self.z_v(z_previous)) # layer dynamics
        v_out += b_b * self.b_v(b_in) # bottom-up signal
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

    def forward(self, b_in, v_previous, z_previous, b_previous):
        v_out = (1 - b_previous) * (self.alpha * v_previous + self.z_v(z_previous)) # layer dynamics
        v_out += self.b_v(b_in) # bottom-up signal
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

    def forward(self, b_in, b_b, v_previous, z_previous, a_previous):

        v_out = self.alpha * v_previous + self.z_v(z_previous) # layer dynamics
        v_out += b_b * self.b_v(b_in) # bottom-up signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        a_out = self.rho * a_previous + z_previous
        z_out = self.spike_grad(v_out - (self.threshold + self.beta * a_out))

        return v_out, z_out, a_out
    
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

    def forward(self, b_in, b_b, v_previous, z_previous, b_previous, a_previous):

        v_out = (1 - b_previous) * (self.alpha * v_previous + self.z_v(z_previous)) # layer dynamics
        v_out += b_b * self.b_v(b_in) # bottom-up signal
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

    def forward(self, b_in, v_previous, z_previous, b_previous, a_previous):
        v_out = (1 - b_previous) * (self.alpha * v_previous + self.z_v(z_previous)) # layer dynamics
        v_out += self.b_v(b_in) # bottom-up signal
        v_out -= (z_previous * self.threshold).detach() # reset mechanism

        a_out = self.rho * a_previous + z_previous
        z_out = self.spike_grad(v_out - (self.threshold + self.beta * a_out))

        b_out = self.z_b(z_out)
        b_out = self.b_grad(b_out - 0.5)

        return v_out, z_out, b_out, a_out

class TwoLayer_HGRSNN(nn.Module):

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

        super(TwoLayer_HGRSNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.out_type = out_type

        if isinstance(neuron_options, dict):
            neuron_options = [neuron_options, neuron_options]

        if neuron_type in ['ALIF', 'alif']:
            self.neuron_type = 'alif'
            self.bottom_cell = ALIF_Bottom_Cell(self.in_dims, self.recurrent_dims[0], self.out_dims, **neuron_options[0])
            self.top_cell = ALIF_Top_Cell(self.recurrent_dims[0], self.out_dims, **neuron_options[1])
        else:
            self.neuron_type = 'lif'
            self.bottom_cell = LIF_Bottom_Cell(self.in_dims, self.recurrent_dims[0], self.out_dims, **neuron_options[0])
            self.top_cell = LIF_Top_Cell(self.recurrent_dims[0], self.out_dims, **neuron_options[1])

    def _forward_lif(self, x, logging):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        if logging:
            b_0s = []

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1)

        v_1 = torch.zeros(B, self.out_dims)
        z_1 = torch.zeros(B, self.out_dims)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, b_0 = self.bottom_cell(x_t, v_0, z_0, b_0) # output spikes are inputs to next layer
            v_1, z_1 = self.top_cell(z_0, b_0, v_1, z_1)

            outputs.append(z_1.clone())

            if logging:
                b_0s.append(b_0.detach().clone())


        outputs = torch.stack(outputs, 1)

        if logging:
            return outputs, z_1, [torch.stack(b_0s, 1)]

        return outputs, z_1
    
    def _forward_alif(self, x, logging):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        if logging:
            b_0s = []

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1)
        a_0 = torch.zeros(B, self.recurrent_dims[0])

        v_1 = torch.zeros(B, self.out_dims)
        z_1 = torch.zeros(B, self.out_dims)
        a_1 = torch.zeros(B, self.out_dims)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, b_0, a_0 = self.bottom_cell(x_t, v_0, z_0, b_0, a_0) # output spikes are inputs to next layer
            v_1, z_1, a_1 = self.top_cell(z_0, b_0, v_1, z_1, a_1)

            outputs.append(z_1.clone())

            if logging:
                b_0s.append(b_0.detach().clone())


        outputs = torch.stack(outputs, 1)

        if logging:
            return outputs, z_1, [torch.stack(b_0s, 1)]

        return outputs, z_1

    
    def forward(self, x, logging=False):
        if self.neuron_type == 'lif':
            return self._forward_lif(x, logging)
        else:
            return self._forward_alif(x, logging)
        
class ThreeLayer_HGRSNN(nn.Module):

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

        super(ThreeLayer_HGRSNN, self).__init__()

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

    def _forward_lif(self, x, logging):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        if logging:
            b_0s = []
            b_1s = []

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1)

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])
        b_1 = torch.zeros(B, 1)

        v_2 = torch.zeros(B, self.recurrent_dims[2])
        z_2 = torch.zeros(B, self.recurrent_dims[2])

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, b_0 = self.bottom_cell(x_t, v_0, z_0, b_0) # output spikes are inputs to next layer
            v_1, z_1, b_1 = self.middle_cell(z_0, b_0, v_1, z_1, b_1)
            v_2, z_2 = self.top_cell(z_1, b_1, v_2, z_2)

            outputs.append(z_2.clone())

            if logging:
                b_0s.append(b_0.detach().clone())
                b_1s.append(b_1.detach().clone())


        outputs = torch.stack(outputs, 1)

        if logging:
            return outputs, z_2, [torch.stack(b_0s, 1), torch.stack(b_1s, 1)]

        return outputs, z_2
    
    def _forward_alif(self, x, logging):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        if logging:
            b_0s = []
            b_1s = []

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
        a_2 = torch.zeros(B, self.recurrent_dims[2])

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, b_0, a_0 = self.bottom_cell(x_t, v_0, z_0, b_0, a_0) # output spikes are inputs to next layer
            v_1, z_1, b_1, a_1 = self.middle_cell(z_0, b_0, v_1, z_1, b_1, a_1)
            v_2, z_2, a_2 = self.top_cell(z_1, b_1, v_2, z_2, a_2)

            outputs.append(z_2.clone())

            if logging:
                b_0s.append(b_0.detach().clone())
                b_1s.append(b_1.detach().clone())


        outputs = torch.stack(outputs, 1)

        if logging:
            return outputs, z_2, [torch.stack(b_0s, 1), torch.stack(b_1s, 1)]

        return outputs, z_2

    
    def forward(self, x, logging=False):
        if self.neuron_type == 'lif':
            return self._forward_lif(x, logging)
        else:
            return self._forward_alif(x, logging)
        
class FourLayer_HGRSNN(nn.Module):

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
        recurrent_dimensions: list of length three with output dimensions of recurrent layers
        neuron_options: kwargs to pass to SNN_Cell
        out_type: readout layer type ('linear', 'lif' or 'alif')
        out_options: kwargs to pass to the readout layer
        '''

        super(FourLayer_HGRSNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.out_type = out_type

        if isinstance(neuron_options, dict):
            neuron_options = [neuron_options, neuron_options, neuron_options]

        # the top cell emits recurrent_dims[2] features; the readout does the classification
        if neuron_type in ['ALIF', 'alif']:
            self.neuron_type = 'alif'
            self.bottom_cell = ALIF_Bottom_Cell(self.in_dims, self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options[0])
            self.middle_cell = ALIF_Middle_Cell(self.recurrent_dims[0], self.recurrent_dims[1], self.recurrent_dims[2], **neuron_options[1])
            self.top_cell = ALIF_Top_Cell(self.recurrent_dims[1], self.recurrent_dims[2], **neuron_options[2])
        else:
            self.neuron_type = 'lif'
            self.bottom_cell = LIF_Bottom_Cell(self.in_dims, self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options[0])
            self.middle_cell = LIF_Middle_Cell(self.recurrent_dims[0], self.recurrent_dims[1], self.recurrent_dims[2], **neuron_options[1])
            self.top_cell = LIF_Top_Cell(self.recurrent_dims[1], self.recurrent_dims[2], **neuron_options[2])

        if self.out_type == 'linear':
            self.fc = nn.Sequential(
                HebbianLinearLayer(self.recurrent_dims[2], self.out_dims, **neuron_options[-1]['linear_options']),
                get_activation_fn(out_act, out_act_kwargs)
            )

        elif self.out_type == 'lif':
            self.fc = LIF_Cell(self.recurrent_dims[2], self.out_dims, **out_options)

        else:
            self.fc = ALIF_Cell(self.recurrent_dims[2], self.out_dims, **out_options)

    def _readout(self, z_2, v_fc, z_fc, a_fc):
        if self.out_type == 'linear':
            return self.fc(z_2), v_fc, z_fc, a_fc
        elif self.out_type == 'lif':
            v_fc, z_fc = self.fc(z_2, v_fc, z_fc)
            return z_fc, v_fc, z_fc, a_fc
        else:
            v_fc, z_fc, a_fc = self.fc(z_2, v_fc, z_fc, a_fc)
            return z_fc, v_fc, z_fc, a_fc

    def _forward_lif(self, x, logging):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        if logging:
            b_0s = []
            b_1s = []

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1)

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])
        b_1 = torch.zeros(B, 1)

        v_2 = torch.zeros(B, self.recurrent_dims[2])
        z_2 = torch.zeros(B, self.recurrent_dims[2])

        v_fc = torch.zeros(B, self.out_dims)
        z_fc = torch.zeros(B, self.out_dims)
        a_fc = torch.zeros(B, self.out_dims)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, b_0 = self.bottom_cell(x_t, v_0, z_0, b_0) # output spikes are inputs to next layer
            v_1, z_1, b_1 = self.middle_cell(z_0, b_0, v_1, z_1, b_1)
            v_2, z_2 = self.top_cell(z_1, b_1, v_2, z_2)

            output, v_fc, z_fc, a_fc = self._readout(z_2, v_fc, z_fc, a_fc)

            outputs.append(output.clone())

            if logging:
                b_0s.append(b_0.detach().clone())
                b_1s.append(b_1.detach().clone())


        outputs = torch.stack(outputs, 1)

        if logging:
            return outputs, output, [torch.stack(b_0s, 1), torch.stack(b_1s, 1)]

        return outputs, output

    def _forward_alif(self, x, logging):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        if logging:
            b_0s = []
            b_1s = []

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
        a_2 = torch.zeros(B, self.recurrent_dims[2])

        v_fc = torch.zeros(B, self.out_dims)
        z_fc = torch.zeros(B, self.out_dims)
        a_fc = torch.zeros(B, self.out_dims)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            v_0, z_0, b_0, a_0 = self.bottom_cell(x_t, v_0, z_0, b_0, a_0) # output spikes are inputs to next layer
            v_1, z_1, b_1, a_1 = self.middle_cell(z_0, b_0, v_1, z_1, b_1, a_1)
            v_2, z_2, a_2 = self.top_cell(z_1, b_1, v_2, z_2, a_2)

            output, v_fc, z_fc, a_fc = self._readout(z_2, v_fc, z_fc, a_fc)

            outputs.append(output.clone())

            if logging:
                b_0s.append(b_0.detach().clone())
                b_1s.append(b_1.detach().clone())


        outputs = torch.stack(outputs, 1)

        if logging:
            return outputs, output, [torch.stack(b_0s, 1), torch.stack(b_1s, 1)]

        return outputs, output

    def forward(self, x, logging=False):
        if self.neuron_type == 'lif':
            return self._forward_lif(x, logging)
        else:
            return self._forward_alif(x, logging)


if __name__ == '__main__':
    x = torch.rand((1, 2, 3))
    net = ThreeLayer_HGRSNN(3, 4, [8, 10, 4])

    net(x)