import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import snntorch as snn
import snntorch.surrogate as surrogate

import math
from typing import Union, List, Dict

from .hebbian_linear import HebbianLinearLayer
from .vanilla_snn import LIF_Cell, ALIF_Cell, get_activation_fn

def init_weights(weight, in_dims, out_dims):
    k = 6 / in_dims
    bound = math.sqrt(k)
    nn.init.uniform_(weight, -bound, bound)


# simple gate module: linear layer -> sigmoid -> subtraction -> spike
class Gate(nn.Module):
    def __init__(self, in_dims, out_dims=1, bias=False, spike_fn=surrogate.atan(alpha=2)):

        super(Gate, self).__init__()
        self.projection = nn.Linear(in_dims, out_dims, bias)
        self.spike_fn = spike_fn

        init_weights(self.projection.weight, in_dims, out_dims)

    def forward(self, x):
        x = self.projection(x)
        x = F.sigmoid(x) - 0.5
        x = self.spike_fn(x)

        return x
    
# Compute dL/dW from dL/d{gate}
def gate_op_local_grads(W, z, dL_dg, spike_fn='atan', alpha=2):
    with torch.no_grad():
        a = z @ W.t()
        sigmoid = F.sigmoid(a)
    
        # atan grad
        dL_dsigmoid = (
            alpha
            / 2
            / (1 + (torch.pi / 2 * alpha * (sigmoid - 0.5)).pow(2))
            * dL_dg
        )

        dL_dprojection = dL_dsigmoid * (sigmoid * (1 - sigmoid))
        dL_dW = dL_dprojection.t().mm(z)

    return dL_dW

# use this syntax to compute something after dL/b0 grad is computed
def compute_dL_dg_dW_grad(W, z, spike_fn='atan', alpha=2):
    def hook(dL_dg):
        with torch.no_grad():
            dL_dW = gate_op_local_grads(W, z, dL_dg, spike_fn, alpha)

            if W.grad is None:
                W.grad = torch.zeros_like(W)

            W.grad += dL_dW
        
        return dL_dg
    return hook

# Will Compute Voltage Separately (to allow for the spike grad to propagate easily)
class LIF_Top_Voltage(Function):

    @staticmethod
    def forward(ctx, bottom_input, bottom_gate, v_previous, z_previous, W_zv, W_bv, W_zv_mask, zv_trace, bv_trace, alpha, v_threshold):

        ungated_bottom_up = bottom_input.mm(W_bv.t())
        bottom_up = bottom_gate * ungated_bottom_up

        masked_W_zv = W_zv * W_zv_mask # mask the self connections
        recurrent_input = z_previous.mm(masked_W_zv.t())

        dynamics = alpha * v_previous + recurrent_input

        v_out = dynamics + bottom_up 
        v_out = v_out - (z_previous * v_threshold).detach() # reset mechanism

        ctx.save_for_backward(bottom_gate, W_bv, zv_trace, bv_trace, ungated_bottom_up, masked_W_zv)
        ctx.alpha = alpha

        return v_out

    @staticmethod # WORKS W EPROP
    def backward(ctx, grad_v):
        bottom_gate, W_bv, zv_trace, bv_trace, ungated_bottom_up, masked_W_zv = ctx.saved_tensors

        grad_bottom_input = grad_bottom_gate = grad_v_previous = grad_z_previous = None 
        grad_W_zv = grad_W_bv = grad_W_zv_mask = grad_zv_trace = grad_bv_trace = grad_alpha = grad_v_threshold = None

        if ctx.needs_input_grad[0]:
            grad_bottom_input = bottom_gate * grad_v.mm(W_bv)

        if ctx.needs_input_grad[1]:
            grad_bottom_gate = torch.sum(grad_v * ungated_bottom_up, dim=1, keepdim=True)

        if ctx.needs_input_grad[2]:
            grad_v_previous = ctx.alpha * grad_v
        
        if ctx.needs_input_grad[3]:
            grad_z_previous = grad_v.mm(masked_W_zv)
        
        # e-prop
        if ctx.needs_input_grad[4]: 
            grad_W_zv = (grad_v.unsqueeze(-1) * zv_trace).sum(0)
            # grad_W_zv = grad_v.t().mm(z_previous)  * W_zv_mask
        
        # e-prop
        if ctx.needs_input_grad[5]:
            grad_W_bv = (grad_v.unsqueeze(-1) * bv_trace).sum(0)
            # grad_W_bv = grad_v.t().mm(bottom_input * bottom_gate)
        
        return (grad_bottom_input, grad_bottom_gate, grad_v_previous, grad_z_previous, grad_W_zv, grad_W_bv, 
                grad_W_zv_mask, grad_zv_trace, grad_bv_trace, grad_alpha, grad_v_threshold)

class LIF_Top_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims,
            alpha = 0.9, 
            threshold = 1.0, 
            spike_grad = surrogate.atan(alpha=2),
            linear_options = None, # DOESNT REFER TO ANYTHING (JUST TO KEEP THINGS CONSISTENT)
            timesteps = 360, # timesteps (to normalize e-prop)
        ):
        
        super(LIF_Top_Cell, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.alpha = alpha
        self.threshold = threshold
        self.spike_grad = spike_grad
        self.timesteps = timesteps
        
        # initialize weights
        self.W_zv = nn.Parameter(torch.Tensor(out_dims, out_dims))
        self.W_bv = nn.Parameter(torch.Tensor(out_dims, in_dims))

        init_weights(self.W_zv, out_dims, out_dims)
        init_weights(self.W_bv, in_dims, out_dims)

        self.gate = Gate(out_dims, 1, bias=False, spike_fn=spike_grad)

        # register buffers
        self.register_buffer('zv_mask', (1 - torch.eye(out_dims, out_dims)).to(dtype=torch.float32))
        self.register_buffer('zv_trace', torch.zeros((1, out_dims, out_dims)).to(dtype=torch.float32))
        self.register_buffer('bv_trace', torch.zeros((1, out_dims, in_dims)).to(dtype=torch.float32))

    def reset_traces(self):
        self.zv_trace = torch.zeros((1, self.out_dims, self.out_dims)).to(dtype=torch.float32)
        self.bv_trace = torch.zeros((1, self.out_dims, self.in_dims)).to(dtype=torch.float32)

    def forward(self, bottom_input, bottom_gate, v_previous, z_previous):

        # update eligibility traces
        if self.training:
            with torch.no_grad():
                self.zv_trace = self.alpha * self.zv_trace + z_previous.unsqueeze(1) * self.zv_mask
                self.bv_trace = self.alpha * self.bv_trace + (bottom_gate * bottom_input).unsqueeze(1)

        v_out = LIF_Top_Voltage.apply(bottom_input, bottom_gate, v_previous, 
                                z_previous, self.W_zv, self.W_bv, 
                                self.zv_mask, self.zv_trace, self.bv_trace, 
                                self.alpha, self.threshold)

        z_out = self.spike_grad(v_out - self.threshold)
        g_out = self.gate(z_out)

        return v_out, z_out, g_out
    
class LIF_Middle_Voltage(Function):

    @staticmethod
    def forward(ctx, bottom_input, bottom_gate, top_input, top_gate, v_previous, z_previous, gate_previous, W_zv, W_bv, W_tv, W_zv_mask, zv_trace, bv_trace, tv_trace, alpha, v_threshold):

        ungated_bottom_up = bottom_input.mm(W_bv.t())
        bottom_up = bottom_gate * ungated_bottom_up

        masked_W_zv = W_zv * W_zv_mask # mask the self connections
        recurrent_input = z_previous.mm(masked_W_zv.t())

        ungated_top_down = top_input.mm(W_tv.t())
        top_down = top_gate * ungated_top_down

        ungated_dynamics = alpha * v_previous + recurrent_input
        v_out = (1 - gate_previous) * ungated_dynamics + bottom_up + top_down
        v_out = v_out - (z_previous * v_threshold).detach() # reset mechanism

        ctx.save_for_backward(bottom_gate, top_gate, gate_previous, W_bv, W_tv, zv_trace, bv_trace, tv_trace, ungated_bottom_up, ungated_top_down, ungated_dynamics, masked_W_zv)
        ctx.alpha = alpha

        return v_out

    @staticmethod # WORK W/O E-PROP
    def backward(ctx, grad_v):
        # grad_v: batch_size x v
        bottom_gate, top_gate, gate_previous, W_bv, W_tv, zv_trace, bv_trace, tv_trace, ungated_bottom_up, ungated_top_down, ungated_dynamics, masked_W_zv = ctx.saved_tensors

        grad_bottom_input = grad_bottom_gate = grad_top_input = grad_top_gate = grad_v_previous = None
        grad_z_previous = grad_gate_previous = grad_W_zv = grad_W_bv = grad_W_tv = grad_W_zv_mask = None
        grad_zv_trace = grad_bv_trace = grad_tv_trace = grad_alpha = grad_v_threshold = None

        self_gate = 1 - gate_previous

        if ctx.needs_input_grad[0]:
            grad_bottom_input = bottom_gate * grad_v.mm(W_bv)

        if ctx.needs_input_grad[1]:
            grad_bottom_gate = torch.sum(grad_v * ungated_bottom_up, dim=1, keepdim=True)

        if ctx.needs_input_grad[2]:
            grad_top_input = top_gate * grad_v.mm(W_tv)

        if ctx.needs_input_grad[3]:
            grad_top_gate = torch.sum(grad_v * ungated_top_down, dim=1, keepdim=True)

        if ctx.needs_input_grad[4]:
            grad_v_previous = self_gate * ctx.alpha * grad_v
        
        if ctx.needs_input_grad[5]:
            grad_z_previous = self_gate * grad_v.mm(masked_W_zv)
        
        if ctx.needs_input_grad[6]:
            grad_gate_previous = -1 * torch.sum(grad_v * ungated_dynamics, dim=1, keepdim=True)

        # e-prop
        if ctx.needs_input_grad[7]: 
            grad_W_zv = (grad_v.unsqueeze(-1) * zv_trace).sum(0)
            # grad_W_zv =  grad_v.t().mm(self_gate * z_previous) * W_zv_mask
        
        # e-prop
        if ctx.needs_input_grad[8]:
            grad_W_bv = (grad_v.unsqueeze(-1) * bv_trace).sum(0)
            # grad_W_bv = grad_v.t().mm(bottom_gate * bottom_input)

        # e-prop
        if ctx.needs_input_grad[9]:
            grad_W_tv = (grad_v.unsqueeze(-1) * tv_trace).sum(0)
            # grad_W_tv = grad_v.t().mm(top_gate * top_input)

        return (grad_bottom_input, grad_bottom_gate, grad_top_input, grad_top_gate, grad_v_previous, 
                grad_z_previous, grad_gate_previous, grad_W_zv, grad_W_bv, grad_W_tv, grad_W_zv_mask, 
                grad_zv_trace, grad_bv_trace, grad_tv_trace, grad_alpha, grad_v_threshold)

class LIF_Middle_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims,
            next_dims,
            alpha = 0.9, 
            threshold = 1.0, 
            spike_grad = surrogate.atan(alpha=2), 
            linear_options = None, # DOESNT REFER TO ANYTHING (JUST TO KEEP THINGS CONSISTENT)
            timesteps = 360, # timesteps (to normalize e-prop)
        ):
        
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
        self.next_dims = next_dims
        self.alpha = alpha
        self.threshold = threshold
        self.spike_grad = spike_grad
        self.timesteps = timesteps

        # initialize weights
        self.W_zv = nn.Parameter(torch.Tensor(out_dims, out_dims))
        self.W_bv = nn.Parameter(torch.Tensor(out_dims, in_dims))
        self.W_tv = nn.Parameter(torch.Tensor(out_dims, next_dims))

        init_weights(self.W_zv, out_dims, out_dims)
        init_weights(self.W_bv, in_dims, out_dims)
        init_weights(self.W_tv, next_dims, out_dims)

        self.gate = Gate(out_dims, 1, bias=False, spike_fn=spike_grad)

        # register buffers
        self.register_buffer('zv_mask', (1 - torch.eye(out_dims, out_dims)).to(dtype=torch.float32))
        self.register_buffer('zv_trace', torch.zeros((1, out_dims, out_dims)).to(dtype=torch.float32))
        self.register_buffer('bv_trace', torch.zeros((1, out_dims, in_dims)).to(dtype=torch.float32))
        self.register_buffer('tv_trace', torch.zeros((1, out_dims, next_dims)).to(dtype=torch.float32))

    def reset_traces(self):
        self.zv_trace = torch.zeros((1, self.out_dims, self.out_dims)).to(dtype=torch.float32)
        self.bv_trace = torch.zeros((1, self.out_dims, self.in_dims)).to(dtype=torch.float32)
        self.tv_trace = torch.zeros((1, self.out_dims, self.next_dims)).to(dtype=torch.float32)

    def forward(self, bottom_input, bottom_gate, top_input, top_gate, v_previous, z_previous, gate_previous):
        
        # update eligibility traces
        if self.training:
            with torch.no_grad():
                self_gate = 1 - gate_previous
                self.zv_trace = self_gate.unsqueeze(-1) * self.alpha * self.zv_trace + (self_gate * z_previous).unsqueeze(1) * self.zv_mask
                self.bv_trace = self_gate.unsqueeze(-1) * self.alpha * self.bv_trace +  (bottom_gate * bottom_input).unsqueeze(1)
                self.tv_trace = self_gate.unsqueeze(-1) * self.alpha * self.tv_trace + (top_gate * top_input).unsqueeze(1)

        v_out = LIF_Middle_Voltage.apply(bottom_input, bottom_gate, top_input, top_gate, v_previous, z_previous, gate_previous, 
                                         self.W_zv, self.W_bv, self.W_tv, self.zv_mask, self.zv_trace, self.bv_trace, 
                                         self.tv_trace, self.alpha, self.threshold)

        z_out = self.spike_grad(v_out - self.threshold)
        g_out = self.gate(z_out)

        return v_out, z_out, g_out

class LIF_Bottom_Voltage(Function):

    @staticmethod
    def forward(ctx, bottom_input, top_input, top_gate, v_previous, z_previous, gate_previous, W_zv, W_bv, W_tv, W_zv_mask, zv_trace, bv_trace, tv_trace, alpha, v_threshold):

        bottom_up = bottom_input.mm(W_bv.t())

        masked_W_zv = W_zv * W_zv_mask # mask the self connections
        recurrent_input = z_previous.mm(masked_W_zv.t())

        ungated_top_down = top_input.mm(W_tv.t())
        top_down = top_gate * ungated_top_down

        ungated_dynamics = alpha * v_previous + recurrent_input
        gated_dynamics = (1 - gate_previous) * ungated_dynamics
        v_out = gated_dynamics + bottom_up + top_down
        v_out = v_out - (z_previous * v_threshold).detach() # reset mechanism

        ctx.save_for_backward(top_gate, gate_previous, W_bv, W_tv, zv_trace, bv_trace, tv_trace, ungated_top_down, ungated_dynamics, masked_W_zv)
        ctx.alpha = alpha

        return v_out

    @staticmethod # WORKING W E-PROP
    def backward(ctx, grad_v):
        # grad_v: batch_size x v
        top_gate, gate_previous, W_bv, W_tv, zv_trace, bv_trace, tv_trace, ungated_top_down, ungated_dynamics, masked_W_zv = ctx.saved_tensors

        grad_bottom_input = grad_top_input = grad_top_gate = grad_v_previous = None
        grad_z_previous = grad_gate_previous = grad_W_zv = grad_W_bv = grad_W_tv = grad_W_zv_mask = None
        grad_zv_trace = grad_bv_trace = grad_tv_trace = grad_alpha = grad_v_threshold = None

        self_gate = 1 - gate_previous

        if ctx.needs_input_grad[0]:
            grad_bottom_input = grad_v.mm(W_bv)

        if ctx.needs_input_grad[1]:
            grad_top_input = top_gate * grad_v.mm(W_tv)

        if ctx.needs_input_grad[2]:
            grad_top_gate = torch.sum(grad_v * ungated_top_down, dim=1, keepdim=True)

        if ctx.needs_input_grad[3]:
            grad_v_previous = self_gate * ctx.alpha * grad_v
        
        if ctx.needs_input_grad[4]:
            grad_z_previous = self_gate * grad_v.mm(masked_W_zv)
        
        if ctx.needs_input_grad[5]:
            grad_gate_previous = -1 * torch.sum(grad_v * ungated_dynamics, dim=1, keepdim=True)

        # e-prop
        if ctx.needs_input_grad[6]: 
            grad_W_zv = (grad_v.unsqueeze(-1) * zv_trace).sum(0)
            # grad_W_zv = grad_v.t().mm(self_gate * z_previous) * W_zv_mask
        
        # e-prop
        if ctx.needs_input_grad[7]:
            grad_W_bv = (grad_v.unsqueeze(-1) * bv_trace).sum(0)
            #grad_W_bv = grad_v.t().mm(bottom_input)

        # e-prop
        if ctx.needs_input_grad[8]:
            grad_W_tv = (grad_v.unsqueeze(-1) * tv_trace).sum(0)
            #grad_W_tv = grad_v.t().mm(top_gate * top_input)

        return (grad_bottom_input, grad_top_input, grad_top_gate, grad_v_previous, 
                grad_z_previous, grad_gate_previous, grad_W_zv, grad_W_bv, grad_W_tv, grad_W_zv_mask, 
                grad_zv_trace, grad_bv_trace, grad_tv_trace, grad_alpha, grad_v_threshold)

class LIF_Bottom_Cell(nn.Module):

    def __init__(
            self, 
            in_dims, 
            out_dims,
            next_dims,
            alpha = 0.9, 
            threshold = 1.0, 
            spike_grad = surrogate.atan(alpha=2), 
            linear_options = None,
            timesteps = 360, # timesteps (to normalize e-prop)
        ):
        
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
        self.next_dims = next_dims
        self.alpha = alpha
        self.threshold = threshold
        self.spike_grad = spike_grad
        self.timesteps = timesteps

        # initialize weights
        self.W_zv = nn.Parameter(torch.Tensor(out_dims, out_dims))
        self.W_bv = nn.Parameter(torch.Tensor(out_dims, in_dims))
        self.W_tv = nn.Parameter(torch.Tensor(out_dims, next_dims))

        init_weights(self.W_zv, out_dims, out_dims)
        init_weights(self.W_bv, in_dims, out_dims)
        init_weights(self.W_tv, next_dims, out_dims)

        self.gate = Gate(out_dims, 1, bias=False, spike_fn=spike_grad)

        # register buffers
        self.register_buffer('zv_mask', (1 - torch.eye(out_dims, out_dims)).to(dtype=torch.float32))
        self.register_buffer('zv_trace', torch.zeros((1, out_dims, out_dims)).to(dtype=torch.float32))
        self.register_buffer('bv_trace', torch.zeros((1, out_dims, in_dims)).to(dtype=torch.float32))
        self.register_buffer('tv_trace', torch.zeros((1, out_dims, next_dims)).to(dtype=torch.float32))

    def reset_traces(self):
        self.zv_trace = torch.zeros((1, self.out_dims, self.out_dims)).to(dtype=torch.float32)
        self.bv_trace = torch.zeros((1, self.out_dims, self.in_dims)).to(dtype=torch.float32)
        self.tv_trace = torch.zeros((1, self.out_dims, self.next_dims)).to(dtype=torch.float32)

    def forward(self, bottom_input, top_input, top_gate, v_previous, z_previous, gate_previous):

        # update eligibility traces
        if self.training:
            with torch.no_grad():
                self_gate = 1 - gate_previous
                self.zv_trace = self_gate.unsqueeze(-1) * self.alpha * self.zv_trace + (self_gate * z_previous).unsqueeze(1) * self.zv_mask
                self.bv_trace = self_gate.unsqueeze(-1) * self.alpha * self.bv_trace +  bottom_input.unsqueeze(1)
                self.tv_trace = self_gate.unsqueeze(-1) * self.alpha * self.tv_trace + (top_gate * top_input).unsqueeze(1)

        v_out = LIF_Bottom_Voltage.apply(bottom_input, top_input, top_gate, v_previous, z_previous, gate_previous, 
                                         self.W_zv, self.W_bv, self.W_tv, self.zv_mask, self.zv_trace, self.bv_trace, 
                                         self.tv_trace, self.alpha, self.threshold)

        z_out = self.spike_grad(v_out - self.threshold)
        g_out = self.gate(z_out)

        return v_out, z_out, g_out
    
class TwoLayer_EpropHierarchicalSNN(nn.Module):

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

        super(TwoLayer_EpropHierarchicalSNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.out_type = out_type

        if isinstance(neuron_options, dict):
            neuron_options = [neuron_options, neuron_options]

        self.neuron_type = 'lif'
        self.bottom_cell = LIF_Bottom_Cell(self.in_dims, self.recurrent_dims[0], self.out_dims, **neuron_options[0])
        self.top_cell = LIF_Top_Cell(self.recurrent_dims[0], self.out_dims, **neuron_options[1])

    def reset_traces(self):
        self.bottom_cell.reset_traces()
        self.top_cell.reset_traces()

    def _forward_lif(self, x, logging):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        if logging:
            b_0s = []
            b_1s = []

        self.reset_traces() # reset etraces

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1, requires_grad=True)

        v_1 = torch.zeros(B, self.out_dims)
        z_1 = torch.zeros(B, self.out_dims)
        b_1 = torch.zeros(B, 1, requires_grad=True)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            # compute the past time's contribution on gradient wrt to projection weights
            if self.training:
                b_0.register_hook(compute_dL_dg_dW_grad(self.bottom_cell.gate.projection.weight, z_0))
                b_1.register_hook(compute_dL_dg_dW_grad(self.top_cell.gate.projection.weight, z_1))

            v_0, z_0, b_0 = self.bottom_cell(x_t, z_1, b_1, v_0, z_0, b_0) # output spikes are inputs to next layer
            v_1, z_1, b_1 = self.top_cell(z_0, b_0, v_1, z_1)

            outputs.append(z_1.clone())

            v_0 = v_0.detach()
            z_0 = z_0.detach()
            b_0 = b_0.detach().requires_grad_()

            v_1 = v_1.detach()
            z_1 = z_1.detach()
            b_1 = b_1.detach().requires_grad_()

            if logging:
                b_0s.append(b_0.clone())
                b_1s.append(b_1.clone())

        outputs = torch.stack(outputs, 1)

        if logging:
            return outputs, z_1, [torch.stack(b_0s, 1), torch.stack(b_1s, 1)]

        return outputs, z_1
    
    
    def forward(self, x, logging=False):
        return self._forward_lif(x, logging)

        
class ThreeLayer_EpropHierarchicalSNN(nn.Module):

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

        super(ThreeLayer_EpropHierarchicalSNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.out_type = out_type

        if isinstance(neuron_options, dict):
            neuron_options = [neuron_options, neuron_options, neuron_options]

        self.neuron_type = 'lif'
        self.bottom_cell = LIF_Bottom_Cell(self.in_dims, self.recurrent_dims[0], self.recurrent_dims[1], **neuron_options[0])
        self.middle_cell = LIF_Middle_Cell(self.recurrent_dims[0], self.recurrent_dims[1], self.recurrent_dims[2], **neuron_options[1])
        self.top_cell = LIF_Top_Cell(self.recurrent_dims[1], self.out_dims, **neuron_options[2])

    def reset_traces(self):
        self.bottom_cell.reset_traces()
        self.middle_cell.reset_traces()
        self.top_cell.reset_traces()

    def _forward_lif(self, x, logging):
        # x shape: B x T x IN_DIMS

        B, T, _ = x.shape

        if logging:
            b_0s = []
            b_1s = []
            b_2s = []

        self.reset_traces() # reset etraces

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1, requires_grad=True)

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])
        b_1 = torch.zeros(B, 1, requires_grad=True)

        v_2 = torch.zeros(B, self.recurrent_dims[2])
        z_2 = torch.zeros(B, self.recurrent_dims[2])
        b_2 = torch.zeros(B, 1, requires_grad=True)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            # compute the past time's contribution on gradient wrt to projection weights
            if self.training:
                b_0.register_hook(compute_dL_dg_dW_grad(self.bottom_cell.gate.projection.weight, z_0))
                b_1.register_hook(compute_dL_dg_dW_grad(self.middle_cell.gate.projection.weight, z_1))
                b_2.register_hook(compute_dL_dg_dW_grad(self.top_cell.gate.projection.weight, z_2))

            v_0, z_0, b_0 = self.bottom_cell(x_t, z_1, b_1, v_0, z_0, b_0) # output spikes are inputs to next layer
            v_1, z_1, b_1 = self.middle_cell(z_0, b_0, z_2, b_2, v_1, z_1, b_1)
            v_2, z_2, b_2 = self.top_cell(z_1, b_1, v_2, z_2)

            outputs.append(z_2.clone())

            if logging:
                b_0s.append(b_0.detach().clone())
                b_1s.append(b_1.detach().clone())
                b_2s.append(b_2.detach().clone())

            v_0 = v_0.detach()
            z_0 = z_0.detach()
            b_0 = b_0.detach().requires_grad_()

            v_1 = v_1.detach()
            z_1 = z_1.detach()
            b_1 = b_1.detach().requires_grad_()

            v_2 = v_2.detach()
            z_2 = z_2.detach()
            b_2 = b_2.detach().requires_grad_()

        outputs = torch.stack(outputs, 1)

        if logging:
            return outputs, z_2, [torch.stack(b_0s, 1), torch.stack(b_1s, 1), torch.stack(b_2s, 1)]

        return outputs, z_2
    
    
    def forward(self, x, logging=False):
        return self._forward_lif(x, logging)


class FourLayer_EpropHierarchicalSNN(nn.Module):

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

        super(FourLayer_EpropHierarchicalSNN, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.recurrent_dims = recurrent_dims
        self.out_type = out_type

        if isinstance(neuron_options, dict):
            neuron_options = [neuron_options, neuron_options, neuron_options]

        # the top cell emits recurrent_dims[2] features; the readout does the classification
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

    def reset_traces(self):
        self.bottom_cell.reset_traces()
        self.middle_cell.reset_traces()
        self.top_cell.reset_traces()

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
            b_2s = []

        self.reset_traces() # reset etraces

        v_0 = torch.zeros(B, self.recurrent_dims[0])
        z_0 = torch.zeros(B, self.recurrent_dims[0])
        b_0 = torch.zeros(B, 1, requires_grad=True)

        v_1 = torch.zeros(B, self.recurrent_dims[1])
        z_1 = torch.zeros(B, self.recurrent_dims[1])
        b_1 = torch.zeros(B, 1, requires_grad=True)

        v_2 = torch.zeros(B, self.recurrent_dims[2])
        z_2 = torch.zeros(B, self.recurrent_dims[2])
        b_2 = torch.zeros(B, 1, requires_grad=True)

        v_fc = torch.zeros(B, self.out_dims)
        z_fc = torch.zeros(B, self.out_dims)
        a_fc = torch.zeros(B, self.out_dims)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            # compute the past time's contribution on gradient wrt to projection weights
            if self.training:
                b_0.register_hook(compute_dL_dg_dW_grad(self.bottom_cell.gate.projection.weight, z_0))
                b_1.register_hook(compute_dL_dg_dW_grad(self.middle_cell.gate.projection.weight, z_1))
                b_2.register_hook(compute_dL_dg_dW_grad(self.top_cell.gate.projection.weight, z_2))

            v_0, z_0, b_0 = self.bottom_cell(x_t, z_1, b_1, v_0, z_0, b_0) # output spikes are inputs to next layer
            v_1, z_1, b_1 = self.middle_cell(z_0, b_0, z_2, b_2, v_1, z_1, b_1)
            v_2, z_2, b_2 = self.top_cell(z_1, b_1, v_2, z_2)

            output, v_fc, z_fc, a_fc = self._readout(z_2, v_fc, z_fc, a_fc)

            outputs.append(output.clone())

            if logging:
                b_0s.append(b_0.detach().clone())
                b_1s.append(b_1.detach().clone())
                b_2s.append(b_2.detach().clone())

            v_0 = v_0.detach()
            z_0 = z_0.detach()
            b_0 = b_0.detach().requires_grad_()

            v_1 = v_1.detach()
            z_1 = z_1.detach()
            b_1 = b_1.detach().requires_grad_()

            v_2 = v_2.detach()
            z_2 = z_2.detach()
            b_2 = b_2.detach().requires_grad_()

            # keep the readout state truncated as well, consistent with the eprop core
            v_fc = v_fc.detach()
            z_fc = z_fc.detach()
            a_fc = a_fc.detach()

        outputs = torch.stack(outputs, 1)

        if logging:
            return outputs, output, [torch.stack(b_0s, 1), torch.stack(b_1s, 1), torch.stack(b_2s, 1)]

        return outputs, output


    def forward(self, x, logging=False):
        return self._forward_lif(x, logging)


if __name__ == '__main__':

    def top_check(bottom_input, bottom_gate, v_previous, z_previous, W_zv, W_bv, W_zv_mask, zv_trace, bv_trace, alpha, v_threshold):
        return LIF_Top_Voltage.apply(bottom_input, bottom_gate, v_previous, z_previous, W_zv, W_bv, W_zv_mask, zv_trace, bv_trace, alpha, v_threshold)
    
    def middle_check(bottom_input, bottom_gate, top_input, top_gate, v_previous, z_previous, gate_previous, W_zv, W_bv, W_tv, W_zv_mask, zv_trace, bv_trace, tv_trace, alpha, v_threshold):
        return LIF_Middle_Voltage.apply(bottom_input, bottom_gate, top_input, top_gate, v_previous, z_previous, gate_previous, W_zv, W_bv, W_tv, W_zv_mask, zv_trace, bv_trace, tv_trace, alpha, v_threshold)
    
    def bottom_check(bottom_input, top_input, top_gate, v_previous, z_previous, gate_previous, W_zv, W_bv, W_tv, W_zv_mask, zv_trace, bv_trace, tv_trace, alpha, v_threshold):
        return LIF_Bottom_Voltage.apply(bottom_input, top_input, top_gate, v_previous, z_previous, gate_previous, W_zv, W_bv, W_tv, W_zv_mask, zv_trace, bv_trace, tv_trace, alpha, v_threshold)


    B = 4
    T = 6
    I = 1
    H1 = 8
    H2 = 12
    O = 1

    z = torch.rand((B, H1))
    W = torch.rand((1, H1), requires_grad=True)
    spike_fn=surrogate.atan(alpha=2)

    x = z @ W.T
    x = F.sigmoid(x) - 0.5
    x = spike_fn(x)

    scale = torch.rand_like(x)

    loss = (scale * x).sum()
    print(torch.autograd.grad(loss, (W)))

    print(gate_op_local_grads(W, z, scale))

    # x = torch.rand((B, T, 1))
    # net = ThreeLayer_EpropHierarchicalSNN(I, O, [H1, H2, O])
    # # net = TwoLayer_EpropHierarchicalSNN(I, O, [H1, O])
    # y, _ = net(x)

    # loss = (y - torch.rand_like(y)).mean()
    # loss.backward()

    # bottom_input = torch.rand((B, I), requires_grad=True, dtype=torch.float64)
    # bottom_gate = torch.rand((B, 1), requires_grad=True, dtype=torch.float64)

    # top_input = torch.rand((B, O), requires_grad=True, dtype=torch.float64)
    # top_gate = torch.rand((B, 1), requires_grad=True, dtype=torch.float64)

    # v_previous = torch.rand((B, H), requires_grad=True, dtype=torch.float64)
    # z_previous = torch.rand((B, H), requires_grad=True, dtype=torch.float64)
    # gate_previous = torch.rand((B, 1), requires_grad=True, dtype=torch.float64)

    # W_zv = torch.rand((H, H), requires_grad=True, dtype=torch.float64)
    # W_bv = torch.rand((H, I), requires_grad=True, dtype=torch.float64)
    # W_tv = torch.rand((H, O), requires_grad=True, dtype=torch.float64)

    # W_zv_mask = (1 - torch.eye(H, H)).to(torch.float64)
    # zv_trace = torch.zeros((1, H, H))
    # bv_trace = torch.zeros((1, H, I))
    # tv_trace = torch.zeros((1, H, O))
    # alpha = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    # v_threshold = torch.tensor(1.0, dtype=torch.float64, requires_grad=False)

    # self_gate = 1 - gate_previous
    # with torch.no_grad():
    #     zv_trace = self_gate.unsqueeze(-1) * alpha * zv_trace + (self_gate * z_previous).unsqueeze(1) * W_zv_mask
    #     bv_trace = self_gate.unsqueeze(-1) * alpha * bv_trace +  bottom_input.unsqueeze(1)
    #     tv_trace = self_gate.unsqueeze(-1) * alpha * tv_trace + (top_gate * top_input).unsqueeze(1)


    # input =(bottom_input, top_input, top_gate, v_previous, z_previous, gate_previous, W_zv, W_bv, W_tv, W_zv_mask, zv_trace, bv_trace, tv_trace, alpha, v_threshold)
    # test = torch.autograd.gradcheck(bottom_check, input, eps=1e-6, atol=1e-4)

    # self_gate = 1 - gate_previous
    # with torch.no_grad():
    #     zv_trace = self_gate.unsqueeze(-1) * alpha * zv_trace + (self_gate * z_previous).unsqueeze(1) * W_zv_mask
    #     bv_trace = self_gate.unsqueeze(-1) * alpha * bv_trace +  (bottom_gate * bottom_input).unsqueeze(1)
    #     tv_trace = self_gate.unsqueeze(-1) * alpha * tv_trace + (top_gate * top_input).unsqueeze(1)

    # input = (bottom_input, bottom_gate, top_input, top_gate, v_previous, z_previous, gate_previous, W_zv, W_bv, W_tv, W_zv_mask, zv_trace, bv_trace, tv_trace, alpha, v_threshold)    
    # test = torch.autograd.gradcheck(middle_check, input, eps=1e-6, atol=1e-4)
    # with torch.no_grad():
    #     zv_trace = alpha * zv_trace + z_previous.unsqueeze(1) * W_zv_mask
    #     bv_trace = alpha * bv_trace +  (bottom_gate * bottom_input).unsqueeze(1)

    # input = (bottom_input, bottom_gate, v_previous, z_previous, W_zv, W_bv, W_zv_mask, zv_trace, bv_trace, alpha, v_threshold)
    # test = torch.autograd.gradcheck(top_check, input, eps=1e-6, atol=1e-4)

    # print(test)

