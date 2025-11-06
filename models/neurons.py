import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
import snntorch.surrogate as surrogate

# basic LIF Neuron: 
# S[t] = V[t] > V_threshold[t]
# V[t+1] = beta * V[t] + X[t+1] - R[t]
# R[t] = beta * V_threshold * S[t] -> detach from the computational graph (it's a "hard-coded reset" during backwards pass [NOT learnable])

class LIF(nn.Module):
    def __init__(self, beta=0.9, threshold=1.0, spike_grad=surrogate.atan(alpha=2)):
        '''
        in_shape: shape of layer input
        beta: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        '''

        super(LIF, self).__init__()

        self.beta = beta
        self.threshold = threshold
        self.spike_grad = spike_grad

    def forward(self, v, x):
        spike = self.spike_grad(v - self.threshold)
        reset = (self.beta * self.threshold * spike).detach()
        v = self.beta * v + x - reset
        return spike, v
    
# ALIF Neuron (https://www.nature.com/articles/s41467-020-17236-y)
# basic idea: we want the spiking threshold to change based on previous spiking activity (harder to spike when it recently spiked)
# V_thresh[t] = V_threshold + beta * A[t]
# S[t] = V[t] > V_thresh[t]
# R[t] = alpha * V_thresh[t] * S[t] -> detach from the computational graph (it's a "hard-coded reset" during backwards pass [NOT learnable])
# V[t+1] = alpha * V[t] + X[t+1] - R[t]
# A[t+1] = rho * A[t] + S[t] -> detach from computational graph (it's a "hard coded" rule, NOT LEARNABLE)

class ALIF(nn.Module):
    def __init__(self, alpha=0.9, beta=0.1, rho=0.1, threshold=1.0, spike_grad=surrogate.atan(alpha=2)):
        '''
        in_shape: shape of layer input
        beta: voltage decay
        threshold: spiking threshold
        spike_grad: spiking function with surrogate gradient (built into snntorch)
        '''

        super(ALIF, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.threshold = threshold
        self.spike_grad = spike_grad

    def forward(self, v, a, x):
        v_thresh = self.beta * a + self.threshold
        spike = self.spike_grad(v - v_thresh)
        reset = (self.alpha * v_thresh * spike).detach()
        v = self.alpha * v + x - reset
        a = (self.rho * a + spike).detach()
        return spike, v, a

if __name__ == '__main__':
    v = torch.randn(4, requires_grad=True)
    x = torch.randn(4)
    y = torch.randn(4)

    neuron = LIF()

    spike_out, v_out = neuron(v, x)

    loss = sum((spike_out - y) ** 2)
    loss.backward()

    print(v, x, y)
    print(spike_out, v_out)