import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

'''
This file introduces a top-down RNN. neurons in this multilayered network maintain two components: a state (s) and read-out (r)
K[t] = K0 * f(x[t])
s[t] = Rs[t-1] + K(s[t-1])x[t]
r[t] = Ur[l+1, t-1] + Ws[t]
output is a concatenation of the bottom-layer readouts

Note that in a multilayer network with R=0, U=I, W=I, and f(x)=1, this degenerates to a 1D convolution. 
The intuition behind this architecture is that we want to combine benefits of both network types: we want to dynamically
leverage different conv filters dependent on sequence position
'''

if __name__ == '__main__':
    k = 3
    out_dim = 32
    in_dim = 16
    batch_size = 4
    timesteps = 8
    padding = 1

    input = torch.rand((batch_size, timesteps, in_dim))

    R = torch.zeros(out_dim, out_dim)
    K = torch.rand((out_dim, in_dim, k))
    U = torch.eye(out_dim)
    W = torch.eye(out_dim)

    s = torch.zeros(k, batch_size, out_dim, 1)
    r = torch.zeros(k, batch_size, out_dim, 1)
    readouts = []

    input_to_network = F.pad(input, (0, 0, 0, padding)).transpose(0, 1)
    k_for_network = K.permute(2, 0, 1).unsqueeze(1)

    for x in input_to_network:
        s = R @ s + k_for_network @ x.unsqueeze(-1)

        td = torch.zeros_like(r)

        #r = U @ s + W @ 
        

