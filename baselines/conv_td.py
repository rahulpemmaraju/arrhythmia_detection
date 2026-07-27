import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

'''
This file introduces a top-down RNN/ConvTD. neurons in this multilayered network maintain two components: a state (s) and read-out (r)
K[t] = K0 * f(s[t])
s[t] = Rs[t-1] + K(s[t-1])x[t]
r[t] = Ur[l+1, t-1] + Ws[t]
output is a concatenation of the bottom-layer readouts

Note that in a multilayer network with R=0, U=I, W=I, and f(x)=1, this degenerates to a 1D convolution.
The intuition behind this architecture is that we want to combine benefits of both network types: we want to dynamically
leverage different conv filters dependent on sequence position
'''


class ConvTD(nn.Module):

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            kernel_size: int = 3,
            padding: int = 0,
            fixed_R: bool = True,
            fixed_U: bool = True,
            fixed_W: bool = True,
            dynamic_K: bool = False,
        ):

        '''
        the top-down recurrence above, as a sequence-to-sequence layer.

        kernel_size: depth of the layer, and therefore the receptive field in samples. equals the kernel size of the equivalent conv.
        padding: zero padding, in the conv1d sense
        fixed_R: whether to make s-to-s recurrence learnable
        fixed_U: whether to make top-down projection learnable
        fixed_W: whether to make s-to-readout projection learnable
        dynamic_K: whether to implement state-dependent kernel reweighting
        * keeping fixed_R = fixed_U = fixed_W = True, dynamic_K = False directly recovers convolution *
        '''

        super(ConvTD, self).__init__()

        if not 0 <= padding <= kernel_size - 1:
            raise ValueError(
                f'padding must be in [0, kernel_size - 1] = [0, {kernel_size - 1}], got {padding}'
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.padding = padding

        # readouts before this index are partial sums: layer 0 at time t is still
        # missing the taps sitting in the layers above it
        self.lag = kernel_size - 1 - padding

        self.K0 = nn.Parameter(torch.empty(self.kernel_size, 1, out_dim, in_dim))

        # R = 0, U = I, W = I is the conv-equivalent point of the parameter space.
        self.R = nn.Parameter(torch.zeros(kernel_size, out_dim, out_dim))
        self.U = nn.Parameter(torch.eye(out_dim).repeat(kernel_size, 1, 1))
        self.W = nn.Parameter(torch.eye(out_dim).repeat(kernel_size, 1, 1))
        self.Ks = nn.Parameter(torch.zeros(out_dim, out_dim))

        if fixed_R:
            self.R.requires_grad = False
        if fixed_U:
            self.U.requires_grad = False
        if fixed_W:
            self.W.requires_grad = False

        if not dynamic_K:
            self.Ks.requires_grad = False
            self.s_norm = nn.Identity()
        else:
            self.s_norm = nn.LayerNorm(out_dim)

        bound = 1 / (in_dim * kernel_size) ** 0.5
        nn.init.uniform_(self.K0, -bound, bound)
        

    def out_len(self, timesteps: int) -> int:
        '''length of the returned sequence, matching conv1d's output length'''

        return timesteps + 2 * self.padding - self.kernel_size + 1

    def kernel(self, s: torch.Tensor) -> torch.Tensor:
        '''
        K[t] for every layer, given the previous state s[t-1]

        K[t] = K[t] * exp(s @ Ks.T) -> s is a layernorm value so this value should stay within a reasonable range
        '''

        weight = self.s_norm(s) @ self.Ks.T
        exp_weight = torch.exp(weight)

        return exp_weight.unsqueeze(-1) * self.K0

    def forward(self, x):
        # x shape: B x T x IN_DIMS
        batch_size = x.shape[0]

        # right pad only -- the left side falls out of the zero init of s and r
        seq = F.pad(x, (0, 0, 0, self.padding)).transpose(0, 1)  # (T + padding, B, in_dim)

        s = x.new_zeros(self.kernel_size, batch_size, self.out_dim)
        r = x.new_zeros(self.kernel_size, batch_size, self.out_dim)

        readouts = []

        for xt in seq:
            # K is read off s[t-1], before the update
            K = self.kernel(s)

            s = s @ self.R.transpose(1, 2) + (K @ xt.unsqueeze(-1)).squeeze(-1)

            td = torch.cat([r[1:], torch.zeros_like(r[:1])], dim=0)

            # states are row vectors now, so matrices apply on the right, transposed
            r = td @ self.U.transpose(1, 2) + s @ self.W.transpose(1, 2)
            readouts.append(r[0])

        # drop the partial sums; what remains is aligned with conv1d's output
        outputs = torch.stack(readouts[self.lag:], dim=1)  # B x T' x out_dim

        return outputs


if __name__ == '__main__':

    torch.manual_seed(0)

    in_dim = 16
    out_dim = 32
    batch_size = 4
    timesteps = 8

    input = torch.rand((batch_size, timesteps, in_dim))
    td = ConvTD(in_dim, out_dim, kernel_size=3, padding=1, fixed_R=True, fixed_U=True, fixed_W=True, dynamic_K=True)
    out = td(input)

    # the equivalence holds for every kernel size and every padding conv1d itself
    # accepts. k=3, padding=1 is the one case where the off-by-ones coincide, so
    # testing only that case hides them
    for kernel_size in (1, 2, 3, 4, 5, 7):
        for padding in range(kernel_size):

            td = ConvTD(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
            readouts = td(input)

            # the reference conv absorbs the reindexing, not the network: the
            # recurrence keeps its own layout (layer l = tap l, time-major, state
            # in columns) and conv1d is bent to meet it. permute maps
            # (kernel_size, out, in) onto conv1d's (out, in, k), and the flip undoes
            # the fact that conv1d cross-correlates -- it reads tap j at x[t + j],
            # while layer l reads x[t - l]
            weight = td.K0.squeeze(1).permute(1, 2, 0).flip(-1)
            conv = F.conv1d(input.transpose(1, 2), weight, padding=padding).transpose(1, 2)

            assert readouts.shape == conv.shape, (readouts.shape, conv.shape)
            assert readouts.shape[1] == td.out_len(timesteps)

            err = (conv - readouts).abs().max()
            assert err < 1e-4, f'kernel_size={kernel_size} padding={padding} err={err}'

            print(f'kernel_size={kernel_size} padding={padding} '
                  f'out_len={readouts.shape[1]:2d} err={err:.2e}')

