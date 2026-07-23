import torch
import torch.nn as nn

from typing import List

'''
non-spiking baselines.

these exist to answer "how much of the performance is the spiking dynamics, and
how much is just having a sequence model on this data?", so they deliberately
keep the same input/output contract as the SNNs in ../models:

    input   (B, T, in_dims)   -- the z-scored waveform
    output  (B, T, out_dims)  -- a logit per timestep, plus the final timestep's
                                 output as the second return value

matching that contract means the beat-span masking, the summed-logit readout and
the metrics helpers all apply to these models unchanged, so the numbers land on
the same footing as the SNN numbers. nothing here imports from ../models and
nothing in ../models imports from here.
'''


class Conv1D_Baseline(nn.Module):

    def __init__(
            self,
            in_dims: int = 1,
            out_dims: int = 1,
            hidden_dims: List[int] = [32, 64, 32],
            kernel_size: int = 7,
            dilate: bool = True,
            dilation_base: int = 2,
            causal: bool = True,
            dropout: float = 0.0,
            norm: str = 'none',
            residual: bool = False,
            readout: str = 'per_timestep',
            beat_bounds=None,
            bias: bool = False,
        ):

        '''
        a dilated 1d conv net (TCN-style) over the raw waveform.

        in_dims: channels of the input waveform (1 for single-lead)
        out_dims: number of readout units -- 1 for binary, class count otherwise
        hidden_dims: channel width of each conv block
        kernel_size: conv width, in samples
        dilate: dilate block i by dilation_base**i. dilation adds no parameters,
            so this is the only way to widen the receptive field at a fixed
            layer count and model size -- without it, n blocks see just
            n*(k-1) + 1 samples, which at n=3, k=7 is 19 samples (~0.11s at the
            dataset's 180Hz), barely a QRS complex and well short of a beat.
        dilation_base: growth factor of the dilation per block. keep it <=
            kernel_size: once a block's dilation exceeds its kernel width, the
            block skips over input samples entirely and the receptive field is
            full of holes rather than densely covered.
        causal: left-pad only, so timestep t never sees t+1. the SNNs are causal
            by construction, so a non-causal baseline would be getting strictly
            more information than the model it is compared against.
        dropout: dropout after each block (0 disables). defaults off -- the SNNs
            train without it, and a baseline with regularization the SNN lacks
            is not measuring the architecture.
        norm: 'batch', 'layer' or 'none'. defaults off for the same reason:
            BatchNorm is a large trainability advantage unrelated to conv vs
            spiking, so it does not belong in an architecture comparison.
        residual: skip connection around each block, projected by a 1x1 conv
            where the width changes. defaults off -- the SNN stacks are plain
            feedforward, and the projections would add parameters too.
        readout: 'per_timestep' emits an independent logit at each t, the direct
            analog of the SNN readout. 'pooled' averages features over the
            supervised window and broadcasts one logit across t -- a stronger
            classifier, but no longer an online/streaming model.
        beat_bounds: (start, end) window-local indices of the labeled beat span,
            from dataloader.get_beat_bounds. only used by the 'pooled' readout,
            and must match the span the loss uses -- pooling over the whole
            window would average in the pre/post context, which does not belong
            to the labeled beat.
        bias: whether the convs carry a bias term. defaults off to match the
            SNN configs, which all run bias: false.
        '''

        super(Conv1D_Baseline, self).__init__()

        if readout not in ('per_timestep', 'pooled'):
            raise ValueError(f"readout must be 'per_timestep' or 'pooled', got {readout!r}")

        if dilate and dilation_base > kernel_size:
            print(f'warning: dilation_base ({dilation_base}) > kernel_size ({kernel_size}), '
                  'so the upper blocks skip input samples and the receptive field is '
                  'not densely covered')

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.causal = causal
        self.readout = readout
        self.beat_bounds = beat_bounds

        self.residual = residual

        self.blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.dilations = []

        widths = [in_dims] + list(hidden_dims)

        for i, (c_in, c_out) in enumerate(zip(widths[:-1], widths[1:])):
            dilation = dilation_base ** i if dilate else 1
            self.dilations.append(dilation)

            # padding is applied by hand in forward (causal needs it one-sided),
            # so the conv itself pads nothing
            layers = [nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation, bias=bias)]

            if norm == 'batch':
                layers.append(nn.BatchNorm1d(c_out))
            elif norm == 'layer':
                # GroupNorm with one group normalizes over (C, T) per sample, and
                # unlike BatchNorm1d does not depend on batch composition
                layers.append(nn.GroupNorm(1, c_out))
            elif norm != 'none':
                raise ValueError(f"norm must be 'batch', 'layer' or 'none', got {norm!r}")

            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            self.blocks.append(nn.Sequential(*layers))

            if residual:
                # a skip across a width change needs a 1x1 projection to add
                self.residuals.append(
                    nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
                )

        self.fc = nn.Linear(hidden_dims[-1], out_dims)

    def receptive_field(self):
        # how many input samples one output timestep can see. worth checking
        # against the window length: if it is much shorter than a beat the model
        # is structurally unable to use morphology, and if it exceeds the window
        # the extra dilation is only padding.
        return 1 + sum((self.kernel_size - 1) * d for d in self.dilations)

    def _pad(self, x, dilation):
        # total padding that keeps T unchanged for this kernel/dilation
        pad = (self.kernel_size - 1) * dilation

        if self.causal:
            return nn.functional.pad(x, (pad, 0))

        left = pad // 2

        return nn.functional.pad(x, (left, pad - left))

    def forward(self, x, logging=False):
        # x shape: B x T x IN_DIMS -> conv1d wants B x C x T
        h = x.transpose(1, 2)

        for i, (block, dilation) in enumerate(zip(self.blocks, self.dilations)):
            out = block(self._pad(h, dilation))
            h = out + self.residuals[i](h) if self.residual else out

        h = h.transpose(1, 2)  # back to B x T x C

        if self.readout == 'pooled':
            start, end = self.beat_bounds if self.beat_bounds is not None else (0, h.shape[1])
            output = self.fc(h[:, start:end].mean(dim=1))
            # the label is constant across the window, so broadcasting one logit
            # over t leaves every downstream readout (sum over t, target[:, 0])
            # meaning exactly what it means for the SNNs
            outputs = output.unsqueeze(1).expand(-1, h.shape[1], -1)

            return outputs, output

        outputs = self.fc(h)

        return outputs, outputs[:, -1]


class ConvNet_Baseline(nn.Module):

    def __init__(
            self,
            in_dims: int = 1,
            out_dims: int = 1,
            hidden_dims: List[int] = [32, 64, 32],
            kernel_size: int = 5,
            pool_size: int = 2,
            pool_type: str = 'max',
            global_pool: str = 'avg',
            dropout: float = 0.0,
            norm: str = 'none',
            beat_bounds=None,
            bias: bool = False,
        ):

        '''
        a conventional 1d CNN classifier: conv -> relu -> pool, repeated, then a
        global pool and a single dense readout. one prediction per window.

        this is the standard shape for ECG beat classification, and differs from
        Conv1D_Baseline in three ways that all matter for the comparison:

        - it downsamples. each pool halves the time axis, so a fixed kernel
          covers twice the input span in the next block. the receptive field
          therefore grows without dilation, and deeper layers work on a coarser,
          smoother signal.
        - it is not causal. a window classifier has the whole beat in hand, so
          there is no reason to hide the future from it -- but that also means it
          is not an online model, and is getting information the SNNs never see.
        - it predicts once per window rather than per timestep, which is what
          "pools and makes one prediction" means and what the pooling buys.

        the output is still broadcast to (B, T, out_dims) so the loss, the
        beat-span masking and the metrics helpers all apply unchanged -- the
        label is constant across a window, so a broadcast logit means exactly
        what a per-timestep logit means to everything downstream.

        kernel_size: conv width in samples, at that block's resolution. must be
            odd so 'same' padding is symmetric.
        pool_size: downsampling factor per block (1 disables pooling)
        pool_type: 'max' or 'avg', the per-block pool
        global_pool: 'avg' or 'max', collapsing time before the dense readout
        beat_bounds: the global pool covers only the labeled beat span, matching
            what the loss supervises. pooling the whole window would fold in the
            pre/post context, which does not belong to the labeled beat.
        bias: defaults off to match the SNN configs
        '''

        super(ConvNet_Baseline, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError(f'kernel_size must be odd for symmetric padding, got {kernel_size}')

        if pool_type not in ('max', 'avg'):
            raise ValueError(f"pool_type must be 'max' or 'avg', got {pool_type!r}")

        if global_pool not in ('avg', 'max'):
            raise ValueError(f"global_pool must be 'avg' or 'max', got {global_pool!r}")

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.global_pool = global_pool
        self.beat_bounds = beat_bounds

        blocks = []
        widths = [in_dims] + list(hidden_dims)

        for c_in, c_out in zip(widths[:-1], widths[1:]):
            layers = [nn.Conv1d(c_in, c_out, kernel_size,
                                padding=(kernel_size - 1) // 2, bias=bias)]

            if norm == 'batch':
                layers.append(nn.BatchNorm1d(c_out))
            elif norm == 'layer':
                layers.append(nn.GroupNorm(1, c_out))
            elif norm != 'none':
                raise ValueError(f"norm must be 'batch', 'layer' or 'none', got {norm!r}")

            layers.append(nn.ReLU())

            if pool_size > 1:
                layers.append(nn.MaxPool1d(pool_size) if pool_type == 'max'
                              else nn.AvgPool1d(pool_size))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            blocks.append(nn.Sequential(*layers))

        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(hidden_dims[-1], out_dims)

    def receptive_field(self):
        # in input samples. each pool multiplies the stride, so the span a kernel
        # covers doubles per block -- reported so it can be checked against the
        # window length the same way Conv1D_Baseline's is.
        rf, stride = 1, 1

        for _ in self.hidden_dims:
            rf += (self.kernel_size - 1) * stride
            stride *= max(1, self.pool_size)

        return rf

    def forward(self, x, logging=False):
        # x shape: B x T x IN_DIMS -> conv1d wants B x C x T
        T = x.shape[1]

        h = self.blocks(x.transpose(1, 2))


        # the pools have shrunk the time axis, but the beat span is defined on the
        # input scale, so it is rescaled by the same factor before slicing
        if self.beat_bounds is not None:
            scale = h.shape[-1] / T
            start, end = (int(round(b * scale)) for b in self.beat_bounds)
            h = h[..., start:max(end, start + 1)]

        pooled = h.mean(dim=-1) if self.global_pool == 'avg' else h.max(dim=-1).values

        output = self.fc(pooled)
        outputs = output.unsqueeze(1).expand(-1, T, -1)

        return outputs, output


class RNN_Baseline(nn.Module):

    def __init__(
            self,
            in_dims: int = 1,
            out_dims: int = 1,
            hidden_dims: List[int] = [32, 64, 32],
            cell_type: str = 'gru',
            nonlinearity: str = 'tanh',
            bidirectional: bool = False,
            dropout: float = 0.0,
            readout: str = 'per_timestep',
            beat_bounds=None,
            bias: bool = False,
        ):

        '''
        a stacked RNN/GRU/LSTM over the raw waveform -- the non-spiking recurrent
        counterpart to the SNNs, so this comparison isolates the spiking neuron
        rather than recurrence itself.

        cell_type: 'rnn' (vanilla Elman), 'gru' or 'lstm'. the vanilla RNN is the
            closest structural match to a LIF layer -- one state vector, one
            recurrent matrix, no gates -- so it isolates the spiking
            nonlinearity, while GRU/LSTM measure what gating buys on top.
        nonlinearity: 'tanh' or 'relu', vanilla RNN only.
        hidden_dims: hidden width per layer. nn.GRU only supports one width for a
            whole stack, so the layers are built separately to allow a taper.
        bidirectional: off by default -- the SNNs are causal, so a bidirectional
            baseline sees the future and is not a like-for-like comparison.
        bias: defaults off to match the SNN configs, which all run bias: false.
        readout: 'per_timestep' and 'pooled' as in Conv1D_Baseline, plus 'last' --
            classify from the hidden state at the final timestep of the beat span
            and broadcast that logit across t. 'last' is the conventional readout
            for a gated RNN, which carries a dense continuous state that has
            already integrated the whole sequence; note it is the opposite of
            'per_timestep' for deployment, since it cannot decide until the span
            has fully arrived.
        beat_bounds: as in Conv1D_Baseline. 'pooled' and 'last' both respect it,
            so with post_window > 0 the readout is taken at the end of the beat
            rather than the end of the window.
        '''

        super(RNN_Baseline, self).__init__()

        if cell_type not in ('rnn', 'gru', 'lstm'):
            raise ValueError(f"cell_type must be 'rnn', 'gru' or 'lstm', got {cell_type!r}")

        if readout not in ('per_timestep', 'pooled', 'last'):
            raise ValueError(
                f"readout must be 'per_timestep', 'pooled' or 'last', got {readout!r}")

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.readout = readout
        self.beat_bounds = beat_bounds

        rnn_cls = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[cell_type]
        directions = 2 if bidirectional else 1

        # nonlinearity is an nn.RNN-only argument; passing it to GRU/LSTM raises
        cell_kwargs = {'nonlinearity': nonlinearity} if cell_type == 'rnn' else {}

        self.layers = nn.ModuleList()
        widths = [in_dims] + [d * directions for d in hidden_dims]

        for c_in, hidden in zip(widths[:-1], hidden_dims):
            self.layers.append(rnn_cls(
                c_in, hidden, num_layers=1, batch_first=True,
                bidirectional=bidirectional, bias=bias, **cell_kwargs,
            ))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dims[-1] * directions, out_dims)

    def forward(self, x, logging=False):
        # x shape: B x T x IN_DIMS -- already what batch_first RNNs expect
        h = x

        for i, layer in enumerate(self.layers):
            h, _ = layer(h)

            # no dropout on the last layer's output; the readout follows it
            if i < len(self.layers) - 1:
                h = self.dropout(h)

        if self.readout in ('pooled', 'last'):
            start, end = self.beat_bounds if self.beat_bounds is not None else (0, h.shape[1])

            if self.readout == 'pooled':
                summary = h[:, start:end].mean(dim=1)
            else:
                # the final state of the beat span. an RNN state is already a
                # running summary of everything before it, so this is a readout
                # of the whole span rather than of one instant.
                summary = h[:, end - 1]

            output = self.fc(summary)
            outputs = output.unsqueeze(1).expand(-1, h.shape[1], -1)

            return outputs, output

        outputs = self.fc(h)

        return outputs, outputs[:, -1]


BASELINE_MODELS = {
    'conv1d': Conv1D_Baseline,
    'convnet': ConvNet_Baseline,
    'rnn': lambda **kw: RNN_Baseline(cell_type='rnn', **kw),
    'gru': lambda **kw: RNN_Baseline(cell_type='gru', **kw),
    'lstm': lambda **kw: RNN_Baseline(cell_type='lstm', **kw),
}


def create_baseline(model_type, out_dims=1, beat_bounds=None, **model_kwargs):
    '''
    build a baseline from a config's model_type + model_kwargs.

    beat_bounds is threaded in from the dataset rather than set independently, so
    a 'pooled' readout can never pool over a wider span than the loss supervises.
    '''

    if model_type not in BASELINE_MODELS:
        raise ValueError(
            f'unknown baseline model_type {model_type!r}, '
            f'expected one of {sorted(BASELINE_MODELS)}'
        )

    model_kwargs.setdefault('beat_bounds', beat_bounds)

    return BASELINE_MODELS[model_type](out_dims=out_dims, **model_kwargs)
