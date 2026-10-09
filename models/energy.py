"""
Operation-level energy accounting for the spiking models and the non-spiking
baselines.

Rather than adding a parallel logging-forward to every model class, forward hooks
count the operations each weight matrix performs during a normal forward pass. The
same meter handles both families, so the SNN and baseline numbers follow one
rulebook (the standard SNN energy comparison, e.g. Horowitz 2014 / Merolla 2014):

  * A projection whose input is binary {0, 1} (a spike train) costs an accumulate
    (AC) per presynaptic spike per postsynaptic target:
        AC = (number of input spikes) * fan_out.   No spike, no op.
  * A projection whose input is analog (the raw ECG, or any ANN activation) costs
    a multiply-accumulate (MAC) on every synapse, for every input vector:
        MAC = rows * synapses.
    Binarity is detected from the data per layer, so the SNN's first projection
    (analog ECG) is billed as MACs automatically.
  * Neuron/hidden state updates are tallied separately as state MACs: the SNN
    membrane leak (alpha * v, plus rho * a and beta * a for ALIF) once per neuron
    per timestep, and the GRU/LSTM elementwise gate products (3 per hidden unit per
    timestep). Vanilla RNN and conv layers have none. The SNN literature usually
    reports synaptic energy alone; synaptic + state is the more hardware-honest
    total. Both are reported.
  * Not counted for any model: additions, nonlinearities (tanh, sigmoid, ReLU,
    spike threshold), pooling, bias terms, and multiplications by a binary gate
    (a mux, not a multiplier).

Gates. In the HGRSNN family (hgrsnn, hgrsnn_no_flush, hgrsnn_no_share, eprop_hgrsnn) a
boundary b is binary, and the order of operations is taken to be: compute the gate, then
run only what it opens. A gated cell sees up to two different gate signals, identified by
argument name (no_flush has only the first, no_share only the second):

    bottom-up gate  (b_b / bottom_gate)          the layer below's boundary at t.
        While 0, the bottom-up projection (b_v / W_bv) is not run.
    transition gate (b_previous / gate_previous) this cell's own boundary at t-1.
        While 1 the state is flushed, so neither the leak alpha * v nor the
        recurrent projection (z_v / W_zv) is run.

The gate's own projection (z_b / gate.projection) always runs, as do ungated terms
(the bottom cell's input projection, the top cell's transition, the readout). Every
count is also kept as if no gate ever closed (the ungated_* fields), so the saving
from gating is reportable. A gated cell from any other model file is not mapped here
and is billed as if every gate were open, with a warning.

Hooked modules:
    HebbianLinearLayer, nn.Linear    AC or MAC, by input binarity
    nn.Conv1d                        MAC
    nn.RNN / nn.GRU / nn.LSTM        MAC, plus state MACs for GRU/LSTM
    e-prop cells (W_bv/W_zv)         AC or MAC -- these weights are multiplied
                                     inside a custom autograd Function, invisible
                                     to the linear hooks, so they are read off the
                                     cell's own inputs
    *_Cell modules                   state MACs, firing rate, and gate values

A layer that is defined but never called (e.g. the HGRSNN cells' unused top-down t_v)
costs nothing and is left out.

Every result is normalized per beat (one input window), so numbers are comparable
across test sets of different sizes.

Energy-per-op defaults are 45 nm, 32-bit CMOS numbers (Horowitz, ISSCC 2014):
    MAC = 4.6 pJ, AC = 0.9 pJ. Override via EnergyMeter(e_mac=..., e_ac=...) to
    match whatever process/precision you cite in the paper.

Usage
-----
    from models.energy import EnergyMeter

    model.eval()
    with EnergyMeter(model) as meter, torch.no_grad():
        for data in batches:
            model(data)            # normal forward; hooks do the counting

    print(meter.report())          # human-readable summary
    stats = meter.summary()        # per-beat dict, safe to serialize
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .hebbian_linear import HebbianLinearLayer

# Energy per operation, in picojoules (Horowitz 2014, 45 nm, 32-bit).
E_MAC_PJ = 4.6
E_AC_PJ = 0.9

# the e-prop cells' raw weight matrices, each paired with the forward argument it
# multiplies and whether it carries the self-connection mask
EPROP_PROJECTIONS = {
    'W_bv': ('bottom_input', False),
    'W_zv': ('z_previous', True),
}

# model files whose gate semantics are mapped and checked against the bottom-up /
# transition description in the module docstring. no_flush has only bottom-up gates and
# no_share only transition gates; the argument names say which each cell has
GATED_MODEL_FILES = {'hgrsnn', 'hgrsnn_no_flush', 'hgrsnn_no_share', 'eprop_hgrsnn'}

# forward arguments carrying each gate
GATE_ARGS = {
    'bottom_up': ('b_b', 'bottom_gate'),
    'transition': ('b_previous', 'gate_previous'),
}

# which gate each projection sits behind
GATED_PROJECTIONS = {'b_v': 'bottom_up', 'W_bv': 'bottom_up', 'z_v': 'transition', 'W_zv': 'transition'}

# any of these in a cell's forward means it is gated somehow
ANY_GATE_ARGS = {'b_b', 'bottom_gate', 'b_previous', 'gate_previous', 't_b', 'top_gate'}

# weight matrices per timestep, and elementwise state products per hidden unit per
# timestep, for torch's recurrent layers. GRU: r*(W_hn h), (1-z)*n, z*h.
# LSTM: f*c, i*g, o*tanh(c).
RNN_KINDS = {'RNN_TANH': 'rnn', 'RNN_RELU': 'rnn', 'GRU': 'gru', 'LSTM': 'lstm'}
RNN_GATES = {'rnn': 1, 'gru': 3, 'lstm': 4}
RNN_STATE_PRODUCTS = {'rnn': 0, 'gru': 3, 'lstm': 3}

_COUNTERS = ('rows', 'rows_run', 'input_elems', 'input_nonzero', 'input_nonzero_run',
             'dense_macs', 'dense_macs_run', 'state_macs', 'state_macs_ungated',
             'neuron_steps', 'spikes_out')


@dataclass
class _Tally:
    name: str
    kind: str                     # 'linear', 'conv1d', 'rnn', 'gru', 'lstm' or 'neuron'
    in_features: int
    out_features: int
    masked: bool = False          # self-connections removed from a square matrix
    binary: bool = True           # flips to False the first time analog input is seen
    gate: str = ''                # '' (always runs), 'bottom_up' or 'transition'
    calls: int = 0
    rows: float = 0.0             # input vectors presented (batch x timesteps/positions)
    rows_run: float = 0.0         # of those, the ones the gate let run
    input_elems: float = 0.0
    input_nonzero: float = 0.0
    input_nonzero_run: float = 0.0
    dense_macs: float = 0.0       # MACs if every synapse were evaluated densely
    dense_macs_run: float = 0.0
    state_macs: float = 0.0       # elementwise multiplies on neuron/hidden state, performed
    state_macs_ungated: float = 0.0
    neuron_steps: float = 0.0     # neurons x rows (neuron tallies only)
    spikes_out: float = 0.0       # neuron tallies only

    @property
    def synaptic(self) -> bool:
        return self.kind != 'neuron'

    @property
    def fan_out(self) -> int:
        return self.out_features - 1 if self.masked else self.out_features

    def ac_ops(self, gated: bool = True) -> float:
        """Accumulates: one per input spike per target, on the rows that ran."""
        if not (self.synaptic and self.binary):
            return 0.0
        return (self.input_nonzero_run if gated else self.input_nonzero) * self.fan_out

    def mac_ops(self, gated: bool = True) -> float:
        if not self.synaptic or self.binary:
            return 0.0
        return self.dense_macs_run if gated else self.dense_macs


def _count_projection(tally: _Tally, x: torch.Tensor, run: torch.Tensor = None) -> None:
    """Tally one call of a projection. run: optional (B,) binary, 1 where the gate let it run."""
    x = x.detach()
    batch = x.shape[0]
    rows = x.numel() // tally.in_features

    synapses = tally.in_features * tally.out_features
    if tally.masked:
        synapses -= min(tally.in_features, tally.out_features)

    nonzero_per_sample = torch.count_nonzero(x.reshape(batch, -1), dim=1)
    nonzero = int(nonzero_per_sample.sum())

    if run is None:
        rows_run, nonzero_run = rows, nonzero
    else:
        rows_run = rows * float(run.sum()) / batch
        nonzero_run = int((nonzero_per_sample * run).sum())

    tally.calls += 1
    tally.rows += rows
    tally.rows_run += rows_run
    tally.input_elems += x.numel()
    tally.input_nonzero += nonzero
    tally.input_nonzero_run += nonzero_run
    tally.dense_macs += rows * synapses
    tally.dense_macs_run += rows_run * synapses

    # analog input (values outside {0, 1}) can't be billed as accumulates. once
    # seen the layer stays analog, so the check is skipped from then on
    if tally.binary and not _is_binary(x):
        tally.binary = False


def _is_cell(module: nn.Module) -> bool:
    return type(module).__name__.endswith('_Cell') and hasattr(module, 'alpha')


def _gate_values(name: str, gate: torch.Tensor) -> torch.Tensor:
    gate = gate.detach().reshape(-1)
    # skipping a term is only free if the gate is a hard 0/1 switch
    if not _is_binary(gate):
        raise ValueError(f'{name}: gate is not binary, so it cannot be billed as a skip')
    return gate


class EnergyMeter:
    """Context manager that tallies synaptic and state operations across a model.

    Works on the SNNs in ``models/`` and the baselines in ``baselines/`` alike.
    Hooks accumulate across forward calls; call ``reset()`` to zero them (e.g.
    between a warm-up batch and the measured run).
    """

    def __init__(
        self,
        model: nn.Module,
        e_mac: float = E_MAC_PJ,
        e_ac: float = E_AC_PJ,
    ):
        self.model = model
        self.e_mac = e_mac
        self.e_ac = e_ac
        self._tallies: Dict[str, _Tally] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        # per gated cell: which gates it has, and their values during its forward
        self._gate_states: Dict[str, dict] = {}
        self.n_beats = 0
        self.timesteps = None

    # -- context manager plumbing --------------------------------------------

    def __enter__(self) -> "EnergyMeter":
        self._handles.append(self.model.register_forward_pre_hook(self._beat_hook))

        # named_modules is pre-order, so a cell is registered before the
        # projections inside it look up its gates
        for name, module in self.model.named_modules():
            if _is_cell(module):
                self._add_cell(name, module)

            if isinstance(module, HebbianLinearLayer):
                self._add_projection(name, module, module.in_channels,
                                     module.out_channels, module.masked)

            elif isinstance(module, nn.Linear):
                self._add_projection(name, module, module.in_features,
                                     module.out_features, False)

            elif isinstance(module, nn.Conv1d):
                tally = self._new(name, 'conv1d', module.in_channels,
                                  module.out_channels, binary=False)
                self._handles.append(module.register_forward_hook(self._conv_hook(tally)))

            elif isinstance(module, nn.RNNBase):
                kind = RNN_KINDS[module.mode]
                directions = 2 if module.bidirectional else 1
                tally = self._new(name, kind, module.input_size,
                                  module.hidden_size * directions, binary=False)
                self._handles.append(module.register_forward_hook(self._rnn_hook(tally)))

        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        # zeroed in place: the hooks hold references to these tallies
        for t in self._tallies.values():
            t.binary = t.kind in ('linear', 'neuron')
            t.calls = 0
            for field in _COUNTERS:
                setattr(t, field, 0.0)
        for state in self._gate_states.values():
            state['bottom_up'] = state['transition'] = None
        self.n_beats = 0
        self.timesteps = None

    # -- hooks ---------------------------------------------------------------

    def _new(self, name, kind, in_features, out_features, masked=False, binary=True, gate=''):
        tally = _Tally(name=name, kind=kind, in_features=in_features,
                       out_features=out_features, masked=masked, binary=binary, gate=gate)
        self._tallies[name] = tally
        return tally

    def _beat_hook(self, module, args) -> None:
        # every model takes (B, T, in_dims) as its first argument
        x = args[0]
        self.n_beats += x.shape[0]
        self.timesteps = x.shape[1]

    def _gate_for(self, cell_name, projection_attr):
        """The gate a projection sits behind: its parent cell's state and role, or (None, '')."""
        state = self._gate_states.get(cell_name)
        role = GATED_PROJECTIONS.get(projection_attr)
        if state is None or role not in state['roles']:
            return None, ''
        return state, role

    def _add_projection(self, name, module, in_features, out_features, masked):
        parent, _, attr = name.rpartition('.')
        state, gate = self._gate_for(parent, attr)
        tally = self._new(name, 'linear', in_features, out_features, masked, gate=gate)

        def hook(module, inputs, output):
            _count_projection(tally, inputs[0], state[gate] if gate else None)

        self._handles.append(module.register_forward_hook(hook))

    def _conv_hook(self, tally):
        def hook(module, inputs, output):
            batch, out_channels, length = output.shape
            kernel = module.kernel_size[0]
            macs = batch * length * out_channels * (module.in_channels // module.groups) * kernel
            tally.calls += 1
            tally.rows += batch * length
            tally.rows_run += batch * length
            tally.dense_macs += macs
            tally.dense_macs_run += macs
        return hook

    def _rnn_hook(self, tally):
        def hook(module, inputs, output):
            if module.proj_size:
                raise NotImplementedError('LSTM proj_size is not accounted for')

            x = inputs[0]
            batch, steps = (x.shape[0], x.shape[1]) if module.batch_first else (x.shape[1], x.shape[0])
            directions = 2 if module.bidirectional else 1
            hidden = module.hidden_size
            gates = RNN_GATES[tally.kind]

            macs_per_step = 0
            for layer in range(module.num_layers):
                in_size = module.input_size if layer == 0 else hidden * directions
                macs_per_step += gates * (in_size + hidden) * hidden * directions

            rows = batch * steps
            state = rows * RNN_STATE_PRODUCTS[tally.kind] * hidden * directions * module.num_layers
            tally.calls += 1
            tally.rows += rows
            tally.rows_run += rows
            tally.dense_macs += rows * macs_per_step
            tally.dense_macs_run += rows * macs_per_step
            tally.state_macs += state
            tally.state_macs_ungated += state
        return hook

    def _add_cell(self, name, module):
        signature = inspect.signature(module.forward)
        params = signature.parameters

        verified = type(module).__module__.rsplit('.', 1)[-1] in GATED_MODEL_FILES
        gate_args = {role: next((a for a in args if a in params), None)
                     for role, args in GATE_ARGS.items()}
        roles = {role for role, arg in gate_args.items() if arg} if verified else set()

        if not verified and ANY_GATE_ARGS & set(params):
            warnings.warn(
                f'{name} ({type(module).__module__}.{type(module).__name__}) is gated, but '
                'its gate semantics are not mapped in models/energy.py; billing it as if '
                'every gate were open'
            )

        state = {'roles': roles, 'bottom_up': None, 'transition': None}
        self._gate_states[name] = state

        # e-prop cells: one projection tally per raw weight matrix present
        projections = []
        for weight_name, (arg_name, masked) in EPROP_PROJECTIONS.items():
            weight = getattr(module, weight_name, None)
            if isinstance(weight, nn.Parameter) and arg_name in params:
                out_features, in_features = weight.shape
                _, gate = self._gate_for(name, weight_name)
                tally = self._new(f'{name}.{weight_name}', 'linear',
                                  in_features, out_features, masked, gate=gate)
                projections.append((tally, arg_name, gate))

        gated_leak = 'transition' in roles
        neuron = self._new(name, 'neuron', module.in_dims, module.out_dims,
                           gate='transition' if gated_leak else '')
        # ALIF's adaptation decay rho * a and beta * a run whatever the gate says
        adaptation_products = 2 if hasattr(module, 'rho') else 0

        def pre_hook(module, args, kwargs):
            if not roles:
                return
            bound = signature.bind(*args, **kwargs).arguments
            if 'bottom_up' in roles:
                state['bottom_up'] = _gate_values(name, bound[gate_args['bottom_up']])
            if 'transition' in roles:
                # a flush (gate 1) skips the transition, so it runs where the gate is 0
                state['transition'] = 1 - _gate_values(name, bound[gate_args['transition']])

        def hook(module, args, kwargs, output):
            if projections:
                bound = signature.bind(*args, **kwargs).arguments
                for tally, arg_name, gate in projections:
                    _count_projection(tally, bound[arg_name], state[gate] if gate else None)

            # every cell returns (v, z, ...): membrane potential, then spikes
            z = output[1].detach()
            batch, n_neurons = z.shape[0], z.numel() // z.shape[0]
            leak_rows = float(state['transition'].sum()) if gated_leak else batch

            neuron.calls += 1
            neuron.rows += batch
            neuron.rows_run += leak_rows
            neuron.neuron_steps += z.numel()
            neuron.spikes_out += int(torch.count_nonzero(z))
            neuron.state_macs += (leak_rows + adaptation_products * batch) * n_neurons
            neuron.state_macs_ungated += (1 + adaptation_products) * z.numel()

            state['bottom_up'] = state['transition'] = None

        self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self._handles.append(module.register_forward_hook(hook, with_kwargs=True))

    # -- results -------------------------------------------------------------

    def summary(self) -> dict:
        """Per-beat totals and a per-layer breakdown, as a plain dict."""
        if not self.n_beats:
            raise RuntimeError('EnergyMeter.summary(): no forward pass was metered')

        n = self.n_beats
        layers = []
        spikes = neuron_steps = 0.0

        for t in self._tallies.values():
            if t.calls == 0:
                continue

            synaptic_pj = t.ac_ops() * self.e_ac + t.mac_ops() * self.e_mac
            ungated_synaptic_pj = t.ac_ops(gated=False) * self.e_ac + t.mac_ops(gated=False) * self.e_mac
            state_pj = t.state_macs * self.e_mac
            ungated_state_pj = t.state_macs_ungated * self.e_mac
            dense_pj = t.dense_macs * self.e_mac if t.synaptic else 0.0

            spikes += t.spikes_out
            neuron_steps += t.neuron_steps

            layers.append({
                'layer': t.name,
                'kind': t.kind,
                'in_features': t.in_features,
                'out_features': t.out_features,
                'input_type': ('binary' if t.binary else 'analog') if t.synaptic else '',
                'gate': t.gate,
                # fraction of rows where the gated computation actually ran
                'run_rate': t.rows_run / t.rows if t.gate and t.rows else float('nan'),
                'rows_per_beat': t.rows / n,
                'input_spike_rate': (t.input_nonzero / t.input_elems
                                     if t.synaptic and t.binary and t.input_elems else float('nan')),
                'firing_rate': t.spikes_out / t.neuron_steps if t.neuron_steps else float('nan'),
                'ac_ops': t.ac_ops() / n,
                'mac_ops': t.mac_ops() / n,
                'state_mac_ops': t.state_macs / n,
                'synaptic_energy_pj': synaptic_pj / n,
                'state_energy_pj': state_pj / n,
                'total_energy_pj': (synaptic_pj + state_pj) / n,
                'ungated_synaptic_energy_pj': ungated_synaptic_pj / n,
                'ungated_total_energy_pj': (ungated_synaptic_pj + ungated_state_pj) / n,
                'dense_equiv_synaptic_energy_pj': dense_pj / n,
            })

        def total(key):
            return float(sum(L[key] for L in layers))

        synaptic = total('synaptic_energy_pj')
        state = total('state_energy_pj')
        ungated_synaptic = total('ungated_synaptic_energy_pj')
        ungated_total = total('ungated_total_energy_pj')
        dense = total('dense_equiv_synaptic_energy_pj')

        return {
            'n_beats': n,
            'timesteps': self.timesteps,
            'ac_ops': total('ac_ops'),
            'mac_ops': total('mac_ops'),
            'state_mac_ops': total('state_mac_ops'),
            'synaptic_energy_pj': synaptic,
            'state_energy_pj': state,
            'total_energy_pj': synaptic + state,
            # the same counts with every gate held open
            'ungated_synaptic_energy_pj': ungated_synaptic,
            'ungated_total_energy_pj': ungated_total,
            # a gate can only skip work, so any negative value is float rounding
            'gating_savings': max(0.0, 1.0 - (synaptic + state) / ungated_total) if ungated_total else 0.0,
            # the same architecture with every synapse evaluated as a dense MAC and
            # no gate ever closed; identical to the ungated energy for a model with
            # no binary inputs
            'dense_equiv_synaptic_energy_pj': dense,
            'dense_equiv_total_energy_pj': dense + (ungated_total - ungated_synaptic),
            'synaptic_savings': 1.0 - synaptic / dense if dense else 0.0,
            'mean_firing_rate': spikes / neuron_steps if neuron_steps else float('nan'),
            'e_mac_pj': self.e_mac,
            'e_ac_pj': self.e_ac,
            'layers': layers,
        }

    def report(self) -> str:
        """Pretty, printable version of :meth:`summary`."""
        s = self.summary()

        def fmt_e(pj: float) -> str:
            for unit, scale in (("mJ", 1e9), ("uJ", 1e6), ("nJ", 1e3)):
                if pj >= scale:
                    return f"{pj / scale:.3f} {unit}"
            return f"{pj:.3f} pJ"

        def fmt_rate(rate: float) -> str:
            return '' if np.isnan(rate) else f'{100 * rate:.2f}%'

        lines = ["Per-layer, per beat (rate: input spikes, or firing for neurons; run: share of steps a gated term ran):"]
        lines.append(f"  {'layer':<30}{'kind':>8}{'in->out':>10}{'input':>8}"
                     f"{'rate':>9}{'run':>9}{'AC':>14}{'MAC':>14}{'state MAC':>12}{'energy':>13}")
        for L in s["layers"]:
            rate = L['firing_rate'] if L['kind'] == 'neuron' else L['input_spike_rate']
            lines.append(
                f"  {L['layer']:<30}{L['kind']:>8}{L['in_features']:>5}->{L['out_features']:<4}"
                f"{L['input_type']:>8}{fmt_rate(rate):>9}{fmt_rate(L['run_rate']):>9}"
                f"{L['ac_ops']:>14,.0f}{L['mac_ops']:>14,.0f}{L['state_mac_ops']:>12,.0f}"
                f"{fmt_e(L['total_energy_pj']):>13}"
            )
        lines.append("")
        lines.append(f"Beats metered        : {s['n_beats']:,} x {s['timesteps']} timesteps")
        if not np.isnan(s['mean_firing_rate']):
            lines.append(f"Mean firing rate     : {100 * s['mean_firing_rate']:.2f}%")
        lines.append(f"AC / MAC / state MAC : {s['ac_ops']:,.0f} / {s['mac_ops']:,.0f} / "
                     f"{s['state_mac_ops']:,.0f} per beat")
        lines.append(f"Synaptic energy      : {fmt_e(s['synaptic_energy_pj'])} per beat")
        lines.append(f"State energy         : {fmt_e(s['state_energy_pj'])} per beat")
        lines.append(f"Total energy         : {fmt_e(s['total_energy_pj'])} per beat")
        if any(L['gate'] for L in s['layers']):
            lines.append(f"Without gating       : {fmt_e(s['ungated_total_energy_pj'])} per beat "
                         f"(gating saves {100 * s['gating_savings']:.2f}%)")
        if s['ac_ops'] > 0:
            lines.append(f"Dense same-shape ANN : {fmt_e(s['dense_equiv_synaptic_energy_pj'])} "
                         f"synaptic per beat ({100 * s['synaptic_savings']:.2f}% saved)")
        return "\n".join(lines)


def _is_binary(x: torch.Tensor) -> bool:
    """True if every element is 0 or 1 (i.e. a spike train)."""
    return bool(torch.all((x == 0) | (x == 1)))


class SpikeRecorder:
    """Capture each neuron cell's continuous membrane potential and binary spikes.

    Every cell class across the model files shares the convention that ``forward``
    returns ``(v_out, z_out, ...)`` -- membrane potential first, spikes second --
    so they can be hooked generically by class-name suffix, the same way
    :class:`EnergyMeter` hooks the linear layers.

    Records a single sample (``sample_idx``) across all timesteps, giving a
    (T, N) trace per layer. Only one sample is kept: these traces are for a
    figure, not a measurement, and holding a whole batch is pointless.
    """

    def __init__(self, model: torch.nn.Module, sample_idx: int = 0):
        self.model = model
        self.sample_idx = sample_idx
        self.traces: Dict[str, Dict[str, list]] = {}
        self.thresholds: Dict[str, float] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "SpikeRecorder":
        for name, module in self.model.named_modules():
            if type(module).__name__.endswith('_Cell'):
                self.traces[name] = {'v': [], 'z': []}
                self.thresholds[name] = getattr(module, 'threshold', None)
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(module, inputs, output) -> None:
            if not isinstance(output, tuple) or len(output) < 2:
                return
            v, z = output[0], output[1]
            if not (torch.is_tensor(v) and torch.is_tensor(z)):
                return
            i = self.sample_idx
            self.traces[name]['v'].append(v[i].detach().cpu())
            self.traces[name]['z'].append(z[i].detach().cpu())
        return hook

    def stacked(self) -> Dict[str, dict]:
        """Return {layer: {'v': (T, N), 'z': (T, N), 'threshold': float|None}}.

        Cells whose second output isn't actually a spike train are dropped rather
        than silently plotted as if they were.
        """
        out = {}
        for name, tr in self.traces.items():
            if not tr['z']:
                continue
            v = torch.stack(tr['v'], 0)
            z = torch.stack(tr['z'], 0)
            if not _is_binary(z):
                continue
            out[name] = {'v': v, 'z': z, 'threshold': self.thresholds.get(name)}
        return out


def _select_neuron(stacked: Dict[str, dict], layer: str = None,
                   neuron_idx: int = None, t_start: int = None,
                   t_end: int = None):
    """Resolve which (layer, neuron) to draw and return its traces.

    Defaults to the first cell and its most active neuron, so a figure is never
    accidentally a silent neuron. The choice is printed rather than titled onto
    the plot, so an auto-picked neuron stays traceable.

    t_start/t_end crop the timestep window (applied after the neuron is chosen by
    activity over the full trace).
    """
    name = layer if layer is not None else next(iter(stacked))
    v = stacked[name]['v'].numpy()
    z = stacked[name]['z'].numpy()
    idx = int(z.sum(0).argmax()) if neuron_idx is None else neuron_idx

    v_i, z_i = v[:, idx], z[:, idx]
    lo = 0 if t_start is None else max(0, t_start)
    hi = len(v_i) if t_end is None else min(len(v_i), t_end)

    # an inverted or out-of-range window silently renders a blank figure, which
    # is worse than failing: say what went wrong instead
    if lo >= hi:
        raise ValueError(
            f'empty timestep window: t_start={t_start} t_end={t_end} resolves to '
            f'[{lo}, {hi}) on a {len(v_i)}-step trace. Need t_start < t_end, both '
            f'within the trace.'
        )

    t = np.arange(lo, hi)
    print(f'  drawing {name}, neuron {idx}, timesteps {lo}-{hi} '
          f'({100 * z_i[lo:hi].mean():.1f}% active in window)')
    return name, idx, t, v_i[lo:hi], z_i[lo:hi]


def plot_voltage(stacked: Dict[str, dict], save_path: str, layer: str = None,
                 neuron_idx: int = None, t_start: int = 200, t_end: int = 275,
                 dpi: int = 600) -> None:
    """Plot one neuron's membrane potential v(t) as a bare trace.

    No axes, box or labels -- just the line. A real value at every timestep, i.e.
    what a dense ANN would transmit. This is the model's own continuous state
    standing in for the dense signal, not a trained ANN's activations.
    """
    import matplotlib.pyplot as plt

    if not stacked:
        print('plot_voltage: no spiking cells captured, skipping plot')
        return

    _, _, t, v_i, _ = _select_neuron(stacked, layer, neuron_idx, t_start, t_end)

    fig, ax = plt.subplots(figsize=(5, 1.2))
    ax.plot(t, v_i, color='#1f4e79', linewidth=4.0)
    ax.axis('off')

    fig.tight_layout(pad=0)
    fig.savefig(save_path + '.png', dpi=dpi, transparent=True)
    plt.close(fig)


def plot_spikes(stacked: Dict[str, dict], save_path: str, layer: str = None,
                neuron_idx: int = None, t_start: int = 200, t_end: int = 275,
                dpi: int = 600) -> None:
    """Plot one neuron's spike train as a bare raster.

    No axes, box or labels -- just the ticks. Only these timesteps cost an
    accumulate; the gap between this and the full v(t) line is the sparsity term
    in the energy math.
    """
    import matplotlib.pyplot as plt

    if not stacked:
        print('plot_spikes: no spiking cells captured, skipping plot')
        return

    _, _, t, _, z_i = _select_neuron(stacked, layer, neuron_idx, t_start, t_end)

    fig, ax = plt.subplots(figsize=(5, 1.2))
    ax.eventplot(t[z_i > 0], colors='#1f4e79', linewidth=2.0)
    ax.set_xlim(t[0], t[-1])
    ax.axis('off')

    fig.tight_layout(pad=0)
    fig.savefig(save_path + '.png', dpi=dpi, transparent=True)
    plt.close(fig)


def plot_energy(snn_pj: float, ann_pj: float, save_path: str, dpi: int = 150,
                title_size: int = 20, label_size: int = 16, tick_size: int = 14,
                value_size: int = 15) -> None:
    """Bar chart of mean-over-seeds energy: dense ANN vs SNN.

    Sized for a slide: the point is the visual asymmetry between the two bars, so
    each bar is labelled with its own value (the SNN bar is otherwise too short to
    read) and the ratio is called out directly. Font sizes are parameters so the
    figure can be retuned without editing the plot.
    """
    import matplotlib.pyplot as plt

    ann_uj, snn_uj = ann_pj / 1e6, snn_pj / 1e6

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(['ANN', 'SNN'], [ann_uj, snn_uj], color=['#2a78d6', '#008300'])

    # the SNN bar is a sliver at a large ratio; label both so it stays readable
    for bar, val in zip(bars, (ann_uj, snn_uj)):
        ax.annotate(f'{val:.2f}', (bar.get_x() + bar.get_width() / 2, val),
                    ha='center', va='bottom', fontsize=value_size,
                    xytext=(0, 4), textcoords='offset points')

    if snn_uj > 0:
        ax.annotate(f'{ann_uj / snn_uj:.0f}x less', (1, snn_uj),
                    ha='center', va='bottom', fontsize=value_size + 3,
                    color='#008300', fontweight='bold',
                    xytext=(0, 28), textcoords='offset points')

    ax.set_ylabel('energy (uJ)', fontsize=label_size)
    ax.set_title('Energy Demand', fontsize=title_size)
    ax.tick_params(labelsize=tick_size)
    ax.set_ylim(0, ann_uj * 1.18)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path + '.png', dpi=dpi, transparent=True)
    plt.close(fig)
