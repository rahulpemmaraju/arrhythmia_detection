"""
Energy accounting for the spiking models.

Estimates the operation-level energy of an SNN forward pass and compares it to
the energy an equivalent (same-shape) dense ANN would spend. Rather than adding a
parallel logging-forward to every model class, this attaches forward hooks to
every ``HebbianLinearLayer`` -- the input to that layer *is* the spike train
crossing a set of synapses, so counting its non-zero entries gives the exact
number of synaptic operations the hardware would perform.

Methodology (the standard SNN energy comparison, e.g. Horowitz 2014 / Merolla
et al. 2014):

  * ANN: a dense multiply-accumulate (MAC) on every synapse, every timestep
        -> B * T * in * out MACs per layer.
  * SNN: an accumulate (AC) only when a presynaptic spike arrives
        -> SOP = (number of input spikes) * out per layer. No spike, no op.
  * The first projection sees analog input (real-valued ECG), so it is billed as
    dense MACs; layers whose input is binary {0, 1} are billed as sparse ACs.
    The meter auto-detects this from the data.

Energy-per-op defaults are 45 nm, 32-bit CMOS numbers (Horowitz, ISSCC 2014):
    MAC = 4.6 pJ, AC = 0.9 pJ. Override via EnergyMeter(e_mac=..., e_ac=...) to
    match whatever process/precision you cite in the paper.

Usage
-----
    from models.energy import EnergyMeter

    model.eval()
    with EnergyMeter(model) as meter:
        for data, _ in test_loader:
            model(data)            # normal forward; hooks do the counting

    print(meter.report())          # human-readable summary
    stats = meter.summary()        # dict, for logging to hdf5/wandb/etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from .hebbian_linear import HebbianLinearLayer

# Energy per operation, in picojoules (Horowitz 2014, 45 nm, 32-bit).
E_MAC_PJ = 4.6
E_AC_PJ = 0.9


@dataclass
class _LayerTally:
    name: str
    in_features: int
    out_features: int
    calls: int = 0                # number of forward calls (== timesteps * batches seen)
    input_elems: int = 0          # total presynaptic values presented
    input_spikes: float = 0.0     # total non-zero presynaptic values
    dense_macs: float = 0.0       # MACs a dense ANN layer would do
    spiking: bool = True          # False if this layer's input is analog (real-valued)

    @property
    def sop(self) -> float:
        """Synaptic operations actually performed by the SNN for this layer."""
        return self.input_spikes * self.out_features

    @property
    def spike_rate(self) -> float:
        """Mean presynaptic activation (fraction of inputs that spiked)."""
        return self.input_spikes / self.input_elems if self.input_elems else 0.0


class EnergyMeter:
    """Context manager that tallies spikes/ops across a spiking model.

    Attaches a forward hook to every ``HebbianLinearLayer`` under ``model``.
    Because the model iterates over timesteps internally, each hook fires once
    per timestep per forward, and counts accumulate. Call ``reset()`` to zero the
    tallies (e.g. between the warm-up batch and the measured run).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        e_mac: float = E_MAC_PJ,
        e_ac: float = E_AC_PJ,
    ):
        self.model = model
        self.e_mac = e_mac
        self.e_ac = e_ac
        self._tallies: Dict[str, _LayerTally] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # -- context manager plumbing --------------------------------------------

    def __enter__(self) -> "EnergyMeter":
        for name, module in self.model.named_modules():
            if isinstance(module, HebbianLinearLayer):
                self._tallies[name] = _LayerTally(
                    name=name,
                    in_features=module.in_channels,
                    out_features=module.out_channels,
                )
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        for t in self._tallies.values():
            t.calls = 0
            t.input_elems = 0
            t.input_spikes = 0.0
            t.dense_macs = 0.0

    # -- the hook ------------------------------------------------------------

    def _make_hook(self, name: str):
        def hook(module: HebbianLinearLayer, inputs, output) -> None:
            x = inputs[0].detach()
            tally = self._tallies[name]

            nonzero = torch.count_nonzero(x).item()
            numel = x.numel()
            batch = x.shape[0]

            tally.calls += 1
            tally.input_elems += numel
            tally.input_spikes += nonzero
            # A dense ANN would MAC every synapse for every row in the batch.
            tally.dense_macs += batch * module.in_channels * module.out_channels

            # Analog input (values outside {0, 1}) can't be billed as accumulates.
            if not _is_binary(x):
                tally.spiking = False

        return hook

    # -- results -------------------------------------------------------------

    def summary(self) -> dict:
        """Return aggregate stats as a plain dict (safe to log/serialize)."""
        snn_energy = 0.0
        ann_energy = 0.0
        total_sop = 0.0
        total_dense_macs = 0.0
        total_spikes = 0.0
        total_elems = 0

        layers = []
        for t in self._tallies.values():
            # SNN cost: sparse ACs for spiking layers, dense MACs for analog input.
            if t.spiking:
                layer_snn = t.sop * self.e_ac
            else:
                layer_snn = t.dense_macs * self.e_mac
            layer_ann = t.dense_macs * self.e_mac

            snn_energy += layer_snn
            ann_energy += layer_ann
            total_sop += t.sop
            total_dense_macs += t.dense_macs
            total_spikes += t.input_spikes
            total_elems += t.input_elems

            layers.append({
                "layer": t.name,
                "in": t.in_features,
                "out": t.out_features,
                "spiking_input": t.spiking,
                "spike_rate": t.spike_rate,
                "sop": t.sop,
                "dense_macs": t.dense_macs,
                "snn_energy_pj": layer_snn,
                "ann_energy_pj": layer_ann,
            })

        return {
            "snn_energy_pj": snn_energy,
            "ann_energy_pj": ann_energy,
            "energy_savings": 1.0 - snn_energy / ann_energy if ann_energy else 0.0,
            "energy_ratio_ann_over_snn": ann_energy / snn_energy if snn_energy else float("inf"),
            "total_sop": total_sop,
            "total_dense_macs": total_dense_macs,
            "mean_spike_rate": total_spikes / total_elems if total_elems else 0.0,
            "e_mac_pj": self.e_mac,
            "e_ac_pj": self.e_ac,
            "layers": layers,
        }

    def report(self) -> str:
        """Pretty, printable version of :meth:`summary`."""
        s = self.summary()

        def fmt_e(pj: float) -> str:
            for unit, scale in (("mJ", 1e9), ("uJ", 1e6), ("nJ", 1e3), ("pJ", 1)):
                if pj >= scale:
                    return f"{pj / scale:.3f} {unit}"
            return f"{pj:.3f} pJ"

        lines = []
        lines.append("Per-layer:")
        header = f"  {'layer':<28}{'in->out':>12}{'spike%':>9}{'SOP':>14}{'denseMAC':>14}"
        lines.append(header)
        for L in s["layers"]:
            kind = "" if L["spiking_input"] else "  (analog->MAC)"
            shape = f"{L['in']}->{L['out']}"
            lines.append(
                f"  {L['layer']:<28}{shape:>12}"
                f"{100 * L['spike_rate']:>8.2f}%"
                f"{L['sop']:>14,.0f}"
                f"{L['dense_macs']:>14,.0f}{kind}"
            )
        lines.append("")
        lines.append(f"Mean spike rate      : {100 * s['mean_spike_rate']:.2f}%")
        lines.append(f"Total SOP (SNN)      : {s['total_sop']:,.0f} accumulates")
        lines.append(f"Total MAC (dense ANN): {s['total_dense_macs']:,.0f} MACs")
        lines.append(f"SNN energy           : {fmt_e(s['snn_energy_pj'])}")
        lines.append(f"ANN energy           : {fmt_e(s['ann_energy_pj'])}")
        lines.append(f"Energy savings       : {100 * s['energy_savings']:.2f}%")
        lines.append(f"ANN / SNN            : {s['energy_ratio_ann_over_snn']:.1f}x")
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
