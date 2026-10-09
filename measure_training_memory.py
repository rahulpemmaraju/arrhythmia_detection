"""
Peak memory of one training step, BPTT vs e-prop, as the input sequence grows.

BPTT has to keep every timestep's activations until its single backward pass, so its
memory should grow linearly with sequence length T. Per-timestep e-prop
(eprop_online.py) backwards each timestep's loss as soon as it exists and carries
only the eligibility traces forward, so its memory should not depend on T. This
script measures both on real ECG.

What is trained
---------------
    bptt    four_layer_hgrsnn (hgrsnn_large_config.yaml) through main.train_step
    eprop   four_layer_hgrsnn_eprop (hgrsnn_eprop_large_config.yaml) through
            eprop_online.online_train_step

Both are the paper's [32, 64, 32] HG-RSNN at their xbest hyperparameters, and each
step is the real training function, run on a one-batch DataLoader. Every timestep is
supervised. The weights are untrained. Autograd saves the same tensor shapes whatever
the weights, spikes and labels are, so none of these change the memory.

Input
-----
A contiguous stretch of one MIT-BIH record, preprocessed as create_beat_dataset.py
does it: lead 1, downsampled 360 -> 180 Hz, neurokit-cleaned, and z-scored by the
whole record's mean and std (patient_norm). The first T samples are one sequence
of length T. With --batch_size B, the batch is B back-to-back segments.

Two measurements per (algorithm, T)
-----------------------------------
Each point runs in a fresh Python process, so one run's allocations can't leak into
the next. After a short warm-up step:

    peak_rss_increase_mb  how far the process's resident memory rose above its
                          pre-step level during the step. This is what the OS sees:
                          tensors, autograd graph nodes and allocator overhead.
                          Resident memory is polled from a background thread, and
                          the OS high-water mark is used when it rose during the
                          step. Allocator reuse makes this noisy (BPTT can differ
                          ~2x between runs at one length), so each point is
                          repeated (--repeats) and the median is reported.
    graph_peak_mb         peak bytes held by the autograd graph for backward, i.e.
                          the tensors passed through saved_tensors_hooks, which
                          includes ctx.save_for_backward in the e-prop Functions.
                          Each storage is counted once and parameters are
                          excluded. This is exact and noise-free, but it leaves
                          out graph bookkeeping. The step is re-run with the hooks
                          installed for this, so the hook overhead doesn't
                          inflate the RSS measurement.

Example (from this directory)
-------
    python measure_training_memory.py
    python measure_training_memory.py --batch_size 64 --seq_lens 90 270 1080 2160

Output (default: ../memory_evaluation)
------
    training_memory.csv   one row per algorithm x sequence length x repeat
    training_memory.png   both measurements (median over repeats) against sequence length
"""

import os
import gc
import sys
import json
import time
import argparse
import resource
import tempfile
import threading
import subprocess

import yaml
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(CODE_DIR), 'memory_evaluation')
DEFAULT_MITDB_DIR = os.path.join(os.path.dirname(CODE_DIR), 'mit-bih-arrhythmia-database-1.0.0')

# matches ../beats/window_config.json
DOWNSAMPLE_FACTOR = 2
FS = 180

ALGORITHMS = {
    'bptt': 'hgrsnn_large_config.yaml',
    'eprop': 'hgrsnn_eprop_large_config.yaml',
}

LABELS = {'bptt': 'BPTT', 'eprop': 'e-prop (per-timestep)'}

# a short step run before measuring, so one-time allocations (thread pools, kernel
# caches, the optimizer's first touch of .grad) aren't billed to the measured step
WARMUP_LEN = 16

MB = 1e6


# --------------------------------------------------------------------------- data

def load_ecg(mitdb_dir, record, num_samples):
    '''
    first num_samples of a record, preprocessed as create_beat_dataset.py does it
    and z-scored by the whole record (patient_norm)
    '''

    import wfdb
    from create_beat_dataset import downsample_ecg, preprocess_ecg

    signal = wfdb.rdrecord(os.path.join(mitdb_dir, str(record))).p_signal[:, 0]
    signal = downsample_ecg(signal, DOWNSAMPLE_FACTOR)
    signal = preprocess_ecg(signal, 'neurokit', FS)

    if num_samples > len(signal):
        raise ValueError(f'record {record} has {len(signal)} samples at {FS} Hz, '
                         f'{num_samples} needed (max seq_len x batch_size)')

    signal = (signal - signal.mean()) / signal.std()

    return signal[:num_samples].astype(np.float32)


def make_loader(signal, seq_len, batch_size):
    x = torch.from_numpy(signal[:seq_len * batch_size]).reshape(batch_size, seq_len)

    # the label only sets loss values, never which tensors autograd keeps
    y = torch.zeros(batch_size, seq_len)

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size)


# ---------------------------------------------------------------- measurement

class _SavedHandle:
    # stands in for a saved tensor inside the graph; autograd drops it when it frees
    # the saved variable, which is what tells the tracker the bytes are gone
    __slots__ = ('tracker', 'key', 'tensor')

    def __init__(self, tracker, key, tensor):
        self.tracker, self.key, self.tensor = tracker, key, tensor

    def __del__(self):
        self.tracker.release(self.key)


class SavedTensorTracker:
    '''
    live and peak bytes of the tensors the autograd graph holds for backward.

    A storage saved by several nodes (e.g. an eligibility trace buffer that is also
    a module attribute) is counted once, and only while the graph holds it.
    Parameters are skipped: every step saves them, but they exist regardless.
    '''

    def __init__(self, parameters):
        self.skip = {p.untyped_storage().data_ptr() for p in parameters}
        self.refs = {}  # storage pointer -> [handles alive, nbytes]
        self.current = 0
        self.peak = 0

    def pack(self, tensor):
        storage = tensor.untyped_storage()
        key = storage.data_ptr()

        if key in self.skip:
            return tensor

        entry = self.refs.setdefault(key, [0, storage.nbytes()])

        if entry[0] == 0:
            self.current += entry[1]
            self.peak = max(self.peak, self.current)

        entry[0] += 1

        return _SavedHandle(self, key, tensor)

    def unpack(self, saved):
        return saved.tensor if isinstance(saved, _SavedHandle) else saved

    def release(self, key):
        entry = self.refs[key]
        entry[0] -= 1

        if entry[0] == 0:
            self.current -= entry[1]
            del self.refs[key]


def peak_rss_bytes():
    # the process's resident high-water mark: bytes on macOS, kilobytes on Linux
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == 'darwin' else peak * 1024


class RSSSampler:
    '''highest resident memory seen by a polling thread while the context is open'''

    def __init__(self, interval=1e-3):
        self.interval = interval
        self.process = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def run_worker(algorithm, seq_len, batch_size, signal_file):
    '''one (algorithm, seq_len) point, in its own process; prints a json row'''

    from models import create_model
    from main import train_step as bptt_train_step
    from eprop_online import online_train_step
    from utils.encoding import Current_Encoder

    torch.set_num_threads(1)
    torch.manual_seed(0)

    with open(os.path.join(CODE_DIR, 'nested_configs', ALGORITHMS[algorithm])) as f:
        model_configs = yaml.safe_load(f)

    model = create_model(model_configs, 'xbest')
    optimizer = optim.SGD(model.parameters(), lr=model_configs['xbest']['lr'])
    loss_fn = nn.BCEWithLogitsLoss()

    # the signal is already z-scored, so the encoder only adds the channel axis
    encoder = Current_Encoder(0.0, 1.0)
    signal = np.load(signal_file)

    def step(length):
        loader = make_loader(signal, length, batch_size)

        if algorithm == 'bptt':
            bptt_train_step(model, loader, encoder, optimizer, loss_fn, 0, 'cpu')
        else:
            online_train_step(model, loader, encoder, optimizer, loss_fn, 'cpu')

    step(min(WARMUP_LEN, seq_len))
    gc.collect()

    rss_before = psutil.Process().memory_info().rss
    high_water_before = peak_rss_bytes()

    start = time.perf_counter()
    with RSSSampler() as sampler:
        step(seq_len)
    step_seconds = time.perf_counter() - start

    # the high-water mark is exact but only informative if this step set it; a
    # higher mark left over from imports would say nothing about the step
    peak_rss = sampler.peak
    if peak_rss_bytes() > high_water_before:
        peak_rss = max(peak_rss, peak_rss_bytes())

    gc.collect()

    tracker = SavedTensorTracker(model.parameters())
    with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
        step(seq_len)

    print(json.dumps({
        'algorithm': algorithm,
        'seq_len': seq_len,
        'seconds_of_ecg': seq_len / FS,
        'batch_size': batch_size,
        'peak_rss_increase_mb': (peak_rss - rss_before) / MB,
        'graph_peak_mb': tracker.peak / MB,
        'rss_before_mb': rss_before / MB,
        'step_seconds': step_seconds,
    }))


# ----------------------------------------------------------------------- driver

def measure(args):
    signal = load_ecg(args.mitdb_dir, args.record, max(args.seq_lens) * args.batch_size)

    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        signal_file = os.path.join(tmp, 'ecg.npy')
        np.save(signal_file, signal)

        for seq_len in sorted(args.seq_lens):
            for algorithm in args.algorithms:
                for repeat in range(args.repeats):
                    result = subprocess.run(
                        [sys.executable, os.path.abspath(__file__), '--worker', algorithm,
                         '--seq_len', str(seq_len), '--batch_size', str(args.batch_size),
                         '--signal_file', signal_file],
                        cwd=CODE_DIR, capture_output=True, text=True,
                    )

                    if result.returncode != 0:
                        sys.stderr.write(result.stderr)
                        raise RuntimeError(f'{algorithm} at seq_len={seq_len} failed')

                    row = json.loads(result.stdout.strip().splitlines()[-1])
                    row['record'] = args.record
                    row['repeat'] = repeat
                    rows.append(row)

                point = pd.DataFrame(rows[-args.repeats:])
                rss = point['peak_rss_increase_mb']

                print('{:>6} ({:5.1f} s)  {:<22} peak RSS +{:8.2f} MB [{:.1f}-{:.1f}]   graph {:8.3f} MB   {:6.1f} s'.format(
                    seq_len, seq_len / FS, LABELS[algorithm], rss.median(), rss.min(), rss.max(),
                    point['graph_peak_mb'].median(), point['step_seconds'].median()))
                sys.stdout.flush()

    return pd.DataFrame(rows)


def median_over_repeats(df):
    return df.groupby(['algorithm', 'seq_len'], sort=False).median(numeric_only=True).reset_index()


def plot(df, path, batch_size):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'bptt': '#2a78d6', 'eprop': '#eb6834'}
    ink, muted, grid = '#1f1f1e', '#6b6a64', '#e4e3dc'

    panels = [
        ('peak_rss_increase_mb', 'Peak process memory increase (RSS)'),
        ('graph_peak_mb', 'Peak memory held for backward (saved tensors)'),
    ]

    df = median_over_repeats(df)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    for ax, (column, title) in zip(axes, panels):
        for algorithm, group in df.groupby('algorithm', sort=False):
            group = group.sort_values('seq_len')
            ax.plot(group['seq_len'], group[column], color=colors[algorithm], linewidth=1.5,
                    marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1.5,
                    label=LABELS[algorithm], solid_capstyle='round')

            last = group.iloc[-1]
            ax.annotate('{:.1f} MB'.format(last[column]), (last['seq_len'], last[column]),
                        xytext=(6, 0), textcoords='offset points', va='center',
                        fontsize=9, color=ink)

        ax.set_title(title, loc='left', fontsize=11, color=ink)
        ax.set_xlabel(f'sequence length (samples at {FS} Hz)', fontsize=9, color=muted)
        ax.set_ylabel('MB', fontsize=9, color=muted)
        ax.set_ylim(bottom=min(0, df[column].min() * 1.1))
        ax.set_xlim(0, df['seq_len'].max() * 1.15)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, labelsize=8.5, length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(grid)

    axes[0].legend(frameon=False, fontsize=9, labelcolor=ink, loc='upper left')
    fig.suptitle(f'Training-step memory vs sequence length (HG-RSNN [32, 64, 32], batch size {batch_size}, '
                 'median over repeats)',
                 x=0.01, ha='left', fontsize=12, color=ink)

    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='peak training-step memory of BPTT vs per-timestep e-prop as sequence length grows')
    parser.add_argument('--seq_lens', type=int, nargs='+',
                        default=[90, 180, 270, 540, 1080, 2160, 4320, 8640],
                        help=f'sequence lengths in samples ({FS} Hz); 270 is the beat window')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='1 measures a single sample; training uses 64')
    parser.add_argument('--algorithms', nargs='+', choices=sorted(ALGORITHMS), default=['bptt', 'eprop'])
    parser.add_argument('--repeats', type=int, default=3,
                        help='fresh-process runs per point; the RSS median is reported')
    parser.add_argument('--record', type=int, default=100, help='MIT-BIH record the ECG is taken from')
    parser.add_argument('--mitdb_dir', type=str, default=DEFAULT_MITDB_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)

    # internal: one measurement point, run in a child process by measure()
    parser.add_argument('--worker', choices=sorted(ALGORITHMS), help=argparse.SUPPRESS)
    parser.add_argument('--seq_len', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--signal_file', type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker, args.seq_len, args.batch_size, args.signal_file)
        sys.exit(0)

    df = measure(args)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'training_memory.csv')
    png_path = os.path.join(args.output_dir, 'training_memory.png')

    df.to_csv(csv_path, index=False)
    plot(df, png_path, args.batch_size)

    print(f'\nwritten {csv_path}\n        {png_path}')
