"""
Estimate per-beat inference energy on the test set for a trained SNN or a
non-spiking baseline, and log it as tidy CSVs for plotting.

The model family is inferred from the config: a nested SNN config (model_type
under kwargs, from nested_configs/) or a baseline config (model_type at the top
level, from baselines/configs/). Inference mirrors main.py / baselines/train.py:
the same split, the same per-record normalization (patient_norm), the same
beat-span readout. The test AUC from that same pass is logged next to the energy
(macro one-vs-rest for multiclass), so energy-vs-performance plots need no join.

Op-counting rules live in models/energy.py; both families are counted by the same
EnergyMeter.

Examples (from this directory)
--------
    python evaluate_energy.py --config_file hgrsnn_large_config.yaml \
        --model_name hgrsnn_wo_td --weight_dir ../remote_results/model_weights \
        --data_dir ../beats --use_best_model

    python evaluate_energy.py --config_file conv1d_config.yaml \
        --model_name conv1d_baseline --weight_dir ../baseline_results/model_weights \
        --data_dir ../beats

    bash jobscripts/evaluate_energy_local.sh     # the whole comparison set

Weights are loaded from {weight_dir}/s_{seed}_{model_name}.pth, the naming main.py
and baselines/train.py save under.

Output (default: ../energy_evaluation)
------
    seeds/{model_name}.csv    one row per seed: energy, op counts, firing rate, AUC
    layers/{model_name}.csv   one row per seed x layer
    runs/{model_name}.json    args, config and the full per-seed summaries
    all_seeds.csv             every seeds/*.csv concatenated
    all_layers.csv            every layers/*.csv concatenated
    summary.csv               mean/std over seeds, one row per model

Energies are picojoules per beat and op counts are per beat, where a beat is one
input window. The whole window is billed, pre-beat context included, because the
model has to run over it. Re-running a model replaces its per-model files rather
than appending, so rows are never duplicated, and parallel runs of different
models don't overwrite each other. The combined files are rebuilt from the
per-model files at the end of every run.

n_params counts every trainable parameter a model builds; n_params_used counts only
those of modules that actually run at inference. They differ for configs that build
weights they never call, e.g. the unused top-down t_v projections in the HGRSNN cells.
"""

import os
import glob
import json
import argparse
import datetime

import yaml
import numpy as np
import pandas as pd
import torch

from dataloader import train_test_val_split, get_dataset_statistics, get_beat_bounds
from models import create_model
from models.energy import (
    EnergyMeter, E_MAC_PJ, E_AC_PJ, SpikeRecorder, plot_voltage, plot_spikes, plot_energy,
)
from utils.encoding import Current_Encoder, unpack_batch
from utils.readout import compute_auc
from main import AAMI_CLASS_NAMES
from baselines.models import create_baseline
# the baselines' own split rules, reused so a baseline is measured on exactly the
# data it was evaluated on
from baselines.train import make_dataset_configs

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_CONFIG_DIR = os.path.join(CODE_DIR, 'nested_configs')
BASELINE_CONFIG_DIR = os.path.join(CODE_DIR, 'baselines', 'configs')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(CODE_DIR), 'energy_evaluation')

# properties of the model, constant across its seeds -- summary.csv groups on these
MODEL_COLS = [
    'model_name', 'family', 'model_type', 'config_file', 'config_key', 'binary',
    'out_dims', 'n_params', 'n_params_used', 'timesteps', 'e_mac_pj', 'e_ac_pj',
]

# per-seed measurements -- summary.csv reports the mean and std of each
METRIC_COLS = [
    'test_auc', 'mean_firing_rate', 'ac_ops', 'mac_ops', 'state_mac_ops',
    'synaptic_energy_pj', 'state_energy_pj', 'total_energy_pj',
    'ungated_synaptic_energy_pj', 'ungated_total_energy_pj', 'gating_savings',
    'dense_equiv_synaptic_energy_pj', 'dense_equiv_total_energy_pj', 'synaptic_savings',
]


def resolve_config_path(config_file):
    """Accept a full path, or a bare filename under nested_configs/ or baselines/configs/."""
    for candidate in (config_file,
                      os.path.join(SNN_CONFIG_DIR, config_file),
                      os.path.join(BASELINE_CONFIG_DIR, config_file)):
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"config '{config_file}' not found as a path, or under "
        f"{SNN_CONFIG_DIR} or {BASELINE_CONFIG_DIR}"
    )


def describe_config(config, use_best_model):
    """Family and readout settings, read the way each family's training script reads them."""
    if 'kwargs' in config:
        kwargs = config['kwargs']
        return {
            'family': 'snn',
            'model_type': kwargs['model_type'],
            'config_key': 'xbest' if use_best_model else 'x0',
            'binary': kwargs.get('binary', True),
            'out_dims': kwargs.get('out_dims', 1),
            'patient_norm': kwargs.get('patient_norm', False),
        }

    if 'model_type' in config:
        binary = config.get('binary', True)
        return {
            'family': 'baseline',
            'model_type': config['model_type'],
            # baselines have no x0/xbest; their hyperparameters are the config
            'config_key': '',
            'binary': binary,
            'out_dims': config.get('out_dims', 1 if binary else len(AAMI_CLASS_NAMES)),
            'patient_norm': config.get('patient_norm', False),
        }

    raise ValueError('config is neither a nested SNN config (kwargs.model_type) '
                     'nor a baseline config (top-level model_type)')


def build_dataset_configs(spec, config, data_dir, seed):
    if spec['family'] == 'baseline':
        return make_dataset_configs(config, data_dir, spec['binary'], seed)

    # main.py's evaluation split, field for field. main.py ignores
    # kwargs.dataset_configs (its paths point at the cluster), so this does too
    return {
        'data_file': f'{data_dir}/data.hdf5',
        'metadata': f'{data_dir}/data.csv',
        'train_prop': 0.6,
        'val_prop': 0.2,
        'test_prop': 0.2,
        'binary': spec['binary'],
        'random_state': seed,
        'balance': True,
        'fourier': False,
        'return_stats': spec['patient_norm'],
    }


def build_model(spec, config, beat_bounds):
    if spec['family'] == 'snn':
        return create_model(config, spec['config_key'])

    return create_baseline(config['model_type'], out_dims=spec['out_dims'],
                           beat_bounds=beat_bounds, **config.get('model_kwargs', {}))


def count_used_params(model, x):
    """Trainable parameters of the modules that actually run in a forward pass over x."""
    called = set()
    handles = [module.register_forward_pre_hook(lambda mod, args: called.add(mod))
               for module in model.modules()]

    model.eval()
    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    # recurse=False, so a parameter is counted once, by the module that owns it
    return sum(p.numel() for module in called
               for p in module.parameters(recurse=False) if p.requires_grad)


def measure_seed(model, test_loader, encoder, beat_bounds, binary, e_mac, e_ac,
                 max_batches=None):
    """One metered pass over the test set.

    Returns the meter and the test AUC, scored from the same outputs the way
    main.get_binary_metrics scores them (mean over the labeled beat span).
    """
    model.eval()
    start, end = beat_bounds
    y_true, y_score = [], []

    with EnergyMeter(model, e_mac=e_mac, e_ac=e_ac) as meter, torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            data, target, stats = unpack_batch(batch)
            data = encoder.encode(data, *stats).float()

            output, _ = model(data)

            y_true.append(target[:, 0].numpy())
            y_score.append(output[:, start:end].mean(1).numpy())

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    # a truncated run (--max_batches) can hold a single class, where AUC is undefined
    auc = compute_auc(y_true, y_score, binary) if len(np.unique(y_true)) > 1 else float('nan')

    return meter, auc


def plot_sample(model, test_loader, encoder, stem, neuron_idx=None, t_start=None, t_end=None):
    """Plot one neuron from one test sample: voltage and spikes, as two figures."""
    model.eval()
    data, _, stats = unpack_batch(next(iter(test_loader)))
    data = encoder.encode(data, *stats).float()

    with SpikeRecorder(model, sample_idx=0) as recorder, torch.no_grad():
        model(data)

    stacked = recorder.stacked()
    window = {'neuron_idx': neuron_idx, 't_start': t_start, 't_end': t_end}
    plot_voltage(stacked, stem + '_voltage', **window)
    plot_spikes(stacked, stem + '_spikes', **window)


def write_csv(df, path):
    # write then rename, so a parallel run rebuilding the combined files never
    # reads a half-written csv
    tmp = f'{path}.{os.getpid()}.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def rebuild_combined(output_dir):
    for subdir, combined in (('seeds', 'all_seeds.csv'), ('layers', 'all_layers.csv')):
        files = sorted(glob.glob(os.path.join(output_dir, subdir, '*.csv')))
        frames = [pd.read_csv(f) for f in files]
        write_csv(pd.concat(frames, ignore_index=True), os.path.join(output_dir, combined))

    seeds = pd.read_csv(os.path.join(output_dir, 'all_seeds.csv'))

    # dropna=False: baselines have an empty config_key, which would otherwise
    # silently drop them from the summary
    group_cols = [c for c in MODEL_COLS if c in seeds.columns]
    grouped = seeds.groupby(group_cols, dropna=False, sort=False)
    summary = grouped[METRIC_COLS].agg(['mean', 'std'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    summary['n_seeds'] = grouped.size()

    write_csv(summary.reset_index(), os.path.join(output_dir, 'summary.csv'))


def evaluate_energy(args):
    config_path = resolve_config_path(args.config_file)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    spec = describe_config(config, args.use_best_model)

    if spec['family'] == 'baseline' and args.use_best_model:
        print('note: --use_best_model does nothing for a baseline config')

    evaluated_at = datetime.datetime.now().isoformat(timespec='seconds')

    # the normalization constants depend only on which beats are in the training
    # split, and reading them means a pass over the whole training set. keyed on
    # the split's row index, so seeds that share a split share one pass without
    # assuming anything about how the split uses random_state
    stats_cache = {}

    seed_rows, layer_rows, run_seeds = [], [], []

    for seed in range(args.num_seeds):
        weight_path = os.path.join(args.weight_dir, f's_{seed}_{args.model_name}.pth')

        if not os.path.exists(weight_path):
            print(f'[seed {seed}] no weights at {weight_path}, skipping')
            continue

        dataset_configs = build_dataset_configs(spec, config, args.data_dir, seed)
        train_dataset, _, test_dataset = train_test_val_split(**dataset_configs)
        beat_bounds = get_beat_bounds(train_dataset)

        stats_key = (dataset_configs['data_file'], tuple(train_dataset.metadata.index))
        if stats_key not in stats_cache:
            stats_cache[stats_key] = get_dataset_statistics(train_dataset)
        encoder = Current_Encoder(*stats_cache[stats_key])

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size)

        model = build_model(spec, config, beat_bounds)
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        probe, _, probe_stats = unpack_batch(next(iter(test_loader)))
        probe = encoder.encode(probe[:1], *(s[:1] for s in probe_stats)).float()
        n_params_used = count_used_params(model, probe)

        meter, auc = measure_seed(model, test_loader, encoder, beat_bounds, spec['binary'],
                                  args.e_mac, args.e_ac, args.max_batches)
        summary = meter.summary()

        print(f'\n===== {args.model_name}, seed {seed} =====')
        print(meter.report())
        print(f"Test AUC             : {auc:.4f}{'' if spec['binary'] else ' (macro OVR)'}")
        print(f'Parameters           : {n_params:,} built, {n_params_used:,} used at inference')

        seed_rows.append({
            'model_name': args.model_name,
            'family': spec['family'],
            'model_type': spec['model_type'],
            'config_file': os.path.basename(config_path),
            'config_key': spec['config_key'],
            'binary': spec['binary'],
            'out_dims': spec['out_dims'],
            'n_params': n_params,
            'n_params_used': n_params_used,
            'timesteps': summary['timesteps'],
            'e_mac_pj': summary['e_mac_pj'],
            'e_ac_pj': summary['e_ac_pj'],
            'seed': seed,
            'n_beats': summary['n_beats'],
            'full_test_set': summary['n_beats'] == len(test_dataset),
            'beat_start': beat_bounds[0],
            'beat_end': beat_bounds[1],
            'test_auc': auc,
            **{key: summary[key] for key in METRIC_COLS if key != 'test_auc'},
            'weight_path': os.path.abspath(weight_path),
            'evaluated_at': evaluated_at,
        })

        for layer in summary['layers']:
            layer_rows.append({'model_name': args.model_name, 'family': spec['family'],
                               'seed': seed, **layer})

        run_seeds.append({'seed': seed, 'weight_path': os.path.abspath(weight_path),
                          'test_auc': auc, 'energy': summary})

        if args.plot and spec['family'] == 'snn':
            plot_dir = os.path.join(args.output_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            stem = os.path.join(plot_dir, f's_{seed}_{args.model_name}')
            plot_sample(model, test_loader, encoder, stem, neuron_idx=args.neuron_idx,
                        t_start=args.t_start, t_end=args.t_end)
            print(f'plots written to {stem}_voltage.png, {stem}_spikes.png')

    if not seed_rows:
        raise SystemExit(f'no weights found for {args.model_name} in {args.weight_dir} '
                         f'(seeds 0-{args.num_seeds - 1}); nothing measured')

    for subdir in ('seeds', 'layers', 'runs'):
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    seeds_df = pd.DataFrame(seed_rows)
    write_csv(seeds_df, os.path.join(args.output_dir, 'seeds', f'{args.model_name}.csv'))
    write_csv(pd.DataFrame(layer_rows), os.path.join(args.output_dir, 'layers', f'{args.model_name}.csv'))

    with open(os.path.join(args.output_dir, 'runs', f'{args.model_name}.json'), 'w') as f:
        json.dump({
            'args': vars(args),
            'config_path': os.path.abspath(config_path),
            'config': config,
            'spec': spec,
            'torch_version': torch.__version__,
            'evaluated_at': evaluated_at,
            'seeds': run_seeds,
        }, f, indent=2, default=str)

    rebuild_combined(args.output_dir)

    if not seeds_df['full_test_set'].all():
        print('\nnote: --max_batches truncated the test set; AUC and firing rate are partial')

    print(f'\n===== {args.model_name}, mean over {len(seeds_df)} seed(s) =====')
    print(f"Test AUC        : {seeds_df['test_auc'].mean():.4f}")
    if spec['family'] == 'snn':
        print(f"Firing rate     : {100 * seeds_df['mean_firing_rate'].mean():.2f}%")
    print(f"Synaptic energy : {seeds_df['synaptic_energy_pj'].mean() / 1e3:,.2f} nJ per beat")
    print(f"Total energy    : {seeds_df['total_energy_pj'].mean() / 1e3:,.2f} nJ per beat")
    if seeds_df['gating_savings'].mean() > 0:
        print(f"Without gating  : {seeds_df['ungated_total_energy_pj'].mean() / 1e3:,.2f} nJ per beat "
              f"(gating saves {100 * seeds_df['gating_savings'].mean():.2f}%)")
    print(f'results written to {os.path.abspath(args.output_dir)}')

    if args.plot and spec['family'] == 'snn':
        stem = os.path.join(args.output_dir, 'plots', f'{args.model_name}_energy')
        plot_energy(float(seeds_df['synaptic_energy_pj'].mean()),
                    float(seeds_df['dense_equiv_synaptic_energy_pj'].mean()), stem)
        print(f'plot written to {stem}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Estimate per-beat inference energy for a trained SNN or baseline.'
    )
    parser.add_argument('--config_file', type=str, required=True,
                        help='config YAML: a full path, or a name under nested_configs/ '
                             'or baselines/configs/')
    parser.add_argument('--model_name', type=str, required=True,
                        help='weights are loaded from s_{seed}_{model_name}.pth')
    parser.add_argument('--weight_dir', type=str, required=True,
                        help='folder of s_{seed}_{model_name}.pth weights')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='dataset folder holding data.hdf5 and data.csv')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='where the CSVs go (default: ../energy_evaluation)')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--use_best_model', action='store_true',
                        help="SNN only: use the config's xbest hyperparameters instead of x0")
    parser.add_argument('--max_batches', type=int, default=None,
                        help='meter only this many test batches, for a quick check '
                             '(logged as full_test_set=False)')
    parser.add_argument('--e_mac', type=float, default=E_MAC_PJ,
                        help=f'energy per multiply-accumulate, pJ (default {E_MAC_PJ})')
    parser.add_argument('--e_ac', type=float, default=E_AC_PJ,
                        help=f'energy per accumulate, pJ (default {E_AC_PJ})')
    parser.add_argument('--plot', action='store_true',
                        help='SNN only: also plot one neuron (voltage vs spikes) and the '
                             'mean SNN vs dense-ANN energy, into {output_dir}/plots')
    parser.add_argument('--neuron_idx', type=int, default=None,
                        help='which neuron to draw (default: the most active one per layer)')
    parser.add_argument('--t_start', type=int, default=150,
                        help='first timestep to draw')
    parser.add_argument('--t_end', type=int, default=275,
                        help='last timestep to draw (clipped to the trace length)')

    evaluate_energy(parser.parse_args())
