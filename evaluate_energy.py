"""
Estimate SNN vs ANN energy for a trained model on the test set, driven by config.

Pick any model architecture/hyperparameter set from a nested config YAML, load
the matching pretrained weights, run the test set through the EnergyMeter, and
print the per-seed energy report. Energy only -- no metrics, thresholds, plots,
or files are produced.

Example
-------
    python evaluate_energy.py \
        --config_file rsnn_config.yaml \
        --model_name optimized_rsnn \
        --data_dir /path/to/processed_beats \
        --weight_dir ./results/model_weights \
        --use_best_model

Weights are loaded from ``{weight_dir}/s_{seed}_{model_name}.pth`` (the naming
``main.py`` and ``nested_train.py`` save under).
"""

import os
import argparse

import yaml
import numpy as np
import torch

from dataloader import train_test_val_split, get_dataset_statistics
from models import create_model
from models.energy import (
    EnergyMeter, SpikeRecorder, plot_voltage, plot_spikes, plot_energy,
)
from utils.encoding import Current_Encoder


def resolve_config_path(config_file):
    """Accept either a full path or a bare filename under nested_configs/."""
    if os.path.exists(config_file):
        return config_file
    candidate = os.path.join('nested_configs', config_file)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"config not found as '{config_file}' or '{candidate}'"
    )


def build_dataset_configs(model_configs, data_dir, seed):
    """Build the dataset_configs for one seed.

    Starts from the config's own kwargs.dataset_configs (if present) so the eval
    split matches how the model was trained, then applies the per-seed random
    state and an optional --data_dir override of the data location.
    """
    base = dict(model_configs['kwargs'].get('dataset_configs', {}))

    base.setdefault('train_prop', 0.6)
    base.setdefault('val_prop', 0.2)
    base.setdefault('test_prop', 0.2)
    base.setdefault('balance', True)
    base.setdefault('fourier', False)

    if data_dir is not None:
        base['data_file'] = os.path.join(data_dir, 'data.hdf5')
        base['metadata'] = os.path.join(data_dir, 'data.csv')

    base['binary'] = True
    base['random_state'] = seed

    return base


def measure_energy(model, test_loader, encoder, device):
    """Run the test set through the EnergyMeter and return the meter."""
    model.eval()
    with EnergyMeter(model) as meter:
        with torch.no_grad():
            for data, _ in test_loader:
                if encoder is not None:
                    data = encoder.encode(data)
                data = data.to(device)
                model(data)
    return meter


def plot_sample(model, test_loader, encoder, device, stem, sample_idx=0,
                neuron_idx=None, t_start=None, t_end=None):
    """Plot one neuron from one test sample: voltage and spikes, as two figures."""
    model.eval()
    data, _ = next(iter(test_loader))
    if encoder is not None:
        data = encoder.encode(data)
    data = data.to(device)

    with SpikeRecorder(model, sample_idx=sample_idx) as recorder:
        with torch.no_grad():
            model(data)

    stacked = recorder.stacked()
    window = {'neuron_idx': neuron_idx, 't_start': t_start, 't_end': t_end}
    plot_voltage(stacked, stem + '_voltage', **window)
    plot_spikes(stacked, stem + '_spikes', **window)


def evaluate_energy(args):
    config_path = resolve_config_path(args.config_file)
    with open(config_path, 'r') as f:
        model_configs = yaml.safe_load(f)

    config_key = 'xbest' if args.use_best_model else 'x0'
    device = 'cpu'

    seeds = list(range(args.num_seeds))
    weight_dir = args.weight_dir

    # one model instance is enough; we just reload weights per seed
    model = create_model(model_configs, config_key)

    savings = []
    ratios = []
    spike_rates = []
    snn_energy = []
    ann_energy = []

    for seed in seeds:
        model_weight = f's_{seed}_{args.model_name}'
        model_path = os.path.join(weight_dir, model_weight) + '.pth'

        if not os.path.exists(model_path):
            print(f'[seed {seed}] no weights at {model_path}, skipping')
            continue

        dataset_configs = build_dataset_configs(model_configs, args.data_dir, seed)
        train_dataset, _, test_dataset = train_test_val_split(**dataset_configs)
        train_mean, train_std = get_dataset_statistics(train_dataset)
        encoder = Current_Encoder(train_mean, train_std)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size)

        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])

        meter = measure_energy(model, test_loader, encoder, device)
        summary = meter.summary()

        print(f'\n===== seed {seed} =====')
        print(meter.report())

        if args.plot:
            os.makedirs(args.plot_dir, exist_ok=True)
            stem = os.path.join(args.plot_dir, f's_{seed}_{args.model_name}')
            plot_sample(model, test_loader, encoder, device, stem,
                        neuron_idx=args.neuron_idx,
                        t_start=args.t_start, t_end=args.t_end)
            print(f'plots written to {stem}_voltage.png, {stem}_spikes.png')

        savings.append(summary['energy_savings'])
        ratios.append(summary['energy_ratio_ann_over_snn'])
        spike_rates.append(summary['mean_spike_rate'])
        snn_energy.append(summary['snn_energy_pj'])
        ann_energy.append(summary['ann_energy_pj'])

    if not savings:
        print('No weights found for any seed; nothing measured.')
        return

    print('\n===== mean over seeds =====')
    print(f'Energy savings : {np.mean(savings) * 100:.2f}%')
    print(f'ANN / SNN      : {np.mean(ratios):.1f}x')
    print(f'Mean spike rate: {np.mean(spike_rates) * 100:.2f}%')

    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)
        stem = os.path.join(args.plot_dir, f'{args.model_name}_energy')
        plot_energy(float(np.mean(snn_energy)), float(np.mean(ann_energy)), stem)
        print(f'plot written to {stem}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Estimate SNN vs ANN energy on the test set, selected via config.'
    )
    parser.add_argument('--config_file', type=str, required=True,
                        help='config YAML (full path or a name under nested_configs/)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='weights are loaded from s_{seed}_{model_name}.pth')
    parser.add_argument('--weight_dir', type=str, required=True,
                        help='folder of s_{seed}_{model_name}.pth weights')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='override the data location; expects data.hdf5 and data.csv inside')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--use_best_model', action='store_true',
                        help="use the config's xbest hyperparameters instead of x0")
    parser.add_argument('--plot', action='store_true',
                        help='also plot one neuron (signal vs spikes) and mean energy over seeds')
    parser.add_argument('--plot_dir', type=str, default='energy_plots',
                        help='where --plot writes figures (default: energy_plots/)')
    parser.add_argument('--neuron_idx', type=int, default=None,
                        help='which neuron to draw (default: the most active one per layer)')
    parser.add_argument('--t_start', type=int, default=150,
                        help='first timestep to draw (default: 0)')
    parser.add_argument('--t_end', type=int, default=275,
                        help='last timestep to draw (default: end of the trace)')

    evaluate_energy(parser.parse_args())
