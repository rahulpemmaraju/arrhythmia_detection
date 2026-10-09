'''
How well gate activity alone separates normal from arrhythmic beats (Section 3.3).

For every test beat, the fraction of the labeled beat span during which each gate
(g0, g1) is open is used as a score, and its AUROC is compared with the AUROC of
the network's own output. Evaluation mirrors main.py: the same split, per-record
normalization and beat-span readout.

Example (from this directory)
-------
    python gate_auc.py --config_file=hgrsnn_large_config.yaml --model_name=hgrsnn \
        --weight_dir={output path}/model_weights --data_dir={path to processed data} \
        --use_best_model
'''

import os
import argparse

import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from dataloader import train_test_val_split, get_dataset_statistics, get_beat_bounds
from models import create_model
from utils.encoding import Current_Encoder, unpack_batch

torch.set_num_threads(1)


def gate_scores(model, test_loader, encoder, beat_bounds):
    # returns (labels, network scores, g0 open fraction, g1 open fraction), one entry per beat
    start, end = beat_bounds
    y_true, score, g0_open, g1_open = [], [], [], []

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            data, target, stats = unpack_batch(batch)
            data = encoder.encode(data, *stats)

            output, _, (g0, g1) = model(data, logging=True)

            y_true.append(target[:, 0].numpy())
            score.append(output[:, start:end].mean(dim=(1, 2)).numpy())
            g0_open.append(g0[:, start:end].mean(dim=(1, 2)).numpy())
            g1_open.append(g1[:, start:end].mean(dim=(1, 2)).numpy())

    return tuple(np.concatenate(x) for x in (y_true, score, g0_open, g1_open))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='hgrsnn_large_config.yaml') # file in nested_configs
    parser.add_argument('--model_name', type=str) # weights are read from s_{seed}_{model_name}.pth
    parser.add_argument('--weight_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--use_best_model', action='store_true')
    parser.add_argument('--output_csv', type=str, default=None) # optional: save the per-seed table

    args = parser.parse_args()

    config_key = 'xbest' if args.use_best_model else 'x0'

    with open(f'nested_configs/{args.config_file}', 'r') as file:
        model_configs = yaml.safe_load(file)

    model = create_model(model_configs, config_key)

    rows = []

    for seed in range(args.num_seeds):
        model_path = os.path.join(args.weight_dir, f's_{seed}_{args.model_name}.pth')

        if not os.path.exists(model_path):
            continue

        train_dataset, _, test_dataset = train_test_val_split(
            data_file=f'{args.data_dir}/data.hdf5',
            metadata=f'{args.data_dir}/data.csv',
            train_prop=0.6,
            val_prop=0.2,
            test_prop=0.2,
            binary=True,
            random_state=seed,
            balance=True,
            return_stats=model_configs['kwargs'].get('patient_norm', False),
        )

        beat_bounds = get_beat_bounds(train_dataset)
        encoder = Current_Encoder(*get_dataset_statistics(train_dataset))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])

        y_true, score, g0_open, g1_open = gate_scores(model, test_loader, encoder, beat_bounds)

        rows.append({
            'seed': seed,
            'network_auc': roc_auc_score(y_true, score),
            'g0_open_auc': roc_auc_score(y_true, g0_open),
            'g1_open_auc': roc_auc_score(y_true, g1_open),
            'g0_open_frac': g0_open.mean(),
            'g1_open_frac': g1_open.mean(),
        })

    results = pd.DataFrame(rows).set_index('seed')
    results.loc['mean'] = results.mean()
    results.loc['std'] = results.iloc[:-1].std()

    print(results.round(4).to_string())

    if args.output_csv is not None:
        results.to_csv(args.output_csv)
