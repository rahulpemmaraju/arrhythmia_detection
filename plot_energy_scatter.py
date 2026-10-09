"""
Energy vs AUC bubble chart from the energy_evaluation results.

x is per-beat inference energy (log scale), y is test AUC, bubble area is the number
of parameters the model uses at inference, and color is the model family.

    python plot_energy_scatter.py                        # core comparison
    python plot_energy_scatter.py --include_ablations    # + the HG-RSNN ablations

Each run writes four PNGs: a light theme (white background) for documents and a dark
theme (black background) for slides, each opaque and transparent:

    {stem}_light.png   {stem}_light_transparent.png
    {stem}_dark.png    {stem}_dark_transparent.png

Energy defaults to total (synaptic + state), gated -- see models/energy.py.
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(CODE_DIR), 'energy_evaluation')

# model_name -> (display name, family). a model missing from summary.csv is skipped.
# each SNN is listed under the model name used in REPRODUCE.md and under the name of
# the paper's checkpoints (from before the model was renamed hgrsnn)
CORE_MODELS = {
    'hgrsnn': ('HG-RSNN', 'spiking'),
    'hgrsnn_wo_td': ('HG-RSNN', 'spiking'),
    'rsnn': ('RSNN', 'spiking'),
    'rsnn_no_stopping': ('RSNN', 'spiking'),
    'conv1d_baseline': ('Conv1D', 'conv'),
    'convnet_baseline': ('ConvNet', 'conv'),
    'rnn_baseline': ('RNN', 'recurrent'),
    'gru_baseline': ('GRU', 'recurrent'),
    'lstm_baseline': ('LSTM', 'recurrent'),
}

ABLATION_MODELS = {
    'hgrsnn_wo_flush': ('HG-RSNN − F', 'spiking'),
    'hgrsnn_wo_td_wo_flush': ('HG-RSNN − F', 'spiking'),
    'hgrsnn_wo_share': ('HG-RSNN − S', 'spiking'),
    'hgrsnn_wo_td_wo_share': ('HG-RSNN − S', 'spiking'),
    'hgrsnn_eprop': ('HG-RSNN (e-prop)', 'spiking'),
    'hgrsnn_wo_td_eprop_large': ('HG-RSNN (e-prop)', 'spiking'),
}

FAMILY_LABELS = {'spiking': 'Spiking', 'conv': 'Convolutional', 'recurrent': 'Recurrent'}

# reference palette slots 1-3, validated all-pairs in each mode against its own
# background (a scatter puts any two families side by side, which caps it at three
# categorical colors). the dark steps are the palette's dark column, not a lightened
# copy of the light ones
THEMES = {
    'light': {
        'families': {'spiking': '#2a78d6', 'conv': '#eb6834', 'recurrent': '#1baf7a'},
        'ink': '#0b0b0b',
        'ink_secondary': '#52514e',
        'ink_muted': '#898781',
        'grid': '#e1e0d9',
        'axis': '#c3c2b7',
        'surface': '#ffffff',
    },
    'dark': {
        'families': {'spiking': '#3987e5', 'conv': '#d95926', 'recurrent': '#199e70'},
        'ink': '#ffffff',
        'ink_secondary': '#c3c2b7',
        'ink_muted': '#898781',
        'grid': '#2c2c2a',
        'axis': '#383835',
        'surface': '#000000',
    },
}

# every text size on the figure, in points. the label offsets below are tuned for
# these, so a large change here may need them re-tuned too
FONT = {
    'axis_label': 20,
    'tick': 17,
    'model_label': 16,
    'legend': 16,
    'legend_title': 16,
    'note': 14,
}

# label placement per model, as (dx, dy) in points from the bubble centre, tuned
# against the rendered figure. a label pushed further than LEADER_MIN_PT from its
# bubble gets a leader line back to it. a model without an entry is labelled right
LABEL_OFFSETS = {
    'hgrsnn_wo_td': (0, 26),
    'hgrsnn_wo_td_wo_share': (12, -44),
    'hgrsnn_wo_td_wo_flush': (-22, -40),
    'hgrsnn_wo_td_eprop_large': (0, -38),
    'rsnn_no_stopping': (20, 0),
    'conv1d_baseline': (0, 30),
    'convnet_baseline': (0, 29),
    'rnn_baseline': (0, 25),
    'gru_baseline': (-24, 0),
    'lstm_baseline': (26, 0),
    'lstm_encoder_baseline': (-26, 0),
}
LEADER_MIN_PT = 32

# square points of bubble area per parameter
AREA_PER_PARAM = 1 / 40


def load_models(results_dir, include_ablations, energy_col):
    summary = pd.read_csv(os.path.join(results_dir, 'summary.csv'))
    wanted = dict(CORE_MODELS, **(ABLATION_MODELS if include_ablations else {}))

    rows = summary[summary.model_name.isin(wanted)].copy()
    missing = sorted({label for label, _ in wanted.values()}
                     - {wanted[m][0] for m in rows.model_name})
    if missing:
        print(f'not in summary.csv, skipped: {missing}')

    if 'n_params_used' in rows and rows.n_params_used.notna().all():
        rows['params'] = rows.n_params_used
    else:
        # built parameters overstate models that construct weights they never call
        print('warning: n_params_used missing -- re-run evaluate_energy.py; '
              'sizing bubbles by n_params instead')
        rows['params'] = rows.n_params

    rows['label'] = rows.model_name.map(lambda m: wanted[m][0])
    rows['group'] = rows.model_name.map(lambda m: wanted[m][1])
    rows['energy_uj'] = rows[f'{energy_col}_mean'] / 1e6
    rows['energy_uj_std'] = rows[f'{energy_col}_std'].fillna(0) / 1e6
    rows['auc'] = rows.test_auc_mean
    rows['auc_std'] = rows.test_auc_std.fillna(0)
    rows['area'] = rows.params * AREA_PER_PARAM
    rows['radius_pt'] = np.sqrt(rows.area / np.pi)

    # draw in the declared order so overlaps stack the same way on every render
    order = {m: i for i, m in enumerate(wanted)}
    return rows.sort_values('model_name', key=lambda s: s.map(order))


def format_uj(value, _):
    return f'{value:g}'


def plot(rows, save_stem, theme='light', dpi=600):
    t = THEMES[theme]
    families = t['families']

    with plt.rc_context({
        'font.family': 'sans-serif',
        'font.size': FONT['tick'],
        'axes.labelsize': FONT['axis_label'],
        'xtick.labelsize': FONT['tick'],
        'ytick.labelsize': FONT['tick'],
        'legend.fontsize': FONT['legend'],
        'text.color': t['ink'],
        'axes.labelcolor': t['ink'],
        'xtick.color': t['ink_secondary'],
        'ytick.color': t['ink_secondary'],
    }):
        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
        # the opaque export paints this; the transparent one drops it
        fig.patch.set_facecolor(t['surface'])
        ax.set_facecolor(t['surface'])

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_color(t['axis'])
            ax.spines[spine].set_linewidth(1)
        ax.grid(True, which='major', color=t['grid'], linewidth=0.8, linestyle='-', zorder=0)
        ax.set_axisbelow(True)

        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(FuncFormatter(format_uj))
        ax.xaxis.set_minor_formatter(NullFormatter())

        # extra room on the right so a label beside the rightmost bubble stays inside
        hi = rows.energy_uj.max() * 3.5
        ax.set_xlim(rows.energy_uj.min() / 2.5, hi)
        y_lo = max(0.0, (rows.auc - rows.auc_std).min() - 0.04)
        ax.set_ylim(y_lo, 1.0)

        # chance is a real threshold, so it is the one dashed rule on the chart
        if y_lo < 0.5:
            ax.axhline(0.5, color=t['ink_muted'], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate('chance', (hi, 0.5), xytext=(-4, 4), textcoords='offset points',
                        ha='right', va='bottom', color=t['ink_muted'], fontsize=FONT['note'])

        for _, r in rows.iterrows():
            # the ring is the background color, so overlapping bubbles stay separable
            ax.scatter(r.energy_uj, r.auc, s=r.area, color=families[r.group],
                       edgecolors=t['surface'], linewidths=1.5, zorder=3)

            dx, dy = LABEL_OFFSETS.get(r.model_name, (r.radius_pt + 8, 0))
            leader = None
            if np.hypot(dx, dy) > LEADER_MIN_PT:
                leader = dict(arrowstyle='-', color=t['ink_muted'], linewidth=0.8,
                              shrinkA=2, shrinkB=r.radius_pt + 2)
            ax.annotate(r.label, (r.energy_uj, r.auc), xytext=(dx, dy), textcoords='offset points',
                        ha='center' if dx == 0 else ('left' if dx > 0 else 'right'),
                        va='center', color=t['ink'], fontsize=FONT['model_label'],
                        arrowprops=leader, zorder=4)

        ax.set_xlabel('Inference energy per beat (µJ, log scale)')
        ax.set_ylabel('Test AUROC')

        family_handles = [Line2D([], [], marker='o', linestyle='', markersize=12,
                                 markerfacecolor=families[g], markeredgecolor=t['surface'],
                                 label=FAMILY_LABELS[g])
                          for g in FAMILY_LABELS if g in set(rows.group)]
        family_legend = ax.legend(handles=family_handles, title='Model family', loc='lower left',
                                  frameon=False, title_fontsize=FONT['legend_title'],
                                  handletextpad=0.4, labelcolor=t['ink'])
        family_legend.get_title().set_color(t['ink'])
        ax.add_artist(family_legend)

        # drawn as markers sized to the same area as the bubbles (markersize is a
        # diameter in points), so each label centres on its circle. placed in the empty
        # band above the family legend, clear of every bubble along the bottom
        size_values = [10_000, 25_000, 40_000]
        size_handles = [Line2D([], [], marker='o', linestyle='',
                               markersize=2 * np.sqrt(v * AREA_PER_PARAM / np.pi),
                               markerfacecolor=t['ink_muted'], markeredgecolor=t['surface'],
                               markeredgewidth=1.5, alpha=0.45, label=f'{v // 1000}k')
                        for v in size_values]
        size_legend = ax.legend(handles=size_handles, title='Parameters', loc='center left',
                                bbox_to_anchor=(0.0, 0.52), ncol=len(size_values), frameon=False,
                                title_fontsize=FONT['legend_title'], handlelength=2.4,
                                handleheight=2.4, columnspacing=1.2, handletextpad=0.5,
                                labelcolor=t['ink'])
        size_legend.get_title().set_color(t['ink'])

        fig.tight_layout()
        os.makedirs(os.path.dirname(save_stem), exist_ok=True)
        written = []
        for transparent in (False, True):
            path = f"{save_stem}_{theme}{'_transparent' if transparent else ''}.png"
            fig.savefig(path, dpi=dpi, transparent=transparent, bbox_inches='tight',
                        facecolor='none' if transparent else t['surface'])
            written.append(path)
        plt.close(fig)
        return written


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Energy vs AUC bubble chart.')
    parser.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR,
                        help='energy_evaluation folder holding summary.csv')
    parser.add_argument('--include_ablations', action='store_true',
                        help='also plot the HG-RSNN ablations present in summary.csv')
    parser.add_argument('--energy', default='total_energy_pj',
                        choices=['total_energy_pj', 'synaptic_energy_pj',
                                 'ungated_total_energy_pj', 'ungated_synaptic_energy_pj'],
                        help='which energy column goes on x (default: total, gated)')
    parser.add_argument('--theme', default='both', choices=['light', 'dark', 'both'],
                        help='light (white background), dark (black background), or both')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--out', default=None,
                        help='output path stem, without theme suffix or extension '
                             '(default: {results_dir}/plots/energy_vs_auc[_ablations])')
    args = parser.parse_args()

    rows = load_models(args.results_dir, args.include_ablations, args.energy)

    stem = args.out or os.path.join(args.results_dir, 'plots',
                                    'energy_vs_auc' + ('_ablations' if args.include_ablations else ''))
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]

    written = []
    for theme in themes:
        written += plot(rows, stem, theme=theme, dpi=args.dpi)

    print(rows[['label', 'group', 'params', 'energy_uj', 'energy_uj_std', 'auc', 'auc_std']]
          .to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('written:\n  ' + '\n  '.join(written))
