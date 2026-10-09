# Reproducing the paper

Commands for every table and figure. They are run from the repository root. `DATA` is the processed beat folder from [the README](README.md#data) and `OUT` is any output folder.

Everything runs on CPU with one thread per process. The scripts in `jobscripts/` run the same commands on a SLURM cluster, where five seeds of one model are given 24 hours.

## Models and configs

| Paper | Config (`nested_configs/`) | Code |
| --- | --- | --- |
| RSNN | `rsnn_large_config.yaml` | `models/vanilla_snn.py` |
| HG-RSNN − F | `hgrsnn_wo_flush_large_config.yaml` | `models/hgrsnn_no_flush.py` |
| HG-RSNN − S | `hgrsnn_wo_share_large_config.yaml` | `models/hgrsnn_no_share.py` |
| HG-RSNN | `hgrsnn_large_config.yaml` | `models/hgrsnn.py` |
| HG-RSNN (e-prop) | `hgrsnn_eprop_large_config.yaml` | `models/eprop_hgrsnn.py` |

In the code the gate is called the boundary (`b`), and the `four_layer_*` model types are the three spiking layers plus the linear readout. Each config holds the search space (`bounds`) and the optimized hyperparameters (`xbest`), which `--use_best_model` selects. The `_multiclass` configs (N/S/V/F) are not in the paper.

## Data and splits

`create_beat_dataset.py` drops the paced records, downsamples lead MLII to 180 Hz, cleans it with NeuroKit2, and cuts a 270-sample window per beat (1 s before to 0.5 s after the R peak). Only the last 180 samples are scored. The test set is de Chazal's DS2 (10,616 beats), and DS1 is split 75:25 into training (7,920) and validation (2,640). The split is fixed; the seeds change only initialization and batch order. `python dataloader.py $DATA` prints the split sizes.

## Table 1: spiking models

```
for pair in rsnn_large:rsnn hgrsnn_wo_flush_large:hgrsnn_wo_flush \
            hgrsnn_wo_share_large:hgrsnn_wo_share hgrsnn_large:hgrsnn; do
  config=${pair%%:*}; name=${pair##*:}
  python main.py --config_file=${config}_config.yaml --model_name=$name \
    --data_dir=$DATA --output_dir=$OUT --num_seeds=5 --train --use_best_model
done
```

E-prop trains with `main_eprop.py` and is evaluated with `main.py`:

```
python main_eprop.py --config_file=hgrsnn_eprop_large_config.yaml --model_name=hgrsnn_eprop \
  --data_dir=$DATA --output_dir=$OUT --num_seeds=5 --use_best_model
python main.py --config_file=hgrsnn_eprop_large_config.yaml --model_name=hgrsnn_eprop \
  --data_dir=$DATA --output_dir=$OUT --num_seeds=5 --use_best_model
```

Models train for 20 epochs with SGD (batch size 64) and the final epoch is evaluated. Each seed's threshold is the lowest false positive rate on validation with sensitivity ≥ 0.90. Results go to `$OUT/model_results/metrics/{model_name}_fixed_sensitivity.csv`. Leave out `--train` (and `main_eprop.py`) to evaluate existing weights, passing `--weight_dir` if they are not in `$OUT/model_weights`.

`python test_eprop_streaming_equivalence.py` and `python -m models.eprop_hgrsnn` check the e-prop gradients.

## Table 2: baselines, parameters and energy

```
for b in conv1d convnet rnn gru lstm; do
  python -m baselines.train --config_file=${b}_config.yaml --model_name=${b}_baseline \
    --data_dir=$DATA --output_dir=$OUT/baselines --num_seeds=5 --train
done
```

The baselines use the same split, readout and threshold rule, train for 20 epochs with SGD (learning rate 0.01), and keep the epoch with the lowest validation loss.

`evaluate_energy.py` counts operations on the test set: an accumulate (0.9 pJ) per input spike for spiking inputs, a multiply-accumulate (4.6 pJ) per synapse for analog inputs, and the state updates. Closed gates skip their operations. Run it once per model with the same output folder:

```
python evaluate_energy.py --config_file=hgrsnn_large_config.yaml --model_name=hgrsnn \
  --weight_dir=$OUT/model_weights --data_dir=$DATA --output_dir=$OUT/energy --use_best_model
python evaluate_energy.py --config_file=conv1d_config.yaml --model_name=conv1d_baseline \
  --weight_dir=$OUT/baselines/model_weights --data_dir=$DATA --output_dir=$OUT/energy
```

In `$OUT/energy/summary.csv`, Table 2's parameters are `n_params_used` and its energy is `total_energy_pj` / 1000. `python plot_energy_scatter.py --results_dir=$OUT/energy` draws the figure in the README.

## Training memory

```
python measure_training_memory.py --batch_size 64 --output_dir=$OUT/memory
```

The paper reports `graph_peak_mb` at sequence length 270. The script reads the raw records from `../mit-bih-arrhythmia-database-1.0.0` unless `--mitdb_dir` is given.

## Figure 2 and gate analysis

`visualize_hsnn.ipynb` draws Figure 2 from the seed-2 HG-RSNN, in the cell under "Normal vs arrhythmic beats, correct and incorrect". Set the dataset, weight and MIT-BIH paths at the top first. The rest of the notebook is not in the paper.

`gate_auc.py` scores each test beat by the fraction of time the g1 gate is open and compares that AUROC with the network's:

```
python gate_auc.py --model_name=hgrsnn --weight_dir=$OUT/model_weights --data_dir=$DATA --use_best_model
```

## Hyperparameter search

`xbest` was found with [nested](https://github.com/neurosutras/nested) (population annealing over the per-layer decay and threshold and the learning rate, minimizing validation loss and maximizing validation AUROC). Set the data paths under `kwargs.dataset_configs` in the config, then run:

```
python -m nested.optimize --config-file-path=nested_configs/hgrsnn_large_config.yaml \
  --output-dir=data --pop_size=200 --max_iter=50 --path_length=3 --framework=serial --disp
```

The paper's runs used MPI on a cluster (`jobscripts/optimize_expanse.sh`). `analyze_nested_results.ipynb` plots an optimization history.

## Results

Re-evaluated from the paper's trained weights with this code and `requirements.txt`. Mean ± standard deviation over five seeds.

| Model | AUROC | Sensitivity | Specificity |
| --- | --- | --- | --- |
| RSNN | 0.878 ± 0.024 | 0.928 ± 0.032 | 0.583 ± 0.055 |
| HG-RSNN − F | 0.785 ± 0.170 | 0.728 ± 0.408 | 0.643 ± 0.212 |
| HG-RSNN − S | 0.913 ± 0.005 | 0.935 ± 0.009 | 0.622 ± 0.022 |
| HG-RSNN | 0.914 ± 0.003 | 0.932 ± 0.005 | 0.650 ± 0.024 |
| HG-RSNN (e-prop) | 0.874 ± 0.019 | 0.916 ± 0.033 | 0.587 ± 0.067 |

In HG-RSNN − F seed 4 the bottom gate never opens, so the upper layers never spike and that seed scores AUROC 0.5.

- **Training memory** (batch size 64, 270 timesteps): BPTT 38.4 MB vs. e-prop 2.8 MB (13.5×). BPTT grows linearly with sequence length; e-prop stays at 2.8–3.0 MB up to 1,080 timesteps.
- **Gate analysis:** the g1 open fraction scores AUROC 0.902 ± 0.004 vs. the network's 0.914 ± 0.003 (seed 2: 0.898 vs. 0.911).

The paper's weights were saved under the model names used before HG-RSNN was renamed. Pass these as `--model_name` to evaluate them: `rsnn_no_stopping`, `hgrsnn_wo_td_wo_flush`, `hgrsnn_wo_td_wo_share`, `hgrsnn_wo_td` and `hgrsnn_wo_td_eprop_large`. The baselines use `{conv1d,convnet,rnn,gru,lstm}_baseline`. `jobscripts/evaluate_energy_local.sh` uses these names.
