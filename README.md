## Hierarchically Gated Spiking Neural Networks for Interpretable Arrhythmia Detection

This repository contains the code for:

> R. Pemmaraju, A. Voleti, A. D. Milstein. Hierarchically Gated Spiking Neural Networks for Interpretable Arrhythmia Detection. *Computing in Cardiology*, 2026. (link to come)

It trains the hierarchically gated recurrent spiking neural network (HG-RSNN) to detect arrhythmic beats in the MIT-BIH Arrhythmia Database, and reproduces every result in the paper:

| Result | Content | Section |
| --- | --- | --- |
| Table 1 | test AUROC, sensitivity and specificity of RSNN, HG-RSNN − F, HG-RSNN − S, HG-RSNN and HG-RSNN (e-prop) | [3](#3-classification-performance-table-1), [4](#4-training-with-e-prop-table-1-and-training-memory) |
| Section 3.2 | training memory of BPTT vs. e-prop (38.4 MB vs. 2.8 MB) | [4](#4-training-with-e-prop-table-1-and-training-memory) |
| Figure 2, Section 3.3 | gate activity on example beats; gate-only AUROC (0.90 vs. 0.91) | [7](#7-gate-interpretability-figure-2) |
| Table 2 | non-spiking baselines: AUROC, parameters and inference energy | [5](#5-non-spiking-baselines-table-2), [6](#6-energy-estimation-table-2) |

The expected numbers are listed in [Results](#results).

The repository also supports a four-class version of the task (N/S/V/F) and two extra baselines. Neither is reported in the paper.

All commands are run from the repository root.

## Model

The HG-RSNN has three layers of leaky integrate-and-fire neurons (widths 32, 64, 32) and a linear readout. The bottom and middle layers each compute a binary gate g from their spikes. When layer l's gate is on:

- **Sharing:** layer l+1 receives layer l's spikes. While the gate is off, layer l+1 gets no input from below.
- **Flushing:** on the next timestep, layer l's membrane potential and recurrent input are cleared.

The bottom layer always receives the ECG, and the top layer is never flushed:

```
v_rec_l[t] = α v_l[t-1] + z_l[t-1] Wᵀ
v_0[t]     = (1 - g_0[t-1]) v_rec_0[t] + x[t] Bᵀ               - v_th z_0[t-1]
v_1[t]     = (1 - g_1[t-1]) v_rec_1[t] + g_0[t] z_0[t] Bᵀ      - v_th z_1[t-1]
v_2[t]     =                v_rec_2[t] + g_1[t] z_1[t] Bᵀ      - v_th z_2[t-1]
z_l[t]     = H(v_l[t] - v_th),    g_l[t] = H(σ(z_l[t] G) - 0.5)
```

The diagonal of each W is masked. Spikes and gates are trained with an arctan surrogate gradient. The readout is applied to the top layer's spikes at every timestep, and a beat's score is the mean readout over the labeled beat span.

The ablations each remove one mechanism. **HG-RSNN − F** never flushes. **HG-RSNN − S** always shares. **RSNN** is the same three-layer recurrent SNN with no gates.

In the code, the gate is called the boundary (`b`), and the `four_layer_*` model types are the three spiking layers plus the readout:

| Paper | `model_type` | Config | Code |
| --- | --- | --- | --- |
| RSNN | `four_layer_snn` | `rsnn_large_config.yaml` | `models/vanilla_snn.py` |
| HG-RSNN − F | `four_layer_hgrsnn_no_flush` | `hgrsnn_wo_flush_large_config.yaml` | `models/hgrsnn_no_flush.py` |
| HG-RSNN − S | `four_layer_hgrsnn_no_share` | `hgrsnn_wo_share_large_config.yaml` | `models/hgrsnn_no_share.py` |
| HG-RSNN | `four_layer_hgrsnn` | `hgrsnn_large_config.yaml` | `models/hgrsnn.py` |
| HG-RSNN (e-prop) | `four_layer_hgrsnn_eprop` | `hgrsnn_eprop_large_config.yaml` | `models/eprop_hgrsnn.py` |

The BPTT HG-RSNN cells also build a top-down projection (`t_v`) that is never used. It is kept so that trained checkpoints load unchanged, and it is not counted in the paper's parameter counts.

## Repository layout

```
create_beat_dataset.py        MIT-BIH records -> windowed, labeled beats (hdf5 + csv)
dataloader.py                 dataset class and train/val/test split
models/
  hgrsnn.py                   HG-RSNN
  hgrsnn_no_flush.py          HG-RSNN - F
  hgrsnn_no_share.py          HG-RSNN - S
  eprop_hgrsnn.py             HG-RSNN trained with e-prop
  vanilla_snn.py              RSNN
  hebbian_linear.py           linear layer used for the SNN weights
  energy.py                   operation counting for energy estimates
  utils.py                    create_model: config -> model
utils/                        input encoding, readout, metrics and training logs
nested_configs/               one config per model (architecture, search space, hyperparameters)
main.py                       train + evaluate an SNN over several seeds
main_eprop.py                 train the e-prop model (evaluate with main.py)
eprop_online.py               per-timestep e-prop training step
measure_training_memory.py    training-step memory, BPTT vs. e-prop, as sequence length grows
nested_train.py               source module for hyperparameter optimization with nested
nested_train_eprop.py         the same, for the e-prop model
baselines/                    non-spiking baselines (models, configs, train/evaluate)
evaluate_energy.py            per-beat inference energy and parameter counts for trained SNNs and baselines
gate_auc.py                   how well gate activity alone separates normal from arrhythmic beats
visualize_hsnn.ipynb          Figure 2 and further gate and spike visualizations
analyze_nested_results.ipynb  optimization history and hyperparameter importance (not in the paper)
jobscripts/                   SLURM scripts used to run the above on a cluster
test_eprop_streaming_equivalence.py
```

## Installation

```
git clone https://github.com/rahulpemmaraju/arrhythmia_detection.git
cd arrhythmia_detection

conda create --name hgrsnn python=3.12.0
conda activate hgrsnn
pip install -r requirements.txt
```

The results below were reproduced with exactly these package versions.

Hyperparameter optimization uses [nested](https://github.com/neurosutras/nested). It has no installer. Clone it and put the folder that contains it on your `PYTHONPATH`:

```
git clone https://github.com/neurosutras/nested.git /path/to/libs/nested
export PYTHONPATH=/path/to/libs:$PYTHONPATH
```

Running nested across many processes (`--framework=mpi`) also needs `pip install mpi4py`.

Everything runs on CPU. The scripts set PyTorch to a single thread, so full training runs are slow. The cluster scripts in `jobscripts/` request 24 hours for five seeds of one model.

## 1. Data preparation

Download the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/):

```
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
```

Build the beat dataset:

```
python create_beat_dataset.py \
  --raw_data_folder=physionet.org/files/mitdb/1.0.0/ \
  --processed_data_folder={path to processed data} \
  --window_length=1 --pre_window=0.5 --post_window=0 --downsample_factor=2
```

The script does the following:

- **Records:** drops the four paced records (102, 104, 107, 217) and uses the first ECG channel (MLII).
- **Signal processing:** downsamples from 360 Hz to 180 Hz, then cleans the signal with NeuroKit2's default method (a 0.5 Hz high-pass Butterworth filter and a powerline filter).
- **Windows:** cuts one 270-sample window per beat, from 1 s before the R peak to 0.5 s after it. The model reads the whole window. The loss and the prediction use only the last 180 samples, which run from 0.5 s before the R peak to 0.5 s after (the shaded span in Figure 2).
- **Labels:** assigns each beat an AAMI class: N (normal), S (supraventricular), V (ventricular) or F (fusion). Beats outside these classes are dropped. The paper's binary task is normal vs. arrhythmic (S, V or F).
- **Normalization:** stores each record's mean and standard deviation. Every config z-scores each beat by its own record (`patient_norm: true`).

It writes `data.hdf5` (the windows), `data.csv` (one row per beat) and `window_config.json` into the processed data folder.

**Splits** (`dataloader.train_test_val_split`):

- **Balancing:** normal beats are subsampled, with a fixed seed, to match the number of arrhythmic beats.
- **Test set:** the DS2 records of the inter-patient split of de Chazal et al. (2004), 10,616 beats.
- **Training and validation:** the DS1 records, split 75:25 at the beat level with class stratification, giving 7,920 training and 2,640 validation beats.

The split is the same for every seed. The five seeds change only weight initialization and batch order. To print the split sizes:

```
python dataloader.py {path to processed data}
```

## 2. Hyperparameter optimization

Each file in `nested_configs/` holds a model's architecture and training settings (`kwargs`), a hyperparameter search space (`bounds`), default values (`x0`), and the optimized values (`xbest`). The optimized hyperparameters are:

- each layer's voltage decay and spike threshold (`alpha_i`, `threshold_i`)
- the learning rate (`lr`)

`xbest` in the binary configs holds the values used in the paper. `--use_best_model` in the sections below uses them, so this section is only needed to rerun the search.

Optimization uses nested's population annealing. Each candidate is trained on every seed and scored on two objectives: mean validation loss and mean validation AUROC. Before running, set `kwargs.dataset_configs.data_file` and `kwargs.dataset_configs.metadata` in the config to your processed data. The later sections take the data folder as a command-line argument instead.

Run it in a single process:

```
mkdir -p data
python -m nested.optimize --config-file-path=nested_configs/hgrsnn_large_config.yaml \
  --output-dir=data --pop_size=200 --max_iter=50 --path_length=3 --framework=serial --disp
```

Or over MPI, as for the paper (`jobscripts/optimize_expanse.sh`; `jobscripts/optimize.sh` is the TACC Frontera version):

```
srun -n 1001 --mpi=pmi2 python -m mpi4py.futures -m nested.optimize \
  --config-file-path=nested_configs/hgrsnn_large_config.yaml \
  --output-dir=data --pop_size=200 --max_iter=50 --path_length=3 --framework=mpi --disp
```

Each run writes two files to `--output-dir`:

- the optimization history, `*_nested_optimization_history_*.hdf5`
- a copy of the config with the best parameters in `xbest`

To use the result, copy that `xbest` block into the config in `nested_configs/`. To continue a finished run, add `--hot-start --history-file-path={history .hdf5}` (`jobscripts/optimize_hot_start_expanse.sh`).

`analyze_nested_results.ipynb` plots an optimization history and estimates hyperparameter importance. Set `file_path` in the notebook to your history file.

## 3. Classification performance (Table 1)

Train and evaluate one model over five seeds:

```
python main.py \
  --config_file=hgrsnn_large_config.yaml \
  --model_name=hgrsnn \
  --data_dir={path to processed data} \
  --output_dir={output path} \
  --num_seeds=5 --train --use_best_model
```

- `--config_file` is a file name in `nested_configs/`.
- `--use_best_model` uses the optimized hyperparameters (`xbest`). Without it, `x0` is used.
- Leave out `--train` to evaluate existing weights. They are read from `{output path}/model_weights` unless `--weight_dir` is given.

The four BPTT models in Table 1:

```
DATA={path to processed data}
OUT={output path}

for pair in rsnn_large:rsnn hgrsnn_wo_flush_large:hgrsnn_wo_flush \
            hgrsnn_wo_share_large:hgrsnn_wo_share hgrsnn_large:hgrsnn; do
  config=${pair%%:*}; name=${pair##*:}
  python main.py --config_file=${config}_config.yaml --model_name=$name \
    --data_dir=$DATA --output_dir=$OUT --num_seeds=5 --train --use_best_model
done
```

Each model trains for 20 epochs with SGD, batch size 64 and binary cross-entropy loss. The weights evaluated are those of the final epoch.

For each seed, the decision threshold is chosen on the validation set: the lowest false positive rate among thresholds with sensitivity ≥ 0.90. The test set is scored at that threshold, and a beat's score is the mean output over its labeled span.

Output in `{output path}`:

```
model_weights/s_{seed}_{model_name}.pth
train_logs/s_{seed}_{model_name}/                          loss and AUROC curves
model_results/metrics/{model_name}_fixed_sensitivity.csv   AUROC, sensitivity, specificity, ...: one row per seed, then mean and std
model_results/plots/                                       per-seed ROC curves and the mean ROC across seeds
```

On a SLURM cluster, `jobscripts/train_serial.sh` runs the same command, and `jobscripts/evaluate_serial.sh` runs it without `--train`. Their arguments are `{config} {job label} {model name} {data dir} {output dir}`. First edit the account, log paths and environment setup at the top of the script.

### Multiclass (not in the paper)

Each BPTT config has a `_multiclass` counterpart (for example `hgrsnn_large_multiclass_config.yaml`) that classifies N/S/V/F with a four-unit readout and cross-entropy loss. Run it with the same command. The prediction is the argmax over classes, and the results go to `model_results/metrics/{model_name}_multiclass.csv`:

- accuracy
- macro and weighted one-vs-rest AUROC
- per-class precision, recall, F1 and specificity

The HG-RSNN − F and HG-RSNN − S multiclass configs were not optimized; their `xbest` is a copy of `x0`.

## 4. Training with e-prop (Table 1 and training memory)

`main_eprop.py` trains the HG-RSNN with e-prop. Each timestep's loss is backpropagated as soon as it is computed, so memory does not grow with the length of the input. The checkpoints use the same layout as `main.py`, which then evaluates them for the HG-RSNN (e-prop) row of Table 1:

```
python main_eprop.py --config_file=hgrsnn_eprop_large_config.yaml --model_name=hgrsnn_eprop \
  --data_dir={path to processed data} --output_dir={output path} --num_seeds=5 --use_best_model

python main.py --config_file=hgrsnn_eprop_large_config.yaml --model_name=hgrsnn_eprop \
  --data_dir={path to processed data} --output_dir={output path} --num_seeds=5 --use_best_model
```

`jobscripts/train_eprop_serial.sh` runs both steps and takes the same arguments as `train_serial.sh`.

Two checks of the e-prop implementation:

```
python test_eprop_streaming_equivalence.py   # per-timestep gradients match a single batched backward pass
python -m models.eprop_hgrsnn                # hand-written backward passes match autograd
```

**Training memory.** `measure_training_memory.py` runs one training step of each algorithm, BPTT (`main.train_step`) and e-prop (`eprop_online.online_train_step`), on a stretch of MIT-BIH record 100 at increasing sequence lengths. Each point runs in a fresh process. The paper's numbers are `graph_peak_mb`, the peak memory held for the backward pass, at batch size 64 and the 270-sample beat window:

```
python measure_training_memory.py --batch_size 64 --output_dir={memory output path}
```

It writes `training_memory.csv` (one row per algorithm, sequence length and repeat) and `training_memory.png`. `peak_rss_increase_mb` is the process's resident memory, which is noisier. It reads the raw records from `../mit-bih-arrhythmia-database-1.0.0` unless `--mitdb_dir` is given.

## 5. Non-spiking baselines (Table 2)

The baselines use the same data split, normalization, beat-span readout and threshold rule as the SNNs. They train for 20 epochs with SGD (learning rate 0.01, batch size 64) and binary cross-entropy loss, with no bias, normalization or dropout. Validation runs every two epochs, and the epoch with the lowest validation loss is kept. (The SNNs keep their final epoch.) Configs are in `baselines/configs/`:

| Paper | Config | Model |
| --- | --- | --- |
| Conv1D | `conv1d_config.yaml` | three dilated 1D convolutions (kernel 7, dilations 1, 4, 16, widths 32, 64, 32), one output per timestep; acausal |
| ConvNet | `convnet_config.yaml` | three convolutions (kernel 5) each followed by max pooling, global average pooling, one output per beat; acausal |
| RNN | `rnn_config.yaml` | three Elman RNN layers (tanh) |
| GRU | `gru_config.yaml` | three GRU layers |
| LSTM | `lstm_config.yaml` | three unidirectional LSTM layers |

`lstm_encoder_config.yaml` (a bidirectional LSTM encoder) and `convnet_mc_config.yaml` (a multiclass ConvNet) are not in the paper.

Train and evaluate the five paper baselines:

```
for b in conv1d convnet rnn gru lstm; do
  python -m baselines.train --config_file=${b}_config.yaml --model_name=${b}_baseline \
    --data_dir={path to processed data} --output_dir={baseline output path} --num_seeds=5 --train
done
```

Leave out `--train` to evaluate existing weights (`--weight_dir` works as in `main.py`). Metrics go to `{baseline output path}` in the same layout as `main.py`. The parameter and energy columns of Table 2 come from Section 6. On a cluster, `jobscripts/train_baselines.sh {job label} {data dir} {output dir}` trains the baselines in parallel.

## 6. Energy estimation (Table 2)

`evaluate_energy.py` runs trained weights over the test set and estimates inference energy per beat. Spiking and non-spiking models are counted with the same rules (`models/energy.py`):

- **Synaptic operations:** a projection whose input is a spike train costs one accumulate (AC, 0.9 pJ) per input spike per target neuron. A projection with analog input costs one multiply-accumulate (MAC, 4.6 pJ) per synapse. Analog inputs are the ECG itself and every layer of a non-spiking model. The per-operation energies are 45 nm CMOS estimates (Horowitz, 2014).
- **State updates:** each neuron or hidden-state update also costs MACs. For the SNNs that is the membrane leak; for the GRU and LSTM it is the gate products.
- **Gating:** in the HG-RSNN models, a bottom-up projection is counted only when the gate below is open (g_{l-1}[t] = 1), and the leak and recurrent input only when the layer is not being flushed (g_l[t-1] = 0).

The whole 270-sample window is billed, pre-beat context included. Run it for each spiking model and each baseline, with the same `--output_dir` every time:

```
DATA={path to processed data}
ENERGY={energy output path}

for pair in rsnn_large:rsnn hgrsnn_wo_flush_large:hgrsnn_wo_flush \
            hgrsnn_wo_share_large:hgrsnn_wo_share hgrsnn_large:hgrsnn hgrsnn_eprop_large:hgrsnn_eprop; do
  config=${pair%%:*}; name=${pair##*:}
  python evaluate_energy.py --config_file=${config}_config.yaml --model_name=$name \
    --weight_dir={output path}/model_weights --data_dir=$DATA --output_dir=$ENERGY --use_best_model
done

for b in conv1d convnet rnn gru lstm; do
  python evaluate_energy.py --config_file=${b}_config.yaml --model_name=${b}_baseline \
    --weight_dir={baseline output path}/model_weights --data_dir=$DATA --output_dir=$ENERGY
done
```

`{energy output path}` then contains:

- `all_seeds.csv`: one row per model and seed
- `summary.csv`: mean and standard deviation per model
- the per-layer counts

| Table 2 column | Column in `all_seeds.csv` / `summary.csv` |
| --- | --- |
| AUROC | `test_auc` |
| Parameters | `n_params_used` |
| Energy (nJ) | `total_energy_pj` / 1000 |

`n_params_used` counts only the weights that run at inference. `n_params` also includes the unused top-down weights of the BPTT HG-RSNN cells.

`--plot` additionally draws one neuron's voltage and spikes (not in the paper). `jobscripts/evaluate_energy_local.sh` runs the whole comparison using the weight names of the paper's runs (see [Results](#results)).

## 7. Gate interpretability (Figure 2)

`visualize_hsnn.ipynb` draws Figure 2 from the seed-2 HG-RSNN. The cell under "Normal vs arrhythmic beats, correct and incorrect" saves it to `../figures/gate_visualization_correct_incorrect.png`. Before running the notebook, set:

- the dataset paths in the "Configure dataset" cell
- the weight folder and weight name (`s_2_...`) in the cells that load models
- the path to the raw MIT-BIH records, which the notebook reads to recover each beat's original annotation symbol

`gate_auc.py` measures the claim in Section 3.3: the fraction of the labeled beat span during which g1 is open, used on its own as a score, separates normal from arrhythmic test beats nearly as well as the network output:

```
python gate_auc.py --config_file=hgrsnn_large_config.yaml --model_name=hgrsnn \
  --weight_dir={output path}/model_weights --data_dir={path to processed data} --use_best_model
```

It prints, per seed, the AUROC of the network output and of the g0 and g1 open fractions.

The rest of the notebook is not in the paper:

- correct and incorrect examples per beat class
- class-averaged gate and spike activity
- which neurons carry the decision

## Results

Mean ± standard deviation over five seeds, re-evaluated from the paper's trained weights with the code and package versions in this repository.

Table 1:

| Model | AUROC | Sensitivity | Specificity |
| --- | --- | --- | --- |
| RSNN | 0.878 ± 0.024 | 0.928 ± 0.032 | 0.583 ± 0.055 |
| HG-RSNN − F | 0.785 ± 0.170 | 0.728 ± 0.408 | 0.643 ± 0.212 |
| HG-RSNN − S | 0.913 ± 0.005 | 0.935 ± 0.009 | 0.622 ± 0.022 |
| HG-RSNN | 0.914 ± 0.003 | 0.932 ± 0.005 | 0.650 ± 0.024 |
| HG-RSNN (e-prop) | 0.874 ± 0.019 | 0.916 ± 0.033 | 0.587 ± 0.067 |

In one HG-RSNN − F seed (seed 4) the bottom layer's gate never opens, so the middle and top layers never spike and the seed scores AUROC 0.5. This causes that row's large spread.

Table 2:

| Model | AUROC | Parameters | Energy (nJ) |
| --- | --- | --- | --- |
| RNN | 0.351 ± 0.008 | 10,305 | 12,798 |
| GRU | 0.908 ± 0.012 | 30,849 | 38,790 |
| LSTM | 0.894 ± 0.009 | 41,121 | 51,548 |
| Conv1D | 0.952 ± 0.006 | 28,929 | 35,929 |
| ConvNet | 0.912 ± 0.002 | 20,673 | 9,714 |
| RSNN | 0.878 ± 0.024 | 10,304 | 1,363 ± 77 |
| HG-RSNN − F | 0.785 ± 0.170 | 10,400 | 1,002 ± 390 |
| HG-RSNN − S | 0.913 ± 0.005 | 10,400 | 275 ± 25 |
| HG-RSNN | 0.914 ± 0.003 | 10,400 | 216 ± 5 |
| HG-RSNN (e-prop) | 0.874 ± 0.019 | 10,400 | 285 ± 18 |

Training memory (`graph_peak_mb`, batch size 64, 270 timesteps): BPTT 38.4 MB, e-prop 2.8 MB (13.5×). From 90 to 1,080 timesteps, BPTT grows linearly from 12.8 MB to 153.5 MB, while e-prop stays between 2.8 and 3.0 MB.

Gate interpretability (Section 3.3): the g1 open fraction scores AUROC 0.902 ± 0.004 against the network's 0.914 ± 0.003 (seed 2, the Figure 2 model: 0.898 vs. 0.911).

The trained weights behind these numbers were saved under the model names used before the HG-RSNN was renamed. To evaluate them, pass these as `--model_name` with the configs above: `rsnn_no_stopping`, `hgrsnn_wo_td_wo_flush`, `hgrsnn_wo_td_wo_share`, `hgrsnn_wo_td` and `hgrsnn_wo_td_eprop_large` for the SNNs, and `{conv1d,convnet,rnn,gru,lstm}_baseline` for the baselines.
