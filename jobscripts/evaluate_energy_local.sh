#!/bin/bash
# per-beat energy for the models in the paper comparison, into one results folder.
# run from the arrhythmia_detection code directory:
#
#   bash jobscripts/evaluate_energy_local.sh [DATA_DIR] [OUTPUT_DIR]
#
# each model writes its own files, so a failure in one does not stop the rest, and
# re-running replaces a model's rows instead of duplicating them.

DATA_DIR=${1:-../beats}
OUTPUT_DIR=${2:-../energy_evaluation}
SNN_WEIGHTS=../remote_results/model_weights
BASELINE_WEIGHTS=../baseline_results/model_weights

baseline () {
  python evaluate_energy.py --config_file "$1_config.yaml" --model_name "$1_baseline" \
    --weight_dir "$BASELINE_WEIGHTS" --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR"
}

# config -> weight name, as trained by train_serial.sh (which always passes --use_best_model)
snn () {
  python evaluate_energy.py --config_file "$1" --model_name "$2" \
    --weight_dir "$SNN_WEIGHTS" --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" --use_best_model
}

for b in conv1d convnet rnn gru lstm lstm_encoder; do
  baseline "$b"
done

# the hgrsnn_wo_td* weight names predate the rename of that model to hgrsnn
snn hgrsnn_large_config.yaml            hgrsnn_wo_td
snn rsnn_large_config.yaml              rsnn_no_stopping

# hgrsnn ablations
snn hgrsnn_wo_flush_large_config.yaml   hgrsnn_wo_td_wo_flush
snn hgrsnn_wo_share_large_config.yaml   hgrsnn_wo_td_wo_share

snn hgrsnn_eprop_large_config.yaml      hgrsnn_wo_td_eprop_large
