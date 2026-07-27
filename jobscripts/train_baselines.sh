#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$1"
export DATA_DIR="$2"
export OUTPUT_DIR="$3"
export JOB_NAME="$LABEL"_"$DATE"
export LOG_DIR=/expanse/lustre/scratch/rpemmaraju/temp_project/logs/arr_detection
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o $LOG_DIR/$JOB_NAME.%j.o
#SBATCH -e $LOG_DIR/$JOB_NAME.%j.e
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -c 1
#SBATCH -t 24:00:00
#SBATCH --mem=48G
#SBATCH --account=sua199
#SBATCH --export=ALL
#SBATCH --mail-user=rp933@rwjms.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH --constraint="lustre"
#SBATCH --no-requeue

source $HOME/cpu_hgrsnn.sh
cd $PROJECT/arrhythmia_detection || exit 1

# baselines/train.py sets torch.set_num_threads(1) and uses num_workers=0, so each
# rank is exactly one core -- all six pack onto a single shared-partition node.
# --exclusive keeps the steps from each grabbing the whole allocation, and the
# per-step --mem is required: without it step 1 reserves all 48G and the rest
# block until it finishes, silently serializing the job.

run_baseline () {
  srun --ntasks=1 --cpus-per-task=1 --mem=8G --exclusive \
    --output=$LOG_DIR/$JOB_NAME.\$1.%j.o --error=$LOG_DIR/$JOB_NAME.\$1.%j.e \
    python -m baselines.train --config_file=\$1_config.yaml --model_name=\$1_baseline \
    --data_dir=$DATA_DIR --output_dir=$OUTPUT_DIR --num_seeds=5 --train &
}

run_baseline conv1d
run_baseline convnet
run_baseline gru
run_baseline lstm
run_baseline lstm_encoder
run_baseline rnn

wait
EOT
