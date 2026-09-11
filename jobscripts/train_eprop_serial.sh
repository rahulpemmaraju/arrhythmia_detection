#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export CONFIG_FILE="$1"
export MODEL_NAME="$3"
export DATA_DIR="$4"
export OUTPUT_DIR="$5"
export LABEL="$2"
export JOB_NAME="$LABEL"_"$DATE"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /expanse/lustre/scratch/rpemmaraju/temp_project/logs/arr_detection/$JOB_NAME.%j.o
#SBATCH -e /expanse/lustre/scratch/rpemmaraju/temp_project/logs/arr_detection/$JOB_NAME.%j.e
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mem=249208M
#SBATCH --account=sua199
#SBATCH --export=ALL
#SBATCH --mail-user=rp933@rwjms.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH --constraint="lustre"
#SBATCH --no-requeue

source $HOME/cpu_hgrsnn.sh

cd $PROJECT/arrhythmia_detection

python main_eprop.py \
--config_file=$CONFIG_FILE \
--model_name=$MODEL_NAME \
--data_dir=$DATA_DIR \
--output_dir=$OUTPUT_DIR \
--num_seeds=5 --use_best_model

python main.py \
--config_file=$CONFIG_FILE \
--model_name=$MODEL_NAME \
--data_dir=$DATA_DIR \
--output_dir=$OUTPUT_DIR \
--num_seeds=5 --use_best_model

EOT