#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME="$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /expanse/lustre/scratch/rpemmaraju/temp_project/logs/arr_detection/$JOB_NAME.%j.o
#SBATCH -e /expanse/lustre/scratch/rpemmaraju/temp_project/logs/arr_detection/$JOB_NAME.%j.e
#SBATCH -p compute
#SBATCH -N 8
#SBATCH --ntasks-per-node=128
#SBATCH -t 48:00:00
#SBATCH --mem=249208M
#SBATCH --export=ALL
#SBATCH --account=sua199
#SBATCH --mail-user=rp933@rwjms.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH --constraint="lustre"
#SBATCH --no-requeue


source $HOME/cpu_hgrsnn.sh
cd $PROJECT/arrhythmia_detection

srun -n 1001 --mpi=pmi2 python -m mpi4py.futures -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=$SCRATCH/data/arr_detection --pop_size=100 --max_iter=2 --path_length=1 --disp \
  --framework=mpi
EOT
