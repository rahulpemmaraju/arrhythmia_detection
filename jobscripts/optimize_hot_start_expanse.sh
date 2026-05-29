#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME="$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
export HISTORY_FILE_PATH="$3"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /expanse/lustre/scratch/rpemmaraju/temp_project/logs/arr_detection/$JOB_NAME.%j.o
#SBATCH -e /expanse/lustre/scratch/rpemmaraju/temp_project/logs/arr_detection/$JOB_NAME.%j.e
#SBATCH -p compute
#SBATCH -N 18
#SBATCH -n 1008
#SBATCH -t 48:00:00
#SBATCH --mem=249208M
#SBATCH --mail-user=rp933@rwjms.rutgers.edu
#SBATCH --mail-type=ALL

source $HOME/cpu_hgrsnn.sh
cd $PROJECT/arrhythmia_detection

export MPI4PY_RC_RECV_MPROBE=false

srun -n 1001 --mpi=pmi2 python -m mpi4py.futures -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=$SCRATCH/data/arr_detection --pop_size=200 --max_iter=50 --path_length=3 --disp \
  --framework=mpi --hot-start --history-file-path=$HISTORY_FILE_PATH
EOT
