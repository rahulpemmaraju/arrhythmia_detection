#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export CONFIG_FILE_PATH="$1"
export HISTORY_FILE="$2"
export MODEL_NAME="$3"
export JOB_NAME="$MODEL_NAME"_"$DATE"

sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch2/11354/rahulpemmaraju/arr_files/logs/$JOB_NAME.%j.o
#SBATCH -e /scratch2/11354/rahulpemmaraju/arr_files/logs/$JOB_NAME.%j.e
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -t 48:00:00
#SBATCH --mail-user=rp933@rwjms.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $WORK/arrhythmia_detection

export MPI4PY_RC_RECV_MPROBE=false

ibrun -n 6 python -m mpi4py.futures -m nested.analyze --framework=mpi --config-file-path=$CONFIG_FILE_PATH --output-dir=$SCRATCH/arr_files/model_weights --history-file-path=$HISTORY_FILE --model-key=best --model_name=$MODEL_NAME --export --disp

EOT

