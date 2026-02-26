#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME="$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch2/11354/rahulpemmaraju/arr_files/logs/$JOB_NAME.%j.o
#SBATCH -e /scratch2/11354/rahulpemmaraju/arr_files/logs/$JOB_NAME.%j.e
#SBATCH -p development
#SBATCH -N 18
#SBATCH -n 1008
#SBATCH -t 2:00:00
#SBATCH --mail-user=rp933@rwjms.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $WORK/arrhythmia_detection

export MPI4PY_RC_RECV_MPROBE=false

ibrun -n 1001 python -m mpi4py.futures -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=$SCRATCH/arr_files/data --pop_size=100 --max_iter=2 --path_length=1 --disp \
  --framework=mpi
EOT

