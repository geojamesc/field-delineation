#!/bin/sh
#$ -N fd_train
#$ -cwd
#$ -l h_rt=09:50:00 # global
# Request 4 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=8G 

# Request one GPU: 
#$ -pe gpu 4


# Initialise the environment modules and load CUDA + anaconda + venv
. /etc/profile.d/modules.sh

module load anaconda/5.3.1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/applications/gridengine/ge-8.6.5/bin/glnxa64:/exports/applications/gridengine/ge-8.6.5/runtime/glnxa64:/exports/applications/apps/SL7/cuda/11.0.2/cuda_cudart/lib64/:/exports/applications/apps/SL7/cuda/11.0.2/:/exports/applications/apps/SL7/cuda/11.0.2/lib64:/exports/applications/apps/SL7/cuda/11.0.2/libcublas/targets/x86_64-linux/lib/:/exports/cmvm/eddie/eb/groups/GAAFS/PythonEnvs/fbound/lib
export CUDA_HOME=/exports/applications/apps/SL7/cuda/11.0.2/
module load cuda/11.0.2
module unload cuda/11.0.2

source activate fbound

python field-delineation/field_delineation_end2end.py #CUDA_VISIBLE_DEVICES=1 
