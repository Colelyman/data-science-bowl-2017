#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=64G  # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH -J "train"   # job name

# Compatibility variables for PBS. Delete if not needed.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

source activate dsb

module load cuda/7.5.18

echo 'Starting training'
THEANO_FLAGS=device=gpu0,floatX=float32 python ./code/model.py --num_epochs 10 --data_path /fslgroup/fslg_dsb2017/compute/data/stage1/ 
echo 'Finished trainig'

source deactivate dsb
