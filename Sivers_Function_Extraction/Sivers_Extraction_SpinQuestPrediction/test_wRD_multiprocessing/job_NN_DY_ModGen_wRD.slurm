#!/bin/sh
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=10
#SBATCH --time=10:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=parallel
#SBATCH -A spinquest_standard
#SBATCH --array=0-20

module load anaconda cuda/11.4.2 cudnn/8.2.4.15
module load gcc/9.2.0 lhapdf/6.3.0
module load openmpi
source activate lhapdf-tf
export PYTHONPATH=~/.conda/envs/lhapdf-tf/lib/python3.8/site-packages:$PYTHONPATH


srun python3 NN_DY_ModGen_wRD.py $SLURM_ARRAY_TASK_ID