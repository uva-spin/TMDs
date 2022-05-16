#!/bin/sh
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard
#SBATCH -A ishara
#SBATCH --array=0-5000

module load anaconda cuda/11.4.2 cudnn/8.2.4.15
module load gcc/9.2.0 lhapdf/6.3.0
source activate lhapdf-tf
export PYTHONPATH=~/.conda/envs/lhapdf-tf/lib/python3.8/site-packages:$PYTHONPATH

python3 NN_SIDIS_ModGen.py $SLURM_ARRAY_TASK_ID
python3 NN_SIDIS_Plots.py