#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 40
#SBATCH --time=12:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard
#SBATCH -A spinquest_standard

module load anaconda cuda/11.4.2 cudnn/8.2.4.15
module load gcc/9.2.0 lhapdf/6.3.0
source activate lhapdf-tf
export PYTHONPATH=~/.conda/envs/lhapdf-tf/lib/python3.8/site-packages:$PYTHONPATH

python3 Data_vs_PseudoData_Parm2_Plots.py
python3 MINUITSiversPlots_Parm2.py