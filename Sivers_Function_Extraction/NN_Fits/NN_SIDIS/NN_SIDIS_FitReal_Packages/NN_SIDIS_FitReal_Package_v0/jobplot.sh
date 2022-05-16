#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=result_%a.out
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -A ishara

module load anaconda cuda/11.4.2 cudnn/8.2.4.15
module load gcc/9.2.0 lhapdf/6.3.0
source activate lhapdf-tf
export PYTHONPATH=~/.conda/envs/lhapdf-tf/lib/python3.8/site-packages:$PYTHONPATH

python NN_SIDIS_Plots.py