#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=terminal_output_%a.out
#SBATCH -c 1
#SBATCH -t 16:30:00
#SBATCH -A spinquest

module load anaconda cuda/11.4.2 cuddnn/8.2.4.15
module load gcc/9.2.0 lhapdf/6.3.0
source PYTHONPATH=~/.conda/envs/lhapdf-tf/lib/python3.8/site-packages:$PYTHONPATH
python Test_1.py

