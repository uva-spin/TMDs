#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --constraint=a100_80gb
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J TMDs_unpol_arc_06
#SBATCH -e output.err
#SBATCH -o output.out
#SBATCH --mem=256000
#SBATCH --array=0-10

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Fit_to_E288pseudo_k_0_2.py $SLURM_ARRAY_TASK_ID