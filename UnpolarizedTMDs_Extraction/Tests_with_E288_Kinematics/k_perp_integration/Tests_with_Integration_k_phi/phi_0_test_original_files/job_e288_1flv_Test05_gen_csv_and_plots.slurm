#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --constraint=a100_80gb
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J TMDs_unpol_Test_05_plots
#SBATCH -e tuner.err
#SBATCH -o output.out
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Evaluate_TMDs_E288_gen_csv.py
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Evaluate_TMDs_E288_only_trained_PDFs.py
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Evaluate_TMDs_E288_only_trained_points.py
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Evaluate_TMDs_with_kval.py
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Set_of_TMD_Plots.py