#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=result_%a.out
#SBATCH --error=result_%a.err
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -A spinquest

module purge
module load singularity
echo $LHAPDF_DATA_PATH
ls $LHAPDF_DATA_PATH
module load gcc/9.2.0 lhapdf
source activate tf-2.7
export LD_LIBRARY_PATH=~/.conda/envs/tf-2.7/lib:$LD_LIBRARY_PATH
conda install matplotlib
python -V

python NN_SIDIS_ModGen.py