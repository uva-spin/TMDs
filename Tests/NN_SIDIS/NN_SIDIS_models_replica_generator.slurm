#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=result_%a.out
#SBATCH --error=result_%a.err
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -A spinquest

module purge
module load gcc/9.2.0 lhapdf
echo $LHAPDF_DATA_PATH
ls $LHAPDF_DATA_PATH
python -c "import lhapdf"
python -V

python NN_SIDIS_ModGen.py