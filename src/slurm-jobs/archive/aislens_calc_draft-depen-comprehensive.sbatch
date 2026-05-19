#!/bin/bash
#SBATCH  --job-name=aislens-dd
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=2 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.calc-dd-aislens.%j
#SBATCH  --error=err.calc-dd-aislens.%j
#SBATCH  --exclusive
#SBATCH  --time=36:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

cd ${AISLENS_REPO}/src/scripts 

#python prepare_data.py
python -u calculate_draft_dependence_comprehensive.py
