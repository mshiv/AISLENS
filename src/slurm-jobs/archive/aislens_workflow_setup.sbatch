#!/bin/bash
#SBATCH  --job-name=aislens-setup
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=4 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.setup-aislens.%j
#SBATCH  --error=err.setup-aislens.%j
#SBATCH  --exclusive
#SBATCH  --time=60:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

cd ${AISLENS_REPO}/src/scripts 

python prepare_data.py
#python calculate_draft_dependence.py
