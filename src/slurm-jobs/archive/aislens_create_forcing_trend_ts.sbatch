#!/bin/bash
#SBATCH  --job-name=forcing-trend-ssp585
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=2 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.trend-ssp585.%j
#SBATCH  --error=err.trend-ssp585.%j
#SBATCH  --exclusive
#SBATCH  --time=24:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

cd ${AISLENS_REPO}/src/scripts 

python prepare_forcing_trend_components.py
#python calculate_draft_dependence.py
