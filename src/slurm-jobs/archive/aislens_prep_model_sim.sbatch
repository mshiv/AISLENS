#!/bin/bash
#SBATCH  --job-name=prep-mpaso-4km
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=4 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.prep-mpaso-4km.%j
#SBATCH  --error=err.prep-mpaso-4km..%j
#SBATCH  --exclusive
#SBATCH  --time=60:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

cd ${AISLENS_REPO}/src/scripts 

python prepare_model_sim.py --start-year 0 --end-year 1000
#python calculate_draft_dependence.py
