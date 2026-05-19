#!/bin/bash
#SBATCH  --job-name=prep-time-mean
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.prep-mpaso.%j
#SBATCH  --error=err.prep-mpaso..%j
#SBATCH  --exclusive
#SBATCH  --time=10:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

cd ${AISLENS_REPO}/src/scripts 

python prepare_model_sim_fast.py --draft-dir ${AISLENS_DATA_DIR}/interim/draft_dependence/model-fast-1000y --output-dir ${AISLENS_DATA_DIR}/processed/model-fast-1000y --start-year 0 --end-year 1000 --init-dirs
