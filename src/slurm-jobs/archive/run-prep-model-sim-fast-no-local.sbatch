#!/bin/bash
#SBATCH --job-name=prep-model-fast-short
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=1
#SBATCH --output=out.prep-mpaso.%j
#SBATCH --error=err.prep-mpaso.%j
#SBATCH --exclusive
#SBATCH --time=30:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis

export CARTOPY_OFFLINE_SHARED=""
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

REPO_ROOT=${AISLENS_REPO}
SCRIPT_DIR=${REPO_ROOT}/src/scripts
cd "${SCRIPT_DIR}"

# persistent output paths (use directly)
OUTPUT_DIR=${AISLENS_DATA_DIR}/processed/model-coarsen-8-smoketest
DRAFT_DIR=${AISLENS_DATA_DIR}/interim/draft_dependence/model-coarsen-8-smoketest
mkdir -p "${OUTPUT_DIR}" "${DRAFT_DIR}"

# set numerics parallelism to avoid oversubscription (recommended)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run prepare_model_sim_fast.py (updated workflow — index-map CLI flags removed)
python prepare_model_sim_fast.py \
  --draft-dir "${DRAFT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --coarsen 8 \
  --start-year 450 --end-year 750 \
  --init-dirs \
  --skip-extrapolation
