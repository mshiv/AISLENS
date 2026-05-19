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
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

set -euo pipefail
module load anaconda3
conda activate mpas-analysis

export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

REPO_ROOT=${AISLENS_REPO}
SCRIPT_DIR=${REPO_ROOT}/src/scripts

# index-map cache path (fast scratch)
INDEX_MAP_CACHE=${AISLENS_DATA_DIR}/interim/index_map_grid_test.npz
mkdir -p "$(dirname "${INDEX_MAP_CACHE}")"

# ensure output and draft dirs exist
OUTPUT_DIR=${AISLENS_DATA_DIR}/processed/model-fast-smoketest
DRAFT_DIR=${AISLENS_DATA_DIR}/interim/draft_dependence/model-fast-smoketest
mkdir -p "${OUTPUT_DIR}" "${DRAFT_DIR}"

cd "${SCRIPT_DIR}"

# small extrapolation test with index-map enabled (cheap)
python prepare_model_sim_fast.py \
  --draft-dir "${DRAFT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --coarsen 4 \
  --start-year 450 --end-year 452 \
  --use-index-map \
  --index-map-cache "${INDEX_MAP_CACHE}"
