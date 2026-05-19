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

#set -euo pipefail

module load anaconda3
conda activate mpas-analysis

# make cartopy conda-activate hook less noisy / avoid activation error
export CARTOPY_OFFLINE_SHARED=""

export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

REPO_ROOT=${AISLENS_REPO}
SCRIPT_DIR=${REPO_ROOT}/src/scripts

# index-map cache path in persistent interim (use same template when creating)
INDEX_MAP_CACHE=${AISLENS_DATA_DIR}/interim/index_map_grid_test.npz
mkdir -p "$(dirname "${INDEX_MAP_CACHE}")"

# shelf-mask cache directory (persistent interim)
SHELF_MASK_CACHE=${AISLENS_DATA_DIR}/interim/shelf_masks
mkdir -p "${SHELF_MASK_CACHE}"

# ensure output and draft dirs exist (persistent processed / interim)
OUTPUT_DIR=${AISLENS_DATA_DIR}/processed/model-fast-smoketest
DRAFT_DIR=${AISLENS_DATA_DIR}/interim/draft_dependence/model-fast-smoketest
mkdir -p "${OUTPUT_DIR}" "${DRAFT_DIR}"

cd "${SCRIPT_DIR}"

# --- quick cache pre-checks (informational) ---
echo "[$(date -Is)] Index map file exists?: " $( [[ -f "${INDEX_MAP_CACHE}" ]] && echo "yes" || echo "no" )
if [[ -f "${INDEX_MAP_CACHE}" ]]; then
  ls -l "${INDEX_MAP_CACHE}"
fi
echo "[$(date -Is)] Shelf mask cache dir exists and contents:"
if [[ -d "${SHELF_MASK_CACHE}" ]]; then
  ls -ld "${SHELF_MASK_CACHE}" "${SHELF_MASK_CACHE}"/* 2>/dev/null || true
else
  echo "no"
fi

# Small extrapolation test with index-map enabled (cheap)
python prepare_model_sim_fast.py \
  --draft-dir "${DRAFT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --coarsen 4 \
  --start-year 450 --end-year 450 \
  --use-index-map \
  --index-map-cache "${INDEX_MAP_CACHE}" \
  --shelf-mask-cache "${SHELF_MASK_CACHE}" \
  --overwrite-shelf-mask-cache
