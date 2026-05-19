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

module load anaconda3
conda activate mpas-analysis

export CARTOPY_OFFLINE_SHARED=""
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

REPO_ROOT=${AISLENS_REPO}
SCRIPT_DIR=${REPO_ROOT}/src/scripts
cd "${SCRIPT_DIR}"

# persistent cache paths (use directly)
INDEX_MAP_CACHE=${AISLENS_DATA_DIR}/interim/index_map_grid_test.npz
SHELF_MASK_CACHE=${AISLENS_DATA_DIR}/interim/shelf_masks

OUTPUT_DIR=${AISLENS_DATA_DIR}/processed/model-fast-smoketest
DRAFT_DIR=${AISLENS_DATA_DIR}/interim/draft_dependence/model-fast-smoketest
mkdir -p "${OUTPUT_DIR}" "${DRAFT_DIR}"

# set numerics parallelism to avoid oversubscription (optional but recommended)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run using persistent caches directly -- no node-local copies
python prepare_model_sim_fast.py \
  --draft-dir "${DRAFT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --coarsen 10 \
  --start-year 450 --end-year 451 \
  # --simple-extrapolate \
  --use-index-map \
  --index-map-cache "${INDEX_MAP_CACHE}" \
  --shelf-mask-cache "${SHELF_MASK_CACHE}"
 
#--log-timings

# If this job should update the shelf-mask cache and you are sure no concurrent writers:
# add --overwrite-shelf-mask-cache (use with caution)
