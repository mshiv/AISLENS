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

# ---- environment and repo paths ----
module load anaconda3
conda activate mpas-analysis

# make cartopy conda-activate hook less noisy / avoid activation error
export CARTOPY_OFFLINE_SHARED=""

export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

REPO_ROOT=${AISLENS_REPO}
SCRIPT_DIR=${REPO_ROOT}/src/scripts
cd "${SCRIPT_DIR}"

# ---- persistent cache locations (persistent storage) ----
PERSIST_INDEX_MAP_CACHE=${AISLENS_DATA_DIR}/interim/index_map_grid_test.npz
mkdir -p "$(dirname "${PERSIST_INDEX_MAP_CACHE}")"

PERSIST_SHELF_MASK_CACHE=${AISLENS_DATA_DIR}/interim/shelf_masks
mkdir -p "${PERSIST_SHELF_MASK_CACHE}"

# ---- output and draft dirs (persistent) ----
OUTPUT_DIR=${AISLENS_DATA_DIR}/processed/model-fast-smoketest
DRAFT_DIR=${AISLENS_DATA_DIR}/interim/draft_dependence/model-fast-smoketest
mkdir -p "${OUTPUT_DIR}" "${DRAFT_DIR}"

# ---- node-local scratch for faster IO (use $TMPDIR if available) ----
NODE_SCRATCH="${TMPDIR:-/tmp}/${USER}_${SLURM_JOB_ID}"
mkdir -p "${NODE_SCRATCH}"

# node-local cache paths (use these when running the Python script)
NODE_INDEX_MAP_CACHE="${NODE_SCRATCH}/index_map_cache.npz"
NODE_SHELF_MASK_CACHE="${NODE_SCRATCH}/shelf_masks"
mkdir -p "${NODE_SHELF_MASK_CACHE}"

# ---- copy persistent caches to node-local if they exist ----
echo "[$(date -Is)] Copying persistent caches (if present) -> node-local scratch"
if [[ -f "${PERSIST_INDEX_MAP_CACHE}" ]]; then
  cp -v "${PERSIST_INDEX_MAP_CACHE}" "${NODE_INDEX_MAP_CACHE}"
fi

if [[ -d "${PERSIST_SHELF_MASK_CACHE}" ]]; then
  rsync -av "${PERSIST_SHELF_MASK_CACHE}/" "${NODE_SHELF_MASK_CACHE}/" || true
fi

# quick diagnostics
echo "[$(date -Is)] Node-local scratch: ${NODE_SCRATCH}"
echo "[$(date -Is)] Node-index-map exists?: " $( [[ -f "${NODE_INDEX_MAP_CACHE}" ]] && echo "yes" || echo "no" )
echo "[$(date -Is)] Node-shelf-mask-cache contents:"
ls -ld "${NODE_SHELF_MASK_CACHE}" "${NODE_SHELF_MASK_CACHE}"/* 2>/dev/null || true

# ---- run the fast prepare script (simple extrapolate) ----
# Use --simple-extrapolate to use the new notebook-style fast path
python prepare_model_sim_fast.py \
  --draft-dir "${DRAFT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --coarsen 8 \
  --start-year 450 --end-year 452 \
  --simple-extrapolate \
  --use-index-map \
  --index-map-cache "${NODE_INDEX_MAP_CACHE}" \
  --shelf-mask-cache "${NODE_SHELF_MASK_CACHE}" \
  --overwrite-shelf-mask-cache \
  --log-timings

RET=$?
echo "[$(date -Is)] prepare_model_sim_fast.py exited with code ${RET}"

# ---- copy node-local caches back to persistent storage (if the run created/updated them) ----
echo "[$(date -Is)] Copying node-local caches back to persistent storage"
if [[ -f "${NODE_INDEX_MAP_CACHE}" ]]; then
  mkdir -p "$(dirname "${PERSIST_INDEX_MAP_CACHE}")"
  cp -v "${NODE_INDEX_MAP_CACHE}" "${PERSIST_INDEX_MAP_CACHE}"
fi

# use rsync to merge shelf mask cache back to persistent; preserve perms
rsync -av --delete "${NODE_SHELF_MASK_CACHE}/" "${PERSIST_SHELF_MASK_CACHE}/" || true

# cleanup node-local
rm -rf "${NODE_SCRATCH}"

exit ${RET}
