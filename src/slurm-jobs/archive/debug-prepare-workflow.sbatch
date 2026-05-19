#!/bin/bash
#SBATCH --job-name=debug-prep-workflow
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=1
#SBATCH --output=out.debug-prepare.%j
#SBATCH --error=err.debug-prepare.%j
#SBATCH --exclusive
#SBATCH --time=30:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

echo "Starting debug_prepare_workflow SLURM job on $(hostname) at $(date)"

module load anaconda3
# adjust the env name to whatever you use on the cluster
conda activate mpas-analysis

# Edit these paths to match your environment
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}
REPO_ROOT=${AISLENS_REPO}
SCRIPT_DIR=${REPO_ROOT}/src/scripts

# Where to write final processed outputs and per-catchment drafts
OUTPUT_DIR=${AISLENS_DATA_DIR}/processed/debug-workflow-smoketest
DRAFT_DIR=${AISLENS_DATA_DIR}/interim/draft_dependence/debug-workflow-smoketest
mkdir -p "${OUTPUT_DIR}" "${DRAFT_DIR}"

cd "${SCRIPT_DIR}"

# Optional toggles (export to override defaults below)
# export SKIP_DEDRAFT=1
# export PRECOMPUTED_MEAN=/path/to/precomputed_time_mean.nc

# CLI arguments for the debug script
START_YEAR=450
END_YEAR=750
COARSEN=8

# Build command
CMD=(python debug_prepare_workflow.py
     --start-year "${START_YEAR}"
     --end-year "${END_YEAR}"
     --coarsen "${COARSEN}"
     --output-dir "${OUTPUT_DIR}"
     --draft-dir "${DRAFT_DIR}"
)

# Optionally use a precomputed mean
if [[ -n "${PRECOMPUTED_MEAN:-}" ]]; then
  CMD+=(--precomputed-mean "${PRECOMPUTED_MEAN}")
fi

# Optionally skip dedraft
if [[ -n "${SKIP_DEDRAFT:-}" ]]; then
  CMD+=(--skip-dedraft)
fi

# Log the full command for debugging
echo "Running command: ${CMD[*]}"
"${CMD[@]}"

echo "Job finished at $(date)"
