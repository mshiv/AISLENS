#!/bin/bash
#SBATCH --job-name=detrend-ismip6
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.detrend-ismip6.%A_%a
#SBATCH --error=err.detrend-ismip6.%A_%a
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis

# Disable HDF5 file locking on some HPC filesystems (as you had)
export HDF5_USE_FILE_LOCKING=FALSE
# Point AISLENS at your data root
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

# Base repo path (adjust if different)
REPO_ROOT=${AISLENS_REPO}
cd "$REPO_ROOT"

OUTDIR=${AISLENS_DATA_DIR}/processed/forcing-trend-scenarios_test
mkdir -p "$OUTDIR"


python src/scripts/extract_draft_and_trend_mali.py \
  --scenarios 585 \
  --regions 33 \
  --outdir ${AISLENS_DATA_DIR}/processed/forcing-trend-scenarios_test \
  --masks ${AISLENS_DATA_DIR}/external/aislens_draftDepen_regionMasks.nc
