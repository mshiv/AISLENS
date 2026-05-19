#!/bin/bash
#SBATCH --job-name=detrend-ismip6
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=logs/out.detrend-ismip6.%A_%a
#SBATCH --error=logs/err.detrend-ismip6.%A_%a
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

# Map SLURM_ARRAY_TASK_ID to scenario string
SCENARIOS=(126 585)
#IDX=${SLURM_ARRAY_TASK_ID}
#SCEN=${SCENARIOS[$IDX]}

# Per-job log / marker
#echo "Running scenario ${SCEN} (array idx ${IDX}) on $(hostname) at $(date)"

# Output directory (you already specified)
OUTDIR=${AISLENS_DATA_DIR}/processed/forcing-trend-scenarios_test
# OUTDIR="${OUTDIR}/SSP${SCEN}"
mkdir -p "$OUTDIR"

# Masks file
MASKS=${AISLENS_DATA_DIR}/external/aislens_draftDepen_regionMasks.nc

# Run the script. We pass the single scenario string to --scenarios so each array task
# processes only its scenario and writes files that include the scenario name in the stem.
python src/scripts/extract_draft_and_trend_mali.py \
  --scenarios 126 \
  --regions 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 \
  --outdir "${OUTDIR}" \
  --masks "${MASKS}"
