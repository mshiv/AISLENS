#!/bin/bash
#SBATCH --job-name=subtract-trend-var
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=2 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --time=02:00:00
#SBATCH --output=out.subtract-trend.%A_%a
#SBATCH --error=err.subtract-trend.%A_%a
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}
#SBATCH --array=0-1

set -euo pipefail
IFS=$'\n\t'

module load anaconda3
conda activate mpas-analysis

export HDF5_USE_FILE_LOCKING=FALSE

# Repo scripts path (adjust if your working directory differs)
REPO_SCRIPTS_DIR="${AISLENS_REPO}/src/scripts"
cd "$REPO_SCRIPTS_DIR"

IDX=${SLURM_ARRAY_TASK_ID:-0}
TMPDIR="${HOME}/scratch/$USER/subtract_trend_idx_${IDX}"
mkdir -p "$TMPDIR"

# Define pairs (index-wise): TREND+VAR file, VAR-only file, OUT file
TREND_FILES=(
  "${AISLENS_DATA_DIR}/processed/vargen_realizations-ssn-trend-ssp585/AIS_4to20km_r01_20220907_AISLENS-Forcing_0.nc"
  "${AISLENS_DATA_DIR}/processed/vargen_realizations-ssn-trend-ssp126/AIS_4to20km_r01_20220907_AISLENS-Forcing_0.nc"
)
VAR_FILES=(
  "${AISLENS_DATA_DIR}/processed/vargen_realizations-ssn/AIS_4to20km_r01_20220907_AISLENS-Forcing_0.nc"
  "${AISLENS_DATA_DIR}/processed/vargen_realizations-ssn/AIS_4to20km_r01_20220907_AISLENS-Forcing_0.nc"
)
OUT_FILES=(
  "${AISLENS_DATA_DIR}/processed/vargen_trend_only/AIS_4to20km_r01_20220907_AISLENS-Forcing_Trend_SSP585.nc"
  "${AISLENS_DATA_DIR}/processed/vargen_trend_only/AIS_4to20km_r01_20220907_AISLENS-Forcing_Trend_SSP126.nc"
)

N=${#TREND_FILES[@]}
if [ $IDX -lt 0 ] || [ $IDX -ge $N ]; then
  echo "Array index $IDX out of range (0..$((N-1)))" >&2
  exit 2
fi

TREND=${TREND_FILES[$IDX]}
VAR=${VAR_FILES[$IDX]}
OUT=${OUT_FILES[$IDX]}
LOG="$TMPDIR/$(basename "$OUT").log"
mkdir -p "$(dirname "$OUT")"

echo "Task $IDX -> TREND=$TREND VAR=$VAR OUT=$OUT"

{
  echo "Starting: $(date)"
  python subtract_trend_var.py --trend "$TREND" --var "$VAR" --out "$OUT" --varname floatingBasalMassBalAdjustment --chunk 32 --verbose
  echo "Finished: $(date)"
} >"$LOG" 2>&1 || {
  echo "Task failed; see log: $LOG"
  exit 10
}

rm -rf "$TMPDIR"
echo "Completed task $IDX"
exit 0

