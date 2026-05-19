#!/bin/bash
#SBATCH --job-name=add-trend-to-vargen-array
#SBATCH --account=gt-pace-pi-account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=8G
#SBATCH --output=out.add-trend.%A_%a
#SBATCH --error=err.add-trend.%A_%a
#SBATCH --time=05:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=email@gatech.edu
#SBATCH --array=0-1

#set -euo pipefail
IFS=$'\n\t'

module load anaconda3
conda activate mpas-analysis

export HDF5_USE_FILE_LOCKING=FALSE

# Forcing variability directory and pattern
VARGEN_DIR="/path/to/scratch/AISLENS/data/processed/vargen_realizations-ssn"
FORCING_PATTERN="AIS_4to20km_r01_20220907_AISLENS-Forcing_*.nc"

# Output directories (one per trend scenario)
OUTDIR_BASE="/path/to/scratch/AISLENS/data/processed"
OUT_SSP126="${OUTDIR_BASE}/vargen_realizations-ssn-ssp126"
OUT_SSP585="${OUTDIR_BASE}/vargen_realizations-ssn-ssp585"
mkdir -p "$OUT_SSP126" "$OUT_SSP585"

# Trend files mapping: index 0 -> SSP126, 1 -> SSP585
TREND_FILES=(
  "/path/to/scratch/AISLENS/data/MALI/ISMIP6/SSP126/output/floatingBMB/floatingBasalMassBalApplied_expAE10-SSP126_Trend_2015-2300.nc"
  "/path/to/scratch/AISLENS/data/MALI/ISMIP6/SSP585/output/floatingBMB/floatingBasalMassBalApplied_expAE05-SSP585_Trend_2015-2300.nc"
)

OUT_DIRS=("$OUT_SSP126" "$OUT_SSP585")

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -lt 0 ] || [ "$IDX" -ge "${#TREND_FILES[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID ($IDX) out of range" 1>&2
  exit 2
fi

TREND_FILE=${TREND_FILES[$IDX]}
OUT_DIR=${OUT_DIRS[$IDX]}

echo "Array index: $IDX"
echo "  TREND: $TREND_FILE"
echo "  OUT_DIR: $OUT_DIR"

# Temporary working directory base
TMPBASE="/path/to/scratch/$USER/add_trend_vargen_tmp"
mkdir -p "$TMPBASE"

# Time index ranges (as in your example)
EARLY_START=0
EARLY_END=167
MID_START=168
MID_END=3599

process_trend_against_vargen() {
  local TREND_FILE="$1"
  local OUT_DIR="$2"
  local TMPDIR="$3"

  echo "Processing trend: $TREND_FILE"
  mkdir -p "$TMPDIR"
  rm -f "$TMPDIR"/* || true

  if [ ! -f "$TREND_FILE" ]; then
    echo "ERROR: trend file not found: $TREND_FILE" >&2
    return 1
  fi

  TREND_WITH_ADJ="$TMPDIR/trend_with_adj.nc"
  echo "Creating trend file with adjustment variable (negated)..."
  ncap2 -O -s 'floatingBasalMassBalAdjustment=-1*floatingBasalMassBalApplied' "$TREND_FILE" "$TREND_WITH_ADJ"

  TREND_LEN=$(ncdump -h "$TREND_WITH_ADJ" | grep "Time =" | sed 's/.*Time = \([0-9]*\).*/\1/' || echo "")
  echo "  Trend Time length: $TREND_LEN"

  for FORCING in "${VARGEN_DIR}"/$FORCING_PATTERN; do
    [ -e "$FORCING" ] || { echo "No forcing files matching pattern in $VARGEN_DIR"; break; }
    base=$(basename "$FORCING")
    OUT_FILE="${OUT_DIR}/${base%.nc}_with_trend.nc"
    echo "Processing forcing: $FORCING -> $OUT_FILE"

    FILETMP="$TMPDIR/$(basename "$base" .nc)"
    mkdir -p "$FILETMP"

    EARLY_FN="$FILETMP/forcing_early.nc"
    ncks -O -d Time,$EARLY_START,$EARLY_END "$FORCING" "$EARLY_FN"

    FORCING_MID_FN="$FILETMP/forcing_mid.nc"
    ncks -O -d Time,$MID_START,$MID_END "$FORCING" "$FORCING_MID_FN"

    FORCING_MID_LEN=$(ncdump -h "$FORCING_MID_FN" | grep "Time =" | sed 's/.*Time = \([0-9]*\).*/\1/' || echo "")
    if [ -z "$TREND_LEN" ] || [ -z "$FORCING_MID_LEN" ]; then
      echo "  Warning: couldn't read Time lengths; skipping file $FORCING"
      continue
    fi
    if [ "$TREND_LEN" != "$FORCING_MID_LEN" ]; then
      echo "  ERROR: Time length mismatch between trend ($TREND_LEN) and forcing mid ($FORCING_MID_LEN). Skipping $FORCING"
      continue
    fi

    if ! ncdump -h "$FORCING_MID_FN" | grep -q "floatingBasalMassBalAdjustment"; then
      echo "  ERROR: forcing file $FORCING does not contain 'floatingBasalMassBalAdjustment'; skipping"
      continue
    fi

    if ! ncdump -h "$TREND_WITH_ADJ" | grep -q "floatingBasalMassBalAdjustment"; then
      echo "  ERROR: trend file does not contain 'floatingBasalMassBalAdjustment' after creation; skipping"
      continue
    fi

    COMBINED_MID_FN="$FILETMP/combined_mid.nc"
    echo "  Adding trend to mid period..."
    ncbo -O -o "$COMBINED_MID_FN" --op_typ=add "$FORCING_MID_FN" "$TREND_WITH_ADJ" || true

    if [ ! -f "$COMBINED_MID_FN" ]; then
      echo "  ncbo failed; attempting ncap2 fallback"
      cp "$FORCING_MID_FN" "$COMBINED_MID_FN"
      TEMP_TREND_VAR_FN="$FILETMP/temp_trend_var.nc"
      ncks -O -v floatingBasalMassBalAdjustment "$TREND_WITH_ADJ" "$TEMP_TREND_VAR_FN"
      ncks -A "$TEMP_TREND_VAR_FN" "$COMBINED_MID_FN"
      ncap2 -O -s 'floatingBasalMassBalAdjustment=floatingBasalMassBalAdjustment + floatingBasalMassBalAdjustment' "$COMBINED_MID_FN" "$COMBINED_MID_FN"
    fi

    echo "  Concatenating early period and combined mid period..."
    ncrcat -O "$EARLY_FN" "$COMBINED_MID_FN" "$OUT_FILE"

    FINAL_TIME_SIZE=$(ncdump -h "$OUT_FILE" | grep "Time =" | sed 's/.*Time = \([0-9]*\).*/\1/' || echo "")
    echo "  Final Time length: $FINAL_TIME_SIZE"

    rm -f "$FILETMP"/* 2>/dev/null || true
    rmdir "$FILETMP" 2>/dev/null || true
    echo "  Finished $base -> $(basename "$OUT_FILE")"
    echo
  done

  rm -f "$TMPDIR"/* 2>/dev/null || true
  rmdir "$TMPDIR" 2>/dev/null || true
  echo "Completed processing for trend file: $TREND_FILE"
}

process_trend_against_vargen "$TREND_FILE" "$OUT_DIR" "$TMPBASE/idx_${IDX}"

echo "All done for array index $IDX."

