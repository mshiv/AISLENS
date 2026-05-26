#!/bin/bash
#SBATCH --job-name=combine-trend-var
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=2 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.combine-trend-var.%A_%a
#SBATCH --error=err.combine-trend-var.%A_%a
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smurugan9@gatech.edu

################################################################################
# 06-combine-trend-var.sbatch
# ===========================
# Combine trend component with variability realizations to create final forcings.
# Supports multiple methods: NCO-based, Python xarray, or batch processing.
#
# Consolidates:
#   - add-trend-forcing-to-variability.sbatch (NCO method)
#   - add_trend_to_vargen_array.sbatch (Python method)
#   - aislens_combine_forcing_components.sbatch (Python combiner)
#   - aislens-create-forcings-trend-combined.sbatch
#
# Usage:
#   # Combine for SSP585:
#   sbatch 06-combine-trend-var.sbatch --scenario SSP585 --method python
#
#   # Array job for multiple realizations:
#   sbatch --array=0-19 06-combine-trend-var.sbatch --scenario SSP585 --method python
#
#   # NCO-based combination:
#   sbatch --array=0-1 06-combine-trend-var.sbatch --scenario all --method nco
#
# Arguments:
#   --scenario SCENARIO   SSP585, SSP126, or 'all' (for both)
#   --method METHOD       python|nco|batch (default: python)
#   --trend-file FILE     Override trend file path
#   --var-dir DIR         Directory with variability realizations
#   --out-dir DIR         Output directory
#   --array-index N       Override SLURM_ARRAY_TASK_ID
#
# Methods:
#   python - Uses add_trend_to_vargen_xarray.py (memory efficient)
#   nco    - Uses ncap2 for arithmetic (faster for small files)
#   batch  - Uses combine_ssp585_forcing_python.py (full directory)
################################################################################

set -euo pipefail
IFS=$'\n\t'

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
SCENARIO=""
METHOD="python"
TREND_FILE=""
VAR_DIR=""
OUT_DIR=""
ARRAY_INDEX=""

# Default directories
DEFAULT_VAR_DIR="${AISLENS_DATA_DIR}/processed/vargen_realizations-ssn"
DEFAULT_TREND_DIR="${AISLENS_DATA_DIR}/processed/deterministic-trend-forcings"

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 --scenario SCENARIO [OPTIONS]

Required:
  --scenario SCENARIO   SSP585, SSP126, or 'all'

Options:
  --method METHOD       python|nco|batch (default: python)
  --trend-file FILE     Override trend file path
  --var-dir DIR         Variability realizations directory
  --out-dir DIR         Output directory
  --array-index N       Process specific array index
  -h, --help            Show this help

Examples:
  sbatch $0 --scenario SSP585 --method python
  sbatch --array=0-19 $0 --scenario SSP585 --method python
  sbatch $0 --scenario all --method batch
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)
            SCENARIO="${2^^}"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --trend-file)
            TREND_FILE="$2"
            shift 2
            ;;
        --var-dir)
            VAR_DIR="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --array-index)
            ARRAY_INDEX="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Validate Arguments
# ============================================================================
if [[ -z "$SCENARIO" ]]; then
    echo "ERROR: --scenario is required"
    print_usage
    exit 1
fi

case "$SCENARIO" in
    SSP585|SSP126|ALL)
        ;;
    *)
        echo "ERROR: Invalid scenario: $SCENARIO"
        exit 1
        ;;
esac

case "$METHOD" in
    python|nco|batch)
        ;;
    *)
        echo "ERROR: Invalid method: $METHOD"
        exit 1
        ;;
esac

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis
export HDF5_USE_FILE_LOCKING=FALSE
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

echo "========================================"
echo "Combine Trend + Variability"
echo "========================================"
echo "Scenario: $SCENARIO"
echo "Method: $METHOD"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "========================================"

# ============================================================================
# Configure Based on Scenario
# ============================================================================
# Set up arrays of trend files and output directories
declare -a TREND_FILES
declare -a OUT_DIRS
declare -a SCENARIO_NAMES

case "$SCENARIO" in
    SSP585)
        TREND_FILES=("${DEFAULT_TREND_DIR}/AIS_4to20km_r01_20220907_AISLENS-Forcing_Trend_SSP585_negAdj.nc")
        OUT_DIRS=("${AISLENS_DATA_DIR}/processed/vargen_realizations-ssn-trend-ssp585")
        SCENARIO_NAMES=("SSP585")
        ;;
    SSP126)
        TREND_FILES=("${DEFAULT_TREND_DIR}/AIS_4to20km_r01_20220907_AISLENS-Forcing_Trend_SSP126_negAdj.nc")
        OUT_DIRS=("${AISLENS_DATA_DIR}/processed/vargen_realizations-ssn-trend-ssp126")
        SCENARIO_NAMES=("SSP126")
        ;;
    ALL)
        TREND_FILES=(
            "${DEFAULT_TREND_DIR}/AIS_4to20km_r01_20220907_AISLENS-Forcing_Trend_SSP126_negAdj.nc"
            "${DEFAULT_TREND_DIR}/AIS_4to20km_r01_20220907_AISLENS-Forcing_Trend_SSP585_negAdj.nc"
        )
        OUT_DIRS=(
            "${AISLENS_DATA_DIR}/processed/vargen_realizations-ssn-trend-ssp126"
            "${AISLENS_DATA_DIR}/processed/vargen_realizations-ssn-trend-ssp585"
        )
        SCENARIO_NAMES=("SSP126" "SSP585")
        ;;
esac

# Override if provided
[[ -n "$TREND_FILE" ]] && TREND_FILES=("$TREND_FILE")
[[ -n "$VAR_DIR" ]] || VAR_DIR="$DEFAULT_VAR_DIR"
[[ -n "$OUT_DIR" ]] && OUT_DIRS=("$OUT_DIR")

# Create output directories
for dir in "${OUT_DIRS[@]}"; do
    mkdir -p "$dir"
done

# ============================================================================
# Get Variability Files
# ============================================================================
mapfile -t VAR_FILES < <(ls -1 "$VAR_DIR"/*.nc 2>/dev/null || true)
N_VAR=${#VAR_FILES[@]}
N_TREND=${#TREND_FILES[@]}

if [[ $N_VAR -eq 0 ]]; then
    echo "ERROR: No variability files found in $VAR_DIR"
    exit 1
fi

echo "Found $N_VAR variability files"
echo "Processing $N_TREND trend scenarios"

# ============================================================================
# Array Job Index Calculation
# ============================================================================
TASK_ID="${SLURM_ARRAY_TASK_ID:-${ARRAY_INDEX:-0}}"
TOTAL=$((N_TREND * N_VAR))

if [[ $TASK_ID -ge $TOTAL ]]; then
    echo "Task ID $TASK_ID exceeds total combinations ($TOTAL)"
    exit 0
fi

# Calculate indices
trend_idx=$((TASK_ID / N_VAR))
var_idx=$((TASK_ID % N_VAR))

CURRENT_TREND="${TREND_FILES[$trend_idx]}"
CURRENT_VAR="${VAR_FILES[$var_idx]}"
CURRENT_OUT_DIR="${OUT_DIRS[$trend_idx]}"
CURRENT_SCENARIO="${SCENARIO_NAMES[$trend_idx]}"

VAR_BASENAME=$(basename "$CURRENT_VAR")
OUT_FILE="${CURRENT_OUT_DIR}/${VAR_BASENAME}"

echo "Task $TASK_ID:"
echo "  Trend: $CURRENT_TREND"
echo "  Variability: $CURRENT_VAR"
echo "  Output: $OUT_FILE"

# ============================================================================
# Processing Functions
# ============================================================================
combine_python() {
    local trend_file="$1"
    local var_file="$2"
    local out_file="$3"
    
    cd "${AISLENS_REPO}/src/scripts"
    
    python add_trend_to_vargen_xarray.py \
        --trend-file "$trend_file" \
        --var-file "$var_file" \
        --output "$out_file"
}

combine_nco() {
    local trend_file="$1"
    local var_file="$2"
    local out_file="$3"
    
    # Create temporary file
    local tmp_file=$(mktemp --suffix=.nc)
    
    # Copy variability file as base
    cp "$var_file" "$tmp_file"
    
    # Add trend variable to the file
    ncks -A -v floatingBasalMassBalAdjustment "$trend_file" "$tmp_file"
    
    # Sum the variables
    ncap2 -O -s 'floatingBasalMassBalAdjustment=floatingBasalMassBalAdjustment+floatingBasalMassBalAdjustment_trend' \
        "$tmp_file" "$out_file"
    
    rm -f "$tmp_file"
}

combine_batch() {
    local trend_file="$1"
    local ensemble_dir="$2"
    
    cd "${AISLENS_REPO}/src/scripts"
    
    python combine_ssp585_forcing_python.py \
        --trend-file "$trend_file" \
        --ensemble-dir "$ensemble_dir"
}

# ============================================================================
# Execute Based on Method
# ============================================================================
case "$METHOD" in
    python)
        combine_python "$CURRENT_TREND" "$CURRENT_VAR" "$OUT_FILE"
        ;;
    nco)
        combine_nco "$CURRENT_TREND" "$CURRENT_VAR" "$OUT_FILE"
        ;;
    batch)
        # Batch mode processes entire directory
        combine_batch "$CURRENT_TREND" "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/${CURRENT_SCENARIO}"
        ;;
esac

echo "========================================"
echo "Completed at $(date)"
echo "========================================"
