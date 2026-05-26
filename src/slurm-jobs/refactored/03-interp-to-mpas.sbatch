#!/bin/bash
#SBATCH --job-name=interp-to-mpas
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=8G
#SBATCH --output=out.interp-to-mpas.%A_%a
#SBATCH --error=err.interp-to-mpas.%A_%a
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smurugan9@gatech.edu
#SBATCH --array=0-10

################################################################################
# 03-interp-to-mpas.sbatch
# ========================
# Interpolate draft dependence parameters or forcing realizations to MPAS-LI grid.
#
# Consolidates:
#   - aislens_interpolate_draft-depen-paramsets_array.sbatch
#   - aislens-interpolate-forcings-array.sbatch
#
# Usage:
#   # Interpolate draft dependence parameter sets (array 0-10):
#   sbatch 03-interp-to-mpas.sbatch --mode params
#
#   # Interpolate forcing realizations (array 0-29):
#   sbatch --array=0-29 03-interp-to-mpas.sbatch --mode forcings --scenario SSP585
#
#   # Single parameter set:
#   sbatch --array=0 03-interp-to-mpas.sbatch --mode params
#
# Arguments:
#   --mode MODE           params|forcings (required)
#   --scenario SCENARIO   CTRL|SSP585|SSP126 (for forcings mode)
#   --input-dir DIR       Input data directory
#   --output-dir DIR      Output directory
#   --method METHOD       Interpolation method (nearest, bilinear, cubic)
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
MODE=""
SCENARIO="SSP585"
INPUT_DIR=""
OUTPUT_DIR=""
METHOD="bilinear"

# Parameter set names for array indices 0-10
PARAM_SETS=(
    "permissive"
    "conservative"
    "baseline"
    "optimal_v1"
    "optimal_v2"
    "optimal_v3"
    "optimal_v4"
    "optimal_v5"
    "optimal_v6"
    "optimal_v7"
    "optimal_v8"
)

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch [--array=N-M] $0 --mode MODE [OPTIONS]

Required:
  --mode MODE           params|forcings

Options:
  --scenario SCENARIO   CTRL|SSP585|SSP126 (for forcings, default: SSP585)
  --input-dir DIR       Input directory (auto-set based on mode)
  --output-dir DIR      Output directory (auto-set based on mode)
  --method METHOD       Interpolation method: nearest|bilinear|cubic (default: bilinear)
  -h, --help            Show this help

Array Job:
  For params mode:   --array=0-10 (11 parameter sets)
  For forcings mode: --array=0-29 (30 realizations)

Parameter Sets (array index):
  0: permissive    1: conservative   2: baseline
  3: optimal_v1    4: optimal_v2     5: optimal_v3
  6: optimal_v4    7: optimal_v5     8: optimal_v6
  9: optimal_v7   10: optimal_v8

Examples:
  sbatch --array=0-10 $0 --mode params
  sbatch --array=0-29 $0 --mode forcings --scenario SSP126
  sbatch --array=5 $0 --mode params  # Only optimal_v3
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
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
if [[ -z "$MODE" ]]; then
    echo "ERROR: --mode is required"
    print_usage
    exit 1
fi

case "$MODE" in
    params|forcings)
        ;;
    *)
        echo "ERROR: Invalid mode: $MODE. Must be params or forcings"
        exit 1
        ;;
esac

case "$SCENARIO" in
    CTRL|SSP585|SSP126)
        ;;
    *)
        echo "ERROR: Invalid scenario: $SCENARIO"
        exit 1
        ;;
esac

# Get array task ID
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}

# ============================================================================
# Set Default Paths
# ============================================================================
MPAS_MESH="${AISLENS_DATA_DIR}/external/AIS-MPAS-4km-mesh/ais4.mali.240km.culled.nc"

if [[ "$MODE" == "params" ]]; then
    PARAM_SET="${PARAM_SETS[$ARRAY_ID]}"
    INPUT_DIR="${INPUT_DIR:-${AISLENS_DATA_DIR}/processed/draft_dependence}"
    OUTPUT_DIR="${OUTPUT_DIR:-${AISLENS_DATA_DIR}/processed/draft_dependence/mpas_interpolated}"
elif [[ "$MODE" == "forcings" ]]; then
    INPUT_DIR="${INPUT_DIR:-${AISLENS_DATA_DIR}/processed/forcing_realizations/${SCENARIO}}"
    OUTPUT_DIR="${OUTPUT_DIR:-${AISLENS_DATA_DIR}/processed/forcing_realizations/${SCENARIO}/mpas_interpolated}"
fi

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis

echo "========================================"
echo "Interpolation to MPAS-LI Grid"
echo "========================================"
echo "Mode: $MODE"
echo "Array ID: $ARRAY_ID"
if [[ "$MODE" == "params" ]]; then
    echo "Parameter Set: $PARAM_SET"
else
    echo "Scenario: $SCENARIO"
    echo "Realization: $ARRAY_ID"
fi
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Method: $METHOD"
echo "========================================"

cd "${AISLENS_REPO}/src/scripts"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Execute Based on Mode
# ============================================================================
case "$MODE" in
    params)
        echo "[$(date)] Interpolating parameter set: $PARAM_SET..."
        
        INPUT_FILE="${INPUT_DIR}/${PARAM_SET}_draft_dependence_params.nc"
        OUTPUT_FILE="${OUTPUT_DIR}/${PARAM_SET}_draft_dependence_params_mpas.nc"
        
        if [[ ! -f "$INPUT_FILE" ]]; then
            echo "ERROR: Input file not found: $INPUT_FILE"
            exit 1
        fi
        
        python interpolate_to_mpasli_grid_new.py \
            --input-file "$INPUT_FILE" \
            --output-file "$OUTPUT_FILE" \
            --mpas-mesh "$MPAS_MESH" \
            --method "$METHOD"
        ;;
    
    forcings)
        echo "[$(date)] Interpolating forcing realization: $ARRAY_ID..."
        
        INPUT_FILE="${INPUT_DIR}/forcing_realization_${ARRAY_ID}.nc"
        OUTPUT_FILE="${OUTPUT_DIR}/forcing_realization_${ARRAY_ID}_mpas.nc"
        
        if [[ ! -f "$INPUT_FILE" ]]; then
            echo "ERROR: Input file not found: $INPUT_FILE"
            exit 1
        fi
        
        python interpolate_to_mpasli_grid_new.py \
            --input-file "$INPUT_FILE" \
            --output-file "$OUTPUT_FILE" \
            --mpas-mesh "$MPAS_MESH" \
            --method "$METHOD"
        ;;
esac

echo "========================================"
echo "Array task $ARRAY_ID completed at $(date)"
echo "========================================"
