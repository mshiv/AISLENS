#!/bin/bash
#SBATCH --job-name=forcing-gen
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=2 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.forcing-gen.%j
#SBATCH --error=err.forcing-gen.%j
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

################################################################################
# 04-forcing-gen.sbatch
# =====================
# Generate basal melt forcing realizations and fill NaN values via extrapolation.
#
# Consolidates:
#   - aislens_gen_forcings.sbatch
#   - aislens-fill-extrapolate.sbatch
#
# Usage:
#   # Generate 30 forcing realizations for SSP585:
#   sbatch 04-forcing-gen.sbatch --mode generate --scenario SSP585 -n 30
#
#   # Fill NaN values via extrapolation:
#   sbatch 04-forcing-gen.sbatch --mode fill --scenario SSP585
#
#   # Full workflow (generate + fill):
#   sbatch 04-forcing-gen.sbatch --mode all --scenario SSP585 -n 30
#
# Arguments:
#   --mode MODE           generate|fill|all (required)
#   --scenario SCENARIO   CTRL|SSP585|SSP126 (required)
#   -n, --num-realizations N   Number of realizations to generate (default: 30)
#   --param-set SET       Parameter set name (default: permissive)
#   --method METHOD       Extrapolation method: nearest|idw|kriging (default: nearest)
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
MODE=""
SCENARIO=""
NUM_REALIZATIONS=30
PARAM_SET="permissive"
EXTRAP_METHOD="nearest"
SEASONALITY_FILE=""
VARIABILITY_FILE=""
TREND_FILE=""
OUTPUT_DIR=""

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 --mode MODE --scenario SCENARIO [OPTIONS]

Required:
  --mode MODE           generate|fill|all
  --scenario SCENARIO   CTRL|SSP585|SSP126

Options:
  -n, --num-realizations N   Number of realizations (default: 30)
  --param-set SET            Parameter set name (default: permissive)
  --method METHOD            Fill method: nearest|idw|kriging (default: nearest)
  --seasonality-file FILE    Path to seasonality file
  --variability-file FILE    Path to variability file
  --trend-file FILE          Path to trend file
  --output-dir DIR           Output directory
  -h, --help                 Show this help

Modes:
  generate - Generate forcing realizations using stochastic sampling
  fill     - Fill NaN values using extrapolation
  all      - Run generate + fill

Examples:
  sbatch $0 --mode generate --scenario SSP585 -n 30
  sbatch $0 --mode fill --scenario SSP126 --method idw
  sbatch $0 --mode all --scenario SSP585 --param-set optimal_v2
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
        -n|--num-realizations)
            NUM_REALIZATIONS="$2"
            shift 2
            ;;
        --param-set)
            PARAM_SET="$2"
            shift 2
            ;;
        --method)
            EXTRAP_METHOD="$2"
            shift 2
            ;;
        --seasonality-file)
            SEASONALITY_FILE="$2"
            shift 2
            ;;
        --variability-file)
            VARIABILITY_FILE="$2"
            shift 2
            ;;
        --trend-file)
            TREND_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
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

if [[ -z "$SCENARIO" ]]; then
    echo "ERROR: --scenario is required"
    print_usage
    exit 1
fi

case "$MODE" in
    generate|fill|all)
        ;;
    *)
        echo "ERROR: Invalid mode: $MODE"
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

# ============================================================================
# Set Default Paths
# ============================================================================
BASE_DIR="${AISLENS_DATA_DIR}/processed"
DRAFT_DEP_DIR="${BASE_DIR}/draft_dependence/${PARAM_SET}"
FORCING_DIR="${BASE_DIR}/forcing_realizations/${SCENARIO}"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$FORCING_DIR"
fi

if [[ -z "$SEASONALITY_FILE" ]]; then
    SEASONALITY_FILE="${DRAFT_DEP_DIR}/seasonality.nc"
fi

if [[ -z "$VARIABILITY_FILE" ]]; then
    VARIABILITY_FILE="${DRAFT_DEP_DIR}/variability.nc"
fi

if [[ -z "$TREND_FILE" ]]; then
    TREND_FILE="${BASE_DIR}/trends/${SCENARIO}/${PARAM_SET}_trend.nc"
fi

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis

echo "========================================"
echo "Forcing Generation Workflow"
echo "========================================"
echo "Mode: $MODE"
echo "Scenario: $SCENARIO"
echo "Parameter set: $PARAM_SET"
echo "Realizations: $NUM_REALIZATIONS"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

cd "${AISLENS_REPO}/src/scripts"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Functions
# ============================================================================
run_generate() {
    echo "[$(date)] Generating forcing realizations..."
    
    CMD="python generate_forcings.py"
    CMD+=" -n $NUM_REALIZATIONS"
    CMD+=" --seasonality-file $SEASONALITY_FILE"
    CMD+=" --variability-file $VARIABILITY_FILE"
    CMD+=" --trend-file $TREND_FILE"
    CMD+=" --output-dir $OUTPUT_DIR"
    CMD+=" --scenario $SCENARIO"
    
    echo "Running: $CMD"
    eval $CMD
}

run_fill() {
    echo "[$(date)] Filling NaN values via extrapolation..."
    
    # Process all realization files
    for i in $(seq 0 $((NUM_REALIZATIONS - 1))); do
        INPUT_FILE="${OUTPUT_DIR}/forcing_realization_${i}.nc"
        OUTPUT_FILE="${OUTPUT_DIR}/forcing_realization_${i}_filled.nc"
        
        if [[ ! -f "$INPUT_FILE" ]]; then
            echo "WARNING: Input file not found: $INPUT_FILE"
            continue
        fi
        
        echo "  Processing realization $i..."
        python fill_extrapolate.py \
            --input-file "$INPUT_FILE" \
            --output-file "$OUTPUT_FILE" \
            --method "$EXTRAP_METHOD"
    done
}

# ============================================================================
# Execute Based on Mode
# ============================================================================
case "$MODE" in
    generate)
        run_generate
        ;;
    fill)
        run_fill
        ;;
    all)
        run_generate
        run_fill
        ;;
esac

echo "========================================"
echo "Completed at $(date)"
echo "========================================"
