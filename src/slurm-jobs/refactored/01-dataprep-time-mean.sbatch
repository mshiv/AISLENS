#!/bin/bash
#SBATCH --job-name=dataprep-time-mean
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.dataprep-time-mean.%j
#SBATCH --error=err.dataprep-time-mean.%j
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

################################################################################
# 01-dataprep-time-mean.sbatch
# ============================
# Compute time-mean products from ocean model (SORRM) data.
# Consolidates: aislens_prep_time_mean_model.sbatch + aislens_prep_time_mean_split.sbatch
#
# Usage:
#   sbatch 01-dataprep-time-mean.sbatch                    # Default: full mode
#   sbatch 01-dataprep-time-mean.sbatch --mode split       # Split (parallel) mode
#   sbatch 01-dataprep-time-mean.sbatch --mode full        # Full mode (original)
#   sbatch 01-dataprep-time-mean.sbatch --coarsen 2        # Spatial coarsening factor
#   sbatch 01-dataprep-time-mean.sbatch --start-year 450   # Custom time range
#
# Modes:
#   full   - Uses prepare_time_mean.py (single file processing)
#   split  - Uses prepare_time_mean_split.py (parallel chunk processing)
#
# History:
#   Consolidated from aislens_prep_time_mean_{model,split}.sbatch
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
MODE="full"
START_YEAR=0
END_YEAR=1000
COARSEN=2
INPUT_PATTERN=""
OUTPUT_FILE=""

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 [OPTIONS]

Options:
  --mode MODE        Processing mode: full|split (default: full)
  --start-year YEAR  Start year for time range (default: 0)
  --end-year YEAR    End year for time range (default: 1000)
  --coarsen N        Spatial coarsening factor (default: 2)
  --input PATTERN    Input file pattern (for split mode)
  --output FILE      Output file path (for split mode)
  -h, --help         Show this help

Examples:
  sbatch $0 --mode full --start-year 450 --end-year 750
  sbatch $0 --mode split --coarsen 4
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --start-year)
            START_YEAR="$2"
            shift 2
            ;;
        --end-year)
            END_YEAR="$2"
            shift 2
            ;;
        --coarsen)
            COARSEN="$2"
            shift 2
            ;;
        --input)
            INPUT_PATTERN="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
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
# Validate Mode
# ============================================================================
case "$MODE" in
    full|split)
        ;;
    *)
        echo "ERROR: Invalid mode: $MODE. Must be 'full' or 'split'"
        exit 1
        ;;
esac

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

echo "========================================"
echo "AISLENS Data Preparation: Time Mean"
echo "========================================"
echo "Mode: $MODE"
echo "Time range: $START_YEAR - $END_YEAR"
echo "Coarsen factor: $COARSEN"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "========================================"

cd "${AISLENS_REPO}/src/scripts"

# ============================================================================
# Execute Based on Mode
# ============================================================================
case "$MODE" in
    full)
        echo "Running prepare_time_mean.py..."
        python prepare_time_mean.py \
            --start-year "$START_YEAR" \
            --end-year "$END_YEAR"
        ;;
    
    split)
        # Set defaults for split mode if not provided
        if [[ -z "$INPUT_PATTERN" ]]; then
            INPUT_PATTERN="${AISLENS_DATA_DIR}/external/aislens_regridded_sorrm/Regridded_SORRMv2.1.ISMF.*-*.6000.0x6000.0km_4.0km_Antarctic_stereo.nc"
        fi
        if [[ -z "$OUTPUT_FILE" ]]; then
            OUTPUT_FILE="${AISLENS_DATA_DIR}/interim/draft_dependence/_temp_time_mean.nc"
        fi
        
        echo "Running prepare_time_mean_split.py..."
        echo "Input pattern: $INPUT_PATTERN"
        echo "Output file: $OUTPUT_FILE"
        
        python prepare_time_mean_split.py \
            --input-pattern "$INPUT_PATTERN" \
            --output "$OUTPUT_FILE" \
            --coarsen "$COARSEN"
        ;;
esac

echo "========================================"
echo "Completed at $(date)"
echo "========================================"
