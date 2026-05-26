#!/bin/bash
#SBATCH --job-name=dataprep-model
#SBATCH --account=${SLURM_ACCOUNT:-gts-arobel3-atlas}
#SBATCH --nodes=2 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.dataprep-model.%j
#SBATCH --error=err.dataprep-model.%j
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL:-smurugan9@gatech.edu}

################################################################################
# 01-dataprep-model.sbatch
# ========================
# Preprocess ocean model data (SORRM) or satellite observations for AISLENS.
# Includes detrending, deseasonalizing, and per-shelf draft dependence removal.
#
# Consolidates:
#   - aislens_prep_satobs.sbatch
#   - aislens_prep_model_sim_fast.sbatch
#
# Usage:
#   # Preprocess satellite observations:
#   sbatch 01-dataprep-model.sbatch --source satobs
#
#   # Preprocess ocean model (SORRM):
#   sbatch 01-dataprep-model.sbatch --source model
#
#   # Custom directories:
#   sbatch 01-dataprep-model.sbatch --source model --draft-dir /path/to/draft --output-dir /path/to/out
#
# Arguments:
#   --source SOURCE       satobs|model (required)
#   --draft-dir DIR       Output directory for draft dependence files
#   --output-dir DIR      Output directory for processed files
#   --init-dirs           Initialize output directories (for model)
#   --start-year YEAR     Start year for processing
#   --end-year YEAR       End year for processing
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
SOURCE=""
DRAFT_DIR=""
OUTPUT_DIR=""
INIT_DIRS=false
START_YEAR=""
END_YEAR=""

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 --source SOURCE [OPTIONS]

Required:
  --source SOURCE       satobs|model

Options:
  --draft-dir DIR       Draft dependence output directory
  --output-dir DIR      Processed data output directory
  --init-dirs           Initialize directories (for model)
  --start-year YEAR     Start year
  --end-year YEAR       End year
  -h, --help            Show this help

Examples:
  sbatch $0 --source satobs
  sbatch $0 --source model --init-dirs
  sbatch $0 --source model --draft-dir /custom/draft --output-dir /custom/out
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --draft-dir)
            DRAFT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --init-dirs)
            INIT_DIRS=true
            shift
            ;;
        --start-year)
            START_YEAR="$2"
            shift 2
            ;;
        --end-year)
            END_YEAR="$2"
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
if [[ -z "$SOURCE" ]]; then
    echo "ERROR: --source is required"
    print_usage
    exit 1
fi

case "$SOURCE" in
    satobs|model)
        ;;
    *)
        echo "ERROR: Invalid source: $SOURCE. Must be satobs or model"
        exit 1
        ;;
esac

# ============================================================================
# Set Default Paths
# ============================================================================
if [[ -z "$DRAFT_DIR" ]]; then
    case "$SOURCE" in
        satobs)
            DRAFT_DIR="${AISLENS_DATA_DIR}/interim/draft_dependence/satobs-fast"
            ;;
        model)
            DRAFT_DIR="${AISLENS_DATA_DIR}/interim/draft_dependence/model-fast"
            ;;
    esac
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    case "$SOURCE" in
        satobs)
            OUTPUT_DIR="${AISLENS_DATA_DIR}/processed/satobs-fast"
            ;;
        model)
            OUTPUT_DIR="${AISLENS_DATA_DIR}/processed/model-fast"
            ;;
    esac
fi

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

echo "========================================"
echo "Data Preprocessing"
echo "========================================"
echo "Source: $SOURCE"
echo "Draft directory: $DRAFT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

cd "${AISLENS_REPO}/src/scripts"

# ============================================================================
# Execute Based on Source
# ============================================================================
case "$SOURCE" in
    satobs)
        echo "Preprocessing satellite observations..."
        python prepare_satobs.py
        ;;
    
    model)
        echo "Preprocessing ocean model data..."
        
        # Build command with optional flags
        CMD="python prepare_model_sim_fast.py"
        CMD+=" --draft-dir $DRAFT_DIR"
        CMD+=" --output-dir $OUTPUT_DIR"
        
        [[ "$INIT_DIRS" == "true" ]] && CMD+=" --init-dirs"
        [[ -n "$START_YEAR" ]] && CMD+=" --start-year $START_YEAR"
        [[ -n "$END_YEAR" ]] && CMD+=" --end-year $END_YEAR"
        
        echo "Running: $CMD"
        eval $CMD
        ;;
esac

echo "========================================"
echo "Completed at $(date)"
echo "========================================"
