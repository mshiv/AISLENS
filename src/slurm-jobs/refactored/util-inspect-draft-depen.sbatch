#!/bin/bash
#SBATCH --job-name=util-inspect-dd
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.util-inspect-dd.%j
#SBATCH --error=err.util-inspect-dd.%j
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smurugan9@gatech.edu

################################################################################
# util-inspect-draft-depen.sbatch
# ===============================
# Debug/validation utility to compare draft dependence results between different
# calculation methods or parameter sets. Uses inspect_shelf_pair.py.
#
# This is a UTILITY script - not part of the main workflow, but useful for:
#   - Validating draft dependence calculations
#   - Comparing fast vs parallel implementations
#   - Debugging parameter set differences
#
# Restored from: archive/aislens_inspect_shelf_pair_draft-depen-comprehensive-debug.sbatch
#
# Usage:
#   sbatch util-inspect-draft-depen.sbatch --shelf Amery
#   sbatch util-inspect-draft-depen.sbatch --shelf "Ross Ice Shelf" --param-set permissive
#
# Arguments:
#   --shelf NAME          Ice shelf name (required)
#   --param-set SET       Parameter set: standard|permissive|fb_A|... (default: standard)
#   --fast-dir DIR        Directory with fast calculation results
#   --parallel-dir DIR    Directory with parallel calculation results
#   --output FILE         Output JSON file for results
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
SHELF=""
PARAM_SET="standard"
FAST_DIR="${AISLENS_DATA_DIR}/interim/draft_dependence/satobs-fast/comprehensive"
PARALLEL_DIR="${AISLENS_DATA_DIR}/interim/parameter-tests"
OUTPUT_FILE=""

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 --shelf NAME [OPTIONS]

Required:
  --shelf NAME          Ice shelf name to inspect

Options:
  --param-set SET       Parameter set (default: standard)
  --fast-dir DIR        Fast calculation results directory
  --parallel-dir DIR    Parallel calculation results directory
  --output FILE         Output JSON file path
  -h, --help            Show this help

Examples:
  sbatch $0 --shelf Amery
  sbatch $0 --shelf "Ross Ice Shelf" --param-set permissive
  sbatch $0 --shelf Filchner --fast-dir /path/to/fast --parallel-dir /path/to/parallel
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shelf)
            SHELF="$2"
            shift 2
            ;;
        --param-set)
            PARAM_SET="$2"
            shift 2
            ;;
        --fast-dir)
            FAST_DIR="$2"
            shift 2
            ;;
        --parallel-dir)
            PARALLEL_DIR="$2"
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
# Validate Arguments
# ============================================================================
if [[ -z "$SHELF" ]]; then
    echo "ERROR: --shelf is required"
    print_usage
    exit 1
fi

# Set default output file if not provided
if [[ -z "$OUTPUT_FILE" ]]; then
    # Sanitize shelf name for filename
    SHELF_SANITIZED=$(echo "$SHELF" | tr ' ' '_' | tr -cd '[:alnum:]_')
    OUTPUT_FILE="${AISLENS_DATA_DIR}/processed/visualizations/inspect_${SHELF_SANITIZED}_${PARAM_SET}.json"
fi

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

echo "========================================"
echo "Draft Dependence Inspection Utility"
echo "========================================"
echo "Shelf: $SHELF"
echo "Parameter Set: $PARAM_SET"
echo "Fast Dir: $FAST_DIR"
echo "Parallel Dir: $PARALLEL_DIR"
echo "Output: $OUTPUT_FILE"
echo "========================================"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# ============================================================================
# Run Inspection
# ============================================================================
cd "${AISLENS_REPO}"

python src/scripts/inspect_shelf_pair.py \
    --shelf "$SHELF" \
    --param-set "$PARAM_SET" \
    --fast-dir "$FAST_DIR" \
    --parallel-dir "$PARALLEL_DIR" \
    --out "$OUTPUT_FILE"

echo "========================================"
echo "Inspection complete: $OUTPUT_FILE"
echo "========================================"
