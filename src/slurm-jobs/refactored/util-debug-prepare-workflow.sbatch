#!/bin/bash
#SBATCH --job-name=util-debug-prep
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.util-debug-prep.%j
#SBATCH --error=err.util-debug-prep.%j
#SBATCH --exclusive
#SBATCH --time=30:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

################################################################################
# util-debug-prepare-workflow.sbatch
# ==================================
# Debug utility for the data preparation workflow. Runs a smoke test or
# full debug run of prepare_model_sim with verbose logging.
#
# This is a UTILITY script - not part of the main workflow, but useful for:
#   - Debugging data preparation issues
#   - Smoke testing with reduced data (coarsened, limited time)
#   - Validating workflow changes
#
# Restored from: archive/debug-prepare-workflow.sbatch
#
# Usage:
#   sbatch util-debug-prepare-workflow.sbatch                    # Default smoke test
#   sbatch util-debug-prepare-workflow.sbatch --full             # Full debug run
#   sbatch util-debug-prepare-workflow.sbatch --coarsen 16       # Very coarse for quick test
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration (smoke test settings)
# ============================================================================
START_YEAR=450
END_YEAR=750
COARSEN=8
OUTPUT_DIR=""
DRAFT_DIR=""
FULL_RUN=0

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 [OPTIONS]

Options:
  --start-year YEAR     Start year (default: 450)
  --end-year YEAR       End year (default: 750)
  --coarsen N           Spatial coarsening factor (default: 8 for smoke test)
  --output-dir DIR      Output directory for processed files
  --draft-dir DIR       Output directory for draft dependence files
  --full                Run full workflow (coarsen=2, full time range)
  -h, --help            Show this help

Examples:
  sbatch $0                          # Quick smoke test (coarsen=8)
  sbatch $0 --full                   # Full debug run
  sbatch $0 --coarsen 16 --end-year 460   # Very quick test
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --draft-dir)
            DRAFT_DIR="$2"
            shift 2
            ;;
        --full)
            FULL_RUN=1
            shift
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
# Apply Full Run Settings
# ============================================================================
if [[ $FULL_RUN -eq 1 ]]; then
    COARSEN=2
    START_YEAR=0
    END_YEAR=1000
fi

# ============================================================================
# Set Default Directories
# ============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${AISLENS_DATA_DIR}/processed/debug-workflow-${TIMESTAMP}"
fi
if [[ -z "$DRAFT_DIR" ]]; then
    DRAFT_DIR="${AISLENS_DATA_DIR}/interim/draft_dependence/debug-workflow-${TIMESTAMP}"
fi

# ============================================================================
# Environment Setup
# ============================================================================
echo "Starting debug_prepare_workflow SLURM job on $(hostname) at $(date)"

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

echo "========================================"
echo "Debug Prepare Workflow Utility"
echo "========================================"
echo "Mode: $([ $FULL_RUN -eq 1 ] && echo 'FULL' || echo 'SMOKE TEST')"
echo "Time range: $START_YEAR - $END_YEAR"
echo "Coarsen factor: $COARSEN"
echo "Output directory: $OUTPUT_DIR"
echo "Draft directory: $DRAFT_DIR"
echo "========================================"

# Create directories
mkdir -p "$OUTPUT_DIR" "$DRAFT_DIR"

# ============================================================================
# Run Debug Workflow
# ============================================================================
cd "${AISLENS_REPO}/src/scripts"

python debug_prepare_workflow.py \
    --start-year "$START_YEAR" \
    --end-year "$END_YEAR" \
    --coarsen "$COARSEN" \
    --output-dir "$OUTPUT_DIR" \
    --draft-dir "$DRAFT_DIR" \
    2>&1 | tee "${OUTPUT_DIR}/debug_workflow.log"

echo "========================================"
echo "Debug workflow complete"
echo "Output: $OUTPUT_DIR"
echo "Log: ${OUTPUT_DIR}/debug_workflow.log"
echo "========================================"
