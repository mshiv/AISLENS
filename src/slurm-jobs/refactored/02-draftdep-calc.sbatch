#!/bin/bash
#SBATCH --job-name=draftdep-calc
#SBATCH --account=${SLURM_ACCOUNT:-gts-arobel3-atlas}
#SBATCH --nodes=2 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.draftdep-calc.%j
#SBATCH --error=err.draftdep-calc.%j
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL:-smurugan9@gatech.edu}

################################################################################
# 02-draftdep-calc.sbatch
# =======================
# Calculate, visualize, and compare draft dependence parameters.
#
# Consolidates:
#   - aislens_calc_draft-depen-comprehensive-fast.sbatch
#   - aislens_visualize_draft-depen-comprehensive.sbatch
#   - aislens_visualize_draft-depen-comprehensive_compare-param-sets.sbatch
#
# Usage:
#   # Calculate draft dependence parameters:
#   sbatch 02-draftdep-calc.sbatch --mode calc
#
#   # Visualize results:
#   sbatch 02-draftdep-calc.sbatch --mode viz
#
#   # Compare parameter sets:
#   sbatch 02-draftdep-calc.sbatch --mode compare --param-sets "permissive,conservative"
#
#   # Full workflow (calc + viz):
#   sbatch 02-draftdep-calc.sbatch --mode all
#
# Arguments:
#   --mode MODE           calc|viz|compare|all (default: calc)
#   --interim-dir DIR     Interim data directory
#   --processed-dir DIR   Processed data directory
#   --no-skip-existing    Recalculate even if output exists
#   --param-sets SETS     Comma-separated list of parameter sets to compare
#   --plot-all-shelves    Generate plots for all individual shelves
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
MODE="calc"
INTERIM_DIR="${AISLENS_DATA_DIR}/interim"
PROCESSED_DIR="${AISLENS_DATA_DIR}/processed"
NO_SKIP_EXISTING=false
PARAM_SETS="permissive"
PLOT_ALL_SHELVES=false
VIZ_SATOBS=false
VIZ_MODEL=false

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 [OPTIONS]

Options:
  --mode MODE           calc|viz|compare|all (default: calc)
  --interim-dir DIR     Interim directory (default: \${AISLENS_DATA_DIR}/interim)
  --processed-dir DIR   Processed directory (default: \${AISLENS_DATA_DIR}/processed)
  --no-skip-existing    Recalculate existing outputs
  --param-sets SETS     Comma-separated parameter sets (e.g., "permissive,conservative")
  --plot-all-shelves    Generate all individual shelf plots
  --satobs              Include satellite observations in visualization
  --model               Include model data in visualization
  -h, --help            Show this help

Modes:
  calc     - Calculate draft dependence parameters
  viz      - Visualize draft dependence results
  compare  - Compare multiple parameter sets
  all      - Run calc + viz

Examples:
  sbatch $0 --mode calc
  sbatch $0 --mode viz --satobs --model
  sbatch $0 --mode compare --param-sets "permissive,conservative,optimal_v2"
  sbatch $0 --mode all --plot-all-shelves
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --interim-dir)
            INTERIM_DIR="$2"
            shift 2
            ;;
        --processed-dir)
            PROCESSED_DIR="$2"
            shift 2
            ;;
        --no-skip-existing)
            NO_SKIP_EXISTING=true
            shift
            ;;
        --param-sets)
            PARAM_SETS="$2"
            shift 2
            ;;
        --plot-all-shelves)
            PLOT_ALL_SHELVES=true
            shift
            ;;
        --satobs)
            VIZ_SATOBS=true
            shift
            ;;
        --model)
            VIZ_MODEL=true
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
# Validate Arguments
# ============================================================================
case "$MODE" in
    calc|viz|compare|all)
        ;;
    *)
        echo "ERROR: Invalid mode: $MODE"
        exit 1
        ;;
esac

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis

echo "========================================"
echo "Draft Dependence Processing"
echo "========================================"
echo "Mode: $MODE"
echo "Interim directory: $INTERIM_DIR"
echo "Processed directory: $PROCESSED_DIR"
echo "========================================"

cd "${AISLENS_REPO}/src/scripts"

# ============================================================================
# Functions
# ============================================================================
run_calc() {
    echo "[$(date)] Running draft dependence calculation..."
    
    CMD="python calculate_draft_dependence_comprehensive_fast.py"
    CMD+=" --interim-dir $INTERIM_DIR"
    CMD+=" --processed-dir $PROCESSED_DIR"
    
    [[ "$NO_SKIP_EXISTING" == "true" ]] && CMD+=" --no-skip-existing"
    
    echo "Running: $CMD"
    eval $CMD
}

run_viz() {
    echo "[$(date)] Running draft dependence visualization..."
    
    CMD="python visualize_draft_dependence_optimized.py"
    
    # Default paths if not specified
    SATOBS_PATH="${PROCESSED_DIR}/satobs-fast/satobs-fast_dedraft_BM_2_5-residuals.nc"
    MODEL_PATH="${PROCESSED_DIR}/model-fast/model-fast_dedraft_BM_2_5-residuals.nc"
    
    CMD+=" --satobs-residuals $SATOBS_PATH"
    CMD+=" --model-residuals $MODEL_PATH"
    CMD+=" --output-dir ${AISLENS_DATA_DIR}/figures/draft_dependence"
    
    [[ "$PLOT_ALL_SHELVES" == "true" ]] && CMD+=" --plot-all-shelves"
    
    echo "Running: $CMD"
    eval $CMD
}

run_compare() {
    echo "[$(date)] Comparing parameter sets: $PARAM_SETS..."
    
    CMD="python compare_parameter_sets.py"
    CMD+=" --parameter-sets $PARAM_SETS"
    
    [[ "$PLOT_ALL_SHELVES" == "true" ]] && CMD+=" --plot-all-shelves"
    
    echo "Running: $CMD"
    eval $CMD
}

# ============================================================================
# Execute Based on Mode
# ============================================================================
case "$MODE" in
    calc)
        run_calc
        ;;
    viz)
        run_viz
        ;;
    compare)
        run_compare
        ;;
    all)
        run_calc
        run_viz
        ;;
esac

echo "========================================"
echo "Completed at $(date)"
echo "========================================"
