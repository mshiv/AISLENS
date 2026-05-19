#!/bin/bash
#SBATCH --job-name=util-thickness-analysis
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1 --ntasks-per-node=4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=out.thickness-analysis.%j
#SBATCH --error=err.thickness-analysis.%j
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

################################################################################
# util-thickness-analysis.sbatch
# ==============================
# Analyze ensemble thickness variability. Computes statistics across ensemble
# members and generates diagnostic plots.
#
# This is a UTILITY script for specialized analysis:
#   - Ensemble thickness variability statistics
#   - Time evolution of thickness spread
#   - Diagnostic plots for ensemble behavior
#
# Restored from: archive/aislens_mali_thickness_analysis.sbatch
#
# Usage:
#   sbatch util-thickness-analysis.sbatch --scenario CTRL --ensemble CTRL-SSN
#   sbatch util-thickness-analysis.sbatch --scenario SSP585 --time-indices "120 240 360"
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
SCENARIO=""
ENSEMBLE_NAME=""
TIME_INDICES="120 240 -1"  # -1 = last timestep
OUTPUT_DIR=""
EXPERIMENT_LIST=""

# Python script
SCRIPT="${AISLENS_REPO}/src/MPAS-Tools/plot_ensemble_thickness_analysis.py"

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 --scenario SCENARIO [OPTIONS]

Required:
  --scenario SCENARIO   CTRL, SSP585, or SSP126

Options:
  --ensemble NAME       Ensemble name (default: derived from scenario)
  --experiments LIST    Comma-separated experiment list
  --time-indices IDX    Space-separated time indices (default: "120 240 -1")
  --output-dir DIR      Output directory for plots
  -h, --help            Show this help

Examples:
  sbatch $0 --scenario CTRL --ensemble CTRL-SSN
  sbatch $0 --scenario SSP585 --time-indices "50 100 150 200"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)
            SCENARIO="${2^^}"
            shift 2
            ;;
        --ensemble)
            ENSEMBLE_NAME="$2"
            shift 2
            ;;
        --experiments)
            EXPERIMENT_LIST="$2"
            shift 2
            ;;
        --time-indices)
            TIME_INDICES="$2"
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
# Validate and Set Defaults
# ============================================================================
if [[ -z "$SCENARIO" ]]; then
    echo "ERROR: --scenario is required"
    print_usage
    exit 1
fi

# Set ensemble name from scenario if not provided
if [[ -z "$ENSEMBLE_NAME" ]]; then
    case "$SCENARIO" in
        CTRL)
            ENSEMBLE_NAME="CTRL-SSN"
            ;;
        SSP585)
            ENSEMBLE_NAME="SSP585-SSNTRND"
            ;;
        SSP126)
            ENSEMBLE_NAME="SSP126-SSNTRND"
            ;;
    esac
fi

# Set data root and output directories
DATA_ROOT="${AISLENS_DATA_DIR}/MALI/ENSEMBLES/${SCENARIO}"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${DATA_ROOT}/${ENSEMBLE_NAME}/thickness_analysis"
fi

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis

echo "========================================"
echo "Thickness Analysis Utility"
echo "========================================"
echo "Scenario: $SCENARIO"
echo "Ensemble: $ENSEMBLE_NAME"
echo "Time indices: $TIME_INDICES"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Discover Experiments
# ============================================================================
if [[ -z "$EXPERIMENT_LIST" ]]; then
    # Auto-discover experiment directories
    EXPERIMENTS=($(find "${DATA_ROOT}" -maxdepth 2 -type d -name "output" -exec dirname {} \; | sort))
    EXPERIMENT_LIST=$(IFS=,; echo "${EXPERIMENTS[*]}")
fi

echo "Experiments: $EXPERIMENT_LIST"

# ============================================================================
# Run Analysis
# ============================================================================
cd "${AISLENS_REPO}/src/MPAS-Tools"

# Check if the analysis script exists
if [[ -f "$SCRIPT" ]]; then
    python "$SCRIPT" \
        --data-root "$DATA_ROOT" \
        --ensemble "$ENSEMBLE_NAME" \
        --experiments "$EXPERIMENT_LIST" \
        --time-indices $TIME_INDICES \
        --output-dir "$OUTPUT_DIR"
else
    echo "WARNING: Analysis script not found: $SCRIPT"
    echo "Attempting alternative analysis..."
    
    # Fall back to basic analysis with plot_ensemble_maps.py
    python plot_ensemble_maps.py \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --variable thickness
fi

echo "========================================"
echo "Analysis complete: $OUTPUT_DIR"
echo "========================================"
