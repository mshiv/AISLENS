#!/bin/bash
#SBATCH --job-name=viz-mali
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=8G
#SBATCH --output=out.viz-mali.%A_%a
#SBATCH --error=err.viz-mali.%A_%a
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smurugan9@gatech.edu

################################################################################
# 08-viz-mali.sbatch
# ==================
# Unified visualization script for MALI outputs. Supports variable maps,
# ratio plots, dH/dt arrays, and ensemble statistics.
#
# Consolidates:
#   - aislens_plot_mali_outputs.sbatch
#   - aislens_plot_dhdt_ratio.sbatch
#   - plot_dhdt_array.sbatch
#
# Usage:
#   # Plot standard variables:
#   sbatch 08-viz-mali.sbatch --type output --scenario CTRL --vars thickness,dHdt
#
#   # Plot ratio maps:
#   sbatch 08-viz-mali.sbatch --type ratio --scenario SSP585
#
#   # Plot dH/dt array:
#   sbatch 08-viz-mali.sbatch --type dhdt --scenario SSP585 --base-dir /path/to/data
#
#   # Array job for ensembles:
#   sbatch --array=0-9 08-viz-mali.sbatch --type output --scenario CTRL
#
# Arguments:
#   --type TYPE           output|ratio|dhdt|ensemble (required)
#   --scenario SCENARIO   CTRL, SSP585, SSP126
#   --vars VARS           Comma-separated variables (default: thickness,dHdt)
#   --base-dir DIR        Override data directory
#   --fig-dir DIR         Override figures directory
#   --array-index N       Override SLURM_ARRAY_TASK_ID
#   --dry-run             List targets and exit
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
PLOT_TYPE=""
SCENARIO=""
VARIABLES="thickness,dHdt,surfaceSpeed,floatingBasalMassBalApplied"
BASE_DATA_DIR=""
BASE_FIG_DIR=""
ARRAY_INDEX=""
DRY_RUN=0

# Statistics defaults
NUMERATORS="thickness_range"
DENOMINATORS="abs_thickness_change"

# Python scripts
SCRIPT_VAR_MAPS="${AISLENS_REPO}/src/MPAS-Tools/plot_var_maps.py"
SCRIPT_OUTPUT_MAPS="${AISLENS_REPO}/src/MPAS-Tools/plot_output_maps_masked.py"
SCRIPT_ENSEMBLE_MAPS="${AISLENS_REPO}/src/MPAS-Tools/plot_ensemble_maps.py"

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 --type TYPE --scenario SCENARIO [OPTIONS]

Required:
  --type TYPE           Plot type: output|ratio|dhdt|ensemble
  --scenario SCENARIO   CTRL, SSP585, or SSP126

Options:
  --vars VARS           Comma-separated variables (default: thickness,dHdt,...)
  --base-dir DIR        Override base data directory
  --fig-dir DIR         Override figures directory
  --array-index N       Process specific array index
  --numerators VARS     Numerator variables for ratio (default: thickness_range)
  --denominators VARS   Denominator variables for ratio (default: abs_thickness_change)
  --dry-run             List targets and exit
  -h, --help            Show this help

Examples:
  sbatch $0 --type output --scenario CTRL --vars thickness
  sbatch --array=0-9 $0 --type output --scenario SSP585
  sbatch $0 --type ratio --scenario SSP585
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)
            PLOT_TYPE="$2"
            shift 2
            ;;
        --scenario)
            SCENARIO="${2^^}"
            shift 2
            ;;
        --vars)
            VARIABLES="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DATA_DIR="$2"
            shift 2
            ;;
        --fig-dir)
            BASE_FIG_DIR="$2"
            shift 2
            ;;
        --array-index)
            ARRAY_INDEX="$2"
            shift 2
            ;;
        --numerators)
            NUMERATORS="$2"
            shift 2
            ;;
        --denominators)
            DENOMINATORS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
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
if [[ -z "$PLOT_TYPE" ]]; then
    echo "ERROR: --type is required"
    print_usage
    exit 1
fi

if [[ -z "$SCENARIO" ]]; then
    echo "ERROR: --scenario is required"
    print_usage
    exit 1
fi

case "$PLOT_TYPE" in
    output|ratio|dhdt|ensemble)
        ;;
    *)
        echo "ERROR: Invalid type: $PLOT_TYPE"
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
if [[ -z "$BASE_DATA_DIR" ]]; then
    BASE_DATA_DIR="${AISLENS_DATA_DIR}/MALI/ENSEMBLES/${SCENARIO}"
fi

if [[ -z "$BASE_FIG_DIR" ]]; then
    BASE_FIG_DIR="${AISLENS_DATA_DIR}/figures/MALI/ENSEMBLES/${SCENARIO}"
fi

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis

cd "${AISLENS_REPO}/src/MPAS-Tools"

echo "========================================"
echo "MALI Visualization"
echo "========================================"
echo "Type: $PLOT_TYPE"
echo "Scenario: $SCENARIO"
echo "Variables: $VARIABLES"
echo "Data directory: $BASE_DATA_DIR"
echo "Figures directory: $BASE_FIG_DIR"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "========================================"

# ============================================================================
# Plot Functions
# ============================================================================
plot_output_vars() {
    local data_dir="$1"
    local fig_dir="$2"
    local vars="$3"
    
    echo "Plotting output variables: $vars"
    
    IFS=',' read -ra VAR_ARRAY <<< "$vars"
    for var in "${VAR_ARRAY[@]}"; do
        echo "  Variable: $var"
        python3 "$SCRIPT_VAR_MAPS" -r "$data_dir" -v "$var" --output-dir "$fig_dir"
    done
}

plot_ratio_maps() {
    local stats_dir="$1"
    local fig_dir="$2"
    
    echo "Plotting ratio maps"
    echo "  Numerators: $NUMERATORS"
    echo "  Denominators: $DENOMINATORS"
    
    python3 "$SCRIPT_ENSEMBLE_MAPS" \
        --stats-dir "$stats_dir" \
        --output-dir "$fig_dir" \
        --numerators "$NUMERATORS" \
        --denominators "$DENOMINATORS" \
        --ratio-mode
}

plot_dhdt_maps() {
    local data_dir="$1"
    local fig_dir="$2"
    
    echo "Plotting dH/dt maps"
    
    # Find all dH/dt files
    mapfile -t DHDT_FILES < <(find "$data_dir" -name "output_flux_all_timesteps_dhdt_*.nc" | sort)
    
    for f in "${DHDT_FILES[@]}"; do
        echo "  Processing: $(basename "$f")"
        python3 "$SCRIPT_OUTPUT_MAPS" -f "$f" -v dHdt --output-dir "$fig_dir"
    done
}

plot_ensemble_stats() {
    local stats_dir="$1"
    local fig_dir="$2"
    
    echo "Plotting ensemble statistics"
    
    python3 "$SCRIPT_ENSEMBLE_MAPS" \
        --stats-dir "$stats_dir" \
        --output-dir "$fig_dir"
}

# ============================================================================
# Discover Targets
# ============================================================================
discover_output_dirs() {
    find "$1" -type d -name "output" 2>/dev/null | sort
}

mapfile -t OUTPUT_DIRS < <(discover_output_dirs "$BASE_DATA_DIR")
NUM_DIRS=${#OUTPUT_DIRS[@]}

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "Discovered directories:"
    for i in "${!OUTPUT_DIRS[@]}"; do
        echo "  [$i] ${OUTPUT_DIRS[$i]}"
    done
    echo ""
    echo "Total: $NUM_DIRS directories"
    exit 0
fi

# ============================================================================
# Array Job Handling
# ============================================================================
IDX="${SLURM_ARRAY_TASK_ID:-${ARRAY_INDEX:-0}}"

if [[ $NUM_DIRS -eq 0 ]]; then
    echo "WARN: No output directories found"
    # For ratio/ensemble, may not need output dirs
fi

if [[ $NUM_DIRS -gt 0 && $IDX -lt $NUM_DIRS ]]; then
    CURRENT_DIR="${OUTPUT_DIRS[$IDX]}"
    RELATIVE_PATH="${CURRENT_DIR#$BASE_DATA_DIR/}"
    FIG_OUTPUT="${BASE_FIG_DIR}/${RELATIVE_PATH}"
    mkdir -p "$FIG_OUTPUT"
else
    CURRENT_DIR="$BASE_DATA_DIR"
    FIG_OUTPUT="$BASE_FIG_DIR"
    mkdir -p "$FIG_OUTPUT"
fi

echo "Processing: $CURRENT_DIR"
echo "Output to: $FIG_OUTPUT"

# ============================================================================
# Execute Based on Type
# ============================================================================
case "$PLOT_TYPE" in
    output)
        plot_output_vars "$CURRENT_DIR" "$FIG_OUTPUT" "$VARIABLES"
        ;;
    ratio)
        STATS_DIR="${BASE_DATA_DIR}/ensemble_stats"
        plot_ratio_maps "$STATS_DIR" "$FIG_OUTPUT"
        ;;
    dhdt)
        plot_dhdt_maps "$CURRENT_DIR" "$FIG_OUTPUT"
        ;;
    ensemble)
        STATS_DIR="${BASE_DATA_DIR}/ensemble_stats"
        plot_ensemble_stats "$STATS_DIR" "$FIG_OUTPUT"
        ;;
esac

echo "========================================"
echo "Completed at $(date)"
echo "========================================"
