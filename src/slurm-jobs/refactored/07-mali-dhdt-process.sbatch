#!/bin/bash
#SBATCH --job-name=mali-dhdt
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.mali-dhdt.%A_%a
#SBATCH --error=err.mali-dhdt.%A_%a
#SBATCH --time=8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smurugan9@gatech.edu

################################################################################
# 07-mali-dhdt-process.sbatch
# ===========================
# Process MALI output to compute dH/dt (thickness change rates) and optionally
# create animations. Supports array jobs for parallel ensemble processing.
#
# Consolidates:
#   - aislens_mali_dhdt_processing_array_updated.sbatch
#   - process_dhdt_array.sbatch
#   - aislens_dhdt_processing_animation.sbatch
#
# Usage:
#   # Process dH/dt for a scenario:
#   sbatch 07-mali-dhdt-process.sbatch --scenario SSP585 --mode process
#
#   # Create animation:
#   sbatch 07-mali-dhdt-process.sbatch --scenario CTRL --mode animate
#
#   # Both processing and animation:
#   sbatch 07-mali-dhdt-process.sbatch --scenario SSP585 --mode all
#
#   # Array job for multiple directories:
#   sbatch --array=0-9 07-mali-dhdt-process.sbatch --scenario SSP585 --mode process
#
#   # Dry run to discover directories:
#   bash 07-mali-dhdt-process.sbatch --scenario SSP585 --dry-run
#
# Arguments:
#   --scenario SCENARIO   Required: CTRL, SSP585, or SSP126
#   --mode MODE           Processing mode: process|animate|all (default: process)
#   --base-dir DIR        Override base data directory
#   --fig-dir DIR         Override figures directory
#   --array-index N       Override SLURM_ARRAY_TASK_ID
#   --min-year-diff N     Minimum year difference for dH/dt (default: 2)
#   --dry-run             List directories and exit
#   --no-resubmit         Process all sequentially (no array)
#
# Workflow:
#   1. Find all output_flux_all_timesteps_YYYY.nc files
#   2. Filter for 5-year intervals after 2000
#   3. Create time-averaged mesh files (ncks)
#   4. Calculate thickness difference (ncdiff)
#   5. Compute dH/dt = thickness_diff / N_years (ncap2)
#   6. Generate plots with consistent colorbar (Python)
#   7. Optionally create animation (ffmpeg)
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
SCENARIO=""
MODE="process"
BASE_DATA_PARENT=""
BASE_FIGURES_PARENT=""
ARRAY_INDEX=""
MIN_YEAR_DIFF=2
DRY_RUN=0
NO_RESUBMIT=0
RESUBMIT=0

# Python scripts
PYTHON_PLOT_SCRIPT="${AISLENS_REPO}/src/MPAS-Tools/plot_output_maps_masked.py"
PYTHON_ANIM_SCRIPT="${AISLENS_REPO}/src/MPAS-Tools/plot_output_maps_masked_animation.py"

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 --scenario SCENARIO [OPTIONS]

Required:
  --scenario SCENARIO   One of: CTRL, SSP585, SSP126

Options:
  --mode MODE           process|animate|all (default: process)
  --base-dir DIR        Override base data directory
  --fig-dir DIR         Override figures directory
  --array-index N       Process specific array index
  --min-year-diff N     Minimum year difference for dH/dt (default: 2)
  --dry-run             List directories and exit
  --no-resubmit         Run sequentially (no array job)
  --resubmit            Auto-submit as array job
  -h, --help            Show this help

Examples:
  sbatch $0 --scenario SSP585 --mode process
  sbatch $0 --scenario CTRL --mode animate --fig-dir /custom/figures
  sbatch --array=0-9 $0 --scenario SSP585 --mode all
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)
            SCENARIO="${2^^}"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DATA_PARENT="$2"
            shift 2
            ;;
        --fig-dir)
            BASE_FIGURES_PARENT="$2"
            shift 2
            ;;
        --array-index)
            ARRAY_INDEX="$2"
            shift 2
            ;;
        --min-year-diff)
            MIN_YEAR_DIFF="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-resubmit)
            NO_RESUBMIT=1
            shift
            ;;
        --resubmit)
            RESUBMIT=1
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
if [[ -z "$SCENARIO" ]]; then
    echo "ERROR: --scenario is required"
    print_usage
    exit 1
fi

case "$SCENARIO" in
    CTRL|SSP585|SSP126)
        ;;
    *)
        echo "ERROR: Invalid scenario: $SCENARIO. Must be CTRL, SSP585, or SSP126"
        exit 1
        ;;
esac

case "$MODE" in
    process|animate|all)
        ;;
    *)
        echo "ERROR: Invalid mode: $MODE. Must be process, animate, or all"
        exit 1
        ;;
esac

# ============================================================================
# Set Paths Based on Scenario
# ============================================================================
if [[ -z "$BASE_DATA_PARENT" ]]; then
    BASE_DATA_PARENT="${AISLENS_DATA_DIR}/MALI/ENSEMBLES/${SCENARIO}"
fi

if [[ -z "$BASE_FIGURES_PARENT" ]]; then
    BASE_FIGURES_PARENT="${AISLENS_DATA_DIR}/figures/MALI/ENSEMBLES/${SCENARIO}"
fi

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis

LOG_FILE="${LOG_FILE:-mali_dhdt_${SCENARIO}_$(date +%Y%m%d_%H%M%S).log}"

echo "========================================"
echo "MALI dH/dt Processing"
echo "========================================"
echo "Scenario: $SCENARIO"
echo "Mode: $MODE"
echo "Data directory: $BASE_DATA_PARENT"
echo "Figures directory: $BASE_FIGURES_PARENT"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-${ARRAY_INDEX:-N/A}}"
echo "========================================"

# ============================================================================
# Discover Output Directories
# ============================================================================
discover_output_dirs() {
    local base_dir="$1"
    local -a dirs=()
    
    while IFS= read -r outdir; do
        [[ -n "$outdir" ]] && dirs+=("$outdir")
    done < <(find "$base_dir" -type d -name "output" 2>/dev/null | sort)
    
    printf '%s\n' "${dirs[@]}"
}

mapfile -t OUTPUT_DIRS < <(discover_output_dirs "$BASE_DATA_PARENT")
NUM_DIRS=${#OUTPUT_DIRS[@]}

if [[ $NUM_DIRS -eq 0 ]]; then
    echo "ERROR: No output directories found under $BASE_DATA_PARENT"
    exit 1
fi

echo "Found $NUM_DIRS output directories"

# ============================================================================
# Dry Run Mode
# ============================================================================
if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "Discovered directories:"
    for i in "${!OUTPUT_DIRS[@]}"; do
        echo "  [$i] ${OUTPUT_DIRS[$i]}"
    done
    echo ""
    echo "To process all as array job:"
    echo "  sbatch --array=0-$((NUM_DIRS-1)) $0 --scenario $SCENARIO --mode $MODE"
    exit 0
fi

# ============================================================================
# Array Job Handling
# ============================================================================
IDX="${SLURM_ARRAY_TASK_ID:-${ARRAY_INDEX:-}}"

if [[ -z "$IDX" ]]; then
    if [[ $RESUBMIT -eq 1 ]]; then
        echo "Resubmitting as array job..."
        sbatch --array=0-$((NUM_DIRS-1)) "$0" --scenario "$SCENARIO" --mode "$MODE" \
            --base-dir "$BASE_DATA_PARENT" --fig-dir "$BASE_FIGURES_PARENT"
        exit 0
    elif [[ $NO_RESUBMIT -eq 1 ]]; then
        echo "Processing all $NUM_DIRS directories sequentially..."
        for i in "${!OUTPUT_DIRS[@]}"; do
            IDX=$i
            OUTPUT_DIR="${OUTPUT_DIRS[$IDX]}"
            echo "Processing [$i] $OUTPUT_DIR"
            # Process inline (see below)
        done
        exit 0
    else
        echo "ERROR: Not running as array job. Use --resubmit, --no-resubmit, or --array-index"
        exit 1
    fi
fi

if [[ $IDX -ge $NUM_DIRS ]]; then
    echo "ERROR: Array index $IDX exceeds available directories ($NUM_DIRS)"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIRS[$IDX]}"
echo "Processing directory [$IDX]: $OUTPUT_DIR"

# Derive figure output path
RELATIVE_PATH="${OUTPUT_DIR#$BASE_DATA_PARENT/}"
FIG_OUTPUT_DIR="${BASE_FIGURES_PARENT}/${RELATIVE_PATH}"
mkdir -p "$FIG_OUTPUT_DIR"

# ============================================================================
# Processing Functions
# ============================================================================
process_dhdt() {
    local out_dir="$1"
    local fig_dir="$2"
    
    echo "Computing dH/dt for: $out_dir"
    
    # Find all output files
    mapfile -t OUTPUT_FILES < <(ls "$out_dir"/output_flux_all_timesteps_*.nc 2>/dev/null | sort)
    
    if [[ ${#OUTPUT_FILES[@]} -lt 2 ]]; then
        echo "WARN: Need at least 2 output files for dH/dt calculation"
        return 1
    fi
    
    # Extract years from filenames
    declare -A YEAR_FILES
    for f in "${OUTPUT_FILES[@]}"; do
        year=$(basename "$f" | sed -n 's/output_flux_all_timesteps_\([0-9]*\)\.nc/\1/p')
        if [[ -n "$year" ]]; then
            YEAR_FILES[$year]="$f"
        fi
    done
    
    YEARS=($(printf '%s\n' "${!YEAR_FILES[@]}" | sort -n))
    echo "Found years: ${YEARS[*]}"
    
    # Process consecutive 5-year pairs after year 2000
    for i in "${!YEARS[@]}"; do
        [[ $i -eq 0 ]] && continue
        
        year1="${YEARS[$((i-1))]}"
        year2="${YEARS[$i]}"
        
        [[ $year1 -lt 2000 ]] && continue
        
        year_diff=$((year2 - year1))
        [[ $year_diff -lt $MIN_YEAR_DIFF ]] && continue
        
        file1="${YEAR_FILES[$year1]}"
        file2="${YEAR_FILES[$year2]}"
        
        echo "Processing dH/dt: $year1 -> $year2 (${year_diff} years)"
        
        # Output file
        dhdt_file="${out_dir}/output_flux_all_timesteps_dhdt_${year1}_${year2}.nc"
        
        if [[ -f "$dhdt_file" ]]; then
            echo "  dH/dt file already exists: $dhdt_file"
            continue
        fi
        
        # Create temporary diff file
        tmp_diff=$(mktemp)
        
        # Calculate thickness difference
        ncdiff -O -v thickness "$file2" "$file1" "$tmp_diff"
        
        # Compute dH/dt
        ncap2 -O -s "dHdt=thickness/${year_diff}.0" "$tmp_diff" "$dhdt_file"
        
        # Clean up
        rm -f "$tmp_diff"
        
        echo "  Created: $dhdt_file"
    done
}

create_animation() {
    local out_dir="$1"
    local fig_dir="$2"
    
    echo "Creating animation for: $out_dir"
    
    cd "${AISLENS_REPO}/src/MPAS-Tools"
    
    python "$PYTHON_ANIM_SCRIPT" \
        --input-dir "$out_dir" \
        --output-dir "$fig_dir" \
        --variable dHdt
}

# ============================================================================
# Execute Based on Mode
# ============================================================================
case "$MODE" in
    process)
        process_dhdt "$OUTPUT_DIR" "$FIG_OUTPUT_DIR"
        ;;
    animate)
        create_animation "$OUTPUT_DIR" "$FIG_OUTPUT_DIR"
        ;;
    all)
        process_dhdt "$OUTPUT_DIR" "$FIG_OUTPUT_DIR"
        create_animation "$OUTPUT_DIR" "$FIG_OUTPUT_DIR"
        ;;
esac

echo "========================================"
echo "Completed at $(date)"
echo "========================================"
