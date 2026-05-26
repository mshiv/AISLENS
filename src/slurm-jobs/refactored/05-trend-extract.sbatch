#!/bin/bash
#SBATCH --job-name=trend-extract
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=2 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.trend-extract.%A_%a
#SBATCH --error=err.trend-extract.%A_%a
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smurugan9@gatech.edu

################################################################################
# 05-trend-extract.sbatch
# =======================
# Extract trend components from MALI output or forcing data.
# Supports multiple methods: from merged output, NCO-based, or single file.
#
# Consolidates:
#   - aislens-extract-trend-from-merged-array.sbatch
#   - extract-trend-forcing.sbatch
#   - calc-trends-nco.sbatch
#   - aislens-create-forcing-trend-array.sbatch
#
# Usage:
#   # Extract from merged MALI output:
#   sbatch 05-trend-extract.sbatch --scenario SSP585 --method merged
#
#   # NCO-based linear trend:
#   sbatch 05-trend-extract.sbatch --scenario SSP585 --method nco --input file.nc
#
#   # Single file extraction:
#   sbatch 05-trend-extract.sbatch --method single --input file.nc --output trend.nc
#
# Arguments:
#   --scenario SCENARIO   CTRL, SSP585, SSP126 (for merged method)
#   --method METHOD       merged|nco|single|dask (default: merged)
#   --input FILE/DIR      Input file or directory
#   --output FILE/DIR     Output file or directory
#   --region N            Region ID (for array jobs)
#   --array-index N       Override SLURM_ARRAY_TASK_ID
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
SCENARIO=""
METHOD="merged"
INPUT_PATH=""
OUTPUT_PATH=""
REGION=""
ARRAY_INDEX=""
START_REGION=33
END_REGION=132

# Python scripts
SCRIPT_EXTRACT_TREND="${AISLENS_REPO}/src/scripts/extract_draft_and_trend_from_merged.py"
SCRIPT_TREND_DASK="${AISLENS_REPO}/src/scripts/create_trend_component_dask.py"
SCRIPT_TREND_MALI="${AISLENS_REPO}/src/scripts/extract_draft_and_trend_mali.py"

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 [OPTIONS]

Options:
  --scenario SCENARIO   CTRL, SSP585, or SSP126 (for merged method)
  --method METHOD       merged|nco|single|dask (default: merged)
  --input FILE/DIR      Input file or directory
  --output FILE/DIR     Output file or directory
  --region N            Region ID for single region processing
  --start-region N      Start region for array (default: 33)
  --end-region N        End region for array (default: 132)
  --array-index N       Override SLURM_ARRAY_TASK_ID
  -h, --help            Show this help

Examples:
  sbatch $0 --scenario SSP585 --method merged
  sbatch --array=0-99 $0 --scenario SSP585 --method merged
  sbatch $0 --method nco --input data.nc --output trend.nc
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)
            SCENARIO="${2^^}"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --input)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --start-region)
            START_REGION="$2"
            shift 2
            ;;
        --end-region)
            END_REGION="$2"
            shift 2
            ;;
        --array-index)
            ARRAY_INDEX="$2"
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
case "$METHOD" in
    merged|nco|single|dask)
        ;;
    *)
        echo "ERROR: Invalid method: $METHOD"
        exit 1
        ;;
esac

if [[ "$METHOD" == "merged" && -z "$SCENARIO" ]]; then
    echo "ERROR: --scenario required for merged method"
    exit 1
fi

if [[ "$METHOD" =~ ^(nco|single)$ && -z "$INPUT_PATH" ]]; then
    echo "ERROR: --input required for $METHOD method"
    exit 1
fi

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

echo "========================================"
echo "Trend Extraction"
echo "========================================"
echo "Method: $METHOD"
echo "Scenario: ${SCENARIO:-N/A}"
echo "Input: ${INPUT_PATH:-auto}"
echo "Output: ${OUTPUT_PATH:-auto}"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "========================================"

# ============================================================================
# Method: Merged Output Processing
# ============================================================================
extract_from_merged() {
    local scenario="$1"
    local region="${2:-}"
    
    local merged_dir="${AISLENS_DATA_DIR}/MALI/ENSEMBLES/${scenario}/merged_output"
    local output_dir="${AISLENS_DATA_DIR}/processed/MALI_forcing_trends/${scenario}"
    mkdir -p "$output_dir"
    
    cd "${AISLENS_REPO}/src/scripts"
    
    if [[ -n "$region" ]]; then
        echo "Processing region: $region"
        python "$SCRIPT_EXTRACT_TREND" \
            --input-dir "$merged_dir" \
            --output-dir "$output_dir" \
            --region "$region"
    else
        echo "Processing all regions ($START_REGION - $END_REGION)"
        python "$SCRIPT_EXTRACT_TREND" \
            --input-dir "$merged_dir" \
            --output-dir "$output_dir" \
            --start-region "$START_REGION" \
            --end-region "$END_REGION"
    fi
}

# ============================================================================
# Method: NCO-based Trend
# ============================================================================
extract_nco() {
    local input="$1"
    local output="${2:-${input%.nc}_trend.nc}"
    
    echo "Computing NCO-based linear trend..."
    echo "Input: $input"
    echo "Output: $output"
    
    # Create temporary files
    local tmp_first=$(mktemp --suffix=.nc)
    local tmp_last=$(mktemp --suffix=.nc)
    local tmp_diff=$(mktemp --suffix=.nc)
    
    # Get number of time steps
    local n_time=$(ncdump -h "$input" | grep -oP 'Time = \K[0-9]+')
    local last_idx=$((n_time - 1))
    
    # Extract first and last time steps
    ncks -O -d Time,0 "$input" "$tmp_first"
    ncks -O -d Time,$last_idx "$input" "$tmp_last"
    
    # Compute difference
    ncdiff -O "$tmp_last" "$tmp_first" "$tmp_diff"
    
    # Normalize by time
    ncap2 -O -s "floatingBasalMassBalAdjustment=floatingBasalMassBalAdjustment/${n_time}.0" \
        "$tmp_diff" "$output"
    
    # Cleanup
    rm -f "$tmp_first" "$tmp_last" "$tmp_diff"
    
    echo "Trend extraction complete: $output"
}

# ============================================================================
# Method: Single File Extraction
# ============================================================================
extract_single() {
    local input="$1"
    local output="${2:-${input%.nc}_trend.nc}"
    
    cd "${AISLENS_REPO}/src/scripts"
    
    python "$SCRIPT_TREND_MALI" \
        --input "$input" \
        --output "$output"
}

# ============================================================================
# Method: Dask-based (Large Files)
# ============================================================================
extract_dask() {
    local input="${1:-$INPUT_PATH}"
    local output="${2:-$OUTPUT_PATH}"
    
    cd "${AISLENS_REPO}/src/scripts"
    
    python "$SCRIPT_TREND_DASK" \
        --input "$input" \
        --output "$output"
}

# ============================================================================
# Array Job Handling
# ============================================================================
IDX="${SLURM_ARRAY_TASK_ID:-${ARRAY_INDEX:-}}"

if [[ -n "$IDX" && "$METHOD" == "merged" ]]; then
    REGION=$((START_REGION + IDX))
    if [[ $REGION -gt $END_REGION ]]; then
        echo "Region $REGION exceeds end region $END_REGION. Skipping."
        exit 0
    fi
fi

# ============================================================================
# Execute
# ============================================================================
case "$METHOD" in
    merged)
        extract_from_merged "$SCENARIO" "${REGION:-}"
        ;;
    nco)
        extract_nco "$INPUT_PATH" "${OUTPUT_PATH:-}"
        ;;
    single)
        extract_single "$INPUT_PATH" "${OUTPUT_PATH:-}"
        ;;
    dask)
        extract_dask "$INPUT_PATH" "${OUTPUT_PATH:-}"
        ;;
esac

echo "========================================"
echo "Completed at $(date)"
echo "========================================"
