#!/bin/bash
#SBATCH --job-name=dataprep-sorrm
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=2 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.dataprep-sorrm.%j
#SBATCH --error=err.dataprep-sorrm.%j
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

################################################################################
# 01-dataprep-sorrm.sbatch
# ========================
# Regrid and concatenate MPAS-Ocean (SORRM) data for AISLENS processing.
#
# Consolidates:
#   - aislens_regrid_sorrm.sbatch
#   - aislens_build_regridded_sorrm.sbatch
#
# Usage:
#   # Regrid SORRM data:
#   sbatch 01-dataprep-sorrm.sbatch --mode regrid
#
#   # Concatenate regridded files:
#   sbatch 01-dataprep-sorrm.sbatch --mode concat
#
#   # Both (regrid then concatenate):
#   sbatch 01-dataprep-sorrm.sbatch --mode all
#
# Arguments:
#   --mode MODE           regrid|concat|all (required)
#   --grid-file FILE      MPAS-O grid file
#   --input-dir DIR       Input directory (raw or regridded)
#   --output-dir DIR      Output directory
#   --dx DX               Grid resolution in km (default: 4.0)
#   --dy DY               Grid resolution in km (default: 4.0)
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
MODE=""
GRID_FILE="${HOME}/data/projects/AISLENS-DATA/data/external/SORRMv2.1.ISMF/mpaso.SOwISC12to60E2r4.rstFromG-anvil.210203.nc"
INPUT_DIR=""
OUTPUT_DIR=""
DX=4.0
DY=4.0
USE_DASK=true
PARALLEL=true
CHUNKS="{'Time': 100}"

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch $0 --mode MODE [OPTIONS]

Required:
  --mode MODE           regrid|concat|all

Options:
  --grid-file FILE      MPAS-O grid file
  --input-dir DIR       Input directory
  --output-dir DIR      Output directory
  --dx DX               Grid resolution X (km, default: 4.0)
  --dy DY               Grid resolution Y (km, default: 4.0)
  --no-dask             Disable dask for concatenation
  --no-parallel         Disable parallel processing
  -h, --help            Show this help

Examples:
  sbatch $0 --mode regrid --dx 4.0 --dy 4.0
  sbatch $0 --mode concat --input-dir /path/to/regridded
  sbatch $0 --mode all
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --grid-file)
            GRID_FILE="$2"
            shift 2
            ;;
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dx)
            DX="$2"
            shift 2
            ;;
        --dy)
            DY="$2"
            shift 2
            ;;
        --no-dask)
            USE_DASK=false
            shift
            ;;
        --no-parallel)
            PARALLEL=false
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
if [[ -z "$MODE" ]]; then
    echo "ERROR: --mode is required"
    print_usage
    exit 1
fi

case "$MODE" in
    regrid|concat|all)
        ;;
    *)
        echo "ERROR: Invalid mode: $MODE"
        exit 1
        ;;
esac

# ============================================================================
# Set Default Paths
# ============================================================================
SORRM_RAW_DIR="${HOME}/data/projects/AISLENS-DATA/data/external/SORRMv2.1.ISMF"
SORRM_REGRID_DIR="${SORRM_RAW_DIR}/aislens_regridded_sorrm"

if [[ -z "$INPUT_DIR" ]]; then
    case "$MODE" in
        regrid)
            INPUT_DIR="$SORRM_RAW_DIR"
            ;;
        concat)
            INPUT_DIR="$SORRM_REGRID_DIR"
            ;;
        all)
            INPUT_DIR="$SORRM_RAW_DIR"
            ;;
    esac
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    case "$MODE" in
        regrid)
            OUTPUT_DIR="$SORRM_REGRID_DIR"
            ;;
        concat)
            OUTPUT_DIR="${AISLENS_DATA_DIR}/external"
            ;;
        all)
            OUTPUT_DIR="${AISLENS_DATA_DIR}/external"
            ;;
    esac
fi

# ============================================================================
# Environment Setup
# ============================================================================
module load anaconda3
conda activate mpas-analysis

echo "========================================"
echo "SORRM Data Preparation"
echo "========================================"
echo "Mode: $MODE"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Grid resolution: ${DX}km x ${DY}km"
echo "========================================"

# ============================================================================
# Processing Functions
# ============================================================================
do_regrid() {
    echo "Regridding MPAS-O data..."
    
    cd "${AISLENS_REPO}/src/MPAS-Tools"
    
    python regrid_mpaso_data.py \
        --grid_file "$GRID_FILE" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$SORRM_REGRID_DIR" \
        --dx "$DX" \
        --dy "$DY"
    
    echo "Regridding complete: $SORRM_REGRID_DIR"
}

do_concat() {
    local input="${1:-$SORRM_REGRID_DIR}"
    local output_file="${OUTPUT_DIR}/Regridded_SORRMv2.1.ISMF.FULL.${DX}km.nc"
    
    echo "Concatenating regridded files..."
    echo "Input: $input"
    echo "Output: $output_file"
    
    mkdir -p "$OUTPUT_DIR"
    
    cd "${AISLENS_REPO}/src/MPAS-Tools"
    
    local dask_flag=""
    local parallel_flag=""
    
    [[ "$USE_DASK" == "true" ]] && dask_flag="--use_dask"
    [[ "$PARALLEL" == "true" ]] && parallel_flag="--parallel"
    
    python concatenate_mpaso_data.py \
        --input_dir "$input" \
        --output_file "$output_file" \
        $dask_flag \
        $parallel_flag \
        --chunks "$CHUNKS"
    
    echo "Concatenation complete: $output_file"
}

# ============================================================================
# Execute Based on Mode
# ============================================================================
case "$MODE" in
    regrid)
        do_regrid
        ;;
    concat)
        do_concat "$INPUT_DIR"
        ;;
    all)
        do_regrid
        do_concat "$SORRM_REGRID_DIR"
        ;;
esac

echo "========================================"
echo "Completed at $(date)"
echo "========================================"
