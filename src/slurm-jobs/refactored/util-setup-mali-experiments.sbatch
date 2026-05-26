#!/bin/bash
#SBATCH --job-name=setup-mali
#SBATCH --account=${SLURM_ACCOUNT:-gts-arobel3-atlas}
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=8G
#SBATCH --output=out.setup-mali.%A_%a
#SBATCH --error=err.setup-mali.%A_%a
#SBATCH --time=01:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL:-smurugan9@gatech.edu}

################################################################################
# util-setup-mali-experiments.sbatch
# ===================================
# Set up MALI experiment directories for either:
#   1) Parameter testing (draft dependence parameter sets)
#   2) Ensemble runs (variability forcing realizations)
#
# Consolidates:
#   - aislens-setup-mali-paramset-experiments.sbatch (parameter testing)
#   - aislens-setup-mali-template.sbatch (ensemble production runs)
#
# Usage:
#   # Parameter testing mode - test different draft-dep parameter sets:
#   sbatch util-setup-mali-experiments.sbatch --mode paramtest \
#       --param-sets "fb_A,fb_B,sc_A,sc_B" \
#       --dest-base /path/to/experiments/
#
#   # Ensemble mode - production runs with different forcings:
#   sbatch --array=0-10 util-setup-mali-experiments.sbatch --mode ensemble \
#       --scenario CTRL \
#       --dest-base /path/to/ensembles/ctrl/
#
# Arguments:
#   --mode MODE           paramtest|ensemble (required)
#   --scenario SCENARIO   CTRL|SSP585|SSP126 (for ensemble mode)
#   --param-sets SETS     Comma-separated parameter sets (for paramtest mode)
#   --dest-base DIR       Base output directory for experiments
#   --template-dir DIR    Template directory to copy from
#   --forcing-dir DIR     Directory containing forcing files (ensemble mode)
#   --processed-dir DIR   Directory with processed init files (paramtest mode)
#   --use-compass         Use compass setup instead of rsync (ensemble mode)
#   --dry-run             Show what would be done without executing
################################################################################

set -euo pipefail

# Source common configuration and utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-common-header.sh"

# ============================================================================
# Default Configuration
# ============================================================================
MODE=""
SCENARIO="CTRL"
PARAM_SETS=""
DEST_BASE=""
TEMPLATE_DIR=""
FORCING_DIR=""
PROCESSED_DIR=""
USE_COMPASS=false
DRY_RUN=false

# Default paths
DEFAULT_TEMPLATE="${HOME}/data/projects/MALI_AISLENS_DIR_TEMPLATE_4KM/landice/ismip6_run/ismip6_ais_proj2300/hist_04/"
DEFAULT_PROCESSED="${AISLENS_DATA_DIR}/processed/"
DEFAULT_FORCING="${AISLENS_DATA_DIR}/interim/forcing_components/proto-ensemble-forcings"
LANDICE_EXEC="${HOME}/shared/MALI/compass/MALI-Dev/components/mpas-albany-landice/landice_model"

# Init file basename (used in paramtest mode)
TEMPLATE_INIT_BASENAME="AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_meanSatObsBMB_Paolo2023_draftDepenPiecewise.nc"

# Ensemble suffixes for array jobs
ENSEMBLE_SUFFIXES=("OG" "00" "01" "02" "03" "04" "05" "06" "07" "08" "09")

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: sbatch [--array=N-M] $0 --mode MODE [OPTIONS]

Required:
  --mode MODE           paramtest|ensemble

Options:
  --scenario SCENARIO   CTRL|SSP585|SSP126 (ensemble mode, default: CTRL)
  --param-sets SETS     Comma-separated param sets (paramtest mode)
                        e.g., "fb_A,fb_B,sc_A,sc_B,permissive,conservative"
  --dest-base DIR       Base directory for experiment outputs
  --template-dir DIR    Template MALI run directory (paramtest mode)
  --forcing-dir DIR     Directory with forcing files (ensemble mode)
  --processed-dir DIR   Directory with processed init files (paramtest mode)
  --use-compass         Use compass setup (ensemble mode)
  --dry-run             Preview without executing
  -h, --help            Show this help

Modes:
  paramtest  - Set up directories for draft-dep parameter testing
               Swaps init files per parameter set
  ensemble   - Set up directories for production ensemble runs
               Swaps forcing files per realization (array job)

Examples:
  # Parameter testing:
  sbatch $0 --mode paramtest --param-sets "fb_A,fb_B,sc_A" \\
      --dest-base \${AISLENS_DATA_DIR}/MALI/paramset-tests/

  # Ensemble runs (11 realizations):
  sbatch --array=0-10 $0 --mode ensemble --scenario SSP585 \\
      --dest-base \${HOME}/scratch/ensembles/ssp585/

  # Ensemble with compass setup:
  sbatch --array=0-10 $0 --mode ensemble --scenario CTRL \\
      --use-compass --dest-base \${HOME}/scratch/ensembles/ctrl/
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --param-sets)
            PARAM_SETS="$2"
            shift 2
            ;;
        --dest-base)
            DEST_BASE="$2"
            shift 2
            ;;
        --template-dir)
            TEMPLATE_DIR="$2"
            shift 2
            ;;
        --forcing-dir)
            FORCING_DIR="$2"
            shift 2
            ;;
        --processed-dir)
            PROCESSED_DIR="$2"
            shift 2
            ;;
        --use-compass)
            USE_COMPASS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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
    paramtest|ensemble)
        ;;
    *)
        echo "ERROR: Invalid mode: $MODE. Must be paramtest or ensemble"
        exit 1
        ;;
esac

# Set defaults based on mode
if [[ "$MODE" == "paramtest" ]]; then
    TEMPLATE_DIR="${TEMPLATE_DIR:-$DEFAULT_TEMPLATE}"
    PROCESSED_DIR="${PROCESSED_DIR:-$DEFAULT_PROCESSED}"
    DEST_BASE="${DEST_BASE:-${AISLENS_DATA_DIR}/MALI/draft-depen-paramset-tests/}"
    
    if [[ -z "$PARAM_SETS" ]]; then
        echo "ERROR: --param-sets is required for paramtest mode"
        exit 1
    fi
elif [[ "$MODE" == "ensemble" ]]; then
    FORCING_DIR="${FORCING_DIR:-$DEFAULT_FORCING}"
    DEST_BASE="${DEST_BASE:-${HOME}/scratch/ensembles/${SCENARIO,,}/}"
fi

# ============================================================================
# Environment Setup
# ============================================================================
echo "========================================"
echo "MALI Experiment Setup"
echo "========================================"
echo "Mode: $MODE"
echo "Destination: $DEST_BASE"
if [[ "$MODE" == "paramtest" ]]; then
    echo "Template: $TEMPLATE_DIR"
    echo "Processed dir: $PROCESSED_DIR"
    echo "Parameter sets: $PARAM_SETS"
else
    echo "Scenario: $SCENARIO"
    echo "Forcing dir: $FORCING_DIR"
    echo "Use compass: $USE_COMPASS"
    echo "Array ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
fi
echo "Dry run: $DRY_RUN"
echo "========================================"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN MODE - No changes will be made]"
fi

# ============================================================================
# Functions
# ============================================================================
setup_paramtest() {
    # Parse comma-separated param sets into array
    IFS=',' read -ra PARAMS <<< "$PARAM_SETS"
    
    # Validate template directory
    if [[ ! -d "$TEMPLATE_DIR" ]]; then
        echo "ERROR: Template directory not found: $TEMPLATE_DIR"
        exit 1
    fi
    
    # Create manifest
    MANIFEST="${DEST_BASE}setup_manifest.csv"
    [[ "$DRY_RUN" == "false" ]] && mkdir -p "$DEST_BASE"
    [[ "$DRY_RUN" == "false" ]] && echo "timestamp,param_set,exp_dir,init_src,notes" > "$MANIFEST"
    
    for param in "${PARAMS[@]}"; do
        ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        exp_dir="${DEST_BASE}${param}/"
        
        echo ""
        echo "=== Setting up: $param -> $exp_dir ==="
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [DRY] Would create: $exp_dir"
            echo "  [DRY] Would rsync from: $TEMPLATE_DIR"
            echo "  [DRY] Would copy init file for param: $param"
            echo "  [DRY] Would symlink landice_model"
            continue
        fi
        
        mkdir -p "$exp_dir"
        
        # Copy template (exclude init file and landice_model)
        rsync -av \
            --exclude "$TEMPLATE_INIT_BASENAME" \
            --exclude "landice_model" \
            "${TEMPLATE_DIR}/" "${exp_dir}/"
        
        # Find and copy processed init file
        src_init=$(find "$PROCESSED_DIR" -name "*${param}*.nc" -type f | head -n 1)
        
        if [[ -n "$src_init" && -f "$src_init" ]]; then
            echo "  Copying init: $src_init -> ${exp_dir}${TEMPLATE_INIT_BASENAME}"
            cp "$src_init" "${exp_dir}${TEMPLATE_INIT_BASENAME}"
            notes="ok"
        else
            echo "  WARNING: No init file found for $param"
            notes="no_init_file"
        fi
        
        # Symlink landice_model
        ln -sf "$LANDICE_EXEC" "${exp_dir}landice_model"
        
        # Set permissions
        chmod 644 "${exp_dir}streams.landice" 2>/dev/null || true
        chmod 644 "${exp_dir}namelist.landice" 2>/dev/null || true
        
        echo "${ts},${param},${exp_dir},${src_init:-none},${notes}" >> "$MANIFEST"
    done
    
    echo ""
    echo "Manifest written to: $MANIFEST"
}

setup_ensemble() {
    # Get current suffix from array index
    ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
    SUFFIX="${ENSEMBLE_SUFFIXES[$ARRAY_ID]}"
    
    WORKING_DIR="${DEST_BASE}${SUFFIX}"
    HIST_DIR="${WORKING_DIR}/landice/ismip6_run/ismip6_ais_proj2300/hist_04"
    
    echo "=== Setting up ensemble: $SUFFIX -> $WORKING_DIR ==="
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY] Would create: $WORKING_DIR"
        if [[ "$USE_COMPASS" == "true" ]]; then
            echo "  [DRY] Would run compass setup"
        fi
        echo "  [DRY] Would copy forcing: ${FORCING_DIR}/AIS_4to20km_r01_20220907_Forcing_${SUFFIX}.nc"
        return
    fi
    
    if [[ "$USE_COMPASS" == "true" ]]; then
        # Source MALI environment
        MALI_ENV="/storage/coda1/p-arobel3/0/shared/Slurm/load_mali_env.sh"
        if [[ -f "$MALI_ENV" ]]; then
            source "$MALI_ENV"
        fi
        
        # Run compass setup
        compass setup \
            -t landice/ismip6_run/ismip6_ais_proj2300 \
            -p ${HOME}/data/compass-env/compass/MALI-Dev/components/mpas-albany-landice \
            -w "$WORKING_DIR" \
            -f ${HOME}/data/MALI_projects/ISMIP6-2300/ISMIP6AE_04_MOD.cfg \
            -m pace
    else
        # Simple directory creation (user must have template ready)
        mkdir -p "$HIST_DIR"
    fi
    
    # Remove default forcing files
    rm -f "${HIST_DIR}/basin_and_coeff_gamma0_DeltaT_quadratic_non_local.nc"
    rm -f "${HIST_DIR}/processed_TF_obs_1995-2017_8km_x_60m.nc"
    
    # Copy the correct forcing file
    SOURCE_FILE="${FORCING_DIR}/AIS_4to20km_r01_20220907_Forcing_${SUFFIX}.nc"
    DEST_FILE="${HIST_DIR}/AIS_4to20km_r01_20220907_Forcing_${SUFFIX}.nc"
    
    if [[ -f "$SOURCE_FILE" ]]; then
        echo "  Copying forcing: $SOURCE_FILE -> $DEST_FILE"
        cp "$SOURCE_FILE" "$DEST_FILE"
    else
        echo "  WARNING: Forcing file not found: $SOURCE_FILE"
    fi
}

# ============================================================================
# Execute Based on Mode
# ============================================================================
case "$MODE" in
    paramtest)
        setup_paramtest
        ;;
    ensemble)
        setup_ensemble
        ;;
esac

echo ""
echo "========================================"
echo "Completed at $(date)"
echo "========================================"
