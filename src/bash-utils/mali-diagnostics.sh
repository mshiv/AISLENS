#!/usr/bin/env bash
# mali-diagnostics.sh
# ====================
# Copy or symlink MALI diagnostic files (globalStats.nc, regionalStats.nc) 
# from ensemble output directories to a central diagnostics folder.
#
# Consolidates:
#   - mali_copy_diagnostics.sh
#   - mali_copy_diagnostics_all.sh
#   - mali_create_diag_symlinks.sh
#
# Usage:
#   # Copy diagnostics for CTRL ensemble:
#   mali-diagnostics.sh --scenario CTRL --mode copy
#
#   # Symlink diagnostics for all scenarios:
#   mali-diagnostics.sh --scenario all --mode symlink
#
#   # Custom paths:
#   mali-diagnostics.sh --base-dir /path/to/MALI --dest-dir /path/to/diagnostics

set -euo pipefail

# ============================================================================
# Default Configuration
# ============================================================================
MODE="copy"                    # copy or symlink
SCENARIO="all"                 # CTRL, SSP126, SSP585, ISMIP6, or all
BASE_DIR="${AISLENS_DATA_DIR:-}/MALI"
DEST_DIR=""
FILES=("globalStats.nc" "regionalStats.nc")

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --mode MODE           copy|symlink (default: copy)
  --scenario SCENARIO   CTRL|SSP126|SSP585|ISMIP6|all (default: all)
  --base-dir DIR        Base MALI directory (default: \${AISLENS_DATA_DIR}/MALI)
  --dest-dir DIR        Destination directory (default: <base-dir>/diagnostics)
  --files "f1 f2"       Files to copy/link (default: "globalStats.nc regionalStats.nc")
  -h, --help            Show this help

Examples:
  $0 --scenario CTRL --mode copy
  $0 --scenario all --mode symlink
  $0 --base-dir /scratch/MALI --scenario SSP585
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
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --dest-dir)
            DEST_DIR="$2"
            shift 2
            ;;
        --files)
            IFS=' ' read -ra FILES <<< "$2"
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

# Set default dest dir
DEST_DIR="${DEST_DIR:-${BASE_DIR}/diagnostics}"

# ============================================================================
# Validation
# ============================================================================
if [[ ! -d "$BASE_DIR" ]]; then
    echo "ERROR: Base directory not found: $BASE_DIR"
    exit 1
fi

case "$MODE" in
    copy|symlink)
        ;;
    *)
        echo "ERROR: Invalid mode: $MODE. Must be copy or symlink"
        exit 1
        ;;
esac

# ============================================================================
# Functions
# ============================================================================
process_dir() {
    local src_output="$1"
    local target_dir="$2"
    local label="$3"
    
    mkdir -p "$target_dir"
    
    for file in "${FILES[@]}"; do
        src_file="${src_output}/${file}"
        target="${target_dir}/${file}"
        
        if [[ -f "$src_file" ]]; then
            if [[ "$MODE" == "copy" ]]; then
                cp "$src_file" "$target"
                echo "  ✓ Copied: $file"
            else
                ln -sf "$src_file" "$target"
                echo "  ✓ Linked: $file"
            fi
        else
            echo "  ⚠ Not found: $file"
        fi
    done
}

process_ensemble() {
    local ensemble_base="$1"
    local scenario_name="$2"
    
    if [[ ! -d "$ensemble_base" ]]; then
        echo "Skipping $scenario_name: directory not found"
        return
    fi
    
    echo "Processing $scenario_name ensembles..."
    
    for member_dir in "${ensemble_base}"/*; do
        [[ -d "$member_dir" ]] || continue
        member_name=$(basename "$member_dir")
        output_dir="${member_dir}/output"
        target_dir="${DEST_DIR}/ENSEMBLES/${scenario_name}/${member_name}"
        
        if [[ -d "$output_dir" ]]; then
            echo "  ${member_name}:"
            process_dir "$output_dir" "$target_dir" "$member_name"
        fi
    done
}

process_ismip6() {
    local scenario="$1"
    local ismip_dir="${BASE_DIR}/ISMIP6/${scenario}"
    
    if [[ ! -d "$ismip_dir" ]]; then
        echo "Skipping ISMIP6/$scenario: directory not found"
        return
    fi
    
    echo "Processing ISMIP6/$scenario..."
    local output_dir="${ismip_dir}/output"
    local target_dir="${DEST_DIR}/ISMIP6/${scenario}"
    
    if [[ -d "$output_dir" ]]; then
        process_dir "$output_dir" "$target_dir" "ISMIP6/$scenario"
    fi
}

# ============================================================================
# Main
# ============================================================================
echo "========================================"
echo "MALI Diagnostics ${MODE^}"
echo "========================================"
echo "Base: $BASE_DIR"
echo "Dest: $DEST_DIR"
echo "Mode: $MODE"
echo "Scenario: $SCENARIO"
echo "========================================"

mkdir -p "$DEST_DIR"

case "$SCENARIO" in
    CTRL)
        process_ensemble "${BASE_DIR}/ENSEMBLES/CTRL" "CTRL"
        ;;
    SSP126)
        process_ensemble "${BASE_DIR}/ENSEMBLES/SSP126" "SSP126"
        process_ismip6 "SSP126"
        ;;
    SSP585)
        process_ensemble "${BASE_DIR}/ENSEMBLES/SSP585" "SSP585"
        process_ismip6 "SSP585"
        ;;
    ISMIP6)
        process_ismip6 "SSP126"
        process_ismip6 "SSP585"
        process_ismip6 "HIST"
        ;;
    all)
        process_ensemble "${BASE_DIR}/ENSEMBLES/CTRL" "CTRL"
        process_ensemble "${BASE_DIR}/ENSEMBLES/SSP126" "SSP126"
        process_ensemble "${BASE_DIR}/ENSEMBLES/SSP585" "SSP585"
        process_ismip6 "HIST"
        process_ismip6 "SSP126"
        process_ismip6 "SSP585"
        ;;
    *)
        echo "ERROR: Unknown scenario: $SCENARIO"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Completed at $(date)"
echo "========================================"
