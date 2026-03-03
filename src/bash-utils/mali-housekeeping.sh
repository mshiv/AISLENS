#!/usr/bin/env bash
# mali-housekeeping.sh
# ====================
# Housekeeping utilities for MALI output directories:
# - Move figure files to organized subdirectories
# - Create symlinks across scenarios
# - Clean up temporary files
#
# Consolidates:
#   - move_aislens_output_figures.sh
#   - mali_create_symlinks.sh
#
# Usage:
#   # Move figures to subdirectory:
#   mali-housekeeping.sh --mode move-figures --scenario CTRL
#
#   # Create symlinks for shared files:
#   mali-housekeeping.sh --mode create-symlinks --source HIST --target "SSP126 SSP585"

set -euo pipefail

# ============================================================================
# Default Configuration
# ============================================================================
MODE=""
SCENARIO=""
BASE_DIR="${AISLENS_DATA_DIR:-}/MALI"
SOURCE_DIR=""
TARGET_DIRS=""
FILE_PATTERN="*.png"

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: $0 --mode MODE [OPTIONS]

Modes:
  move-figures     Move output figures to organized subdirectory
  create-symlinks  Create symlinks for shared files across scenarios
  cleanup          Remove temporary files

Options:
  --scenario SCENARIO    CTRL|SSP126|SSP585 (for move-figures)
  --base-dir DIR         Base MALI directory
  --source DIR           Source directory for symlinks
  --target "D1 D2"       Target directories for symlinks
  --pattern PATTERN      File pattern (default: *.png)
  -h, --help             Show this help

Examples:
  $0 --mode move-figures --scenario CTRL
  $0 --mode create-symlinks --source HIST --target "SSP126 SSP585"
  $0 --mode cleanup --scenario SSP585
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
        --source)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --target)
            TARGET_DIRS="$2"
            shift 2
            ;;
        --pattern)
            FILE_PATTERN="$2"
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

if [[ -z "$MODE" ]]; then
    echo "ERROR: --mode is required"
    print_usage
    exit 1
fi

# ============================================================================
# Mode: Move Figures
# ============================================================================
move_figures() {
    if [[ -z "$SCENARIO" ]]; then
        echo "ERROR: --scenario required for move-figures mode"
        exit 1
    fi
    
    local ensemble_dir="${BASE_DIR}/ENSEMBLES/${SCENARIO}"
    
    if [[ ! -d "$ensemble_dir" ]]; then
        echo "ERROR: Ensemble directory not found: $ensemble_dir"
        exit 1
    fi
    
    echo "Moving figures for $SCENARIO ensembles..."
    
    for member_dir in "${ensemble_dir}"/*; do
        [[ -d "$member_dir" ]] || continue
        local member_name
        member_name=$(basename "$member_dir")
        local output_dir="${member_dir}/output"
        
        if [[ ! -d "$output_dir" ]]; then
            echo "  $member_name: No output directory"
            continue
        fi
        
        # Create figures subdirectory
        local figures_dir="${output_dir}/figures"
        mkdir -p "$figures_dir"
        
        # Move matching files
        local count=0
        for f in "${output_dir}"/${FILE_PATTERN}; do
            [[ -f "$f" ]] || continue
            mv "$f" "$figures_dir/"
            ((count++))
        done
        
        if [[ $count -gt 0 ]]; then
            echo "  $member_name: Moved $count files to figures/"
        else
            echo "  $member_name: No matching files"
        fi
    done
}

# ============================================================================
# Mode: Create Symlinks
# ============================================================================
create_symlinks() {
    if [[ -z "$SOURCE_DIR" || -z "$TARGET_DIRS" ]]; then
        echo "ERROR: --source and --target required for create-symlinks mode"
        exit 1
    fi
    
    local source_path="${BASE_DIR}/ISMIP6/${SOURCE_DIR}/output"
    
    if [[ ! -d "$source_path" ]]; then
        echo "ERROR: Source directory not found: $source_path"
        exit 1
    fi
    
    echo "Creating symlinks from $SOURCE_DIR to targets..."
    
    IFS=' ' read -ra targets <<< "$TARGET_DIRS"
    
    for target in "${targets[@]}"; do
        local target_path="${BASE_DIR}/ISMIP6/${target}/output"
        
        echo "  Target: $target"
        mkdir -p "$target_path"
        
        local count=0
        for f in "${source_path}"/${FILE_PATTERN}; do
            [[ -f "$f" ]] || continue
            local fname
            fname=$(basename "$f")
            ln -sf "$f" "${target_path}/${fname}"
            ((count++))
        done
        
        echo "    Created $count symlinks"
    done
}

# ============================================================================
# Mode: Cleanup
# ============================================================================
cleanup_temp() {
    if [[ -z "$SCENARIO" ]]; then
        echo "ERROR: --scenario required for cleanup mode"
        exit 1
    fi
    
    local ensemble_dir="${BASE_DIR}/ENSEMBLES/${SCENARIO}"
    
    echo "Cleaning up temporary files in $SCENARIO..."
    
    # Common temp patterns
    local patterns=("*.tmp" "temp_*" "*.bak" "core.*")
    local total=0
    
    for pattern in "${patterns[@]}"; do
        local count
        count=$(find "$ensemble_dir" -name "$pattern" -type f 2>/dev/null | wc -l)
        if [[ $count -gt 0 ]]; then
            echo "  Found $count files matching: $pattern"
            find "$ensemble_dir" -name "$pattern" -type f -delete
            total=$((total + count))
        fi
    done
    
    echo "  Removed $total temporary files"
}

# ============================================================================
# Main
# ============================================================================
echo "========================================"
echo "MALI Housekeeping: $MODE"
echo "========================================"

case "$MODE" in
    move-figures)
        move_figures
        ;;
    create-symlinks)
        create_symlinks
        ;;
    cleanup)
        cleanup_temp
        ;;
    *)
        echo "ERROR: Unknown mode: $MODE"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Completed at $(date)"
echo "========================================"
