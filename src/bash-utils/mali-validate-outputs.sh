#!/usr/bin/env bash
# mali-validate-outputs.sh
# ========================
# Validate MALI output files exist and contain expected variables.
# Useful for checking ensemble run completion before post-processing.
#
# Consolidates:
#   - mali_check_thickness_files.sh
#
# Usage:
#   # Check all CTRL ensembles:
#   mali-validate-outputs.sh --scenario CTRL
#
#   # Check specific experiments:
#   mali-validate-outputs.sh --experiments "SSP585-EM1 SSP585-EM2"
#
#   # Check for specific variable:
#   mali-validate-outputs.sh --scenario SSP585 --var thickness

set -euo pipefail

# ============================================================================
# Default Configuration
# ============================================================================
SCENARIO=""
EXPERIMENTS=""
BASE_DIR="${AISLENS_DATA_DIR:-}/MALI/ENSEMBLES"
VAR="thickness"
CHECK_TIME=true

# ============================================================================
# Parse Arguments
# ============================================================================
print_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --scenario SCENARIO      CTRL|SSP126|SSP585 (checks all members)
  --experiments "E1 E2"    Space-separated list of specific experiments
  --base-dir DIR           Base ensembles directory
  --var VARIABLE           Variable to check for (default: thickness)
  --no-time                Skip time dimension check
  -h, --help               Show this help

Examples:
  $0 --scenario CTRL
  $0 --scenario SSP585 --var floatingBasalMassBal
  $0 --experiments "SSP585-EM1 SSP585-EM2 SSP585-EM4"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --experiments)
            EXPERIMENTS="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --var)
            VAR="$2"
            shift 2
            ;;
        --no-time)
            CHECK_TIME=false
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
# Build Experiment List
# ============================================================================
if [[ -n "$SCENARIO" && -n "$EXPERIMENTS" ]]; then
    echo "ERROR: Specify either --scenario or --experiments, not both"
    exit 1
fi

if [[ -z "$SCENARIO" && -z "$EXPERIMENTS" ]]; then
    echo "ERROR: Must specify --scenario or --experiments"
    print_usage
    exit 1
fi

EXP_LIST=()
if [[ -n "$SCENARIO" ]]; then
    SCENARIO_DIR="${BASE_DIR}/${SCENARIO}"
    if [[ ! -d "$SCENARIO_DIR" ]]; then
        echo "ERROR: Scenario directory not found: $SCENARIO_DIR"
        exit 1
    fi
    for d in "${SCENARIO_DIR}"/*; do
        [[ -d "$d" ]] && EXP_LIST+=("$(basename "$d")")
    done
else
    IFS=' ' read -ra EXP_LIST <<< "$EXPERIMENTS"
fi

# ============================================================================
# Validation Functions
# ============================================================================
check_experiment() {
    local exp="$1"
    local exp_dir
    
    # Find the experiment directory
    if [[ -n "$SCENARIO" ]]; then
        exp_dir="${BASE_DIR}/${SCENARIO}/${exp}/output"
    else
        # Try to find in any scenario
        for scen in CTRL SSP126 SSP585; do
            if [[ -d "${BASE_DIR}/${scen}/${exp}/output" ]]; then
                exp_dir="${BASE_DIR}/${scen}/${exp}/output"
                break
            fi
        done
    fi
    
    echo "=== $exp ==="
    
    if [[ ! -d "$exp_dir" ]]; then
        echo "  ✗ ERROR: Output directory not found"
        return 1
    fi
    
    # Count output files
    local file_count
    file_count=$(find "$exp_dir" -maxdepth 1 -name "output*.nc" 2>/dev/null | wc -l)
    echo "  Output files: $file_count"
    
    if [[ "$file_count" -eq 0 ]]; then
        echo "  ✗ No output files found"
        return 1
    fi
    
    # Check latest file
    local latest
    latest=$(ls -t "${exp_dir}"/output*.nc 2>/dev/null | head -1)
    echo "  Latest: $(basename "$latest")"
    
    # Check for variable
    if ncdump -h "$latest" 2>/dev/null | grep -q "float ${VAR}"; then
        echo "  ✓ Has '${VAR}' variable"
    else
        echo "  ✗ Missing '${VAR}' variable"
        echo "    Available variables:"
        ncdump -h "$latest" 2>/dev/null | grep "float" | head -5 | sed 's/^/      /'
    fi
    
    # Check time dimension
    if [[ "$CHECK_TIME" == "true" ]]; then
        local time_dim
        time_dim=$(ncdump -h "$latest" 2>/dev/null | grep "Time = " | sed 's/.*Time = \([0-9]*\).*/\1/' || echo "0")
        echo "  Time dimension: $time_dim"
        
        if [[ "$time_dim" -lt 10 ]]; then
            echo "  ⚠ Warning: Time dimension seems small"
        fi
    fi
    
    echo ""
}

# ============================================================================
# Main
# ============================================================================
echo "========================================"
echo "MALI Output Validation"
echo "========================================"
echo "Checking variable: $VAR"
echo "Experiments: ${#EXP_LIST[@]}"
echo "========================================"
echo ""

PASSED=0
FAILED=0

for exp in "${EXP_LIST[@]}"; do
    if check_experiment "$exp"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
done

echo "========================================"
echo "Summary: $PASSED passed, $FAILED failed"
echo "========================================"
