#!/bin/bash
################################################################################
# AISLENS Common Script Header
# ============================
# This file is sourced by all AISLENS SLURM scripts.
# It sets up the environment, logging, and common functions.
#
# Usage: source "${SCRIPT_DIR}/00-common-header.sh"
################################################################################

# Strict mode
set -euo pipefail

# Get script directory (for sourcing config)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/config"

# Source environment config if it exists
if [[ -f "${CONFIG_DIR}/aislens_env.sh" ]]; then
    source "${CONFIG_DIR}/aislens_env.sh"
fi

# Allow per-user overrides stored in ~/.aislens_env (untracked, user-specific)
if [[ -f "${HOME}/.aislens_env" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.aislens_env"
fi

# Verify required environment variables
: "${AISLENS_DATA_DIR:?ERROR: AISLENS_DATA_DIR not set. Source config/aislens_env.sh}"
: "${AISLENS_REPO:?ERROR: AISLENS_REPO not set. Source config/aislens_env.sh}"

# Load modules and conda environment
load_aislens_env() {
    module load anaconda3 2>/dev/null || true
    conda activate mpas-analysis
    export HDF5_USE_FILE_LOCKING=FALSE
}

# Logging utilities
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

# Validation utilities
validate_scenario() {
    local scenario="$1"
    case "${scenario,,}" in
        ctrl|ssp585|ssp126)
            echo "${scenario^^}"
            ;;
        *)
            log_error "Invalid scenario: $scenario. Must be CTRL, SSP585, or SSP126"
            exit 1
            ;;
    esac
}

validate_file_exists() {
    local file="$1"
    local desc="${2:-File}"
    if [[ ! -f "$file" ]]; then
        log_error "$desc not found: $file"
        exit 1
    fi
}

validate_dir_exists() {
    local dir="$1"
    local desc="${2:-Directory}"
    if [[ ! -d "$dir" ]]; then
        log_error "$desc not found: $dir"
        exit 1
    fi
}

# Create temp directory with cleanup trap
setup_tempdir() {
    local prefix="${1:-aislens}"
    TMPDIR="${HOME}/scratch/${prefix}_${SLURM_JOB_ID:-$$}"
    mkdir -p "$TMPDIR"
    trap "rm -rf $TMPDIR" EXIT
    echo "$TMPDIR"
}

# Array job utilities
get_array_index() {
    echo "${SLURM_ARRAY_TASK_ID:-${1:-0}}"
}

# Print usage helper
print_script_header() {
    local script_name="$1"
    local description="$2"
    log_info "========================================"
    log_info "$script_name"
    log_info "$description"
    log_info "========================================"
    log_info "SLURM Job ID: ${SLURM_JOB_ID:-interactive}"
    log_info "Running on: $(hostname)"
    log_info "Working directory: $(pwd)"
}
