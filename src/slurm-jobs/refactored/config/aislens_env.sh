#!/bin/bash
# =============================================================================
# AISLENS Environment Configuration
# =============================================================================
# Source this file from SLURM scripts to set up the environment.
# Users should set these environment variables in their ~/.bashrc or
# export them before submitting jobs.
#
# Usage in SLURM scripts:
#   source "$(dirname "$0")/config/aislens_env.sh"
# =============================================================================

# Data Directories
export AISLENS_DATA_DIR="${AISLENS_DATA_DIR:-${SCRATCH:-$HOME/scratch}/AISLENS/data}"
export AISLENS_REPO="${AISLENS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export AISLENS_SCRIPTS="${AISLENS_REPO}/src/scripts"
export AISLENS_MPAS_TOOLS="${AISLENS_REPO}/src/MPAS-Tools"
export AISLENS_MALI_DIR="${AISLENS_DATA_DIR}/MALI"
export AISLENS_MALI_ENSEMBLES="${AISLENS_MALI_DIR}/ENSEMBLES"

# SLURM Configuration
# Do NOT hardcode site- or user-specific SLURM settings in this tracked file.
# Users should create a local, untracked file at ~/.aislens_env to set their
# cluster account and email, or export the variables in their shell before
# invoking the wrapper. Example ~/.aislens_env contents:
#
#   export SLURM_ACCOUNT="my-hpc-account"
#   export SLURM_EMAIL="me@my.domain"
#
# This file intentionally does not provide defaults for SLURM_ACCOUNT or
# SLURM_EMAIL so that users must set them in their local environment.

# Conda Environment
export AISLENS_CONDA_ENV="${AISLENS_CONDA_ENV:-mpas-analysis}"

if command -v module &> /dev/null; then
    module load anaconda3 2>/dev/null || true
fi

if command -v conda &> /dev/null; then
    conda activate "$AISLENS_CONDA_ENV" 2>/dev/null || true
fi

echo "=== AISLENS Environment ==="
echo "AISLENS_DATA_DIR: $AISLENS_DATA_DIR"
echo "AISLENS_REPO:     $AISLENS_REPO"
echo "=========================="
