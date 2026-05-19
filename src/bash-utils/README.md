# AISLENS Bash Utilities

> Interactive CLI tools for MALI ensemble management and file operations.

## Overview

This directory contains **bash utilities** designed to be run interactively from the command line (not via SLURM). These complement the SLURM batch jobs in `slurm-jobs/refactored/`.

**Key distinction:**
- `slurm-jobs/` → Batch jobs for HPC compute (data processing, analysis)
- `bash-utils/` → Interactive CLI tools for setup, validation, housekeeping

## Active Scripts

### Ensemble Setup

| Script | Purpose |
|--------|---------|
| `create-ensemble-dirs.sh` | Create ensemble directories with proper file structure |

| `create-restart-ensemble-dirs.sh` | Create ensemble dirs from restart files |
| `ensemble-job-script.sh` | Submit jobs across all ensemble member directories |

### Diagnostics & Validation

| Script | Purpose |
|--------|---------|
| `mali-diagnostics.sh` | Copy/symlink globalStats.nc, regionalStats.nc |
| `mali-validate-outputs.sh` | Verify output files and variables exist |
| `mali-housekeeping.sh` | Move figures, create symlinks, cleanup temp files |

## Usage Examples

### Create Ensemble Directories

```bash
# Create 10 CTRL ensemble members
./create-ensemble-dirs.sh \
    -i /path/to/init.nc \
    -f /path/to/forcings \
    -t /path/to/template \
    -p CTRL- \
    -n 10 \
    -d /scratch/ensembles/CTRL \
    -e /path/to/landice_model \
    -b "AIS_4to20km_r01_20220907_AISLENS-Forcing_%02d.nc"
```

### Submit Ensemble Jobs

```bash
# Submit all 10 CTRL jobs
./ensemble-job-script.sh -p CTRL -d /scratch/ensembles/CTRL -n 10

# Submit specific members
./ensemble-job-script.sh -p CTRL -d /scratch/ensembles/CTRL -l 0,3,7
```

### Collect Diagnostics

```bash
# Copy diagnostics for all scenarios
./mali-diagnostics.sh --scenario all --mode copy

# Symlink instead of copy
./mali-diagnostics.sh --scenario CTRL --mode symlink
```

### Validate Outputs

```bash
# Check all SSP585 ensembles
./mali-validate-outputs.sh --scenario SSP585

# Check specific experiments
./mali-validate-outputs.sh --experiments "SSP585-EM1 SSP585-EM2"
```

### Housekeeping

```bash
# Move PNG files to figures/ subdirectory
./mali-housekeeping.sh --mode move-figures --scenario CTRL

# Create symlinks for shared ISMIP6 files
./mali-housekeeping.sh --mode create-symlinks --source HIST --target "SSP126 SSP585"
```

## Archived Scripts

Scripts in `archive/` are superseded by refactored versions or exist in `slurm-jobs/`:

| Archived Script | Reason | Replacement |
|-----------------|--------|-------------|
| `combine_forcing_components.sh` | Duplicate of SLURM job | `slurm-jobs/aislens_combine_forcing_components.sbatch` |
| `create-forced-ensemble-forcings.sh` | Duplicate of SLURM job | `slurm-jobs/add_trend_to_vargen_array.sbatch` |
| `setup_forcing_trend_files.sh` | Duplicate of SLURM job | `slurm-jobs/archive/aislens_create_forcing_trend*.sbatch` |
| `setup_aislens.sh` | Hardcoded, less flexible | `slurm-jobs/refactored/util-setup-mali-experiments.sbatch` |
| `create_mali_init_cond_file.sh` | One-off NCO script | Kept for reference |
| `mali_copy_diagnostics*.sh` | Consolidated | `mali-diagnostics.sh` |
| `mali_check_thickness_files.sh` | Consolidated | `mali-validate-outputs.sh` |
| `move_aislens_output_figures.sh` | Consolidated | `mali-housekeeping.sh` |

## Relationship to SLURM Jobs

| Task | Use This | Not This |
|------|----------|----------|
| **HPC compute** (processing, analysis) | `slurm-jobs/refactored/*.sbatch` | - |
| **Interactive setup** (create dirs) | `bash-utils/*.sh` | - |
| **Submit ensemble jobs** | `bash-utils/ensemble-job-script.sh` | - |
| **Collect output files** | `bash-utils/mali-diagnostics.sh` | - |
| **Validate runs** | `bash-utils/mali-validate-outputs.sh` | - |

## Environment

These scripts expect:
- `$AISLENS_DATA_DIR` environment variable set
- NCO tools available (for some scripts)
- Access to HPC filesystem
