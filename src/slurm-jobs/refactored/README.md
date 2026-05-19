# AISLENS Refactored SLURM Scripts

> **Status**: Refactored version of SLURM job scripts with standardized naming, consolidated functionality, and improved documentation.

## Overview

This directory contains the refactored SLURM scripts, reducing the original 43 active scripts to ~20 consolidated scripts through parameterization.

## Naming Convention

Scripts follow a **phase-prefixed naming scheme**:

```
XX-phase-task.sbatch
```

| Prefix | Phase | Description |
|--------|-------|-------------|
| `00-` | Common | Shared utilities and headers |
| `01-` | Data Prep | Data preprocessing |
| `02-` | Draft Dep | Draft dependence calculation |
| `03-` | Interp | Grid interpolation |
| `04-` | Forcing | Forcing generation |
| `05-` | Trend | Trend extraction |
| `06-` | Combine | Trend + variability combination |
| `07-` | MALI | MALI processing |
| `08-` | Viz | Visualization |
| `util-` | Utilities | Debug/validation tools |

## Script Inventory

### Core Workflow Scripts

| Script | Replaces | Purpose |
|--------|----------|---------|
| `01-dataprep-time-mean.sbatch` | `aislens_prep_time_mean_{model,split}.sbatch` | Compute time-mean products |
| `01-dataprep-sorrm.sbatch` | `aislens_regrid_sorrm.sbatch`, `aislens_build_regridded_sorrm.sbatch` | Regrid/concatenate ocean model data |
| `01-dataprep-model.sbatch` | `aislens_prep_satobs.sbatch`, `aislens_prep_model_sim_fast.sbatch` | Preprocess model/sat-obs data |
| `02-draftdep-calc.sbatch` | `calc_draft-depen-*.sbatch`, `visualize_draft-depen-*.sbatch` | Calculate and visualize draft dependence |
| `03-interp-to-mpas.sbatch` | `interpolate_draft-depen-paramsets_array.sbatch`, `interpolate-forcings-array.sbatch` | Interpolate to MPAS-LI grid |
| `04-forcing-gen.sbatch` | `gen_forcings.sbatch`, `fill-extrapolate.sbatch` | Generate forcing realizations |
| `05-trend-extract.sbatch` | `extract-trend-*.sbatch`, `calc-trends-nco.sbatch` | Extract trend components |
| `06-combine-trend-var.sbatch` | `add-trend-*.sbatch`, `combine_forcing_components.sbatch` | Combine trend + variability |
| `07-mali-dhdt-process.sbatch` | `*dhdt_processing*.sbatch`, `process_dhdt_array.sbatch` | Compute dH/dt from MALI |
| `08-viz-mali.sbatch` | `plot_mali_outputs.sbatch`, `plot_dhdt_*.sbatch` | All MALI visualization |

### Utility Scripts (Restored from Archive)

| Script | Original | Purpose |
|--------|----------|---------|
| `util-inspect-draft-depen.sbatch` | `aislens_inspect_shelf_pair_*.sbatch` | Validate draft-dep calculations |
| `util-debug-prepare-workflow.sbatch` | `debug-prepare-workflow.sbatch` | Debug data preparation |
| `util-thickness-analysis.sbatch` | `aislens_mali_thickness_analysis.sbatch` | Ensemble thickness analysis |
| `util-setup-mali-experiments.sbatch` | `aislens-setup-mali-paramset-experiments.sbatch`, `aislens-setup-mali-template.sbatch` | Setup MALI experiment directories |

### Support Files

| File | Purpose |
|------|---------|
| `00-common-header.sh` | Shared functions, logging, validation || `config/aislens_env.sh` | Environment setup (paths, conda) |
| `config/slurm_config.example` | Template for user-specific SLURM settings |

## Directory Structure

```
refactored/
├── 00-common-header.sh       # Common utilities (sourced by all scripts)
├── config/
│   ├── aislens_env.sh        # AISLENS_DATA_DIR, AISLENS_REPO, conda activation
│   └── slurm_config.example  # Template: copy to slurm_config.local
├── 01-dataprep-*.sbatch      # Phase 1: Data preparation (3 scripts)
├── 02-draftdep-calc.sbatch   # Phase 2: Draft dependence calculation
├── 03-interp-to-mpas.sbatch  # Phase 3: MPAS grid interpolation
├── 04-forcing-gen.sbatch     # Phase 4: Forcing generation
├── 05-trend-extract.sbatch   # Phase 5: Trend extraction
├── 06-combine-trend-var.sbatch # Phase 6: Combine trend + variability
├── 07-mali-dhdt-process.sbatch # Phase 7: MALI dH/dt processing
├── 08-viz-mali.sbatch        # Phase 8: Visualization
├── util-*.sbatch             # Utility scripts (4)
└── README.md
```
## Usage Examples

### 1. Data Preparation

```bash
# Regrid ocean model data to regular grid
sbatch 01-dataprep-sorrm.sbatch --mode regrid --dx 5000 --dy 5000

# Concatenate regridded files
sbatch 01-dataprep-sorrm.sbatch --mode concat

# Full workflow (regrid + concat)
sbatch 01-dataprep-sorrm.sbatch --mode all

# Preprocess satellite observations
sbatch 01-dataprep-model.sbatch --source satobs

# Preprocess ocean model (SORRM)
sbatch 01-dataprep-model.sbatch --source model --init-dirs

# Full time-mean calculation
sbatch 01-dataprep-time-mean.sbatch --mode full --start-year 450 --end-year 750

# Split (parallel) mode
sbatch 01-dataprep-time-mean.sbatch --mode split --coarsen 2
```

### 2. Draft Dependence Calculation

```bash
# Calculate draft dependence parameters
sbatch 02-draftdep-calc.sbatch --mode calc

# Visualize results
sbatch 02-draftdep-calc.sbatch --mode viz --satobs --model

# Compare parameter sets
sbatch 02-draftdep-calc.sbatch --mode compare --param-sets "permissive,conservative"

# Full workflow (calc + viz)
sbatch 02-draftdep-calc.sbatch --mode all --plot-all-shelves
```

### 3. Grid Interpolation

```bash
# Interpolate parameter sets to MPAS-LI grid (array job for 11 sets)
sbatch --array=0-10 03-interp-to-mpas.sbatch --mode params

# Single parameter set
sbatch --array=5 03-interp-to-mpas.sbatch --mode params  # optimal_v3

# Interpolate forcing realizations (array job for 30 realizations)
sbatch --array=0-29 03-interp-to-mpas.sbatch --mode forcings --scenario SSP585
```

### 4. Forcing Generation

```bash
# Generate 30 forcing realizations
sbatch 04-forcing-gen.sbatch --mode generate --scenario SSP585 -n 30

# Fill NaN values via extrapolation
sbatch 04-forcing-gen.sbatch --mode fill --scenario SSP585 --method nearest

# Full workflow (generate + fill)
sbatch 04-forcing-gen.sbatch --mode all --scenario SSP585 --param-set permissive
```

### 5. Trend Extraction

```bash
# From merged MALI output (array job)
sbatch --array=0-99 05-trend-extract.sbatch --scenario SSP585 --method merged

# NCO-based linear trend
sbatch 05-trend-extract.sbatch --method nco --input data.nc --output trend.nc
```

### 6. Combine Trend + Variability

```bash
# Python method for SSP585 (array job)
sbatch --array=0-19 06-combine-trend-var.sbatch --scenario SSP585 --method python

# NCO method for both scenarios
sbatch --array=0-1 06-combine-trend-var.sbatch --scenario all --method nco
```

### 7. MALI dH/dt Processing

```bash
# Process dH/dt for SSP585 ensembles
sbatch --array=0-9 07-mali-dhdt-process.sbatch --scenario SSP585 --mode process

# Create animations
sbatch 07-mali-dhdt-process.sbatch --scenario CTRL --mode animate

# Dry run to see directories
bash 07-mali-dhdt-process.sbatch --scenario SSP585 --dry-run
```

### 8. Visualization

```bash
# Plot output variables
sbatch --array=0-9 08-viz-mali.sbatch --type output --scenario CTRL --vars thickness,dHdt

# Ratio maps
sbatch 08-viz-mali.sbatch --type ratio --scenario SSP585

# Ensemble statistics
sbatch 08-viz-mali.sbatch --type ensemble --scenario SSP585
```

### 9. Utility Scripts

```bash
# Debug draft dependence
sbatch util-inspect-draft-depen.sbatch --shelf Amery --param-set standard

# Smoke test data prep
sbatch util-debug-prepare-workflow.sbatch --coarsen 16 --end-year 460

# Setup MALI experiments for parameter testing (test different draft-dep params):
sbatch util-setup-mali-experiments.sbatch --mode paramtest \
    --param-sets "fb_A,fb_B,sc_A,sc_B,permissive" \
    --dest-base ${AISLENS_DATA_DIR}/MALI/paramset-tests/

# Setup MALI experiments for production ensemble runs (array job for 11 realizations):
sbatch --array=0-10 util-setup-mali-experiments.sbatch --mode ensemble \
    --scenario SSP585 \
    --dest-base ${HOME}/scratch/ensembles/ssp585/

# Ensemble with compass setup:
sbatch --array=0-10 util-setup-mali-experiments.sbatch --mode ensemble \
    --scenario CTRL --use-compass \
    --dest-base ${HOME}/scratch/ensembles/ctrl/
# Full debug run
sbatch util-debug-prepare-workflow.sbatch --full

# Thickness analysis
sbatch util-thickness-analysis.sbatch --scenario CTRL --ensemble CTRL-SSN
```

## Common Patterns

### Scenario Selection

All scripts accept `--scenario` with values: `CTRL`, `SSP585`, `SSP126`, or `all`.

```bash
sbatch script.sbatch --scenario SSP585
```

### Array Jobs

Most scripts support SLURM array jobs:

```bash
# Submit array
sbatch --array=0-9 script.sbatch --scenario SSP585

# Or specify index directly
sbatch script.sbatch --scenario SSP585 --array-index 3
```

### Dry Run Mode

Preview what will be processed without executing:

```bash
bash script.sbatch --scenario SSP585 --dry-run
```

### Mode Selection

Many scripts have multiple modes:

```bash
# Time-mean: full or split
sbatch 01-dataprep-time-mean.sbatch --mode split

# Trend extraction: merged, nco, single, dask
sbatch 05-trend-extract.sbatch --method merged

# Visualization: output, ratio, dhdt, ensemble
sbatch 08-viz-mali.sbatch --type ratio
```

## Migration Guide

### From Old Scripts

| Old Script | New Script | Command |
|------------|------------|---------|
| `aislens_regrid_sorrm.sbatch` | `01-dataprep-sorrm.sbatch` | `--mode regrid` |
| `aislens_build_regridded_sorrm.sbatch` | `01-dataprep-sorrm.sbatch` | `--mode concat` |
| `aislens_prep_satobs.sbatch` | `01-dataprep-model.sbatch` | `--source satobs` |
| `aislens_prep_model_sim_fast.sbatch` | `01-dataprep-model.sbatch` | `--source model` |
| `aislens_prep_time_mean_model.sbatch` | `01-dataprep-time-mean.sbatch` | `--mode full` |
| `aislens_prep_time_mean_split.sbatch` | `01-dataprep-time-mean.sbatch` | `--mode split` |
| `aislens_calc_draft-depen-comprehensive-fast.sbatch` | `02-draftdep-calc.sbatch` | `--mode calc` |
| `aislens_visualize_draft-depen-comprehensive.sbatch` | `02-draftdep-calc.sbatch` | `--mode viz` |
| `aislens_visualize_draft-depen-comprehensive_compare-param-sets.sbatch` | `02-draftdep-calc.sbatch` | `--mode compare` |
| `aislens_interpolate_draft-depen-paramsets_array.sbatch` | `03-interp-to-mpas.sbatch` | `--mode params` |
| `aislens-interpolate-forcings-array.sbatch` | `03-interp-to-mpas.sbatch` | `--mode forcings` |
| `aislens_gen_forcings.sbatch` | `04-forcing-gen.sbatch` | `--mode generate` |
| `aislens-fill-extrapolate.sbatch` | `04-forcing-gen.sbatch` | `--mode fill` |
| `aislens_mali_dhdt_processing_array_updated.sbatch` | `07-mali-dhdt-process.sbatch` | `--mode process` |
| `aislens_dhdt_processing_animation.sbatch` | `07-mali-dhdt-process.sbatch` | `--mode animate` |
| `process_dhdt_array.sbatch` | `07-mali-dhdt-process.sbatch` | `--mode process` |
| `add-trend-forcing-to-variability.sbatch` | `06-combine-trend-var.sbatch` | `--method nco` |
| `add_trend_to_vargen_array.sbatch` | `06-combine-trend-var.sbatch` | `--method python` |
| `aislens_plot_mali_outputs.sbatch` | `08-viz-mali.sbatch` | `--type output` |
| `aislens_plot_dhdt_ratio.sbatch` | `08-viz-mali.sbatch` | `--type ratio` |
| `plot_dhdt_array.sbatch` | `08-viz-mali.sbatch` | `--type dhdt` |

## Environment Setup

All scripts automatically source `00-common-header.sh`, which loads the config files.

### First-Time Setup

```bash
cd src/slurm-jobs/refactored

# 1. Edit aislens_env.sh with your paths
vim config/aislens_env.sh

# 2. Create your local SLURM config from template
cp config/slurm_config.example config/slurm_config.local
vim config/slurm_config.local
```

### Config Files

| File | Purpose | Git-tracked? |
|------|---------|-------------|
| `config/aislens_env.sh` | Shared environment (AISLENS_DATA_DIR, conda) | ✅ Yes |
| `config/slurm_config.example` | Template for user-specific settings | ✅ Yes |
| `config/slurm_config.local` | Your SLURM account/email (create from template) | ❌ No |

### Required Environment Variables

```bash
# In config/aislens_env.sh (shared):
export AISLENS_DATA_DIR="/path/to/data"
export AISLENS_REPO="/path/to/repo"

# In config/slurm_config.local (user-specific):
export SLURM_ACCOUNT="your-account"
export SLURM_EMAIL="your-email@example.com"
```

## Workflow Diagram

```
01-dataprep-*     →  02-draftdep-*   →  03-interp-*
     ↓                    ↓                 ↓
04-forcing-*      ←───────┴─────────→  05-trend-*
     ↓                                      ↓
     └──────────────→ 06-combine-* ←────────┘
                           ↓
                      07-mali-*
                           ↓
                      08-viz-*
```

## Contributing

When adding new scripts:
1. Follow the naming convention: `XX-phase-task.sbatch`
2. Include comprehensive header documentation
3. Support `--help`, `--dry-run`, and `--scenario` flags
4. Use functions from `00-common-header.sh`
5. Update this README
