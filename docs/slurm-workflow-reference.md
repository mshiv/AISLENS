# AISLENS Workflow Reference

> **Purpose**: Reference for the AISLENS forcing-generation workflow, centered on the refactored SLURM scripts.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Workflow Phases](#2-workflow-phases)
3. [SLURM Script Reference](#3-slurm-script-reference)
4. [Python Script Reference](#4-python-script-reference)
5. [AISLENS Package Reference](#5-aislens-package-reference)
6. [Data Flow](#6-data-flow)
7. [Consolidation Plan](#7-consolidation-plan)

---

## 1. Project Overview

This is the workflow used to build forcing files for the MALI (MPAS-Albany Land Ice) ice sheet model. It covers six steps:

1. Ingest satellite observations (Paolo et al. 2023) and MPAS-Ocean SORRMv2.1 output
2. Preprocess the data with detrending, deseasonalization, and draft-dependence removal
3. Fit draft-melt relationships with changepoint detection and piecewise linear models
4. Generate ensemble forcings with EOF decomposition and phase randomization
5. Combine variability with the climate trend cases (SSP585/SSP126)
6. Regrid everything onto the MPAS-LI (MALI) mesh for model input

### Key Data Sources

| Source | File | Variables |
|--------|------|-----------|
| Satellite Obs | `ANT_G1920V01_IceShelfMeltDraft_Time.nc` | melt, draft |
| Ocean Model | `Regridded_SORRMv2.1.ISMF.FULL.nc` | ssh (draft proxy), landIceFreshwaterFlux |
| Ice Shelf Masks | `iceShelves.geojson` | 100 Antarctic ice shelves (regions 33-133) |

### Key Output Products

| Product | Description |
|---------|-------------|
| Draft Dependence Parameters | α₀, α₁, minDraft, constantMelt per ice shelf |
| Forcing Realizations | Ensemble BMB forcings on regular grid |
| MALI Forcings | Regridded forcings on MPAS-LI mesh |
| Trend Scenarios | SSP585/SSP126 climate-driven trends |

---

## 2. Workflow Phases

The refactored workflow is split into 8 phases. The older script set was broader, but this version keeps the core steps and folds related jobs together:  (If necessary, refer to the GT AISLENS gh repo for archival scripts used on PACE earlier)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: Data Preparation (6 scripts)                                      │
│  ↓                                                                          │
│  Phase 2: Draft Dependence Calculation (3 scripts)                          │
│  ↓                                                                          │
│  Phase 3: Interpolation to MPAS Grid (2 scripts)                            │
│  ↓                                                                          │
│  Phase 4: Forcing Generation (5 scripts)                                    │
│  ↓                                                                          │
│  Phase 5: Trend Extraction (7 scripts)                                      │
│  ↓                                                                          │
│  Phase 6: Trend + Variability Combination (7 scripts)                       │
│  ↓                                                                          │
│  Phase 7: MALI Processing (9 scripts)                                       │
│  ↓                                                                          │
│  Phase 8: Visualization (4 scripts)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. SLURM Script Reference

### Phase 1: Data Preparation (6 scripts)

| Script | Purpose | Python/NCO | Key Inputs | Key Outputs |
|--------|---------|------------|------------|-------------|
| `aislens-regrid_sorrm.sbatch` | Regrid MPAS-O to regular grid | `regrid_mpaso_data.py` | SORRM raw files | Regridded netCDF |
| `aislens-build_regridded_sorrm.sbatch` | Concatenate regridded SORRM | `concatenate_mpaso_data.py` | Regridded files | Single merged file |
| `aislens-prep_satobs.sbatch` | Preprocess satellite obs | `prepare_satobs.py` | Paolo et al. data | Preprocessed satobs |
| `aislens-prep_time_mean_model.sbatch` | Time-mean ocean model | `prepare_time_mean.py` | SORRM merged | Time-mean products |
| `aislens-prep_time_mean_split.sbatch` | Efficient time-mean (split) | `prepare_time_mean_split.py` | SORRM merged | Time-mean products |
| `aislens-prep_model_sim_fast.sbatch` | Fast model preprocessing | `prepare_model_sim_fast.py` | SORRM merged | Detrended/dedrafted |

**Workflow:**
```
Raw SORRM files → regrid → concatenate → preprocess (detrend/deseason/dedraft)
Paolo sat obs → preprocess → variability/seasonality products
```

---

### Phase 2: Draft Dependence Calculation (3 scripts)

| Script | Purpose | Python/NCO | Key Inputs | Key Outputs |
|--------|---------|------------|------------|-------------|
| `aislens-calc-draft-dependence-comprehensive-fast.sbatch` | Calculate per-shelf params | `calculate_draft_dependence_comprehensive_fast.py` | Preprocessed data | α₀, α₁, thresholds |
| `aislens-visualize-draft-dependence.sbatch` | Visualize draft-melt relationships | `visualize_draft_dependence.py` | Draft dep. params | Diagnostic plots |
| `aislens-compare-draft-dependence-param-sets.sbatch` | Compare parameter sets | `compare_param_sets.py` | Multiple param sets | Comparison report |

**Draft dependence algorithm:**
1. For each ice shelf (100 total):
   - Calculate draft-melt correlation
   - Detect changepoint in draft (if exists)
   - Fit piecewise linear model: `melt = α₀ + α₁ × draft`
   - Classify as LINEAR, PIECEWISE, or CONSTANT_MELT
2. Output per-shelf: `α₀`, `α₁`, `minDraft`, `constantMeltValue`, `paramType`

---

### Phase 3: Interpolation to MPAS Grid (2 scripts)

| Script | Purpose | Python/NCO | Key Inputs | Key Outputs |
|--------|---------|------------|------------|-------------|
| `aislens-draft-depen-paramsets_array.sbatch` | Interpolate draft-dep params | `interpolate_to_mpasli_grid_new.py` | Grid params | MALI-grid params |
| `aislens-forcings-array.sbatch` | Interpolate forcings to MALI | `interpolate_to_mpasli_grid_new.py` | Regular grid forcings | MALI-grid forcings |

**Interpolation Methods:**
- Bilinear interpolation
- Barycentric interpolation
- ESMF weight-based interpolation

---

### Phase 4: Forcing Generation (5 scripts)

| Script | Purpose | Python/NCO | Key Inputs | Key Outputs |
|--------|---------|------------|------------|-------------|
| `aislens-gen_forcings.sbatch` | Generate ensemble realizations | `generate_forcings.py` | Variability, seasonality | N realizations |
| `aislens-forcing_prep_for_regrid.sbatch` | Prepare forcings for regrid | NCO tools | Generated forcings | Prepped forcings |
| `aislens-fill-extrapolate.sbatch` | Fill NaN values | NCO/Python | Forcings w/ gaps | Filled forcings |
| `aislens-prune-forcing-vars-array.sbatch` | Remove unnecessary vars | `ncks` | Full forcings | Pruned forcings |
| `aislens-xtime-py.sbatch` | Add xtime dimension | Python | Forcings | xtime-enabled |

**EOF-based forcing generation:**
```python
# 1. EOF decomposition of variability field
eofs, pcs = eof_decomposition(variability, n_modes=50)

# 2. Phase randomization of PCs
randomized_pcs = phase_randomize(pcs)  # Preserves power spectrum

# 3. Reconstruct and add seasonality
forcing = reconstruct(eofs, randomized_pcs) + seasonality
```

---

### Phase 5: Trend Extraction (7 scripts)

| Script | Purpose | Python/NCO | Key Inputs | Key Outputs |
|--------|---------|------------|------------|-------------|
| `aislens-mali_extract_and_concat_vars.sbatch` | Extract vars from MALI output | `ncks` | MALI output | Extracted vars |
| `aislens-create-forcing-trend-array.sbatch` | Create trend files | Python | MALI output | Trend NetCDF |
| `aislens-merge-output-flux-state.sbatch` | Merge flux/state vars | `merge_state_to_flux.py` | Flux, state files | Merged output |
| `aislens-extract-trend-from-merged-array.sbatch` | Extract trend from merged | `extract_draft_and_trend_from_merged.py` | Merged output | Dedrafted + trend |
| `aislens-extract-trend-forcing.sbatch` | Single trend extraction | Python | MALI output | Trend component |
| `aislens-create-forcing-trend-component-dask.sbatch` | Dask-based trend (large data) | `create_trend_component_dask.py` | Large forcing | Trend component |
| `aislens-calc-trends-nco.sbatch` | NCO-based trend calculation | `ncap2`, `ncdiff` | Time series | Linear trend |

**Trend Extraction Methods:**
1. **Breakpoint detrending**: Detect changepoints, fit piecewise trends
2. **Linear regression**: Simple linear trend over time
3. **Draft-dependent detrending**: Remove draft effect before trend

---

### Phase 6: Trend + Variability Combination (7 scripts)

| Script | Purpose | Python/NCO | Key Inputs | Key Outputs |
|--------|---------|------------|------------|-------------|
| `aislens-add-trend-forcing-to-variability.sbatch` | Add trend to var (NCO) | `ncap2` | Trend, variability | Combined forcing |
| `aislens-add_trend_to_vargen_array.sbatch` | Add trend (Python) | `add_trend_to_vargen_xarray.py` | Trend, vargen | Combined forcing |
| `aislens-combine_forcing_components.sbatch` | Combine all components | `combine_ssp585_forcing_python.py` | Components | Final forcing |
| `aislens-create-forcings-trend-combined.sbatch` | Create combined trends | Python | Multiple trends | Merged trends |
| `aislens-create-trend-deterministic-forcing-files.sbatch` | Deterministic (no var) | NCO | Trend only | Deterministic forcing |
| `aislens-subtract-trend-var-xarray.sbatch` | Subtract trend | `subtract_trend.py` | Trend, data | Detrended data |
| `aislens-change-sign-bmb.sbatch` | Flip BMB sign convention | `ncap2` | BMB file | Sign-corrected |
| `aislens-resample-trend-forcing-to-monthly.sbatch` | Resample to monthly | `ncrcat` | Daily/yearly | Monthly |

**Combination Logic (SSP scenarios):**
```
Year 2000-2014: Variability only (baseline period)
Year 2015-2299: Variability + Trend × scale_factor
```

---

### Phase 7: MALI Processing (9 scripts)

| Script | Purpose | Python/NCO | Key Inputs | Key Outputs |
|--------|---------|------------|------------|-------------|
| `aislens-setup-mali-template.sbatch` | Setup MALI experiment | compass | Config | MALI template |
| `aislens-setup-mali-paramset-experiments.sbatch` | Setup paramset expts | `setup_mali_experiments.py` | Param sets | Experiment dirs |
| `aislens-mali_dhdt_processing_array_updated.sbatch` | Process dHdt array | Python | MALI output | dHdt fields |
| `aislens-dhdt_processing_animation.sbatch` | Animate dHdt | Python | dHdt fields | Animation |
| `aislens-process_dhdt_array.sbatch` | Process dHdt (alt) | Python | MALI output | dHdt fields |
| `aislens-mali_ensemble_processing_fast.sbatch` | Fast ensemble stats | Python | Ensemble output | Mean, std |
| `aislens-check-forcings-timeseries-array.sbatch` | Validate timeseries | Python | Forcings | Validation report |
| `aislens-create_final_forcings_with_xtime.sbatch` | Add xtime to forcings | Python | Forcings | xtime-enabled |
| `aislens-zero-variability.sbatch` | Zero out region var | `zero_region_variability.py` | Var file, mask | Zeroed var file |

**dHdt Processing:**
```
MALI output (thickness) → Δthickness/Δt → dHdt (m/yr)
→ Statistics (mean, std across ensemble)
→ Ratio maps (SSP/CTRL)
```

---

### Phase 8: Visualization (4 scripts)

| Script | Purpose | Python/NCO | Key Inputs | Key Outputs |
|--------|---------|------------|------------|-------------|
| `aislens-plot_mali_outputs.sbatch` | General MALI plotting | `plot_output_maps_masked.py` | MALI output | Maps |
| `aislens-plot_dhdt_ratio.sbatch` | dHdt ratio plots | `plot_ensemble_maps.py` | Ratio fields | Ratio maps |
| `aislens-plot_dhdt_array.sbatch` | dHdt array plots | `plot_var_maps.py` | dHdt array | Array plots |
| (unified script pending) | Animation | `create_animation.py` | Frames | MP4/GIF |

---

## 4. Python Script Reference

### Core Processing Scripts (`src/scripts/`)

| Script | Purpose | aislens Imports |
|--------|---------|-----------------|
| `calculate_draft_dependence_comprehensive_fast.py` | Per-shelf draft-melt params | `config`, `dataprep`, `utils` |
| `generate_forcings.py` | EOF-based ensemble generation | `config`, `generator`, `utils` |
| `prepare_model_sim_fast.py` | Optimized ocean model preprocessing | `config`, `dataprep`, `utils` |
| `combine_ssp585_forcing_python.py` | Combine trend + variability | `utils` |
| `add_trend_to_vargen_xarray.py` | Add trend to vargen files | (standalone) |
| `extract_draft_and_trend_from_merged.py` | Extract trend from MALI output | `config`, `dataprep` |
| `merge_state_to_flux.py` | Merge MALI flux/state vars | (standalone) |
| `zero_region_variability.py` | Zero anomalies in regions | `config`, `geospatial` |

### MPAS Tools (`src/MPAS-Tools/`)

| Script | Purpose | Notes |
|--------|---------|-------|
| `regrid_mpaso_data.py` | MPAS-O → regular grid | Uses pyremap |
| `interpolate_to_mpasli_grid_new.py` | Regular → MPAS-LI grid | Bilinear/ESMF |
| `plot_var_maps.py` | Plot variables on maps | Triangulation-based |
| `plot_output_maps_masked.py` | Masked MALI output plots | Ice extent contours |
| `plot_ensemble_maps.py` | Ensemble statistics plots | Mean/std visualization |

---

## 5. AISLENS Package Reference

### Module Structure (`src/aislens/`)

```
aislens/
├── __init__.py
├── config.py      # 125 lines - Configuration dataclass
├── dataprep.py    # 1644 lines - Data preprocessing
├── utils.py       # 1177 lines - Utility functions
├── geospatial.py  # 100 lines - Geospatial operations
├── generator.py   # 130 lines - EOF/forcing generation
└── viz.py         # 280 lines - Visualization functions
```

### Key Configuration Parameters (`config.py`)

```python
@dataclass
class Config:
    # Directory Structure
    BASE_DIR: Path              # Project root
    DATA_ROOT: Path             # Data root (AISLENS_DATA_DIR env var)
    DIR_VARGENS: Path           # Generated variability realizations
    DIR_FORCINGS: Path          # Seasonality + variability + trends
    DIR_MALI_FORCINGS: Path     # Final MALI grid forcings
    
    # Variable Names
    SATOBS_DRAFT_VAR = "draft"
    SATOBS_FLUX_VAR = "melt"
    SORRM_DRAFT_VAR = "timeMonthly_avg_ssh"
    SORRM_FLUX_VAR = "timeMonthly_avg_landIceFreshwaterFlux"
    MALI_FLOATINGBMB_VAR = "floatingBasalMassBalApplied"
    
    # Processing Parameters
    ICE_SHELF_REGIONS: range = range(33, 133)  # 100 ice shelves
    N_REALIZATIONS: int = 10
    
    # Physical Constants
    RHO_ICE: float = 910.0  # kg m⁻³
    SECONDS_PER_YEAR: float = 365 × 24 × 3600
```

### Key Functions by Module

#### `dataprep.py` - Core Processing

| Function | Purpose |
|----------|---------|
| `detrend_dim()` | Remove polynomial trend |
| `deseasonalize()` | Remove seasonal cycle |
| `dedraft_catchment()` | Linear draft-melt regression |
| `dedraft_catchment_comprehensive()` | With changepoint detection |
| `dedraft_unstructured_region()` | For MPAS unstructured grids |
| `fill_nans_by_timestep()` | Nearest-neighbor NaN fill |
| `detrend_with_breakpoints_vectorized()` | Breakpoint-aware detrending |

#### `generator.py` - Forcing Generation

| Function | Purpose |
|----------|---------|
| `eof_decomposition()` | EOF analysis via xeofs |
| `phase_randomization()` | Randomize PC phases |
| `band_limited_phase_randomization()` | Frequency-limited randomization |
| `generate_data()` | Reconstruct from randomized PCs |

#### `utils.py` - Utilities

| Category | Functions |
|----------|-----------|
| Data I/O | `save_netcdf()`, `load_iceshelf_geojson()` |
| Statistics | `compute_statistics()`, `linear_regression()` |
| Subsetting | `subset_dataset_by_time()`, `subset_by_region()` |
| Merging | `merge_catchment_data()`, `merge_datasets()` |
| Geospatial | `create_shelf_mask()`, `create_region_mask_mali()` |

---

## 6. Data Flow

### Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT DATA SOURCES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Satellite Obs              MPAS-Ocean Model            Ice Shelf Masks     │
│  (Paolo 2023)               (SORRMv2.1)                 (GeoJSON)           │
│  └─ melt, draft             └─ ssh, freshwaterFlux      └─ 100 ice shelves  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING (Phase 1-2)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Regrid MPAS-O → regular grid (pyremap)                                  │
│  2. Detrend (polynomial fit)                                                │
│  3. Deseasonalize (monthly groupby)                                         │
│  4. Dedraft per ice shelf (linear regression)                               │
│  5. Calculate draft dependence params (changepoint + piecewise linear)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INTERMEDIATE PRODUCTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │ draft_dependence │  │ sorrm_seasonality│  │ sorrm_variability        │   │
│  │ α₀, α₁, minDraft │  │ monthly climate  │  │ detrended + dedrafted    │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FORCING GENERATION (Phase 4)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. EOF decomposition of variability (xeofs, n_modes=50)                    │
│  2. Phase randomization of PCs (preserves power spectrum)                   │
│  3. Reconstruct N realizations                                              │
│  4. Add seasonality: forcing = variability_realization + seasonality        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TREND COMBINATION (Phase 5-6)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Extract trend from MALI CTRL/SSP outputs                                │
│  2. Dedraft the MALI output                                                 │
│  3. Combine: final_forcing = variability + seasonality + trend × scale      │
│                                                                             │
│     Timeline: 2000-2014 (baseline) │ 2015-2299 (trend applied)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GRID INTERPOLATION (Phase 3)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Regular grid → MPAS-LI unstructured mesh                                   │
│  Methods: Bilinear, Barycentric, ESMF weights                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FINAL OUTPUTS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  MALI Forcings (MPAS-LI grid)    Draft Dep. Params        Trend Scenarios   │
│  └─ floatingBasalMassBalAdj      └─ Alpha0, Alpha1        └─ SSP585/SSP126  │
│  └─ N ensemble realizations      └─ minDraft, paramType   └─ Per-region     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key File Paths

```
data/
├── external/
│   ├── ANT_G1920V01_IceShelfMeltDraft_Time.nc    # Satellite obs
│   ├── Regridded_SORRMv2.1.ISMF.FULL.nc          # Ocean model
│   └── iceShelves.geojson                         # Ice shelf masks
├── interim/
│   ├── sorrm_seasonality*.nc                      # Seasonal climatology
│   └── sorrm_variability*.nc                      # Dedrafted variability
├── processed/
│   ├── draft_dependence/                          # Per-shelf params
│   ├── vargens/                                   # Generated realizations
│   └── forcings/                                  # Final forcings
└── MALI/
    ├── forcings/                                  # MALI-grid forcings
    └── outputs/                                   # MALI run outputs
```

---

## 7. Consolidation Plan

This section records the cleanup plan used while refactoring the old scripts. It serves as a record of what scripts used on PACE HPC got merged, and what remains the same from earlier versions (~2023 onwards vs. 2026)

### Current State: 43 Active Scripts → ~12 Refactored Scripts

The refactoring is implemented in `src/slurm-jobs/refactored/`.

### New Naming Convention

Scripts now have a **phase-prefixed naming scheme**:

```
XX-phase-task.sbatch
```

| Prefix | Phase | Example |
|--------|-------|---------|
| `00-` | Common | `00-common-header.sh` |
| `01-` | Data Prep | `01-dataprep-time-mean.sbatch` |
| `05-` | Trend | `05-trend-extract.sbatch` |
| `06-` | Combine | `06-combine-trend-var.sbatch` |
| `07-` | MALI | `07-mali-dhdt-process.sbatch` |
| `08-` | Viz | `08-viz-mali.sbatch` |
| `util-` | Utilities | `util-debug-prepare-workflow.sbatch` |

### Implemented Consolidations

| Old Scripts | New Script | Key Modes |
|-------------|------------|-----------|
| `aislens_prep_time_mean_model.sbatch`<br>`aislens_prep_time_mean_split.sbatch` | `01-dataprep-time-mean.sbatch` | `--mode full\|split` |
| `aislens_mali_dhdt_processing_array_updated.sbatch`<br>`process_dhdt_array.sbatch`<br>`aislens_dhdt_processing_animation.sbatch` | `07-mali-dhdt-process.sbatch` | `--mode process\|animate\|all` |
| `add-trend-forcing-to-variability.sbatch`<br>`add_trend_to_vargen_array.sbatch`<br>`aislens_combine_forcing_components.sbatch` | `06-combine-trend-var.sbatch` | `--method python\|nco\|batch` |
| `aislens-extract-trend-from-merged-array.sbatch`<br>`extract-trend-forcing.sbatch`<br>`calc-trends-nco.sbatch` | `05-trend-extract.sbatch` | `--method merged\|nco\|single\|dask` |
| `aislens_plot_mali_outputs.sbatch`<br>`aislens_plot_dhdt_ratio.sbatch`<br>`plot_dhdt_array.sbatch` | `08-viz-mali.sbatch` | `--type output\|ratio\|dhdt\|ensemble` |

### Restored Utility Scripts

These unique scripts from the older archive have been restored:

| Script | Purpose | Original |
|--------|---------|----------|
| `util-inspect-draft-depen.sbatch` | Validate draft-dep calculations | `aislens_inspect_shelf_pair_*.sbatch` |
| `util-debug-prepare-workflow.sbatch` | Debug data preparation | `debug-prepare-workflow.sbatch` |
| `util-thickness-analysis.sbatch` | Ensemble thickness analysis | `aislens_mali_thickness_analysis.sbatch` |

### Usage Examples

```bash
# Time-mean with mode selection
sbatch refactored/01-dataprep-time-mean.sbatch --mode split --coarsen 2

# dH/dt processing for SSP585
sbatch --array=0-9 refactored/07-mali-dhdt-process.sbatch --scenario SSP585 --mode process

# Combine trend + variability (array job)
sbatch --array=0-19 refactored/06-combine-trend-var.sbatch --scenario SSP585 --method python

# Visualization
sbatch refactored/08-viz-mali.sbatch --type ratio --scenario SSP585

# Dry run to preview
bash refactored/07-mali-dhdt-process.sbatch --scenario SSP585 --dry-run
```

### Migration Path

1. Test the refactored scripts in `src/slurm-jobs/refactored/`
2. Compare outputs against the original scripts
3. Move the old scripts into `src/slurm-jobs/legacy/` once the replacements are verified
4. Promote the refactored scripts if they should become the default entry point
5. Update any downstream workflows that still point at the old names

### Standardization Patterns

All refactored scripts follow these patterns:

```bash
#!/bin/bash
#SBATCH --job-name=<phase>-<task>
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --mail-user=${SLURM_EMAIL}

# Standard argument parsing
--scenario SCENARIO   # CTRL, SSP585, SSP126
--mode MODE           # Task-specific mode
--dry-run             # Preview without executing
--array-index N       # Override SLURM_ARRAY_TASK_ID
-h, --help            # Show usage

# Validation and logging
set -euo pipefail
echo "========================================"
echo "Job description"
echo "========================================"
```

### NCO Tool Usage Summary

| Tool | Count | Primary Uses |
|------|-------|--------------|
| `ncks` | 8 | Variable extraction, subsetting |
| `ncap2` | 7 | Variable arithmetic, sign flips |
| `ncrcat` | 3 | Time concatenation |
| `ncecat` | 2 | Ensemble concatenation |
| `ncwa` | 2 | Weighted averaging |
| `ncdiff` | 2 | File differencing |

### Potentially Orphaned Scripts (Original)

| Script | Issue | Recommendation |
|--------|-------|----------------|
| `aislens-xtime-py.sbatch` | References external path | Verify/update path or archive |
| `aislens-setup-mali-template.sbatch` | Requires compass dependency | Document dependency or archive |

---

## 8. Archived Scripts Analysis

### Overview

The `src/slurm-jobs/archive/` directory contains **62 archived scripts**.

### Classification Summary

| Category | Count | Description |
|----------|-------|-------------|
| REDUNDANT | 36 | Fully replaced by active scripts |
| PARTIALLY COVERED | 10 | Some features missing from active scripts |
| UNIQUE/ORPHANED | 9 | **No active replacement - important!** |
| ONE-OFF UTILITY | 7 | Specific tasks that might be useful again |

---

### UNIQUE/ORPHANED Scripts (Requires Attention)

These scripts have functionality **NOT present in any active script**:

| Script | Unique Functionality | Python Script | Notes |
|--------|---------------------|---------------|-------|
| `aislens-inspect-draft-depen.sbatch` | Compare fast vs parallel draft-dep results per shelf | `inspect_shelf_pair.py` | Validation/debugging tool |
| `aislens-forcing-trend-ismip6-nco.sbatch` | NCO time-series averaging for ISMIP6 outputs | NCO only | References Trevor's data |
| `aislens-forcing-trend-adjust-mean.sbatch` | Subtract initial timestep from trend (normalization) | NCO (`ncap2`) | Unique algorithm |
| `aislens-prep_model_sim_fast_test_nodelocal.sbatch` | Node-local scratch caching strategy | `prepare_model_sim_fast.py` | HPC optimization |
| `aislens-debug_prepare_workflow.sbatch` | Debug workflow runner | `debug_prepare_workflow.py` | Troubleshooting |
| `aislens-mali_ensemble_thickness_analysis.sbatch` | Ensemble thickness variability analysis | `plot_ensemble_thickness_analysis.py` | Specialized analysis |
| `aislens-forcing-trend-ismip6-expAE10.sbatch` | Extract BMB from ISMIP6 expAE10 | NCO only | External data |
| `aislens-forcing-trend-ismip6-expAE05.sbatch` | Extract BMB from ISMIP6 expAE05 | NCO only | External data |
| `aislens-convert-to-ts.sbatch` | Time-series conversion | `convert-to-ts.py` | External script |

#### Associated Python Scripts (All Still Exist!)

| Script | Location | Purpose |
|--------|----------|---------|
| `inspect_shelf_pair.py` | `src/scripts/` | Compare draft-dep results between methods |
| `debug_prepare_workflow.py` | `src/scripts/` | Debug data preparation pipeline |
| `plot_ensemble_thickness_analysis.py` | `src/MPAS-Tools/` | Ensemble thickness statistics |
| `plot_ensembleDiffs.py` | `src/MPAS-Tools/` | Ensemble difference visualization |
| `plot_output_maps_masked_animation.py` | `src/MPAS-Tools/` | Animation generation |
| `prepare_model_sim_simple.py` | `src/scripts/` | Simplified model prep (vs fast) |

---

### PARTIALLY COVERED Scripts (Features to Consider)

| Script | Missing Feature in Active | Recommendation |
|--------|--------------------------|----------------|
| `aislens-calc-draft-depen-comprehensive_DEBUG.sbatch` | Debug mode, paramset variants (fb_A/B/C, sc_A/B) | Add `--debug` flag |
| `aislens-forcing-trend-ctrl-single.sbatch` | CTRL-specific year range detection | Add CTRL mode |
| `aislens-forcing-trend-create-forcing-trend.sbatch` | TF-INITIAL-ONLY variant processing | Document edge case |
| `aislens-dhdt_processing_animation.sbatch` | Uses different animation script | Verify active script |
| `aislens-interpolate-draft-depen-piecewise.sbatch` | Piecewise interpolation variant | May be covered |
| `aislens-mali_ensemble_processing.sbatch` | Uses `plot_ensembleDiffs.py` | Check vs `plot_ensemble_maps.py` |
| `aislens-fillnan.sbatch` | Simple `ncap2` NaN→0 replacement | Simpler than fill-extrapolate |
| `aislens-mali_plot_output.sbatch` | Uses `plot_var_maps.py` | May differ from active |
| `aislens-prep_model_sim_simple.sbatch` | Uses different Python script | Check if simpler is needed |
| `aislens-forcing-trend-nco.sbatch` | 12-month repetition logic | Document or merge |

---

### ONE-OFF UTILITY Scripts (Keep for Reference)

| Script | When Useful |
|--------|-------------|
| `aislens-interpolate-draft-depen.sbatch` | Setting up new MPAS meshes |
| `aislens-prep_model_sim_fast_test_smoketest.sbatch` | Quick extrapolation testing |
| `aislens-prep_model_sim_fast_test_simple_extrap.sbatch` | Testing extrapolation variants |
| `aislens-prep_model_sim_fast_test_indexmap.sbatch` | Cache debugging |
| `aislens-prep_model_sim_fast_test_persistent.sbatch` | Cache optimization |
| `aislens-forcing-trend-ismip6-nco-test.sbatch` | Quick NCO data inspection |
| `aislens-mali_extract_forcings.sbatch` | One-time variable extraction |

---

### Scripts Referencing External Data

These scripts depend on external data sources that may need documentation:

| Script | External Reference | Status |
|--------|-------------------|--------|
| `aislens-forcing-trend-ismip6-expAE10.sbatch` | `ISMIP6_outputs_from_Trevor/expAE10/` | Document source |
| `aislens-forcing-trend-ismip6-expAE05.sbatch` | `ISMIP6_outputs_from_Trevor/expAE05/` | Document source |
| `aislens-forcing-trend-ismip6-nco.sbatch` | Trevor's ISMIP6 outputs | Document source |
| `aislens-interpolate-draft-depen.sbatch` | `AIS_8to30km_20221027/` mesh | MALI project file |
| `aislens-convert-to-ts.sbatch` | `${HOME}/scratch/convert-to-ts.py` | External script |
| `aislens-mali_plot_output.sbatch` | `${HOME}/scratch/ISMIP6-SSP585-4KM/` | External output dir |

---

### Recommendations going forward

#### HIGH - possibly restore?

1. **`aislens-inspect-draft-depen.sbatch`** → Valuable validation tool
   - Python script `inspect_shelf_pair.py` exists
   - Useful for verifying draft dependence calculations

2. **`aislens-mali_ensemble_thickness_analysis.sbatch`** → Unique analysis
   - Python script `plot_ensemble_thickness_analysis.py` exists
   - No equivalent in active scripts

3. **`aislens-debug_prepare_workflow.sbatch`** → Debugging utility
   - Python script `debug_prepare_workflow.py` exists
   - Useful for troubleshooting data prep issues

#### MEDIUM - document for future use

4. **`aislens-forcing-trend-adjust-mean.sbatch`**
   - Contains unique NCO pattern: `ncap2 -s 'floatingBasalMassBalAdjustment = floatingBasalMassBalAdjustment - floatingBasalMassBalAdjustment(0,:)'`
   - Documents trend normalization approach

5. **`aislens-prep_model_sim_fast_test_nodelocal.sbatch`**
   - HPC optimization: copies cache to `$TMPDIR` for node-local I/O
   - Useful pattern for large-scale runs

#### LOW PRIORITY - notes on archiving

6. All scenario-specific variants (SSP585, SSP126, CTRL) are now consolidated into parameterized active scripts, don't need them anymore.
7. Test/smoke-test scripts documented here for reference if needed

---

## Appendix A: Quick Reference

### Environment Setup

```bash
# Source configuration
source src/slurm-jobs/config/aislens_env.sh

# Required environment variables (in slurm_config)
export SLURM_ACCOUNT="your-account"
export SLURM_EMAIL="your-email@example.com"

# Activate conda environment
conda activate mpas-analysis
```

### Common Commands

```bash
# Run a single script
sbatch src/slurm-jobs/aislens-<script>.sbatch

# Run array job
sbatch --array=0-9 src/slurm-jobs/aislens-<script>_array.sbatch

# Check job status
squeue -u $USER

# View output
cat slurm-*.out
```

### Physical Constants

| Constant | Value | Units |
|----------|-------|-------|
| Ice density (ρ_ice) | 910.0 | kg m⁻³ |
| Seconds per year | 31,536,000 | s |
| Ice shelves | 100 | (regions 33-133) |
| SORRM time range | 450-750 | model years |

---

## Appendix B: Archived Scripts Full List

### By Category (62 total)

#### Draft Dependence (6)
- `aislens-calc-draft-depen.sbatch` → REDUNDANT
- `aislens-calc-draft-depen-comprehensive.sbatch` → REDUNDANT
- `aislens-calc-draft-depen-comprehensive_PARALLEL.sbatch` → REDUNDANT
- `aislens-calc-draft-depen-comprehensive_DEBUG.sbatch` → PARTIALLY COVERED
- `aislens-inspect-draft-depen.sbatch` → **UNIQUE**
- `aislens-visualize-draft-depen-comprehensive_compare-param-sets_DEBUG.sbatch` → PARTIALLY COVERED

#### Forcing/Trend Creation (12)
- `aislens-add-forcing-trend.sbatch` → REDUNDANT
- `aislens-forcing-trend-create-test.sbatch` → REDUNDANT
- `aislens-forcing-trend-create-ssp126.sbatch` → REDUNDANT
- `aislens-forcing-trend-ctrl-single.sbatch` → PARTIALLY COVERED
- `aislens-forcing-trend-create-components.sbatch` → REDUNDANT
- `aislens-forcing-trend-create-forcings-trend-combined.sbatch` → REDUNDANT
- `aislens-forcing-trend-create-forcing-trend.sbatch` → PARTIALLY COVERED
- `aislens-forcing-trend-components.sbatch` → REDUNDANT
- `aislens-forcing-trend-ismip6-nco.sbatch` → **UNIQUE**
- `aislens-forcing-trend-nco.sbatch` → PARTIALLY COVERED
- `aislens-subtract-trend-var.sbatch` → REDUNDANT
- `aislens-forcing-trend-adjust-mean.sbatch` → **UNIQUE**

#### MALI Ensemble Processing (15)
- `aislens-mali_dhdt_processing.sbatch` → REDUNDANT
- `aislens-mali_dhdt_processing_array.sbatch` → REDUNDANT
- `aislens-mali_dhdt_processing_array_ssp585.sbatch` → REDUNDANT
- `aislens-mali_dhdt_processing_array_ssp126.sbatch` → REDUNDANT
- `aislens-mali_ensemble_processing.sbatch` → PARTIALLY COVERED
- `aislens-mali_ensemble_processing_ctrl.sbatch` → REDUNDANT
- `aislens-mali_ensemble_processing_ctrl-ssn.sbatch` → REDUNDANT
- `aislens-mali_ensemble_processing_ctrl-ssn_fast.sbatch` → REDUNDANT
- `aislens-mali_ensemble_processing_ssp126.sbatch` → REDUNDANT
- `aislens-mali_ensemble_processing_ssp585.sbatch` → REDUNDANT
- `aislens-process_dhdt.sbatch` → REDUNDANT
- `aislens-process_dhdt_array.sbatch` → REDUNDANT
- `aislens-plot_dhdt_array.sbatch` → REDUNDANT
- `aislens-plot_dhdt_array_ssp585.sbatch` → REDUNDANT
- `aislens-plot_mali_thickness.sbatch` → REDUNDANT

#### Interpolation/Regridding (5)
- `aislens-interpolate-draft-depen.sbatch` → ONE-OFF UTILITY
- `aislens-interpolate-draft-depen-piecewise.sbatch` → PARTIALLY COVERED
- `aislens-interpolate-forcings_array.sbatch` → REDUNDANT
- `aislens-interpolate-draft-depen-param.sbatch` → REDUNDANT
- `aislens-regrid-8to30km.sbatch` → REDUNDANT

#### Data Preparation (11)
- `aislens-data_prep.sbatch` → REDUNDANT
- `aislens-prep_model_sim.sbatch` → REDUNDANT
- `aislens-prep_model_sim_custom.sbatch` → REDUNDANT
- `aislens-prep_model_sim_fast_test_smoketest.sbatch` → ONE-OFF UTILITY
- `aislens-prep_model_sim_fast_test_simple_extrap.sbatch` → ONE-OFF UTILITY
- `aislens-prep_model_sim_fast_test_indexmap.sbatch` → ONE-OFF UTILITY
- `aislens-prep_model_sim_fast_test_nodelocal.sbatch` → **UNIQUE**
- `aislens-prep_model_sim_fast_test_persistent.sbatch` → ONE-OFF UTILITY
- `aislens-prep_model_sim_fast_test_skip_extrap.sbatch` → ONE-OFF UTILITY
- `aislens-prep_model_sim_simple.sbatch` → PARTIALLY COVERED
- `aislens-debug_prepare_workflow.sbatch` → **UNIQUE**

#### Visualization (5)
- `aislens-dhdt_processing_animation.sbatch` → PARTIALLY COVERED
- `aislens-visualize-draft-depen.sbatch` → REDUNDANT
- `aislens-plot_dhdt_ratio_ctrl.sbatch` → REDUNDANT
- `aislens-plot_dhdt_ratio_ssp126.sbatch` → REDUNDANT
- `aislens-plot_dhdt_ratio_ssp585.sbatch` → REDUNDANT

#### ISMIP6 Specific (4)
- `aislens-mali_plot_output.sbatch` → PARTIALLY COVERED
- `aislens-forcing-trend-ismip6-expAE10.sbatch` → **UNIQUE**
- `aislens-forcing-trend-ismip6-expAE05.sbatch` → **UNIQUE**
- `aislens-forcing-trend-ismip6-nco-test.sbatch` → ONE-OFF UTILITY

#### Miscellaneous (4)
- `aislens-fillnan.sbatch` → PARTIALLY COVERED
- `aislens-mali_ensemble_thickness_analysis.sbatch` → **UNIQUE**
- `aislens-workflow.sbatch` → REDUNDANT
- `aislens-convert-to-ts.sbatch` → **UNIQUE**