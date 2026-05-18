## Draft Dependence Parameter Optimization Notes

After running through multiple parameter sets, two configurations stood out: `fine_binning` and `sensitive_changepoint`. Both got the model outputs closer to observed regional melt rates (lower RMSE/MAE, higher R²) compared to the original conservative settings.

I also unified the units across both the fast and parallel pipelines so everything uses SI mass-flux units (kg m^-2 s^-1). This makes it easier to compare results across runs.

## What was changed

**Units** — I centralized `RHO_ICE` and `SECONDS_PER_YEAR` in `src/aislens/config.py` so the conversion factor (m yr^-1 → kg m^-2 s^-1) is consistent everywhere. The parallel workers now auto-detect the SATOBS units and convert on-the-fly. All output NetCDFs now include SI unit attributes.

**Tools for validation** — a few scripts to inspect results:
- `src/scripts/convert_param_units.py` — handy CLI when you need to batch-convert old NetCDF files
- `tests/test_units_conversion.py` — unit test to catch any conversion regressions
- `src/MPAS-Tools/plot_ensembleRegionalStats.py` — upgraded to support per-experiment colors, `--colormap`, and `--legend-per-experiment` options
- `src/scripts/compare_regionalStats_metrics.py` — computes RMSE/MAE/R² across regions for any two experiments
- `src/scripts/per_shelf_diagnostics.py` — plots per-shelf scatter, residuals, bin counts, and parameter snippets

What the top parameter sets changed (behavioral summary)
- fine_binning
  - Increases the number of bins used to aggregate draft→melt relationships (e.g., `n_bins=100`).
  - Pros: resolves finer spatial heterogeneity; can capture localized relationships.
  - Cons: requires sufficient samples per bin; low `min_points_per_bin` risks overfitting.

- sensitive_changepoint
  - Lowers the changepoint detection penalty so the model accepts more breakpoints in piecewise fits.
  - Pros: detects abrupt regime changes better and reduces residuals where the relationship is nonstationary.
  - Cons: can overfit by fitting to noise if penalties are too small.

## How to interpret "best fit"

Lower RMSE/MAE and higher R² are good signals, but they are not enough. Also check:
- **Per-shelf residuals and parameter maps** — salt-and-pepper patterns are a red flag for overfitting
- **Bin sample counts** — lots of empty bins or bins with small number of samples aren't reliable
- **Out-of-sample RMSE** — if possible, do a quick cross-val to make sure it generalizes beyond the training data (see next section for the algorithm)
Verificatoin steps:
  Run `per_shelf_diagnostics.py` for a small sample of important shelves to inspect scatter, residuals,
   parameter maps, and bin-count histograms.
  Run a cross-validation-based per-shelf model-selection routine (see algorithm sketch below) to ensure
   chosen parameter sets generalize out-of-sample.

## Cross-validation quick algorithm

1. For each shelf, grab all (draft, observed flux) pairs
2. Split into K spatial folds (K=3–5 works well)
3. For each candidate parameter set and each fold:
   - fit the params using training folds only
   - test on the held-out fold, compute RMSE
4. Average CV-RMSE across folds — winner is the parameter set with lowest CV-RMSE
5. Only apply this to shelves with >30–50 samples; use a default for sparse shelves

**Speed vs robustness trade-off:** Full CV (retraining per fold) is most robust but slow. In the interest of time, one approximation is to just evaluate held-out points against the full-run parameter maps — faster but can bias toward overfitting.

Files of interest
- `src/scripts/convert_param_units.py` — conversion CLI
- `src/aislens/config.py` — centralized constants (`RHO_ICE`, `SECONDS_PER_YEAR`)
- `src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py` — worker conversion and scalar writing
## Key files

- `src/aislens/config.py` — constants go here (`RHO_ICE`, `SECONDS_PER_YEAR`)
- `src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py` — the worker does unit conversion here
- `src/scripts/convert_param_units.py` — batch-convert old param files if needed
- `src/scripts/compare_regionalStats_metrics.py` — go-to for comparing two experiments
- `src/scripts/per_shelf_diagnostics.py` — visual inspection of individual shelves
- `src/MPAS-Tools/plot_ensembleRegionalStats.py` — updated plotting with nicer legend and colormap options
- `tests/test_units_conversion.py` — catches unit conversion bugs