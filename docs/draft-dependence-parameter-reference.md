## Draft-dependence parameter sets — notes and guidance

Purpose
-------

This note summarizes the parameter presets in
`src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py`, explains what each parameter controls, give practical ranges and search strategies, and show example commands I use when testing parameter sweeps.

Parameters (what they control)
-----------------------------

- `min_r2_threshold`: the minimum R² required for a fit to be treated as meaningful. Higher values -> more conservative (fewer 'meaningful' shelves).
- `min_correlation`: minimum Pearson correlation between draft and melt. Set this based on the expected sign/magnitude of the relationship.
- `ruptures_penalty`: penalty used by the ruptures changepoint detector. Lower => more breakpoints; higher => fewer breakpoints.
- `n_bins`: number of draft bins when aggregating data. More bins = finer vertical resolution but fewer samples per bin.
- `min_points_per_bin`: minimum samples required for a bin to be kept. Raise this to avoid noisy bin statistics.
- `noisy_fallback`: behavior when no fit passes thresholds. Typical choices are `mean` (use observed mean melt) or `zero` (use zero melt).
- `model_selection`: the selection strategy used to pick a parameterization (e.g., `threshold_intercept`). The dataprep code uses this with the fit metrics to choose between constant, linear, piecewise, etc.
- `description`: human-readable text for the preset.

Presets in the code (short interpretation)
---------------------------------------------------

- `standard` — Balanced defaults. Good starting point for broad runs.
- `permissive` — Low thresholds and looser bin rules. Finds more candidate relationships, at the cost of noisier fits.
- `strict` — Conservative thresholds and higher `min_points_per_bin`. Keeps only high-confidence parameterizations.
- `sensitive_changepoint` — Low `ruptures_penalty` to pick up many breakpoints. Use if you expect multi-stage behavior.
- `robust_changepoint` — High `ruptures_penalty`, so only strong changepoints are kept.
- `fine_binning` — More bins (n_bins = 100) and lower per-bin sample requirement; use only when data are dense.

How the pipeline uses these values
----------------------------------

1. Bin the data by draft (`n_bins`). Discard bins with fewer than `min_points_per_bin`.
2. Compute binned statistics and/or use raw samples for fitting.
3. Run changepoint detection with `ruptures_penalty` to propose piecewise structure.
4. Fit candidate models (constant, linear, piecewise) and score them (R², correlation).
5. Use `model_selection` plus thresholds to pick a final parameterization; if none pass, use `noisy_fallback`.

Quick advice for testing
------------------------

- Start with a small pilot: pick 3–6 representative shelves (large, medium, small, noisy) and sweep parameters there.
- Search strategies I find useful:
  - Grid search over 2–3 parameters (easy to interpret).
  - Random/Latin hypercube sampling for broader coverage if parameters are many.
  - Sequential refinement: coarse grid → inspect → refine.

Reasonable starting ranges
-------------------------

- `ruptures_penalty`: try [0.1, 0.3, 0.5, 1.0, 2.0]
- `min_r2_threshold`: try [0.001, 0.005, 0.01, 0.05]
- `min_correlation`: if you expect a negative relationship, try [-0.9, -0.7, -0.5, -0.3]
- `n_bins`: try [25, 50, 100]
- `min_points_per_bin`: try [3, 5, 10]

Preset ideas to add
-------------------

If you want more presets, I recommend adding:

- `conservative`: higher `min_r2_threshold` and `min_points_per_bin`, larger `ruptures_penalty`.
- `very_fine`: many bins (n_bins=200) and a low per-bin threshold — use only with dense data.
- `noisy_zero`: same detection settings as `standard` but `noisy_fallback='zero'`.

Evaluating parameter sets
-------------------------

- Fraction of shelves labeled 'meaningful' (not falling back).
- Distribution of `paramType` (constant / linear / piecewise).
- Fit metrics among meaningful shelves: mean/median R² and correlation.
- Prediction diagnostics (MAE, RMSE, bias) if you have holdout data or clear validation splits.
- Visual checks: maps of `alpha1`, `minDraft`, etc. and per-shelf predicted vs observed plots.
- Stability: bootstrap samples per shelf to assess parameter spread.

Common pitfalls
--------------

- Overfitting from very low `ruptures_penalty` or overly permissive thresholds.
- Unstable estimates on sparse shelves; guard with `min_points_per_bin`.
- Binning artifacts: too many bins with sparse data is noisy; too few bins hides structure.
- Unit mismatches: older grid outputs may be in `kg m^-2 s^-1` while the parallel scalars are written in `m yr^-1` — pay attention to units when comparing.

Example commands
----------------

Run the `standard` preset (skips existing outputs by default):

```zsh
python src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py --n-workers 8 --parameter-sets standard
```

Run all presets:

```zsh
python src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py --n-workers 8 --test-all-sets
```

Run a hand-picked list:

```zsh
python src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py --n-workers 8 --parameter-sets standard permissive strict
```

Rerun only selected shelves:

```zsh
python src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py --n-workers 8 --parameter-sets standard --rerun-shelves Amery Brahms Pine_Island
```

Or use a file with one shelf per line:

```zsh
python src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py --n-workers 8 --parameter-sets standard --rerun-file rerun_list.txt
```

Comparing results
-----------------

After a run the parallel pipeline writes per-shelf scalars as `draftDepenBasalMelt_params_{shelf}.nc`. I then use the compare script (it prefers those scalars):

```zsh
python src/scripts/compare_parameter_sets_debug.py --base-dir <interim-dir> --param-sets standard permissive strict --shelves Amery Brahms Pine_Island
```

Pilot experiment plan
---------------------

1. Pick 3–4 shelves that represent the range of data density/quality.
2. Sweep `ruptures_penalty` ∈ {0.1, 0.5, 1.0} and `min_r2_threshold` ∈ {0.001, 0.005, 0.01} with `n_bins` = 50 and `min_points_per_bin` = 5.
3. Use `compare_parameter_sets_debug.py` and `inspect_shelf_pair.py` to inspect scalars, grid stats and sample arrays.
4. Scale up promising parameter regions.

Automation suggestions
--------------------

If many sweeps are needed:

- Write a small driver that generates parameter-set dicts, runs the parallel script in a loop or via a scheduler, and collects summary CSVs and scalar files for automated evaluation.
- Keep presets documented in `define_parameter_sets()` and keep the `description` field accurate.
- Consider adding a `debug_dump_full_results` flag to persist per-shelf `full_results` for debugging.

---

For more details, see: `src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py`
