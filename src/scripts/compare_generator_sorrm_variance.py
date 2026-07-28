#!/usr/bin/env python3
"""
compare_generator_sorrm_variance.py — compares generator vs SORRM band-limited variance.

Measures per-cell band variance of detrended melt-anomaly fields and reports the
MEDIAN ratio generator/SORRM over a strict valid-cell subset. Uses a relative threshold
to handle different null-cell fractions between MALI mesh and SORRM grid.

CRITICAL: SORRM is only 50 annual values — periods >25 yr are NOT testable.
Generator-vs-generator ratios are mesh-independent and more defensible than
generator-vs-SORRM. Run --self-test before trusting output.

Author: Shivaprakash Muruganandham
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np


# --------------------------------------------------------------------------------------
# core numerics
# --------------------------------------------------------------------------------------

def linear_detrend(arr: np.ndarray) -> np.ndarray:
    """Remove a per-column linear trend along axis=0. arr: (nTime, nCells) float32."""
    n = arr.shape[0]
    t = np.arange(n, dtype=np.float32)
    t = t - t.mean()
    denom = float((t * t).sum())
    arr_mean = arr.mean(axis=0, keepdims=True)
    arr_c = arr - arr_mean
    if denom == 0:
        return arr_c
    slope = (t[:, None] * arr_c).sum(axis=0) / denom
    return arr_c - slope[None, :] * t[:, None]


def bandpass_variance(arr: np.ndarray, period_lo: float, period_hi: float,
                       dt: float = 1.0) -> np.ndarray:
    """
    Per-cell variance of arr restricted to periods in [period_lo, period_hi] years.
    arr: (nYears, nCells) float32, already detrended. Returns (nCells,) float32.

    Implementation: FFT along axis=0, zero out bins outside the frequency band
    [1/period_hi, 1/period_lo], inverse FFT, variance of the filtered series. Equivalent
    (Parseval) to integrating the periodogram over the band, but vectorized across all
    cells at once instead of looping scipy.signal.welch per cell.
    """
    n = arr.shape[0]
    freqs = np.fft.rfftfreq(n, d=dt)  # cycles / year
    f_lo = 1.0 / period_hi
    f_hi = 1.0 / period_lo
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)
    spec = np.fft.rfft(arr, axis=0)
    spec_filt = np.where(band_mask[:, None], spec, 0)
    filtered = np.fft.irfft(spec_filt, n=n, axis=0)
    return filtered.astype(np.float32).var(axis=0)


def modal_fraction(values: np.ndarray, sigfigs: int = 6) -> tuple[float, float]:
    """
    Fraction of cells sitting at the single most common value (grouped to `sigfigs`
    significant figures), plus that modal value itself.

    DIAGNOSTIC PURPOSE: a regridded field whose output cells inherit near-duplicate values
    from the same coarser source cell -- or a field padded with a constant fill value --
    shows a large modal fraction. That is a bookkeeping artifact, not physical structure,
    and it warns that quantiles computed over such a field describe the duplication rather
    than the melt signal. Grouping is done to a RELATIVE precision (significant figures,
    not absolute rounding) because near-duplicates in these files differ only in the last
    floating-point ULP.
    """
    v = values[np.isfinite(values) & (values > 0)]
    if v.size == 0:
        return float("nan"), float("nan")
    exp = np.floor(np.log10(v))
    mant = np.round(v / 10.0 ** exp, sigfigs - 1)
    # stable integer key = (mantissa digits, decade), avoids float-equality fragility
    key = (mant * 10 ** (sigfigs - 1)).astype(np.int64) * 10000 + exp.astype(np.int64)
    uniq, counts = np.unique(key, return_counts=True)
    imax = int(np.argmax(counts))
    frac = float(counts[imax] / v.size)
    modal_val = float(np.median(v[key == uniq[imax]]))
    return frac, modal_val


def field_stats(annual: np.ndarray, band: tuple[float, float], weights: np.ndarray,
                 valid_floor: float) -> dict:
    """
    annual: (nYears, nCells) float32 raw (not yet detrended) annual-mean series.

    Produces the per-cell band variance and then selects a STRICT valid subset using a
    RELATIVE, self-adapting threshold:

        keep cells with band_var > valid_floor * p99.9(band_var over candidate cells)

    WHY RELATIVE, NOT ABSOLUTE: an absolute "variance > epsilon" test lets through a large
    degenerate mass of cells whose variance is numerically zero (float noise from regridded
    duplicates, masked/grounded cells, fill values). Those null cells dilute any MEAN, and
    -- critically -- they occur in very different proportions on the MALI mesh (~385k cells,
    mostly grounded) than on the SORRM 601x601 grid. A ratio computed over such sets is
    driven by grid bookkeeping rather than by melt amplitude. Scaling the threshold to each
    field's OWN p99.9 makes both sides describe the same thing: "cells that actually carry
    melt variability", in each grid's own dynamic range.
    """
    with warnings.catch_warnings():
        # all-NaN columns (masked / out-of-domain cells) are EXPECTED and handled via the
        # isfinite() check below; suppress numpy's noisy per-column RuntimeWarning for them.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw_std = np.nanstd(annual, axis=0)

    n_total = int(annual.shape[1])
    # candidate = "has any variability at all" (the OLD, dilution-prone criterion)
    candidate = np.isfinite(raw_std) & (raw_std > 1e-12)

    detrended = linear_detrend(np.nan_to_num(annual, nan=0.0))
    band_var = bandpass_variance(detrended, band[0], band[1])

    if candidate.sum() == 0:
        return {"n_total": n_total, "n_candidate": 0, "n_valid": 0, "valid_frac": float("nan"),
                "band_var": band_var, "strict": candidate, "p999": float("nan"),
                "threshold": float("nan"), "mean": float("nan"), "wmean": float("nan"),
                "median": float("nan"), "p75": float("nan"), "p90": float("nan"),
                "modal_frac": float("nan"), "modal_val": float("nan")}

    p999 = float(np.percentile(band_var[candidate], 99.9))
    threshold = valid_floor * p999
    strict = candidate & (band_var > threshold)

    v = band_var[strict]
    if v.size == 0:
        med = p75 = p90 = mean = wmean = float("nan")
    else:
        med, p75, p90 = (float(x) for x in np.percentile(v, [50, 75, 90]))
        mean = float(v.mean())
        w = weights[strict]
        wsum = float(w.sum())
        wmean = float((v * w).sum() / wsum) if wsum > 0 else mean

    modal_frac, modal_val = modal_fraction(band_var[candidate])

    return {
        "n_total": n_total,
        "n_candidate": int(candidate.sum()),
        "n_valid": int(v.size),
        "valid_frac": float(v.size / n_total),
        "band_var": band_var,
        "strict": strict,
        "p999": p999,
        "threshold": threshold,
        "mean": mean,
        "wmean": wmean,
        "median": med,
        "p75": p75,
        "p90": p90,
        "modal_frac": modal_frac,
        "modal_val": modal_val,
    }


# --------------------------------------------------------------------------------------
# SORRM reader (small file, ~150MB -- load fully via xarray as specified)
# --------------------------------------------------------------------------------------

def load_sorrm(path: str):
    import xarray as xr

    ds = xr.open_dataset(path)
    varname = list(ds.data_vars)[0]
    da = ds[varname]
    units = da.attrs.get("units")

    data = da.values  # (Time, y, x)
    nt = data.shape[0]
    flat = data.reshape(nt, -1).astype(np.float32)  # (Time, nGridCells)

    print(f"[SORRM] file: {path}")
    print(f"[SORRM] data variable: '{varname}'  units attr: {units!r}")
    print(f"[SORRM] shape: {data.shape} -> flattened to (Time={nt}, nGridCells={flat.shape[1]})")

    ds.close()
    note = ("SORRM (regular polar-stereo grid; unweighted mean approximates "
            "area-weighted mean since grid spacing is uniform)")
    return flat, units, note


# --------------------------------------------------------------------------------------
# generator reader (large file, streamed via netCDF4)
# --------------------------------------------------------------------------------------

def stream_generator_annual(path: str, varname: str, chunk_months: int):
    """
    Stream (Time, nCells) monthly data from `path`, returning an (nYears, nCells) float32
    array of annual means, skipping garbage (all-zero / all-NaN) months. Never loads the
    whole variable at once.
    """
    from netCDF4 import Dataset

    ds = Dataset(path)
    if varname not in ds.variables:
        candidates = [v for v in ds.variables
                      if "Time" in ds.variables[v].dimensions and ds.variables[v].ndim == 2]
        ds.close()
        raise SystemExit(f"'{varname}' not found in {path}. 2D (Time,*) candidates: {candidates}")

    v = ds.variables[varname]
    units = v.getncattr("units") if "units" in v.ncattrs() else None
    nt, ncells = v.shape
    print(f"[generator] file: {path}")
    print(f"[generator] variable: '{varname}'  units attr: {units!r}  shape: {v.shape}")

    annual_rows = []
    n_dropped_months = 0
    n_dropped_years = 0
    leftover = None

    i0 = 0
    while i0 < nt:
        i1 = min(i0 + chunk_months, nt)
        raw = v[i0:i1, :]
        block = np.ma.filled(raw, np.nan) if np.ma.isMaskedArray(raw) else np.asarray(raw)
        block = block.astype(np.float32)

        if leftover is not None:
            block = np.concatenate([leftover, block], axis=0)
            leftover = None

        n_full_years = block.shape[0] // 12
        rem = block.shape[0] % 12

        for y in range(n_full_years):
            seg = block[y * 12:(y + 1) * 12, :]
            garbage = np.all(seg == 0, axis=1) | np.all(np.isnan(seg), axis=1)
            n_valid = int((~garbage).sum())
            if n_valid == 0:
                n_dropped_years += 1
                continue
            if n_valid < 12:
                n_dropped_months += (12 - n_valid)
            annual_rows.append(np.nanmean(seg[~garbage], axis=0))

        if rem:
            leftover = block[n_full_years * 12:, :].copy()

        i0 = i1

    if leftover is not None and leftover.shape[0] > 0:
        n_dropped_months += leftover.shape[0]
        print(f"[generator]   trailing partial year ({leftover.shape[0]} months) dropped, "
              f"incomplete final year")

    ds.close()

    if not annual_rows:
        raise SystemExit(f"[generator] no valid annual data recovered from {path}")

    annual = np.stack(annual_rows, axis=0)
    n_years = annual.shape[0]
    print(f"[generator]   recovered {n_years} annual values "
          f"(dropped {n_dropped_years} whole garbage years, "
          f"{n_dropped_months} individual garbage months)")

    return annual, units, ncells


def load_area(mesh_path: str, area_var: str, ncells: int):
    from netCDF4 import Dataset

    ds = Dataset(mesh_path)
    if area_var not in ds.variables:
        ds.close()
        print(f"[mesh] WARNING: '{area_var}' not in {mesh_path}; falling back to uniform weights")
        return np.ones(ncells, dtype=np.float64)
    area = np.asarray(ds.variables[area_var][:], dtype=np.float64)
    ds.close()
    if area.shape[0] != ncells:
        print(f"[mesh] WARNING: {area_var} length {area.shape[0]} != generator nCells {ncells}; "
              f"falling back to uniform weights")
        return np.ones(ncells, dtype=np.float64)
    return area


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

def print_field_block(label, st, has_mesh):
    print(f"  {label}")
    print(f"     cells: total={st['n_total']}  any-variability={st['n_candidate']}  "
          f"STRICT valid={st['n_valid']}  (valid fraction of total = {st['valid_frac']:.4%})")
    print(f"     strict threshold = valid_floor * p99.9 = {st['threshold']:.6g}  "
          f"(p99.9 = {st['p999']:.6g})")
    wm = f"{st['wmean']:.6g}" if has_mesh else "n/a (no --mesh)"
    print(f"     over STRICT subset: mean={st['mean']:.6g}  area-wtd mean={wm}")
    print(f"                         median={st['median']:.6g}  p75={st['p75']:.6g}  "
          f"p90={st['p90']:.6g}")
    print(f"     modal-value fraction = {st['modal_frac']:.4%} at {st['modal_val']:.6g}  "
          f"(duplicate/fill-value domination diagnostic)")
    if st["modal_frac"] > 0.20:
        print(f"     *** WARNING: {st['modal_frac']:.1%} of cells share one value -- this field is")
        print(f"     *** dominated by regridding duplicates or a fill value. Quantiles over it")
        print(f"     *** partly describe that duplication, not the melt signal.")


def run_comparison(sorrm_path, generators, varname, mesh_path, area_var, band,
                    chunk_months, valid_floor=1e-3):
    print("=" * 88)
    print(f"BAND: {band[0]:g}-{band[1]:g} years  "
          f"(SORRM-constrainable range: need >=2 cycles in the 50yr SORRM record, "
          f"i.e. period <= 25yr)")
    print(f"VALID-CELL FLOOR: keep cells with band variance > {valid_floor:g} * p99.9 of that "
          f"field's own\n  band variance -- a RELATIVE, self-adapting criterion so that grids "
          f"with very different\n  null-cell fractions (MALI mesh vs SORRM 601x601) are "
          f"compared on the same footing.")
    if band[1] > 25:
        print(f"  *** CAVEAT: requested band upper bound {band[1]:g}yr > 25yr. Content at "
              f"periods > 25yr cannot be constrained by the 50yr SORRM record -- any "
              f"comparison involving those periods is NOT decisive. ***")
    print("=" * 88)

    has_mesh = bool(mesh_path)

    sorrm_annual, sorrm_units, sorrm_note = load_sorrm(sorrm_path)
    sorrm_weights = np.ones(sorrm_annual.shape[1], dtype=np.float64)
    sorrm_st = field_stats(sorrm_annual, band, sorrm_weights, valid_floor)
    print(f"[SORRM] {sorrm_note}")

    results = {"SORRM": (sorrm_st, sorrm_units)}

    for label, path in generators:
        annual, gen_units, ncells = stream_generator_annual(path, varname, chunk_months)
        if annual.shape[0] < 2 * band[1]:
            print(f"[{label}] WARNING: only {annual.shape[0]} annual values recovered; "
                  f"< 2x the {band[1]:g}yr upper band period. This generator's own record "
                  f"may not fully resolve the requested band either.")
        weights = (load_area(mesh_path, area_var, ncells) if mesh_path
                   else np.ones(ncells, dtype=np.float64))
        results[label] = (field_stats(annual, band, weights, valid_floor), gen_units)

    # ---- units check --------------------------------------------------------------
    print()
    print("-" * 88)
    print("UNITS CHECK")
    print("-" * 88)
    all_units = {label: u for label, (_, u) in results.items()}
    for label, u in all_units.items():
        print(f"  {label:>10s} units attr: {u!r}")
    unit_values = list(all_units.values())
    if any(u is None for u in unit_values) or len(set(unit_values)) > 1:
        print("  *** LOUD WARNING: units are missing and/or inconsistent across files. ***")
        print("  *** The absolute generator/SORRM ratio below may reflect a UNIT MISMATCH ***")
        print("  *** rather than (or in addition to) a genuine amplitude difference. This  ***")
        print("  *** script applies NO unit conversion. Verify units by hand before trusting ***")
        print("  *** the SORRM-referenced ratios. Generator-vs-generator ratios (below) are  ***")
        print("  *** unit-independent and are NOT affected by this warning.                 ***")

    # ---- per-field detail -----------------------------------------------------------
    print()
    print("-" * 88)
    print(f"PER-CELL BAND VARIANCE ({band[0]:g}-{band[1]:g} yr) -- STRICT valid subset")
    print("-" * 88)
    for label, (st, _) in results.items():
        print_field_block(label, st, has_mesh)
        print()

    # ---- domain comparability -------------------------------------------------------
    sorrm_frac = sorrm_st["valid_frac"]
    domain_warned = set()
    for label, (st, _) in results.items():
        if label == "SORRM":
            continue
        f = st["valid_frac"]
        if sorrm_frac > 0 and f > 0:
            r = max(f / sorrm_frac, sorrm_frac / f)
            if r > 3.0:
                domain_warned.add(label)
                print(f"  *** WARNING: valid-cell FRACTION differs by {r:.1f}x between "
                      f"{label} ({f:.3%}) and SORRM ({sorrm_frac:.3%}). ***")
                print(f"  *** The two spatial domains are not describing comparable cell "
                      f"populations; DISTRUST the absolute {label}/SORRM ratio below. ***")
    if domain_warned:
        print()

    # ---- headline ratio table (MEDIAN-based) ----------------------------------------
    print("-" * 88)
    print("RATIOS vs SORRM -- headline = MEDIAN over strict subset (robust to right-skew);")
    print("                   p90 ratio printed as an independent shape cross-check.")
    print("-" * 88)
    sorrm_med = sorrm_st["median"]
    sorrm_p90 = sorrm_st["p90"]
    ratios = {}
    for label, (st, _) in results.items():
        if label == "SORRM":
            continue
        r_med = st["median"] / sorrm_med if sorrm_med else float("nan")
        r_p90 = st["p90"] / sorrm_p90 if sorrm_p90 else float("nan")
        ratios[label] = {"median": r_med, "p90": r_p90}
        flag = "  [domain mismatch]" if label in domain_warned else ""
        print(f"  {label:>10s}  median ratio = {r_med:10.4g}   p90 ratio = {r_p90:10.4g}{flag}")
        if np.isfinite(r_med) and np.isfinite(r_p90) and r_med > 0 and r_p90 > 0:
            disagree = max(r_med / r_p90, r_p90 / r_med)
            if disagree > 1.5:
                print(f"     *** WARNING: median-ratio and p90-ratio disagree by {disagree:.2f}x. "
                      f"The two distributions")
                print(f"     *** have DIFFERENT SHAPES, not just different scales -- this "
                      f"comparison is shape-sensitive")
                print(f"     *** and no single scalar ratio fully summarizes it.")

    # ---- verdict ----------------------------------------------------------------------
    print()
    print("-" * 88)
    if ratios:
        def closeness(k):
            r = ratios[k]["median"]
            return abs(np.log(r)) if (np.isfinite(r) and r > 0) else np.inf
        best_label = min(ratios, key=closeness)
        print(f"VERDICT: '{best_label}' has the generator/SORRM MEDIAN ratio closest to 1.0 "
              f"(ratio={ratios[best_label]['median']:.4g})")
        print(f"         in the {band[0]:g}-{band[1]:g}yr band.")
        for label, r in ratios.items():
            tag = "  <-- closest to SORRM" if label == best_label else ""
            print(f"    {label}: median ratio={r['median']:.4g}  p90 ratio={r['p90']:.4g}{tag}")
    else:
        print("VERDICT: no --generator files supplied; SORRM reference computed only.")

    print()
    print("  WHICH NUMBER TO TRUST:")
    print("    * GENERATOR-vs-GENERATOR ratio is ROBUST. Both generators live on the SAME MALI")
    print("      mesh with the SAME null/grounded cells and the SAME storage units, so grid")
    print("      bookkeeping and unit questions cancel exactly. If the two generators bracket")
    print("      SORRM ambiguously, the generator/generator number is STILL trustworthy and is")
    print("      the defensible statement about their relative amplitude.")
    print("    * GENERATOR-vs-SORRM ratio is the DOMAIN- and UNIT-SENSITIVE one. It crosses")
    print("      grids (MALI mesh vs SORRM 601x601 polar-stereo) and possibly unit conventions.")
    print("      Treat it as an order-of-magnitude check, not a precision calibration, and")
    print("      heed any domain/unit warnings printed above.")
    print()
    print("  CAVEAT (sampling uncertainty): SORRM's per-cell band variance is estimated from")
    print("  only 50 annual values. Autocorrelation-inflated sampling uncertainty on a")
    print("  variance estimate at this length is roughly +/-30-35%. Ratios within ~1.5x of")
    print("  1.0 are statistically indistinguishable at this sample size; a ~3x discrepancy")
    print("  (the magnitude that motivated this comparison) is decisive, not noise.")
    print()
    print("  CAVEAT (band restriction): periods above 25yr complete fewer than 2 cycles in")
    print("  the 50yr SORRM record and are therefore NOT constrained by SORRM at all, in")
    print("  either generator. Do not extrapolate this verdict to longer-period content.")

    # ---- generator-vs-generator (unit independent, robust regardless of units above) --
    gen_labels = [label for label in results if label != "SORRM"]
    if len(gen_labels) >= 2:
        print()
        print("-" * 88)
        print("GENERATOR vs GENERATOR -- ROBUST (same mesh, same null cells, unit-independent)")
        print("-" * 88)
        for i in range(len(gen_labels)):
            for j in range(i + 1, len(gen_labels)):
                a, b = gen_labels[i], gen_labels[j]
                sa, sb = results[a][0], results[b][0]
                fa, fb = sa["valid_frac"], sb["valid_frac"]
                if fa > 0 and fb > 0 and max(fa / fb, fb / fa) > 3.0:
                    print(f"  *** WARNING: {a} and {b} valid-cell fractions differ by "
                          f"{max(fa / fb, fb / fa):.1f}x -- unexpected on a shared mesh; "
                          f"check the inputs. ***")
                if sb["median"]:
                    print(f"    {a} / {b}:  median ratio = {sa['median'] / sb['median']:.4g}"
                          f"   p90 ratio = "
                          f"{sa['p90'] / sb['p90'] if sb['p90'] else float('nan'):.4g}")
                if sa["median"]:
                    print(f"    {b} / {a}:  median ratio = {sb['median'] / sa['median']:.4g}"
                          f"   p90 ratio = "
                          f"{sb['p90'] / sa['p90'] if sa['p90'] else float('nan'):.4g}")

    return results, ratios


# --------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------

def _synth(n_years, n_cells, amp, seed, period=10.0, trend_slope=0.05):
    rng = np.random.default_rng(seed)
    t = np.arange(n_years, dtype=np.float32)
    phase = rng.uniform(0, 2 * np.pi, size=n_cells).astype(np.float32)
    signal = amp * np.sin(2 * np.pi * t[:, None] / period + phase[None, :])
    # linear trend added to both fields; must be removed by detrend and not affect the ratio
    return (signal + trend_slope * t[:, None]).astype(np.float32)


def self_test():
    band = (2.0, 25.0)
    floor = 1e-3
    n_years, n_cells = 50, 200
    failures = []

    # ---- CASE 1: clean 4x amplitude-squared recovery --------------------------------
    print("Running --self-test")
    print()
    print("CASE 1: synthetic reference vs. a 4x-band-variance 'generator' (clean, no nulls)")
    ref = _synth(n_years, n_cells, 1.0, seed=0)
    gen = _synth(n_years, n_cells, 2.0, seed=0)  # variance ratio = (2/1)^2 = 4.0
    w = np.ones(n_cells, dtype=np.float64)
    ref_st = field_stats(ref, band, w, floor)
    gen_st = field_stats(gen, band, w, floor)
    r_med = gen_st["median"] / ref_st["median"]
    r_p90 = gen_st["p90"] / ref_st["p90"]
    print(f"  reference median band variance: {ref_st['median']:.6f}")
    print(f"  generator median band variance: {gen_st['median']:.6f}")
    print(f"  MEDIAN ratio (expect ~4.0): {r_med:.4f}")
    print(f"  p90    ratio (expect ~4.0): {r_p90:.4f}")
    if abs(r_med - 4.0) >= 0.05:
        failures.append(f"CASE 1 median ratio {r_med:.4f} != ~4.0")
    if abs(r_p90 - 4.0) >= 0.05:
        failures.append(f"CASE 1 p90 ratio {r_p90:.4f} != ~4.0")

    # ---- CASE 2: null-cell dilution immunity (the bug this criterion fixes) ----------
    # Pad the reference with a large mass of near-zero "null" cells (float-noise variance,
    # as produced by regridding duplicates / masked-out cells) but leave the generator
    # un-padded. A MEAN over an absolute-threshold valid set would be dragged down by the
    # padding and inflate the ratio; the relative floor + median must be unaffected.
    print()
    print("CASE 2: same 4x pair, but reference padded with 5x as many near-null cells")
    print("        (mimics MALI-mesh vs SORRM-grid null-fraction mismatch)")
    rng = np.random.default_rng(7)
    n_null = n_cells * 5
    null_block = (rng.standard_normal((n_years, n_null)) * 1e-9).astype(np.float32)
    ref_padded = np.concatenate([ref, null_block], axis=1)
    w_padded = np.ones(ref_padded.shape[1], dtype=np.float64)
    ref_pad_st = field_stats(ref_padded, band, w_padded, floor)
    r_med_pad = gen_st["median"] / ref_pad_st["median"]
    print(f"  padded reference: total cells={ref_pad_st['n_total']}  "
          f"strict valid={ref_pad_st['n_valid']}  (valid frac={ref_pad_st['valid_frac']:.4%})")
    print(f"  padded reference median band variance: {ref_pad_st['median']:.6f}")
    print(f"  MEDIAN ratio with padding (expect still ~4.0): {r_med_pad:.4f}")
    # the diluted statistic, for contrast: mean over the OLD any-variability criterion
    old_valid = ref_pad_st["band_var"] > 0
    old_mean_padded = float(ref_pad_st["band_var"][old_valid].mean())
    old_ratio = gen_st["mean"] / old_mean_padded
    print(f"  (for contrast, the OLD dilution-prone mean-based ratio would be: "
          f"{old_ratio:.4f})")
    if abs(r_med_pad - 4.0) >= 0.05:
        failures.append(f"CASE 2 median ratio {r_med_pad:.4f} != ~4.0 (null padding leaked in)")
    if ref_pad_st["n_valid"] != ref_st["n_valid"]:
        failures.append(f"CASE 2 strict subset changed under padding: "
                        f"{ref_pad_st['n_valid']} vs {ref_st['n_valid']}")

    # ---- CASE 3: modal-fraction diagnostic fires on duplicate domination -------------
    print()
    print("CASE 3: modal-fraction diagnostic on a field with 60% duplicated cells")
    dup = ref.copy()
    n_dup = int(0.6 * n_cells)
    dup[:, :n_dup] = ref[:, [0]]  # force 60% of cells to be exact copies of cell 0
    dup_st = field_stats(dup, band, w, floor)
    print(f"  modal-value fraction detected: {dup_st['modal_frac']:.2%} (expect >=60%)")
    if dup_st["modal_frac"] < 0.55:
        failures.append(f"CASE 3 modal fraction {dup_st['modal_frac']:.2%} failed to detect "
                        f"duplicate domination")

    print()
    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SELF-TEST PASSED (4x recovered on median and p90; immune to null-cell padding;")
    print("                  duplicate-domination diagnostic fires correctly)")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_generator_arg(s: str):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--generator must be LABEL=PATH, got: {s}")
    label, path = s.split("=", 1)
    return label, path


def parse_band(s: str):
    lo, hi = s.split(",")
    return (float(lo), float(hi))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sorrm",
                     default="data/processed/SORRMv21_flux_detrend_uniform_deseasonalize_uniform_dedraft_anm_annual_50.nc",
                     help="SORRM annual anomaly file, dims (Time=50,y=601,x=601)")
    ap.add_argument("--generator", action="append", default=[], type=parse_generator_arg,
                     help="repeatable LABEL=PATH, e.g. --generator '300yr=/path/to/..._Forcing_0.nc'")
    ap.add_argument("--var", default="floatingBasalMassBalAdjustment",
                     help="generator variable name (Time,nCells), monthly")
    ap.add_argument("--mesh", default=None, help="optional MALI mesh file with areaCell")
    ap.add_argument("--area-var", default="areaCell")
    ap.add_argument("--band", default="2,25", type=parse_band,
                     help="period band in years, 'lo,hi'. Default 2,25 = the SORRM-constrainable "
                          "range (>=2 cycles in the 50yr SORRM record).")
    ap.add_argument("--chunk-months", type=int, default=1200,
                     help="streaming chunk size (months) for memory-safe generator reads")
    ap.add_argument("--valid-floor", type=float, default=1e-3,
                     help="strict valid-cell criterion: keep cells whose band variance exceeds "
                          "VALID_FLOOR * p99.9 of that field's own band variance. Relative and "
                          "self-adapting, so grids with very different null-cell fractions are "
                          "compared on the same footing. Default 1e-3.")
    ap.add_argument("--self-test", action="store_true",
                     help="run a synthetic 4x-variance self-test and exit")
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return

    run_comparison(a.sorrm, a.generator, a.var, a.mesh, a.area_var, a.band, a.chunk_months,
                    valid_floor=a.valid_floor)


if __name__ == "__main__":
    main()
