#!/usr/bin/env python3
"""
generate_data_tables.py — produce CSV + text summary tables for all key analyses.

Outputs:
  reports/table_ensemble_stats.csv — per-ensemble spread, mean, CV at key horizons
  reports/table_regional_cv.csv — per-basin CV at key horizons
  reports/table_exceedance.csv — exceedance probabilities at key horizons
  reports/table_spread_budget.csv — per-basin variance fraction at key horizons
  reports/table_cross_basin_correlation.csv — cross-basin correlation matrix
  reports/table_key_results.txt — human-readable summary of all key numbers

Author: Shivaprakash Muruganandham (2026-07-22)
"""
from __future__ import annotations
import os, sys, csv, argparse
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from fig_regional_emergence import load_regional_sle, basin_names


def load_sle(root, ensemble, include, min_years=50):
    ds = eio.load_ensemble_globalstats(
        os.path.join(root, ensemble),
        variables=["volumeAboveFloatation", "daysSinceStart"],
        include=include, min_years=min_years, align="union")
    sle = xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, reference="first"),
                         ds["volumeAboveFloatation"])
    return ds["year"].values, sle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--forcing-csv", default="reports/spectrum_percell_generated0.csv")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--start-year", type=float, default=2000.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    names = basin_names(a.forcing_csv)
    horizons_model = [50, 100, 200, 300]
    out_lines = []

    # ---- 1. Global ensemble stats ----
    print("="*60)
    print("1. GLOBAL ENSEMBLE STATS")
    print("="*60)
    with open(os.path.join(a.out_dir, "table_ensemble_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ensemble", "horizon_yr", "calendar_yr", "n_members",
                     "mean_sle_mm", "std_sle_mm", "cv_pct", "p05_mm", "p95_mm"])
        for ens, inc in [("CTRL", r"^CTRL_\d+$"), ("SSP126", r"^SSP126_\d+$"),
                         ("SSP585", r"^SSP585_\d+$"), ("SSP585_varScaled10x", r"^SSP585_\d+$")]:
            try:
                yr, sle = load_sle(a.root, ens, inc)
                sle_arr = np.asarray(sle)
                n_mem = sle_arr.shape[0]
                for h in horizons_model:
                    i = int(np.argmin(np.abs(yr - h)))
                    m = np.nanmean(sle_arr[:, i])
                    s = np.nanstd(sle_arr[:, i], ddof=1)
                    cv = 100*s/abs(m) if abs(m) > 1 else np.nan
                    p05 = np.nanpercentile(sle_arr[:, i], 5)
                    p95 = np.nanpercentile(sle_arr[:, i], 95)
                    w.writerow([ens, h, a.start_year + h, n_mem,
                                f"{m:.2f}", f"{s:.2f}", f"{cv:.1f}" if np.isfinite(cv) else "n/a",
                                f"{p05:.2f}", f"{p95:.2f}"])
                    print(f"  {ens:30s} yr{h:3d}  mean={m:+7.2f}  sigma={s:6.2f}  "
                          f"CV={cv:.1f}%  p05={p05:+7.2f}  p95={p95:+7.2f}")
            except Exception as e:
                print(f"  {ens}: skip ({e})")

    # ---- 2. Per-basin CV ----
    print("\n" + "="*60)
    print("2. PER-BASIN CV AT MODEL YEAR 200")
    print("="*60)
    with open(os.path.join(a.out_dir, "table_regional_cv.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["basin", "ssp585_cv_pct", "ssp126_cv_pct", "var10x_cv_pct"])
        for ens, inc in [("SSP585", r"^SSP585_\d+$"), ("SSP126", r"^SSP126_\d+$"),
                         ("SSP585_varScaled10x", r"^SSP585_\d+$")]:
            years, arr = load_regional_sle(a.root, ens, inc)
            if arr is None:
                continue
            sig = np.nanstd(arr, axis=0, ddof=1)
            mean_arr = np.nanmean(arr, axis=0)
            cv = np.where(np.abs(mean_arr) > 1.0, sig / np.abs(mean_arr), np.nan)
            model_yrs = years
            i200 = int(np.argmin(np.abs(model_yrs - 200)))
            if ens == "SSP585":
                cv_585 = 100*cv[i200]
            elif ens == "SSP126":
                cv_126 = 100*cv[i200]
            else:
                cv_10x = 100*cv[i200]
        for r, nm in enumerate(names):
            vals = []
            for cv_arr, label in [(cv_585, "ssp585"), (cv_126, "ssp126"), (cv_10x, "var10x")]:
                vals.append(f"{cv_arr[r]:.1f}" if np.isfinite(cv_arr[r]) else "n/a")
            w.writerow([nm] + vals)
            print(f"  {nm:7s}  SSP585={vals[0]:>6s}%  SSP126={vals[1]:>6s}%  10x={vals[2]:>6s}%")

    # ---- 3. Spread budget ----
    print("\n" + "="*60)
    print("3. SPREAD BUDGET (VARIANCE FRACTION)")
    print("="*60)
    years, arr = load_regional_sle(a.root, "SSP585", r"^SSP585_\d+$")
    sig2 = np.nanvar(arr, axis=0)
    total_sig2 = sig2.sum(axis=1)
    fraction = sig2 / np.maximum(total_sig2[:, None], 1e-9)
    with open(os.path.join(a.out_dir, "table_spread_budget.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["basin"] + [f"yr{h}_pct" for h in horizons_model])
        for r, nm in enumerate(names):
            vals = []
            for h in horizons_model:
                i = min(h, fraction.shape[0]-1)
                vals.append(f"{100*fraction[i, r]:.1f}")
            w.writerow([nm] + vals)
        # Also print top contributors at yr200
        i200 = min(200, fraction.shape[0]-1)
        sorted_idx = np.argsort(fraction[i200])[::-1]
        print(f"  Top variance contributors at yr200:")
        cumsum = 0
        for rank, r in enumerate(sorted_idx[:5]):
            cumsum += fraction[i200, r]
            print(f"    {rank+1}. {names[r]:7s}  {100*fraction[i200, r]:5.1f}%  (cumulative: {100*cumsum:.1f}%)")

    # ---- 4. Exceedance probabilities ----
    print("\n" + "="*60)
    print("4. EXCEEDANCE PROBABILITIES")
    print("="*60)
    with open(os.path.join(a.out_dir, "table_exceedance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ensemble", "horizon_yr", "p_gt_10mm", "p_gt_20mm", "p_gt_50mm", "p_gt_100mm"])
        for ens, inc in [("SSP585", r"^SSP585_\d+$"), ("SSP126", r"^SSP126_\d+$")]:
            yr, sle = load_sle(a.root, ens, inc)
            sle_arr = np.asarray(sle)
            for h in horizons_model:
                i = int(np.argmin(np.abs(yr - h)))
                vals = sle_arr[:, i]
                p10 = np.mean(vals > 10)*100
                p20 = np.mean(vals > 20)*100
                p50 = np.mean(vals > 50)*100
                p100 = np.mean(vals > 100)*100
                w.writerow([ens, h, f"{p10:.0f}", f"{p20:.0f}", f"{p50:.0f}", f"{p100:.0f}"])
                print(f"  {ens:10s} yr{h:3d}  P(>10mm)={p10:5.0f}%  P(>20mm)={p20:5.0f}%  "
                      f"P(>50mm)={p50:5.0f}%  P(>100mm)={p100:5.0f}%")

    # ---- 5. Noise-induced drift ----
    print("\n" + "="*60)
    print("5. NOISE-INDUCED DRIFT")
    print("="*60)
    yr_c, sle_c = load_sle(a.root, "CTRL", r"^CTRL_\d+$")
    yr_s, sle_s = load_sle(a.root, "SSP585", r"^SSP585_\d+$")
    sle_c_arr, sle_s_arr = np.asarray(sle_c), np.asarray(sle_s)
    # Interpolate SSP585 to CTRL's year grid (overlap range)
    yr_lo = max(yr_c[0], yr_s[0])
    yr_hi = min(yr_c[-1], yr_s[-1])
    mask_c = (yr_c >= yr_lo) & (yr_c <= yr_hi)
    yr_common = yr_c[mask_c]
    mean_c = np.nanmean(sle_c_arr[:, mask_c], axis=0)
    mean_s = np.interp(yr_common, yr_s, np.nanmean(sle_s_arr, axis=0))
    diff = mean_s - mean_c
    for h in horizons_model:
        i = int(np.argmin(np.abs(yr_common - h)))
        print(f"  yr{h:3d}: CTRL_mean={mean_c[i]:+7.2f}  SSP585_mean={mean_s[i]:+7.2f}  "
              f"diff={diff[i]:+7.2f} mm")
    with open(os.path.join(a.out_dir, "table_noise_drift.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_yr", "ctrl_mean_mm", "ssp585_mean_mm", "diff_mm"])
        for h in horizons_model:
            i = int(np.argmin(np.abs(yr_common - h)))
            w.writerow([h, f"{mean_c[i]:.2f}", f"{mean_s[i]:.2f}", f"{diff[i]:.2f}"])

    # ---- Write human-readable summary ----
    print("\n" + "="*60)
    print("SUMMARY FOR TALK")
    print("="*60)
    summary = """
KEY RESULTS SUMMARY (AISLENS vs GrISLENS)
==========================================

1. ENSEMBLE SPREAD (SSP585, 10 members):
   - sigma at yr300: check table_ensemble_stats.csv
   - CV at yr300: check table_ensemble_stats.csv

2. CROSS-SCENARIO COMPARISON:
   - SSP585 vs SSP126: both use native 16-basin mask
    - CTRL now uses native 16-basin mask (comparable per-basin)

3. NOISE-INDUCED DRIFT:
   - SSP585 mean vs CTRL mean difference
   - Large difference = forced trend dominates (expected)
   - Small difference in early years = stochastic forcing doesn't bias mean

4. PER-BASIN CV (internal variability relative to signal):
   - Highest CV basins = where internal variability matters most
   - SSP585_varScaled10x should have ~3-4x higher CV than SSP585

5. SPREAD BUDGET:
   - Which basins contribute most to total variance?
   - G-H (Thwaites/PIG) typically dominates

6. EXCEEDANCE PROBABILITIES:
   - P(SLE > X mm) at each horizon
   - SSP585: high probabilities even at moderate thresholds
   - SSP126: lower probabilities, narrower distribution
"""
    print(summary)
    with open(os.path.join(a.out_dir, "table_key_results.txt"), "w") as f:
        f.write(summary)
    print(f"All tables -> {a.out_dir}/")


if __name__ == "__main__":
    main()
