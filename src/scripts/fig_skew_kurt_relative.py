#!/usr/bin/env python3
"""
fig_skew_kurt_relative.py — Higher-order statistics of ensemble VAF distribution.

Panel (a): skewness (distribution asymmetry). Panel (b): excess kurtosis (tail weight).
Panel (c): spread/|mean| (relative uncertainty). All four ensembles on the same axes.
Also outputs a relative uncertainty ratio figure (SSP585/CTRL, varScaled10x/CTRL).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio

ROOT = eio.default_ensembles_root()

ENSEMBLES = {
    "CTRL":               {"include": r"^CTRL_\d+$",            "min_years": 50, "color": "black"},
    "SSP585":             {"include": r"^SSP585_\d+$",          "min_years": 50, "color": "#D55E00"},
    "SSP126":             {"include": r"^SSP126_\d+$",          "min_years": 50, "color": "#0072B2"},
    "SSP585_varScaled10x":{"include": r"^SSP585_\d+$",          "min_years": 50, "color": "#CC79A7", "ls": "--"},
}


def load_sle(ensemble_dir, include, min_years):
    ds = eio.load_ensemble_globalstats(
        ensemble_dir,
        variables=["volumeAboveFloatation", "daysSinceStart"],
        include=include, min_years=min_years, align="union")
    sle = eio.vaf_to_sle_mm(ds["volumeAboveFloatation"], reference="first")
    # Year grid may have sub-annual steps; bin to annual for cleaner stats
    yr_raw = np.array(ds.year.values, dtype=float)
    sle_raw = np.asarray(sle)  # (n_member, n_year)
    yr_int = np.round(yr_raw).astype(int)
    # Average within each integer year
    unique_yr = np.unique(yr_int)
    n_mem = sle_raw.shape[0]
    sle_annual = np.full((n_mem, len(unique_yr)), np.nan)
    for i, y in enumerate(unique_yr):
        mask = yr_int == y
        sle_annual[:, i] = np.nanmean(sle_raw[:, mask], axis=1)
    return unique_yr.astype(float), sle_annual


def compute_stats(year, sle, cv_floor_mm=5.0):
    """Compute skewness, kurtosis, spread/|mean| at each timestep.
    year: numpy array of years, sle: DataArray(member, year) or 2D numpy array.
    spread/|mean| is masked where |mean| < cv_floor_mm (mm SLE): below that the ratio is a
    |mean|->0 division artifact (spikes), not a variability peak."""
    if hasattr(sle, 'values'):
        sle_vals = sle.values
    else:
        sle_vals = np.asarray(sle)
    n_year = len(year)
    skewness = np.full(n_year, np.nan)
    kurtosis = np.full(n_year, np.nan)
    rel_unc = np.full(n_year, np.nan)
    spread = np.full(n_year, np.nan)

    for t in range(n_year):
        vals = sle_vals[:, t]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 3:
            continue
        spread[t] = np.std(vals, ddof=1)
        mean_abs = np.abs(np.mean(vals))
        if mean_abs >= cv_floor_mm:            # mask |mean|->0 artifact, not just div-by-zero
            rel_unc[t] = spread[t] / mean_abs
        skewness[t] = sp_stats.skew(vals)
        kurtosis[t] = sp_stats.kurtosis(vals)  # excess kurtosis

    return skewness, kurtosis, rel_unc, spread


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect stats for all ensembles
    results = {}
    for ens_name, cfg in ENSEMBLES.items():
        ens_dir = os.path.join(ROOT, ens_name)
        try:
            year, sle = load_sle(ens_dir, cfg["include"], cfg["min_years"])
            skew, kurt, rel, sprd = compute_stats(year, sle)
            results[ens_name] = {"year": year, "skew": skew, "kurt": kurt,
                                 "rel": rel, "spread": sprd, "color": cfg["color"],
                                 "ls": cfg.get("ls", "-")}
            _f = rel[np.isfinite(rel)]
            print(f"  {ens_name}: {len(year)} years, spread/|mean| @end = {_f[-1]:.3f}"
                  if _f.size else f"  {ens_name}: {len(year)} years, no valid spread/|mean|")
        except Exception as e:
            print(f"  skipping {ens_name}: {e}")

    # --- Figure 1: 3-panel (skewness, kurtosis, relative uncertainty) ---
    fig, (ax_sk, ax_ku, ax_ru) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Higher-Order Ensemble Statistics of VAF → SLE", fontsize=14, y=0.98)

    for ens_name, r in results.items():
        ls = r.get("ls", "-")
        ax_sk.plot(r["year"], r["skew"], color=r["color"], lw=2, ls=ls, label=ens_name)
        ax_ku.plot(r["year"], r["kurt"], color=r["color"], lw=2, ls=ls, label=ens_name)
        ax_ru.plot(r["year"], r["rel"], color=r["color"], lw=2, ls=ls, label=ens_name)

    ax_sk.axhline(0, color="0.5", ls="--", lw=0.8)
    ax_sk.set_ylabel("Skewness")
    ax_sk.set_title("(a) Distribution asymmetry")
    ax_sk.legend(fontsize=9, loc="upper left")
    ax_sk.grid(True, alpha=0.3)

    ax_ku.axhline(0, color="0.5", ls="--", lw=0.8)
    ax_ku.set_ylabel("Excess Kurtosis")
    ax_ku.set_title("(b) Tail heaviness")
    ax_ku.grid(True, alpha=0.3)

    ax_ru.set_yscale("log")
    ax_ru.set_ylabel("Spread / |Mean|")
    ax_ru.set_title("(c) Relative uncertainty")
    ax_ru.set_xlabel("Years since simulation start")
    ax_ru.grid(True, alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    f1 = os.path.join(args.out_dir, "skew_kurt_relative.png")
    fig.savefig(f1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {f1}")

    # --- Figure 2: Relative uncertainty ratio (SSP585/CTRL and varScaled10x/SSP585) ---
    if "CTRL" in results and "SSP585" in results:
        fig2, ax = plt.subplots(figsize=(9, 5))
        # Interpolate SSP585 onto CTRL's year grid
        r_ctrl = results["CTRL"]
        r_ssp = results["SSP585"]
        common_yr = r_ctrl["year"]
        rel_ssp_interp = np.interp(common_yr, r_ssp["year"], r_ssp["rel"])
        ratio = rel_ssp_interp / r_ctrl["rel"]
        ratio[~np.isfinite(ratio)] = np.nan
        ax.plot(common_yr, ratio, color="#D55E00", lw=2, label="SSP585 / CTRL")

        if "SSP585_varScaled10x" in results:
            r_10x = results["SSP585_varScaled10x"]
            rel_10x_interp = np.interp(common_yr, r_10x["year"], r_10x["rel"])
            ratio_10x = rel_10x_interp / r_ctrl["rel"]
            ratio_10x[~np.isfinite(ratio_10x)] = np.nan
            ax.plot(common_yr, ratio_10x, color="#CC79A7", lw=2,
                    label="SSP585_varScaled10x / CTRL")

        ax.set_yscale("log")
        ax.axhline(1, color="0.5", ls="--", lw=0.8, label="No change")
        ax.set_xlabel("Years since simulation start")
        ax.set_ylabel("Relative Uncertainty Ratio")
        ax.set_title("Does forcing amplify relative uncertainty?")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        fig2.tight_layout()
        f2 = os.path.join(args.out_dir, "relative_uncertainty_ratio.png")
        fig2.savefig(f2, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {f2}")


if __name__ == "__main__":
    main()
