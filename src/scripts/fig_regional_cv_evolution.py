#!/usr/bin/env python3
"""
fig_regional_cv_evolution.py — Per-basin coefficient of variation (σ/|mean|) over time.

4×4 grid of panels, one per ISMIP6 basin, showing CV for SSP585, SSP126, and varScaled10x.
High CV basins = where internal variability matters most relative to the forced signal.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from fig_regional_emergence import load_regional_sle
from ismip6_regions import BASIN_NAMES, SHORT_LABELS

# Grid positions for 4x4 layout (row, col)
BASIN_POSITIONS = {name: (r // 4, r % 4) for r, name in enumerate(BASIN_NAMES)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--start-year", type=float, default=2000.0)
    ap.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    nreg = len(BASIN_NAMES)

    # Get reference year grid from SSP585 (most complete)
    years_ref, arr_ref = load_regional_sle(a.root, "SSP585", r"^SSP585_\d+$")
    if arr_ref is None:
        sys.exit("SSP585: no usable members")

    ensembles = {
        "SSP585":                (r"^SSP585_\d+$", "C3"),
        "SSP126":                (r"^SSP126_\d+$", "C0"),
        "SSP585_varScaled10x":  (r"^SSP585_\d+$", "C4"),
    }
    data = {}
    for ens, (inc, col) in ensembles.items():
        years, arr = load_regional_sle(a.root, ens, inc)
        if arr is None:
            print(f"{ens}: skip"); continue
        n_mem, n_yr, nr = arr.shape
        sig = np.nanstd(arr, axis=0)  # (year, region)
        mean_arr = np.nanmean(arr, axis=0)
        cv = np.where(np.abs(mean_arr) >= 5.0, sig / np.abs(mean_arr), np.nan)  # 5 mm floor: mask |mean|->0 artifact
        data[ens] = {"sig": sig, "cv": cv, "n": n_mem, "cal": a.start_year + years}
        print(f"{ens}: {n_mem} members, {n_yr} years, {nr} basins")

    # ---- Figure: 4x4 grid, one per basin ----
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=False)
    fig.suptitle("Per-basin coefficient of variation (CV = σ / |mean|) over time\n"
                 "High CV = internal variability matters most relative to forced signal",
                 fontsize=11, y=1.01)

    for r, nm in enumerate(BASIN_NAMES):
        row, col_idx = BASIN_POSITIONS[nm]
        ax = axes[row][col_idx]
        for ens, style in [("SSP585", "-"), ("SSP126", "--"), ("SSP585_varScaled10x", ":")]:
            if ens not in data:
                continue
            cal_e = data[ens]["cal"]
            cv = data[ens]["cv"][:, r]
            valid = np.isfinite(cv) & (cal_e > a.start_year + 20)  # skip spin-up
            if valid.any():
                ax.plot(cal_e[valid], 100*cv[valid], style,
                        color={"SSP585": "C3", "SSP126": "C0", "SSP585_varScaled10x": "C4"}[ens],
                        lw=1.5, label=ens)
        ax.set_title(nm, fontsize=9, fontweight="bold")
        ax.set_ylim(0, min(500, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 500))
        ax.grid(alpha=0.2)
        if r == 0 and col_idx == 0:
            ax.legend(fontsize=7, loc="upper left")
        if row == 3:
            ax.set_xlabel("year", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel("CV (%)", fontsize=8)

    fig.tight_layout()
    out = os.path.join(a.out_dir, "regional_cv_evolution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure -> {out}")

    target_model_yr = 200
    print(f"\nPer-basin CV at model year {target_model_yr} (calendar {a.start_year + target_model_yr:.0f}):")
    for r, nm in enumerate(BASIN_NAMES):
        vals = []
        for ens in ["SSP585", "SSP126", "SSP585_varScaled10x"]:
            if ens in data:
                cal_e = data[ens]["cal"]
                model_yrs = cal_e - a.start_year
                i200 = int(np.argmin(np.abs(model_yrs - target_model_yr)))
                cv = data[ens]["cv"][i200, r]
                vals.append(f"  {ens}={100*cv:.1f}%" if np.isfinite(cv) else f"  {ens}=n/a")
        print(f"  {nm:20s}" + "".join(vals))


if __name__ == "__main__":
    main()
