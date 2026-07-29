#!/usr/bin/env python3
"""
fig_cumulative_calving.py — Cumulative calving per basin over time.

Integrates regionalSumCalvingFlux to show total iceberg production per basin.
Layout: 3 panels (SSP585, varScaled10x, SSP126), each showing 16 basins.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from ismip6_regions import BASIN_NAMES, SHORT_LABELS

RHO_ICE = eio.RHO_ICE
OCEAN_AREA = eio.OCEAN_AREA

SCENARIO_DIRS = {"SSP585": "SSP585", "varScaled10x": "SSP585_varScaled10x", "SSP126": "SSP126"}
SCENARIO_INCLUDE = {"SSP585": r"^SSP585_\d+$", "varScaled10x": r"^SSP585_\d+$", "SSP126": r"^SSP126_\d+$"}
SCENARIO_COLORS = {"SSP585": "#C62828", "varScaled10x": "#E65100", "SSP126": "#1565C0"}


def kg_yr_to_mm_sle(kg_yr):
    return kg_yr * (1.0 / (RHO_ICE * OCEAN_AREA)) * 1000.0


def load_regional_var(root, ensemble, variable, include):
    members = eio.discover_members(
        os.path.join(root, ensemble), stats_filename="regionalStats.nc", include=include
    )
    stacks, nmin = [], None
    for name, path in members:
        try:
            ds = eio.to_year_dim(eio.load_member_regionalstats(path))
        except Exception:
            continue
        if variable not in ds:
            continue
        yr = ds["year"].values
        if yr[0] > 5.0 or len(yr) < 10:
            continue
        nreg = ds.dims["nRegions"]
        vals = np.column_stack([kg_yr_to_mm_sle(ds[variable].isel(nRegions=r).values) for r in range(nreg)])
        stacks.append((yr, vals))
        nmin = len(yr) if nmin is None else min(nmin, len(yr))
    if len(stacks) < 3:
        return None, None
    years = stacks[0][0][:nmin]
    arr = np.stack([s[:nmin] for _, s in stacks], axis=0)
    return years, arr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--out-dir", default="/Users/smurugan9/research/aislens/AISLENS/reports/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scenarios = list(SCENARIO_DIRS.keys())
    nsc = len(scenarios)

    fig, axes = plt.subplots(1, nsc, figsize=(5.5 * nsc, 5))
    if nsc == 1:
        axes = [axes]

    for col, sc in enumerate(scenarios):
        ax = axes[col]
        yrs, arr = load_regional_var(args.root, SCENARIO_DIRS[sc], "regionalSumCalvingFlux",
                                     SCENARIO_INCLUDE[sc])
        if arr is None:
            ax.set_visible(False)
            continue

        nreg = arr.shape[2]
        # Cumulative integral (trapezoid)
        cumul = np.zeros_like(arr)
        for t in range(1, len(yrs)):
            dt = yrs[t] - yrs[t - 1]
            cumul[:, t, :] = cumul[:, t - 1, :] + 0.5 * (arr[:, t, :] + arr[:, t - 1, :]) * dt

        ens_mean = np.nanmean(cumul, axis=0)  # (year, nRegions)
        ens_std = np.nanstd(cumul, axis=0)

        for r in range(nreg):
            name = BASIN_NAMES[r]
            lbl = SHORT_LABELS.get(name, name)
            c = plt.cm.tab20(r / nreg)
            ax.plot(yrs, ens_mean[:, r], color=c, lw=1.2, label=lbl)
            ax.fill_between(yrs, ens_mean[:, r] - ens_std[:, r],
                            ens_mean[:, r] + ens_std[:, r], color=c, alpha=0.12)

        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative calving (mm SLE)" if col == 0 else "")
        ax.set_title(sc, fontsize=11, fontweight="bold", color=SCENARIO_COLORS.get(sc, "k"))
        ax.set_xlim(yrs[0], yrs[-1])
        ax.tick_params(labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=7,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cumulative Calving — AISLENS Regional",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = os.path.join(args.out_dir, "cumulative_calving_regional.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out}")

    # Print final cumulative totals
    for sc in scenarios:
        yrs, arr = load_regional_var(args.root, SCENARIO_DIRS[sc], "regionalSumCalvingFlux",
                                     SCENARIO_INCLUDE[sc])
        if arr is None:
            continue
        cumul = np.zeros_like(arr)
        for t in range(1, len(yrs)):
            dt = yrs[t] - yrs[t - 1]
            cumul[:, t, :] = cumul[:, t - 1, :] + 0.5 * (arr[:, t, :] + arr[:, t - 1, :]) * dt
        final = np.nanmean(cumul[:, -1, :], axis=0)
        top = np.argsort(final)[::-1][:5]
        print(f"\n  {sc} final cumulative calving (top 5 basins):")
        for i in top:
            name = BASIN_NAMES[i]
            print(f"    {SHORT_LABELS.get(name, name):14s}: {final[i]:8.1f} mm SLE")


if __name__ == "__main__":
    main()
