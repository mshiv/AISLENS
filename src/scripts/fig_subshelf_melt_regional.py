#!/usr/bin/env python3
"""
fig_subshelf_melt_regional.py — Subshelf melt rate and floating BMB per basin.

2×3 grid: row 1 = mean subshelf melt rate (m/yr), row 2 = floating BMB (mm/yr SLE).
Columns: SSP585, varScaled10x, SSP126.
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
        if variable in ("regionalAvgSubshelfMelt",):
            vals = np.column_stack([ds[variable].isel(nRegions=r).values for r in range(nreg)])
        else:
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
    variables = [
        ("regionalAvgSubshelfMelt", "Subshelf melt rate (m/yr)", True),
        ("regionalSumFloatingBasalMassBal", "Floating BMB (mm/yr SLE)", False),
    ]

    nsc = len(scenarios)
    fig, axes = plt.subplots(2, nsc, figsize=(4.5 * nsc, 7), sharex=True)

    for row, (varname, var_label, is_rate) in enumerate(variables):
        for col, sc in enumerate(scenarios):
            ax = axes[row, col]
            yrs, arr = load_regional_var(args.root, SCENARIO_DIRS[sc], varname, SCENARIO_INCLUDE[sc])
            if arr is None:
                ax.set_visible(False)
                continue
            nreg = arr.shape[2]
            ens_mean = np.nanmean(arr, axis=0)
            ens_std = np.nanstd(arr, axis=0)
            for r in range(nreg):
                name = BASIN_NAMES[r]
                lbl = SHORT_LABELS.get(name, name)
                c = plt.cm.tab20(r / nreg)
                ax.plot(yrs, ens_mean[:, r], color=c, lw=1.0, label=lbl)
                ax.fill_between(yrs, ens_mean[:, r] - ens_std[:, r],
                                ens_mean[:, r] + ens_std[:, r], color=c, alpha=0.12)
            ax.axhline(0, color="k", lw=0.5, ls="--")
            if row == 0:
                ax.set_title(sc, fontsize=11, fontweight="bold", color=SCENARIO_COLORS.get(sc, "k"))
            if col == 0:
                ax.set_ylabel(var_label, fontsize=9)
            if row == 1:
                ax.set_xlabel("Year")
            ax.set_xlim(yrs[0], yrs[-1])
            ax.tick_params(labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=7,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Subshelf Melt Rate & Floating BMB — AISLENS Regional",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = os.path.join(args.out_dir, "subshelf_melt_regional.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out}")


if __name__ == "__main__":
    main()
