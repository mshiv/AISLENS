#!/usr/bin/env python3
"""
fig_gl_migration.py — grounding line migration flux and discharge flux per basin.

Layout: 2 rows (migration flux, discharge flux) × 3 columns (SSP585, varScaled10x, SSP126).
Uses regionalStats variables regionalSumGroundingLineMigrationFlux and regionalSumGroundingLineFlux.
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

SCENARIOS = [
    ("SSP585",          r"^SSP585_\d+$",       "#C62828"),
    ("varScaled10x",    r"^SSP585_\d+$",       "#E65100"),  # same member pattern but different dir
    ("SSP126",          r"^SSP126_\d+$",       "#1565C0"),
    ("CTRL",            r"^CTRL_\d+$",          "#616161"),
]
# varScaled10x lives in a separate directory
SCENARIO_DIRS = {
    "SSP585":       "SSP585",
    "varScaled10x": "SSP585_varScaled10x",
    "SSP126":       "SSP126",
    "CTRL":         "CTRL",
}

PLOT_SCENARIOS = ["CTRL", "SSP585", "varScaled10x", "SSP126"]


def kg_yr_to_mm_sle(kg_yr):
    """Convert area-integrated mass flux (kg/yr) to SLE rate (mm/yr).
    Positive = ice loss → sea-level rise."""
    return kg_yr * (1.0 / (RHO_ICE * OCEAN_AREA)) * 1000.0


def load_regional_var(root, ensemble, variable, include):
    """Load a single regionalStats variable for all members.
    Returns (years, (member, year, nRegions)) in mm/yr SLE equivalent."""
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
        vals = np.column_stack([
            kg_yr_to_mm_sle(ds[variable].isel(nRegions=r).values)
            for r in range(nreg)
        ])
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

    # Load data for each scenario
    data = {}
    for sc in PLOT_SCENARIOS:
        d = SCENARIO_DIRS[sc]
        for varname in ["regionalSumGroundingLineMigrationFlux", "regionalSumGroundingLineFlux"]:
            key = (sc, varname)
            yrs, arr = load_regional_var(args.root, d, varname,
                                         include=SCENARIOS[[s[0] for s in SCENARIOS].index(sc)][1])
            if arr is not None:
                data[key] = (yrs, arr)
                print(f"  loaded {sc} / {varname}: {arr.shape[0]} members, {arr.shape[1]} years, "
                      f"{arr.shape[2]} basins")

    # Figure
    nsc = len(PLOT_SCENARIOS)
    fig, axes = plt.subplots(2, nsc, figsize=(4.5 * nsc, 7), sharex=True)
    var_labels = [
        "regionalSumGroundingLineMigrationFlux",
        "regionalSumGroundingLineFlux",
    ]
    row_titles = [
        "GL migration flux (mm/yr SLE)\n(+ = grounded → floating)",
        "GL discharge flux (mm/yr SLE)\n(+ = grounded → floating)",
    ]

    colors = {
        "SSP585":       "#C62828",
        "varScaled10x": "#E65100",
        "SSP126":       "#1565C0",
    }

    for row, varname in enumerate(var_labels):
        for col, sc in enumerate(PLOT_SCENARIOS):
            ax = axes[row, col]
            key = (sc, varname)
            if key not in data:
                ax.set_visible(False)
                continue
            yrs, arr = data[key]  # (member, year, nRegions)
            nreg = arr.shape[2]
            ens_mean = np.nanmean(arr, axis=0)  # (year, nRegions)
            ens_std = np.nanstd(arr, axis=0)

            for r in range(nreg):
                name = BASIN_NAMES[r]
                lbl = SHORT_LABELS.get(name, name)
                c = plt.cm.tab20(r / nreg)
                ax.plot(yrs, ens_mean[:, r], color=c, lw=1.0, label=lbl)
                ax.fill_between(yrs, ens_mean[:, r] - ens_std[:, r],
                                ens_mean[:, r] + ens_std[:, r], color=c, alpha=0.12)

            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_yscale("symlog", linthresh=0.01)
            if row == 0:
                ax.set_title(sc, fontsize=11, fontweight="bold", color=colors.get(sc, "k"))
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=9)
            if row == 1:
                ax.set_xlabel("Year")
            ax.set_xlim(yrs[0], yrs[-1])
            ax.tick_params(labelsize=8)

    # Shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=7,
               frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Grounding Line Migration & Discharge Flux — AISLENS Regional",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    out = os.path.join(args.out_dir, "gl_migration_flux.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved -> {out}")

    # Print key GL migration stats
    for sc in PLOT_SCENARIOS:
        key = (sc, "regionalSumGroundingLineMigrationFlux")
        if key not in data:
            continue
        yrs, arr = data[key]
        # Find year-100 and year-200 indices
        for yr_target in [100, 200]:
            idx = np.argmin(np.abs(yrs - yr_target))
            mean_vals = np.nanmean(arr[:, idx, :], axis=0)  # (nRegions,)
            top = np.argsort(np.abs(mean_vals))[::-1][:3]
            print(f"\n  {sc} yr{yr_target} — top 3 GL migration flux basins:")
            for i in top:
                name = BASIN_NAMES[i]
                print(f"    {SHORT_LABELS.get(name, name):14s}: {mean_vals[i]:+.4f} mm/yr SLE")


if __name__ == "__main__":
    main()
