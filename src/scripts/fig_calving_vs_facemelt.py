#!/usr/bin/env python3
"""
fig_calving_vs_facemelt.py — Compare calving flux vs face melt flux per basin.

Layout: 3 panels (SSP585, varScaled10x, SSP126), each showing scatter of
calving flux vs face melt flux per basin at yr100, yr200, yr300.

Shows which basins are calving-dominated vs melt-dominated and how this evolves.
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
    horizons = [100, 200, 300]
    nsc = len(scenarios)

    # Load data
    calving_data = {}
    facemelt_data = {}
    for sc in scenarios:
        yrs_c, arr_c = load_regional_var(args.root, SCENARIO_DIRS[sc], "regionalSumCalvingFlux", SCENARIO_INCLUDE[sc])
        yrs_f, arr_f = load_regional_var(args.root, SCENARIO_DIRS[sc], "regionalSumFaceMeltingFlux", SCENARIO_INCLUDE[sc])
        if arr_c is not None:
            calving_data[sc] = (yrs_c, arr_c)
        if arr_f is not None:
            facemelt_data[sc] = (yrs_f, arr_f)

    fig, axes = plt.subplots(1, nsc, figsize=(5 * nsc, 5))
    if nsc == 1:
        axes = [axes]

    for col, sc in enumerate(scenarios):
        ax = axes[col]
        if sc not in calving_data or sc not in facemelt_data:
            ax.set_visible(False)
            continue
        yrs_c, arr_c = calving_data[sc]
        yrs_f, arr_f = facemelt_data[sc]

        for hr_idx, hr in enumerate(horizons):
            idx_c = np.argmin(np.abs(yrs_c - hr))
            idx_f = np.argmin(np.abs(yrs_f - hr))
            calving_mean = np.nanmean(arr_c[:, idx_c, :], axis=0)
            facemelt_mean = np.nanmean(arr_f[:, idx_f, :], axis=0)
            markers = ["o", "s", "^"]
            for r in range(len(calving_mean)):
                name = BASIN_NAMES[r]
                lbl = SHORT_LABELS.get(name, name)
                ax.scatter(calving_mean[r], facemelt_mean[r],
                           marker=markers[hr_idx], s=60, zorder=3,
                           color=plt.cm.tab20(r / len(calving_mean)),
                           edgecolor="k", lw=0.5,
                           label=lbl if hr_idx == 0 else "")
                if hr == 300:
                    ax.annotate(lbl, (calving_mean[r], facemelt_mean[r]),
                                fontsize=5.5, ha="left", va="bottom",
                                xytext=(3, 3), textcoords="offset points")

        # 1:1 line
        lims = [ax.get_xlim(), ax.get_ylim()]
        mn = min(min(l[0] for l in lims), 0)
        mx = max(max(l[1] for l in lims), 0)
        ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, alpha=0.5, label="1:1")
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_xlabel("Calving flux (mm/yr SLE)")
        ax.set_ylabel("Face melt flux (mm/yr SLE)")
        ax.set_title(sc, fontsize=11, fontweight="bold", color=SCENARIO_COLORS.get(sc, "k"))
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    # Legend for horizons
    from matplotlib.lines import Line2D
    horizon_handles = [Line2D([0], [0], marker=m, color="gray", linestyle="None",
                              markersize=7, label=f"yr{h}")
                       for m, h in zip(["o", "s", "^"], horizons)]
    fig.legend(handles=horizon_handles, loc="upper left", fontsize=8, frameon=True,
               title="Horizon", bbox_to_anchor=(0.01, 0.98))

    fig.suptitle("Calving vs Face Melt — AISLENS Regional",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "calving_vs_facemelt.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out}")


if __name__ == "__main__":
    main()
