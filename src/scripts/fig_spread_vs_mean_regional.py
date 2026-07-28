#!/usr/bin/env python3
"""
fig_spread_vs_mean_regional.py — Per-basin σ vs |μ| scatter at multiple horizons.

Tests correlation between mean loss and spread (MISI evidence). Colored by basin, markers by horizon.
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


def load_regional_sle(root, ensemble, include):
    members = eio.discover_members(
        os.path.join(root, ensemble), stats_filename="regionalStats.nc", include=include
    )
    stacks, nmin = [], None
    for name, path in members:
        try:
            ds = eio.to_year_dim(eio.load_member_regionalstats(path))
        except Exception:
            continue
        vaf = ds.get("regionalVolumeAboveFloatation")
        if vaf is None:
            continue
        yr = ds["year"].values
        if yr[0] > 5.0 or len(yr) < 10:
            continue
        nreg = vaf.sizes["nRegions"]
        sle = np.column_stack([eio.vaf_to_sle_mm(vaf.isel(nRegions=r).values, reference="first")
                               for r in range(nreg)])
        stacks.append((yr, sle))
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

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.5 * len(scenarios), 5))
    if len(scenarios) == 1:
        axes = [axes]

    for col, sc in enumerate(scenarios):
        ax = axes[col]
        yrs, arr = load_regional_sle(args.root, SCENARIO_DIRS[sc], SCENARIO_INCLUDE[sc])
        if arr is None:
            ax.set_visible(False)
            continue
        nreg = arr.shape[2]
        for hr in horizons:
            idx = np.argmin(np.abs(yrs - hr))
            mean_vals = np.abs(np.nanmean(arr[:, idx, :], axis=0))
            std_vals = np.nanstd(arr[:, idx, :], axis=0)
            markers = {"SSP585": "o", "varScaled10x": "s", "SSP126": "^"}
            for r in range(nreg):
                name = BASIN_NAMES[r]
                lbl = SHORT_LABELS.get(name, name)
                ax.scatter(mean_vals[r], std_vals[r], marker=markers.get(sc, "o"),
                           s=80, zorder=3, color=plt.cm.tab20(r / nreg),
                           edgecolor="k", lw=0.5)
                if hr == horizons[-1]:
                    ax.annotate(lbl, (mean_vals[r], std_vals[r]),
                                fontsize=6, ha="left", va="bottom",
                                xytext=(3, 3), textcoords="offset points")

        # Correlation at last horizon
        idx = np.argmin(np.abs(yrs - horizons[-1]))
        mean_all = np.abs(np.nanmean(arr[:, idx, :], axis=0))
        std_all = np.nanstd(arr[:, idx, :], axis=0)
        ok = np.isfinite(mean_all) & np.isfinite(std_all) & (mean_all > 0.01)
        if ok.sum() > 3:
            r_corr = np.corrcoef(mean_all[ok], std_all[ok])[0, 1]
            ax.text(0.05, 0.95, f"r = {r_corr:.2f} (yr{horizons[-1]})",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel(f"|Mean dVAF| (mm SLE) — {sc}", fontsize=9)
        ax.set_ylabel("σ (mm SLE)" if col == 0 else "")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # Horizon legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=m, color="gray", linestyle="None", markersize=7,
                       label=f"yr{h}")
               for m, h in zip(["o", "s", "^"], horizons)]
    fig.legend(handles=handles, loc="upper left", fontsize=8, frameon=True,
               title="Horizon", bbox_to_anchor=(0.01, 0.98))

    fig.suptitle("Spread vs Mean — Per Basin σ vs |μ|",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "spread_vs_mean_regional.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out}")


if __name__ == "__main__":
    main()
