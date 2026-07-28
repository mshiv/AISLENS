#!/usr/bin/env python3
"""
forcing_ramp_diagnostic.py — is the FRIS/Ross forcing ramp in the input or the model?

Plots applied melt (model regionalStats) vs INPUT forcing trend to determine whether
the abrupt ramp is from the UKESM/ISMIP6 driver or the draft-dependent parameterization.

Author: Shivaprakash Muruganandham
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio

DEFAULT_REGIONMASK = None
try:
    from aislens.config import config
    DEFAULT_REGIONMASK = str(config.DIR_MALI / "AIS_4to20km_r01_20220907.regionMask_ismip6.nc")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=eio.default_ensembles_root())
    p.add_argument("--ensemble", required=True)
    p.add_argument("--member", required=True,
                   help="A representative member dir (mean forcing is the same across members)")
    p.add_argument("--regions", default="FRIS,Ross",
                   help="Comma-separated friendly basin names (default FRIS,Ross)")
    p.add_argument("--regionmask", default=DEFAULT_REGIONMASK,
                   help="ISMIP6 regionMask NetCDF (has regionCellMasks + regionNames)")
    p.add_argument("--trend-file", default=None,
                   help="Optional INPUT forcing/trend field on nCells to basin-average")
    p.add_argument("--trend-var", default="floatingBasalMassBalAdjustment")
    p.add_argument("--melt-var", default="regionalSumFloatingBasalMassBal",
                   help="regionalStats variable for applied floating BMB")
    p.add_argument("--out-fig-dir", default=None)
    return p.parse_args()


def load_regionmask(path):
    ds = xr.open_dataset(path, decode_times=False)
    names = eio.read_region_names(ds)              # friendly names, file order
    masks = ds["regionCellMasks"].values            # (nCells, nRegions) int
    return names, masks


def main():
    args = parse_args()
    fig_dir = args.out_fig_dir or os.path.join(args.root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    regions = [r.strip() for r in args.regions.split(",")]

    # --- regionalStats for the chosen member ---
    reg_path = os.path.join(args.root, args.ensemble, args.member, "regionalStats.nc")
    if not os.path.isfile(reg_path):
        sys.exit(f"regionalStats not found: {reg_path}")
    rds = eio.to_year_dim(eio.load_member_regionalstats(reg_path))
    reg_names = eio.read_region_names(rds) if "regionNames" in rds else None
    if reg_names is None and args.regionmask:
        reg_names, _ = load_regionmask(args.regionmask)
    year = rds["year"].values

    # optional INPUT trend field basin-averaging
    trend_ds = None
    mask_names = mask_arr = None
    if args.trend_file:
        if not args.regionmask or not os.path.isfile(args.regionmask):
            print("WARNING: --trend-file given but no valid --regionmask; skipping input curve.")
        else:
            mask_names, mask_arr = load_regionmask(args.regionmask)
            trend_ds = xr.open_dataset(args.trend_file, decode_times=False)

    fig, axs = plt.subplots(len(regions), 1, figsize=(9, 4 * len(regions)),
                            squeeze=False)
    for i, rname in enumerate(regions):
        ax = axs[i, 0]
        # applied melt (model) from regionalStats
        ridx = eio.region_index(reg_names, rname)
        melt = rds[args.melt_var].isel(nRegions=ridx).values  # kg/yr (sign: gain +)
        ax.plot(year, -melt / 1e12, "C0", lw=1.8,
                label="applied floating BMB (model, Gt/yr melt)")
        ax.set_ylabel("melt (Gt/yr)"); ax.set_title(f"{rname}")
        ax.set_xlabel("year")

        # input trend field, basin-mean
        if trend_ds is not None:
            midx = eio.region_index(mask_names, rname)
            cell_mask = mask_arr[:, midx].astype(bool)
            tv = trend_ds[args.trend_var]
            # expect dims (Time, nCells)
            cells_dim = [d for d in tv.dims if d != "Time"][0]
            basin_mean = tv.isel({cells_dim: np.where(cell_mask)[0]}).mean(cells_dim).values
            t_year = (trend_ds["daysSinceStart"].values / 365.0
                      if "daysSinceStart" in trend_ds else np.arange(len(basin_mean)))
            ax2 = ax.twinx()
            ax2.plot(t_year, basin_mean, "C3", lw=1.5, alpha=0.8,
                     label="INPUT forcing basin-mean")
            ax2.set_ylabel("input forcing (native units)", color="C3")
            ax2.tick_params(axis="y", labelcolor="C3")
            ax2.legend(loc="upper right", fontsize=8)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"FRIS/Ross forcing-ramp diagnostic: {args.ensemble}:{args.member}")
    fig.tight_layout()
    out = os.path.join(fig_dir, f"forcing_ramp_{args.ensemble}_{args.member}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Figure -> {out}")
    print("Read the plot: a ramp visible in the RED (input) curve => it's in the driver;")
    print("a ramp only in the BLUE (applied model melt) curve => suspect the parameterization.")


if __name__ == "__main__":
    main()
