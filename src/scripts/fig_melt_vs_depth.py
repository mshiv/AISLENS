#!/usr/bin/env python3
"""
fig_melt_vs_depth.py — Relates subshelf melt rate to water depth per ISMIP6 basin.

Panel (a): bar chart of regional mean melt rate, SSP585 vs varScaled10x.
Panel (b): scatter of melt-rate spread vs mean melt rate. Uses regionalStats.nc.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio


# Approximate mean ocean depth per ISMIP6 basin (m), used as x-axis proxy for depth.
# Values from observational compilations (e.g. Fretwell et al. 2013 Bedmap2).
BASIN_DEPTH = {
    "Dronning Maud Land": 1200,
    "Enderby Land":       1400,
    "Amery-Lambert":      1100,
    "Phillipi, Denman":   1300,
    "Totten":             1500,
    "Mertz":               900,
    "Victoria Land":       600,
    "Ross":                800,
    "Getz":               1200,
    "Thwaites/PIG":       1600,
    "Bellingshausen":     1100,
    "George VI":           900,
    "Larsen A-C":          700,
    "Larsen E":            500,
    "FRIS":               1000,
    "Brunt-Stancomb":      800,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=eio.default_ensembles_root())
    p.add_argument("--min-years", type=float, default=50.0)
    p.add_argument("--melt-year-start", type=float, default=200.0,
                   help="Start year for melt-rate averaging (sim-year)")
    p.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    return p.parse_args()


def load_regional_melt(ensemble_dir, include, min_years=50, year_start=200.0):
    members = eio.discover_members(ensemble_dir, stats_filename="regionalStats.nc",
                                   include=include)
    stacks = []
    names_list = []
    for name, path in members:
        try:
            ds = eio.to_year_dim(eio.load_member_regionalstats(path))
        except Exception:
            continue
        if "regionalAvgSubshelfMelt" not in ds:
            continue
        yr = ds["year"].values
        if yr[-1] - yr[0] < min_years or yr[0] > 5.0:
            continue
        # Average melt rate over the last portion of the simulation
        melt = ds["regionalAvgSubshelfMelt"].sel(year=slice(year_start, None)).mean("year")
        stacks.append(melt.values)
        names_list.append(name)
    if not stacks:
        return None, None
    return np.array(stacks), names_list  # (n_members, n_regions)


def get_region_names(ds_path):
    ds = xr.open_dataset(ds_path, decode_times=False)
    # Try multi-char format (nRegions, StrLen) first
    if "regionNames" in ds:
        raw = ds["regionNames"].values
        if raw.ndim == 2:
            names = []
            for row in raw:
                if isinstance(row, bytes):
                    s = row.decode("utf-8", "ignore")
                else:
                    s = b"".join([c if isinstance(c, bytes) else bytes(c) for c in row]).decode("utf-8", "ignore")
                key = "".join(filter(str.isalnum, s.strip()))
                names.append(eio.ISMIP6_BASIN_NAMES.get(key, key))
            return names
        elif raw.ndim == 1:
            names = []
            for v in raw:
                if isinstance(v, bytes):
                    s = v.decode("utf-8", "ignore")
                else:
                    s = str(v)
                key = "".join(filter(str.isalnum, s.strip()))
                names.append(eio.ISMIP6_BASIN_NAMES.get(key, key))
            return names
    return eio.region_names_default() or []


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scenarios = {
        "SSP585":       (os.path.join(args.root, "SSP585"),       r"^SSP585_\d+$"),
        "varScaled10x": (os.path.join(args.root, "SSP585_varScaled10x"), r"^SSP585_\d+$"),
    }

    data = {}
    for label, (ens_dir, inc) in scenarios.items():
        arr, member_names = load_regional_melt(ens_dir, inc, args.min_years,
                                                year_start=args.melt_year_start)
        if arr is None:
            print(f"  WARNING: no regional melt data for {label}")
            continue
        data[label] = arr
        print(f"  {label:16s}  {arr.shape[0]} members, {arr.shape[1]} regions")

    if len(data) < 2:
        sys.exit("Need at least 2 scenarios with data.")

    # Get region names from the regionMask file (regionalStats.nc doesn't contain them)
    region_mask = os.path.join(args.root, "..", "..", "AIS_4to20km_r01_20220907.regionMask_ismip6.nc")
    if not os.path.isfile(region_mask):
        region_mask = "/Users/smurugan9/research/aislens/AISLENS/data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"
    region_names = get_region_names(region_mask)
    n_regions = data[list(data.keys())[0]].shape[1]
    region_names = region_names[:n_regions]
    print(f"\n  Regions ({n_regions}): {region_names}")

    # ---- Compute per-basin stats ----
    stats = {}
    for label, arr in data.items():
        mean_per_basin = np.nanmean(arr, axis=0)  # (n_regions,)
        std_per_basin = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(n_regions)
        stats[label] = {"mean": mean_per_basin, "std": std_per_basin}

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): side-by-side bars
    ax = axes[0]
    labels = list(data.keys())
    x = np.arange(n_regions)
    width = 0.35
    colors = {"SSP585": "C0", "varScaled10x": "C3"}
    for k, label in enumerate(labels):
        offset = (k - (len(labels) - 1) / 2) * width
        ax.bar(x + offset, stats[label]["mean"], width, label=label,
               color=colors.get(label, f"C{k}"), alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(region_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean subshelf melt rate (m/yr)")
    ax.set_title("(a) Regional mean subshelf melt rate")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    # Panel (b): spread vs mean
    ax = axes[1]
    for label in labels:
        ax.scatter(stats[label]["mean"], stats[label]["std"],
                   s=40, label=label, color=colors.get(label, "C0"),
                   edgecolors="k", linewidth=0.5, zorder=3)
    for i, nm in enumerate(region_names):
        ax.annotate(nm, (stats[labels[0]]["mean"][i], stats[labels[0]]["std"][i]),
                    fontsize=6.5, alpha=0.75, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean subshelf melt rate (m/yr)")
    ax.set_ylabel("melt rate spread σ across members (m/yr)")
    ax.set_title("(b) Melt-rate spread vs mean\n(deep basins with more spread → depth-dependent melt)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    fig.suptitle("Subshelf melt vs water depth: does variability amplify melt at depth?",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "melt_vs_depth.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
