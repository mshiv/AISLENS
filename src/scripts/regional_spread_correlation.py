#!/usr/bin/env python3
"""
regional_spread_correlation.py — per-basin spread ranking and cross-basin correlation.

Uses regionalStats.nc to compute ensemble spread (std of Delta-VAF->SLE) per ISMIP6
basin at the latest common year, plus the Pearson correlation matrix across basins.

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=eio.default_ensembles_root(),
                   help="ENSEMBLES root dir")
    p.add_argument("--ensemble", default="SSP585_varScaled10x",
                   help="Ensemble sub-directory name")
    p.add_argument("--members", default=r"^SSP585_\d+$",
                   help="Regex to select a clean member subset")
    p.add_argument("--min-years", type=float, default=50.0,
                   help="Drop members whose record spans fewer than this many years")
    p.add_argument("--out-dir", default=None,
                   help="Figure output dir (default: <root>/figures)")
    return p.parse_args()


def load_regional_delta_vaf(ensemble_dir, include, min_years):
    """Load regionalVolumeAboveFloatation for every member, align onto the
    SHORTEST member's year axis (intersection -- every member present at every
    year), and return Delta-VAF->SLE (mm) with dims (member, year, nRegions)."""
    members = eio.discover_members(ensemble_dir, stats_filename="regionalStats.nc",
                                   include=include)
    if not members:
        raise RuntimeError(f"No members with regionalStats.nc under {ensemble_dir}")

    per_member, names, dropped = [], [], []
    for name, path in members:
        ds = eio.load_member_regionalstats(path)
        ds = eio.to_year_dim(ds)
        raw_year0 = float(ds["year"].values[0])
        if abs(raw_year0) > 5.0:          # restart-continuation segment: drop, don't relabel to yr 0
            dropped.append((name, f"start@{raw_year0:.0f}"))
            continue
        ds = ds.assign_coords(year=ds["year"] - raw_year0)
        span = float(ds["year"].values[-1])
        if min_years is not None and span < min_years:
            dropped.append((name, round(span, 1)))
            continue
        per_member.append(ds[["regionalVolumeAboveFloatation"]])
        names.append(name)

    if dropped:
        print(f"  dropped {len(dropped)} member(s) shorter than {min_years} yr: {dropped}")
    if not per_member:
        raise RuntimeError("All members dropped by min_years/include filters.")

    ref_year = min(per_member, key=lambda d: float(d["year"].values[-1]))["year"].values
    aligned = [d.interp(year=ref_year) for d in per_member]
    out = xr.concat(aligned, dim="member")
    out = out.assign_coords(member=("member", names))

    vaf = out["regionalVolumeAboveFloatation"]  # (member, year, nRegions)
    vaf0 = vaf.isel(year=0)
    delta_vaf = vaf - vaf0
    delta_sle = xr.apply_ufunc(
        lambda a: -a * (eio.RHO_ICE / eio.RHO_OCEAN) / eio.OCEAN_AREA * 1000.0,
        delta_vaf,
    )
    delta_sle.name = "delta_sle_mm"
    return delta_sle, ref_year, names


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    ens_dir = os.path.join(args.root, args.ensemble)
    delta_sle, year, member_names = load_regional_delta_vaf(
        ens_dir, args.members, args.min_years)
    n_members = delta_sle.sizes["member"]
    latest_year = float(year[-1])
    print(f"Loaded {n_members} members for {args.ensemble}; latest common year "
          f"= {latest_year:.1f}")

    region_names = eio.region_names_default()
    n_regions = delta_sle.sizes["nRegions"]
    if region_names is None or len(region_names) != n_regions:
        print(f"  WARNING: region_names_default() returned "
              f"{None if region_names is None else len(region_names)} names for "
              f"{n_regions} basins; falling back to generic labels.")
        region_names = [f"basin{i}" for i in range(n_regions)]

    # values at the latest common year, all members, all basins: (member, nRegions)
    final = delta_sle.isel(year=-1).values
    basin_std = np.std(final, axis=0, ddof=1)  # ensemble spread per basin (mm SLE)
    basin_mean = np.mean(final, axis=0)

    order = np.argsort(basin_std)[::-1]  # largest spread first
    print(f"Per-basin ensemble spread (std across {n_members} members) of "
          f"Delta-VAF->SLE at year {latest_year:.1f}, ranked largest-to-smallest:")
    for rank, i in enumerate(order, 1):
        print(f"  {rank:2d}. {region_names[i]:20s} std={basin_std[i]:7.4f} mm SLE  "
              f"mean={basin_mean[i]:8.4f} mm SLE")

    top3 = [region_names[i] for i in order[:3]]
    print(f"Top-3 highest-spread basins: {top3}")

    # correlation matrix of member anomalies across basins
    anom = final - basin_mean[np.newaxis, :]  # (member, nRegions)
    corr = np.corrcoef(anom, rowvar=False)  # (nRegions, nRegions)

    # ---- figure ----
    fig, axs = plt.subplots(1, 2, figsize=(15, 7),
                            gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axs[0]
    ranked_names = [region_names[i] for i in order]
    ranked_std = basin_std[order]
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, n_regions))
    ax.barh(range(n_regions), ranked_std[::-1], color=colors[::-1])
    ax.set_yticks(range(n_regions))
    ax.set_yticklabels(ranked_names[::-1])
    ax.set_xlabel("ensemble spread: std(Delta-VAF->SLE) across members (mm SLE)")
    ax.set_title(f"Per-basin ensemble spread at year {latest_year:.0f}\n"
                 f"({args.ensemble}, {n_members} members)")

    ax2 = axs[1]
    im = ax2.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(n_regions))
    ax2.set_yticks(range(n_regions))
    ax2.set_xticklabels(region_names, rotation=90, fontsize=8)
    ax2.set_yticklabels(region_names, fontsize=8)
    ax2.set_title("Cross-basin correlation of member Delta-VAF anomalies\n"
                 "(spatial coherence of ensemble spread)")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{args.ensemble}_regional_spread_correlation.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure -> {out_path}")


if __name__ == "__main__":
    main()
