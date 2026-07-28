#!/usr/bin/env python3
"""Per-basin ensemble time series for VAF and total ice volume.

CTRL: 100 ice shelf basins (133-region mask). SSP585/SSP126: 16 ISMIP6 basins.
Each subplot: ensemble mean ± spread + per-member lines. Also produces skewness, spread,
and relative uncertainty diagnostics.
"""

from __future__ import annotations
import os, sys, argparse, glob, re
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from ismip6_regions import BASIN_NAMES

ROOT = eio.default_ensembles_root()

# --- Region mask for 133-region CTRL ---
REGION_MASK_133 = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "projects",
    "MALI_AISLENS_DIR_TEMPLATE_4KM", "DRAFT-DEPEN-TEMPLATE-DIR",
    "aislens_draftDepen_regionMasks.nc")

ENSEMBLES = {
    "CTRL":    {"include": r"^CTRL_\d+$",   "min_years": 50, "color": "black",
                "n_regions": 133, "shelf_start": 33},  # 0-indexed: regions 34-133
    "SSP585":  {"include": r"^SSP585_\d+$", "min_years": 50, "color": "#D55E00",
                "n_regions": 16, "shelf_start": None},
    "SSP126":  {"include": r"^SSP126_\d+$", "min_years": 50, "color": "#0072B2",
                "n_regions": 16, "shelf_start": None},
}

# Names for the 100 individual shelf basins (read from file at runtime)
def load_133_region_names():
    """Load region names from the 133-region mask file."""
    ds = xr.open_dataset(REGION_MASK_133, decode_times=False)
    raw = ds["regionNames"].values
    names = []
    for row in raw:
        if isinstance(row, bytes):
            s = row.decode("utf-8", "ignore")
        else:
            s = b"".join([c if isinstance(c, bytes) else bytes(c) for c in row]).decode(
                "utf-8", "ignore")
        names.append(s.strip())
    ds.close()
    return names  # 133 names, 0-indexed


def load_regional_ensemble(ens_cfg, ens_dir):
    """Load regional VAF and total ice volume for all members.
    Returns (year, vaf, total_vol) where each is (n_member, n_year, n_region)."""
    members = eio.discover_members(ens_dir, stats_filename="regionalStats.nc",
                                   include=ens_cfg["include"])
    if not members:
        return None, None, None

    all_vaf = []
    all_vol = []
    all_year = []
    for name, path in members:
        try:
            ds = eio.load_member_regionalstats(path)
            ds = eio.to_year_dim(ds)
        except Exception:
            continue
        if "regionalVolumeAboveFloatation" not in ds:
            continue
        yr = ds.year.values.astype(float)
        if len(yr) < ens_cfg.get("min_years", 10) or yr[0] > 5.0:
            continue
        vaf = ds["regionalVolumeAboveFloatation"].values  # (year, region)
        vol = ds["regionalIceVolume"].values if "regionalIceVolume" in ds else None
        all_vaf.append(vaf)
        all_vol.append(vol)
        all_year.append(yr)
        ds.close()

    if not all_vaf:
        return None, None, None

    # Align to shortest year axis (for clean plotting)
    min_len = min(len(y) for y in all_year)
    year = all_year[0][:min_len]
    vaf = np.stack([v[:min_len] for v in all_vaf])  # (member, year, region)
    vol = np.stack([v[:min_len] for v in all_vol]) if all_vol[0] is not None else None
    return year, vaf, vol


def vaf_to_sle(vaf_m3):
    """Convert VAF (m^3) to SLE (mm)."""
    RHO_ICE, RHO_OCEAN, OCEAN_AREA = 910.0, 1028.0, 3.62e14
    return -vaf_m3 * (RHO_ICE / RHO_OCEAN) / OCEAN_AREA * 1000.0


def compute_member_stats(data):
    """Compute mean, std, skewness across members at each timestep.
    data: (n_member, n_year) or (n_member, n_year, n_region)"""
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1) if data.shape[0] > 1 else np.zeros_like(mean)
    skew = np.full(mean.shape, np.nan)
    if data.shape[0] >= 3:
        if data.ndim == 2:
            for t in range(data.shape[1]):
                vals = data[:, t]
                vals = vals[np.isfinite(vals)]
                if len(vals) >= 3:
                    skew[t] = sp_stats.skew(vals)
        elif data.ndim == 3:
            for t in range(data.shape[1]):
                for r in range(data.shape[2]):
                    vals = data[:, t, r]
                    vals = vals[np.isfinite(vals)]
                    if len(vals) >= 3:
                        skew[t, r] = sp_stats.skew(vals)
    return mean, std, skew


def plot_basin_timeseries(year, vaf, vol, basin_names, region_slice, ens_name,
                          color, out_dir, var="vaf", max_basins=20):
    """Plot per-basin ensemble timeseries in a grid of subplots.
    region_slice: slice object for which regions to plot."""
    n_region_total = vaf.shape[2]
    if isinstance(region_slice, slice):
        indices = list(range(*region_slice.indices(n_region_total)))
    else:
        indices = list(region_slice)
    indices = indices[:max_basins]

    n = len(indices)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    if var == "vaf":
        data = vaf_to_sle(vaf)
        ylabel = "SLE (mm)"
        title_var = "VAF"
    else:
        data = vol / 1e12  # Convert to 10^6 km^3 → 10^3 km^3
        ylabel = "Volume (×10³ km³)"
        title_var = "Total Ice Volume"

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              sharex=True)
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        basin_data = data[:, :, idx]  # (member, year)
        mean, std, _ = compute_member_stats(basin_data)

        for m in range(basin_data.shape[0]):
            ax.plot(year[:len(basin_data[m])], basin_data[m], color=color,
                    alpha=0.15, lw=0.5)
        # Mean
        ax.plot(year[:len(mean)], mean, color=color, lw=2, ls="--", label="Mean")
        # ±1σ band
        ax.fill_between(year[:len(mean)], mean - std, mean + std,
                        color=color, alpha=0.2)

        name = basin_names[idx] if idx < len(basin_names) else f"Region {idx+1}"
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if i >= n - ncols:
            ax.set_xlabel("Years", fontsize=8)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{ens_name} — {title_var} by Basin", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = os.path.join(out_dir, f"{ens_name}_{var}_regional_timeseries.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_diagnostics(year, vaf, basin_names, region_slice, ens_name,
                     color, out_dir, max_basins=20):
    """Plot skewness, spread, and relative uncertainty per basin."""
    n_region_total = vaf.shape[2]
    if isinstance(region_slice, slice):
        indices = list(range(*region_slice.indices(n_region_total)))
    else:
        indices = list(region_slice)
    indices = indices[:max_basins]

    data = vaf_to_sle(vaf)
    n = len(indices)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              sharex=True)
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        basin_data = data[:, :, idx]
        mean, std, skew = compute_member_stats(basin_data)
        mean_abs = np.abs(mean)
        rel = np.where(mean_abs > 1e-10, std / mean_abs, np.nan)

        ax.plot(year[:len(skew)], skew, color=color, lw=1.5, label="Skewness")
        ax.plot(year[:len(std)], std / (np.abs(mean) + 1e-10), color=color,
                lw=1.5, ls="--", label="Spread/|mean|")
        ax.axhline(0, color="0.5", ls=":", lw=0.5)

        name = basin_names[idx] if idx < len(basin_names) else f"Region {idx+1}"
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel("Diagnostic", fontsize=8)
        if i >= n - ncols:
            ax.set_xlabel("Years", fontsize=8)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{ens_name} — Basin Diagnostics", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = os.path.join(out_dir, f"{ens_name}_regional_diagnostics.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT/regional")
    parser.add_argument("--max-basins", type=int, default=20,
                        help="Max basins per figure (top by spread)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_names_133 = load_133_region_names()
    shelf_names = all_names_133[33:]  # Regions 34-133 (100 shelves)

    for ens_name, cfg in ENSEMBLES.items():
        print(f"\n=== {ens_name} ===")
        ens_dir = os.path.join(ROOT, ens_name)
        year, vaf, vol = load_regional_ensemble(cfg, ens_dir)
        if year is None:
            print(f"  No data loaded for {ens_name}")
            continue

        print(f"  Loaded: {vaf.shape[0]} members, {vaf.shape[1]} years, {vaf.shape[2]} regions")

        if ens_name == "CTRL":
            names = shelf_names
            # Sort by spread to show most interesting basins first
            data_sle = vaf_to_sle(vaf)
            spreads = np.nanstd(data_sle, axis=(0, 1))  # per-region total spread
            top_idx = np.argsort(spreads)[::-1][:args.max_basins]
            region_indices = top_idx

            # Also show full 100-basin figure (first 40 by spread)
            plot_basin_timeseries(year, vaf, vol, names,
                                  list(np.argsort(spreads)[::-1][:40]),
                                  ens_name, cfg["color"], args.out_dir, "vaf")
            if vol is not None:
                plot_basin_timeseries(year, vaf, vol, names,
                                      list(np.argsort(spreads)[::-1][:40]),
                                      ens_name, cfg["color"], args.out_dir, "vol")
            plot_diagnostics(year, vaf, names, list(np.argsort(spreads)[::-1][:40]),
                             ens_name, cfg["color"], args.out_dir)
        else:
            names = BASIN_NAMES
            # Sort by spread
            data_sle = vaf_to_sle(vaf)
            spreads = np.nanstd(data_sle, axis=(0, 1))
            top_idx = np.argsort(spreads)[::-1]

            plot_basin_timeseries(year, vaf, vol, names, list(top_idx),
                                  ens_name, cfg["color"], args.out_dir, "vaf")
            if vol is not None:
                plot_basin_timeseries(year, vaf, vol, names, list(top_idx),
                                      ens_name, cfg["color"], args.out_dir, "vol")
            plot_diagnostics(year, vaf, names, list(top_idx),
                             ens_name, cfg["color"], args.out_dir)


if __name__ == "__main__":
    main()
