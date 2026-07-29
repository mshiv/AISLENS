#!/usr/bin/env python3
"""Global VAF and total ice volume with specific ice shelves removed.

CTRL: removes Eastern_Ross, Western_Ross, Filchner, Ronne.
SSP585/SSP126: removes Ross and FRIS (ISMIP6 equivalents).
Three panels per ensemble: global vs removed VAF, individual shelf contributions, volume.
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio

ROOT = eio.default_ensembles_root()

REGION_MASK_133 = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "projects",
    "MALI_AISLENS_DIR_TEMPLATE_4KM", "DRAFT-DEPEN-TEMPLATE-DIR",
    "aislens_draftDepen_regionMasks.nc")

# Shelf names to remove (from 133-region mask, 0-indexed)
SHELF_REMOVE_133 = {
    "Eastern Ross": "Eastern_Ross",
    "Western Ross": "Western_Ross",
    "Filchner": "Filchner",
    "Ronne": "Ronne",
}

# ISMIP6 basin equivalents for SSP585/SSP126
SHELF_REMOVE_ISMIP6 = {
    "Ross": "Ross",        # Contains both Eastern + Western Ross
    "FRIS": "FRIS",        # Contains both Filchner + Ronne
}

ENSEMBLES = {
    "CTRL":    {"include": r"^CTRL_\d+$",   "min_years": 50, "color": "black",
                "n_regions": 133, "use_133": True},
    "SSP585":  {"include": r"^SSP585_\d+$", "min_years": 50, "color": "#D55E00",
                "n_regions": 16, "use_133": False},
    "SSP126":  {"include": r"^SSP126_\d+$", "min_years": 50, "color": "#0072B2",
                "n_regions": 16, "use_133": False},
}


def load_133_region_names():
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
    return names


def load_regional(ens_cfg, ens_dir):
    """Load regional data for all members. Returns (year, vaf, vol) arrays."""
    members = eio.discover_members(ens_dir, stats_filename="regionalStats.nc",
                                   include=ens_cfg["include"])
    if not members:
        return None, None, None

    all_vaf, all_vol, all_year = [], [], []
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
        vaf = ds["regionalVolumeAboveFloatation"].values
        vol = ds["regionalIceVolume"].values if "regionalIceVolume" in ds else None
        all_vaf.append(vaf)
        all_vol.append(vol)
        all_year.append(yr)
        ds.close()

    if not all_vaf:
        return None, None, None

    min_len = min(len(y) for y in all_year)
    year = all_year[0][:min_len]
    vaf = np.stack([v[:min_len] for v in all_vaf])
    vol = np.stack([v[:min_len] for v in all_vol]) if all_vol[0] is not None else None
    return year, vaf, vol


def vaf_to_sle(vaf_m3):
    RHO_ICE, RHO_OCEAN, OCEAN_AREA = 910.0, 1028.0, 3.62e14
    return -vaf_m3 * (RHO_ICE / RHO_OCEAN) / OCEAN_AREA * 1000.0


def compute_timeseries_stats(data):
    """data: (n_member, n_year). Returns mean, std."""
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1) if data.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def shelf_removal_analysis(year, vaf, vol, region_names, shelf_remove_dict,
                            use_133, ens_name, color, out_dir):
    """Compute and plot VAF/volume with shelves removed."""
    n_mem, n_yr, n_reg = vaf.shape

    # Map shelf names to region indices (exact match preferred)
    shelf_indices = {}
    for label, match_name in shelf_remove_dict.items():
        # First try exact match
        for i, rn in enumerate(region_names):
            if rn == match_name:
                shelf_indices[label] = i
                break
        # If no exact match, try contains but exclude combined regions
        if label not in shelf_indices:
            for i, rn in enumerate(region_names):
                if match_name.lower() in rn.lower() and "-" not in rn:
                    shelf_indices[label] = i
                    break

    if not shelf_indices:
        print(f"  Warning: no shelf regions found for {ens_name}")
        return

    print(f"  Found shelves: {list(shelf_indices.keys())} at indices {list(shelf_indices.values())}")

    sle_all = vaf_to_sle(vaf)  # (member, year, region)

    # Global total per member
    sle_global = sle_all.sum(axis=2)  # (member, year)

    # Shelves removed
    removed_indices = list(shelf_indices.values())
    sle_shelves_removed = sle_all.copy()
    sle_shelves_removed[:, :, removed_indices] = 0
    sle_reduced = sle_shelves_removed.sum(axis=2)  # (member, year)

    # Individual shelf contributions
    shelf_sle = {}
    for label, idx in shelf_indices.items():
        shelf_sle[label] = sle_all[:, :, idx]  # (member, year)

    # --- Figure 1: VAF (SLE) timeseries ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel (a): Global vs reduced
    ax = axes[0]
    mean_g, std_g = compute_timeseries_stats(sle_global)
    mean_r, std_r = compute_timeseries_stats(sle_reduced)

    ax.plot(year[:len(mean_g)], mean_g, color=color, lw=2, label="Global (all shelves)")
    ax.fill_between(year[:len(mean_g)], mean_g - std_g, mean_g + std_g,
                    color=color, alpha=0.15)
    ax.plot(year[:len(mean_r)], mean_r, color=color, lw=2, ls="--",
            label="Without key shelves")
    ax.fill_between(year[:len(mean_r)], mean_r - std_r, mean_r + std_r,
                    color=color, alpha=0.1)

    for m in range(sle_global.shape[0]):
        ax.plot(year[:sle_global.shape[1]], sle_global[m], color=color,
                alpha=0.08, lw=0.5)

    ax.set_ylabel("SLE (mm)")
    ax.set_title(f"{ens_name} — VAF: Global vs Shelves Removed")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel (b): Individual shelf contributions
    ax = axes[1]
    shelf_colors = ["#E76F51", "#2A9D8F", "#E9C46A", "#264653"]
    for i, (label, data) in enumerate(shelf_sle.items()):
        mean_s, std_s = compute_timeseries_stats(data)
        c = shelf_colors[i % len(shelf_colors)]
        ax.plot(year[:len(mean_s)], mean_s, color=c, lw=2, label=label)
        ax.fill_between(year[:len(mean_s)], mean_s - std_s, mean_s + std_s,
                        color=c, alpha=0.15)

    ax.set_xlabel("Years since simulation start")
    ax.set_ylabel("SLE contribution (mm)")
    ax.set_title(f"{ens_name} — Individual Shelf Contributions")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    f1 = os.path.join(out_dir, f"{ens_name}_shelf_removal_vaf.png")
    fig.savefig(f1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {f1}")

    # --- Figure 2: Total ice volume ---
    if vol is not None:
        fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        vol_global = vol.sum(axis=2) / 1e6  # m^3 → 10^6 km^3
        vol_reduced = vol.copy()
        vol_reduced[:, :, removed_indices] = 0
        vol_red = vol_reduced.sum(axis=2) / 1e6

        ax = axes2[0]
        mean_g, std_g = compute_timeseries_stats(vol_global)
        mean_r, std_r = compute_timeseries_stats(vol_red)

        ax.plot(year[:len(mean_g)], mean_g, color=color, lw=2, label="Global")
        ax.fill_between(year[:len(mean_g)], mean_g - std_g, mean_g + std_g,
                        color=color, alpha=0.15)
        ax.plot(year[:len(mean_r)], mean_r, color=color, lw=2, ls="--",
                label="Without key shelves")
        ax.fill_between(year[:len(mean_r)], mean_r - std_r, mean_r + std_r,
                        color=color, alpha=0.1)
        ax.set_ylabel("Ice Volume (×10⁶ km³)")
        ax.set_title(f"{ens_name} — Total Ice Volume: Global vs Shelves Removed")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Individual shelf volumes
        ax = axes2[1]
        for i, (label, idx) in enumerate(shelf_indices.items()):
            shelf_vol = vol[:, :, idx] / 1e6
            mean_s, std_s = compute_timeseries_stats(shelf_vol)
            c = shelf_colors[i % len(shelf_colors)]
            ax.plot(year[:len(mean_s)], mean_s, color=c, lw=2, label=label)
            ax.fill_between(year[:len(mean_s)], mean_s - std_s, mean_s + std_s,
                            color=c, alpha=0.15)
        ax.set_xlabel("Years since simulation start")
        ax.set_ylabel("Volume (×10⁶ km³)")
        ax.set_title(f"{ens_name} — Individual Shelf Volumes")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig2.tight_layout()
        f2 = os.path.join(out_dir, f"{ens_name}_shelf_removal_volume.png")
        fig2.savefig(f2, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {f2}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT/regional")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_names_133 = load_133_region_names()
    ismip6_names = [
        "Dronning Maud Land", "Enderby Land", "Amery-Lambert",
        "Phillipi, Denman", "Totten", "Mertz", "Victoria Land",
        "Ross", "Getz", "Thwaites/PIG", "Bellingshausen",
        "George VI", "Larsen A-C", "Larsen E", "FRIS", "Brunt-Stancomb"
    ]

    for ens_name, cfg in ENSEMBLES.items():
        print(f"\n=== {ens_name} ===")
        ens_dir = os.path.join(ROOT, ens_name)
        year, vaf, vol = load_regional(cfg, ens_dir)
        if year is None:
            print(f"  No data loaded")
            continue
        print(f"  Loaded: {vaf.shape[0]} members, {vaf.shape[1]} years, {vaf.shape[2]} regions")

        if cfg["use_133"]:
            region_names = all_names_133
            shelf_remove = SHELF_REMOVE_133
        else:
            region_names = ismip6_names
            shelf_remove = SHELF_REMOVE_ISMIP6

        shelf_removal_analysis(year, vaf, vol, region_names, shelf_remove,
                               cfg["use_133"], ens_name, cfg["color"], args.out_dir)


if __name__ == "__main__":
    main()
