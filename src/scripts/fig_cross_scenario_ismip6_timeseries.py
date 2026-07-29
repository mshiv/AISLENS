#!/usr/bin/env python3
"""Compare SSP585, SSP126, and CTRL per ISMIP6 basin (all use native 16-basin mask)."""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import ensemble_io as eio

ROOT = os.path.abspath(eio.default_ensembles_root())
OUT = os.path.join(os.path.dirname(__file__), "..", "..",
    "reports", "figures", "presentations", "20260722-IceT", "regional")
os.makedirs(OUT, exist_ok=True)

ISMIP6_NAMES = [
    "A-Ap (DML)", "Ap-B (Enderby)", "B-C (Amery)", "C-Cp (Denman)",
    "Cp-D (Totten)", "D-Dp (Mertz)", "Dp-E (Vic.Land)", "E-F (Ross)",
    "F-G (Getz)", "G-H (Thwaites)", "H-Hp (Belling.", "Hp-I (GeorgeVI)",
    "I-Ipp (LarsenAC)", "Ipp-J (LarsenE)", "J-K (FRIS)", "K-A (Brunt)"
]

COLORS = {"CTRL": "#888888", "SSP585": "#D55E00", "SSP126": "#0072B2"}


def load_ismip6_ensemble(ens_name, include, min_years=50):
    """Load 16-basin regional VAF and volume from SSP scenarios (native mask)."""
    ens_dir = os.path.join(ROOT, ens_name)
    members = eio.discover_members(ens_dir, stats_filename="regionalStats.nc",
                                   include=include)
    all_vaf, all_vol, all_year = [], [], []
    for name, path in members:
        try:
            ds = eio.load_member_regionalstats(path)
            ds = eio.to_year_dim(ds)
        except Exception:
            continue
        yr = ds.year.values.astype(float)
        if len(yr) < min_years or yr[0] > 5.0:
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


def main():
    ensemble_data = {}
    for ens_name, include in [("CTRL", r"^CTRL_\d+$"),
                               ("SSP585", r"^SSP585_\d+$"),
                               ("SSP126", r"^SSP126_\d+$")]:
        print(f"Loading {ens_name}...")
        year, vaf, vol = load_ismip6_ensemble(ens_name, include)
        if year is not None:
            print(f"  {vaf.shape[0]} members, {vaf.shape[1]} years, {vaf.shape[2]} basins")
            ensemble_data[ens_name] = (year, vaf, vol)

    # --- VAF timeseries ---
    # Sort basins by total spread across SSP scenarios
    all_sle = {k: vaf_to_sle(v[1]) for k, v in ensemble_data.items()}
    total_spread = np.zeros(16)
    for d in all_sle.values():
        total_spread += np.nanstd(d, axis=(0, 1))
    order = np.argsort(total_spread)[::-1]

    fig, axes = plt.subplots(4, 4, figsize=(18, 15), sharex=True)
    axes = axes.flatten()

    for plot_idx, basin_idx in enumerate(order):
        ax = axes[plot_idx]
        for ens_name, (yr, dat, _) in ensemble_data.items():
            basin_data = vaf_to_sle(dat[:, :, basin_idx])
            mean = np.nanmean(basin_data, axis=0)
            std = np.nanstd(basin_data, axis=0, ddof=1) if basin_data.shape[0] > 1 else np.zeros_like(mean)
            color = COLORS[ens_name]
            for m in range(basin_data.shape[0]):
                ax.plot(yr[:basin_data.shape[1]], basin_data[m], color=color, alpha=0.12, lw=0.4)
            ax.plot(yr[:len(mean)], mean, color=color, lw=2, ls="--", label=ens_name)
            ax.fill_between(yr[:len(mean)], mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(ISMIP6_NAMES[basin_idx], fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if plot_idx % 4 == 0:
            ax.set_ylabel("SLE (mm)", fontsize=8)
        if plot_idx >= 12:
            ax.set_xlabel("Years", fontsize=8)
        if plot_idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    # Panel 16: Global total with all scenarios
    ax = axes[15]
    for ens_name, (yr, dat, _) in ensemble_data.items():
        global_data = vaf_to_sle(dat).sum(axis=2)
        mean = np.nanmean(global_data, axis=0)
        std = np.nanstd(global_data, axis=0, ddof=1) if global_data.shape[0] > 1 else np.zeros_like(mean)
        color = COLORS[ens_name]
        ax.plot(yr[:len(mean)], mean, color=color, lw=2, ls="--", label=ens_name)
        ax.fill_between(yr[:len(mean)], mean - std, mean + std, color=color, alpha=0.15)
    ax.set_title("GLOBAL TOTAL", fontsize=9, fontweight="bold", color="red")
    ax.legend(fontsize=7)

    fig.suptitle("VAF (SLE) by ISMIP6 Basin — SSP585 vs SSP126 vs CTRL", fontsize=14, y=1.01)
    fig.tight_layout()
    outpath = os.path.join(OUT, "cross_scenario_ismip6_vaf_timeseries.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

    # --- Volume timeseries ---
    fig, axes = plt.subplots(4, 4, figsize=(18, 15), sharex=True)
    axes = axes.flatten()

    total_spread_vol = np.zeros(16)
    for k, (yr, dat, vol) in ensemble_data.items():
        if vol is not None:
            total_spread_vol += np.nanstd(vol / 1e6, axis=(0, 1))
    order_vol = np.argsort(total_spread_vol)[::-1]

    for plot_idx, basin_idx in enumerate(order_vol):
        ax = axes[plot_idx]
        for ens_name, (yr, dat, vol) in ensemble_data.items():
            if vol is None:
                continue
            basin_data = vol[:, :, basin_idx] / 1e6
            mean = np.nanmean(basin_data, axis=0)
            std = np.nanstd(basin_data, axis=0, ddof=1) if basin_data.shape[0] > 1 else np.zeros_like(mean)
            color = COLORS[ens_name]
            for m in range(basin_data.shape[0]):
                ax.plot(yr[:basin_data.shape[1]], basin_data[m], color=color, alpha=0.12, lw=0.4)
            ax.plot(yr[:len(mean)], mean, color=color, lw=2, ls="--", label=ens_name)
            ax.fill_between(yr[:len(mean)], mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(ISMIP6_NAMES[basin_idx], fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if plot_idx % 4 == 0:
            ax.set_ylabel("Volume (×10⁶ km³)", fontsize=8)
        if plot_idx >= 12:
            ax.set_xlabel("Years", fontsize=8)
        if plot_idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    # Panel 16: Global total volume
    ax = axes[15]
    for ens_name, (yr, dat, vol) in ensemble_data.items():
        if vol is None:
            continue
        global_vol = vol.sum(axis=2) / 1e6
        mean = np.nanmean(global_vol, axis=0)
        std = np.nanstd(global_vol, axis=0, ddof=1) if global_vol.shape[0] > 1 else np.zeros_like(mean)
        color = COLORS[ens_name]
        ax.plot(yr[:len(mean)], mean, color=color, lw=2, ls="--", label=ens_name)
        ax.fill_between(yr[:len(mean)], mean - std, mean + std, color=color, alpha=0.15)
    ax.set_title("GLOBAL TOTAL", fontsize=9, fontweight="bold", color="red")
    ax.legend(fontsize=8)

    fig.suptitle("Total Ice Volume by ISMIP6 Basin — SSP585 vs SSP126", fontsize=14, y=1.01)
    fig.tight_layout()
    outpath = os.path.join(OUT, "cross_scenario_ismip6_vol_timeseries.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
