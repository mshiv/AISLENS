#!/usr/bin/env python3
"""Cross-scenario regional comparison: SSP585 vs SSP126 (both native 16-basin mask).

CTRL uses a different 133-region mask, so it's excluded from basin-level comparisons.
CTRL is shown as a global reference only in Figure 1.

NOTE: CTRL's 133-region mask and SSP's 16-basin mask define different regional
boundaries. Mapping CTRL→ISMIP6 via overlap introduces ~20% per-basin bias.
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio

ROOT = eio.default_ensembles_root()

from ismip6_regions import BASIN_NAMES, SHORT_LABELS, letter_from_index

COLORS = {"CTRL": "#888888", "SSP585": "#D55E00", "SSP126": "#0072B2"}


def load_ismip6_regional(ens_name, include, min_years=50):
    """Load 16-basin regional data from SSP scenarios (native mask)."""
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
        if "regionalVolumeAboveFloatation" not in ds:
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


def load_ctrl_global(include, min_years=50):
    """Load CTRL global VAF from globalStats for reference."""
    ens_dir = os.path.join(ROOT, "CTRL")
    members = eio.discover_members(ens_dir, stats_filename="globalStats.nc",
                                   include=include)
    all_vaf, all_year = [], []
    for name, path in members:
        try:
            ds = eio.load_member_globalstats(path)
            ds = eio.to_year_dim(ds)
        except Exception:
            continue
        yr = ds.year.values.astype(float)
        if len(yr) < min_years or yr[0] > 5.0:
            continue
        vaf = ds["volumeAboveFloatation"].values
        all_vaf.append(vaf)
        all_year.append(yr)
        ds.close()
    if not all_vaf:
        return None, None
    min_len = min(len(y) for y in all_year)
    year = all_year[0][:min_len]
    vaf = np.stack([v[:min_len] for v in all_vaf])
    return year, vaf


def vaf_to_sle(vaf_m3):
    RHO_ICE, RHO_OCEAN, OCEAN_AREA = 910.0, 1028.0, 3.62e14
    return -vaf_m3 * (RHO_ICE / RHO_OCEAN) / OCEAN_AREA * 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT/regional")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load SSP585 and SSP126 (both native 16-basin)
    ssp_data = {}
    for ens_name, include in [("SSP585", r"^SSP585_\d+$"), ("SSP126", r"^SSP126_\d+$")]:
        print(f"Loading {ens_name}...")
        year, vaf, vol = load_ismip6_regional(ens_name, include)
        if year is not None:
            print(f"  {vaf.shape[0]} members, {vaf.shape[1]} years, {vaf.shape[2]} basins")
            ssp_data[ens_name] = (year, vaf, vol)

    # Load CTRL global for reference
    print("Loading CTRL (global)...")
    ctrl_year, ctrl_vaf = load_ctrl_global(r"^CTRL_\d+$")
    if ctrl_year is not None:
        print(f"  {ctrl_vaf.shape[0]} members, {ctrl_year.shape[0]} steps")

    # --- Figure 1: Cross-scenario per-basin VAF (SLE) ---
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=True)
    axes = axes.flatten()

    for b in range(16):
        ax = axes[b]
        for ens_name in ["SSP585", "SSP126"]:
            if ens_name not in ssp_data:
                continue
            yr, vaf, _ = ssp_data[ens_name]
            sle = vaf_to_sle(vaf[:, :, b])
            mean = np.nanmean(sle, axis=0)
            std = np.nanstd(sle, axis=0, ddof=1) if sle.shape[0] > 1 else np.zeros_like(mean)
            color = COLORS[ens_name]
            for m in range(sle.shape[0]):
                ax.plot(yr[:sle.shape[1]], sle[m], color=color, alpha=0.12, lw=0.4)
            ax.plot(yr[:len(mean)], mean, color=color, lw=2, label=ens_name)
            ax.fill_between(yr[:len(mean)], mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(BASIN_NAMES[b], fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if b % 4 == 0:
            ax.set_ylabel("SLE (mm)", fontsize=8)
        if b >= 12:
            ax.set_xlabel("Years", fontsize=8)
        if b == 0:
            ax.legend(fontsize=8)

    # Panel 16: Global total with all scenarios
    ax = axes[15]
    if ctrl_year is not None:
        ctrl_sle = vaf_to_sle(ctrl_vaf)
        ctrl_mean = np.nanmean(ctrl_sle, axis=0)
        ctrl_std = np.nanstd(ctrl_sle, axis=0, ddof=1) if ctrl_sle.shape[0] > 1 else np.zeros_like(ctrl_mean)
        ax.plot(ctrl_year[:len(ctrl_mean)], ctrl_mean, color=COLORS["CTRL"], lw=2, ls="--", label="CTRL")
        ax.fill_between(ctrl_year[:len(ctrl_mean)], ctrl_mean - ctrl_std, ctrl_mean + ctrl_std, color=COLORS["CTRL"], alpha=0.1)
    for ens_name in ["SSP585", "SSP126"]:
        if ens_name not in ssp_data:
            continue
        yr, vaf, _ = ssp_data[ens_name]
        global_sle = vaf_to_sle(vaf).sum(axis=2)
        mean = np.nanmean(global_sle, axis=0)
        std = np.nanstd(global_sle, axis=0, ddof=1) if global_sle.shape[0] > 1 else np.zeros_like(mean)
        color = COLORS[ens_name]
        ax.plot(yr[:len(mean)], mean, color=color, lw=2, label=ens_name)
        ax.fill_between(yr[:len(mean)], mean - std, mean + std, color=color, alpha=0.15)
    ax.set_title("GLOBAL TOTAL", fontsize=9, fontweight="bold", color="red")
    ax.legend(fontsize=8)

    fig.suptitle("Cross-Scenario VAF by ISMIP6 Basin (SSP585 vs SSP126)", fontsize=14, y=1.01)
    fig.tight_layout()
    f1 = os.path.join(args.out_dir, "cross_scenario_regional_vaf.png")
    fig.savefig(f1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {f1}")

    # --- Figure 2: Basin importance bar chart ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    x = np.arange(16)
    for i, ens_name in enumerate(["SSP585", "SSP126"]):
        if ens_name not in ssp_data:
            continue
        yr, vaf, _ = ssp_data[ens_name]
        sle = vaf_to_sle(vaf)
        final_mean = np.nanmean(sle[:, -1, :], axis=0)
        ax2.bar(x + i * bar_width, final_mean, bar_width, color=COLORS[ens_name], alpha=0.8, label=ens_name)

    ax2.set_xticks(x + bar_width / 2)
    ax2.set_xticklabels(BASIN_NAMES, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Cumulative SLE at final horizon (mm)")
    ax2.set_title("Basin Contribution to Total VAF Change")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(0, color="0.5", lw=0.5)
    fig2.tight_layout()
    f2 = os.path.join(args.out_dir, "cross_scenario_basin_importance.png")
    fig2.savefig(f2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {f2}")

    # --- Figure 3: Ranking preservation (SSP585 vs SSP126) ---
    if "SSP585" in ssp_data and "SSP126" in ssp_data:
        from scipy.stats import spearmanr
        fig3, axes3 = plt.subplots(4, 4, figsize=(14, 14))
        axes3 = axes3.flatten()

        for b in range(16):
            ax = axes3[b]
            ssp585_sle = vaf_to_sle(ssp_data["SSP585"][1][:, :, b])
            ssp126_sle = vaf_to_sle(ssp_data["SSP126"][1][:, :, b])

            ssp585_final = ssp585_sle[:, -1] if ssp585_sle.ndim > 1 else np.array([ssp585_sle[-1]])
            ssp126_final = ssp126_sle[:, -1] if ssp126_sle.ndim > 1 else np.array([ssp126_sle[-1]])

            if len(ssp585_final) < 3:
                ax.set_visible(False)
                continue

            rho, pval = spearmanr(ssp585_final, ssp126_final)
            ax.scatter(ssp585_final, ssp126_final, c="black", s=30, alpha=0.7)
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            ax.set_title(f"{BASIN_NAMES[b]}\nr={rho:.2f}{sig}", fontsize=8)
            if b % 4 == 0:
                ax.set_ylabel("SSP126 final SLE (mm)", fontsize=7)
            if b >= 12:
                ax.set_xlabel("SSP585 final SLE (mm)", fontsize=7)

        for i in range(16, len(axes3)):
            axes3[i].set_visible(False)

        fig3.suptitle("Ranking Preservation: SSP585 vs SSP126 by Basin", fontsize=13, y=1.01)
        fig3.tight_layout()
        f3 = os.path.join(args.out_dir, "cross_scenario_basin_ranking.png")
        fig3.savefig(f3, dpi=200, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved: {f3}")


if __name__ == "__main__":
    main()
