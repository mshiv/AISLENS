#!/usr/bin/env python3
"""
fig_map_spread_budget_covariance.py — map-based versions of spread budget and basin covariance.

  Panel (a): per-basin σ² fraction of total variance at yr200, painted on MALI mesh.
  Panel (b): per-basin Pearson correlation with Thwaites/PIG (reference basin), painted on mesh.

Uses the same mesh/mask approach as fig_dynamic_gating.py.

Author: Shivaprakash Muruganandham (2026-07-22)
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from fig_regional_emergence import load_regional_sle
from ismip6_regions import BASIN_NAMES

MESH = ("data/MALI/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_"
        "Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_"
        "meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
MASK = "data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--mesh", default=MESH)
    ap.add_argument("--mask", default=MASK)
    ap.add_argument("--start-year", type=float, default=2000.0)
    ap.add_argument("--horizon", type=float, default=200.0)
    ap.add_argument("--ref-basin", default="Thwaites/PIG",
                    help="Reference basin for covariance map")
    ap.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # Load SSP585 regional data
    years, arr = load_regional_sle(a.root, "SSP585", r"^SSP585_\d+$")
    if arr is None:
        sys.exit("SSP585: no usable members")
    n_mem, n_yr, nreg = arr.shape
    cal = a.start_year + years
    hy = a.start_year + a.horizon

    # Find horizon index
    idx = int(np.argmin(np.abs(years - a.horizon)))
    arr_h = arr[:, idx, :]  # (member, region) at horizon

    # ---- Per-basin sigma and variance fraction ----
    sig = np.nanstd(arr_h, axis=0, ddof=1)  # (nreg,)
    sig2 = sig ** 2
    total_sig2 = sig2.sum()
    fraction = sig2 / total_sig2

    # ---- Per-basin correlation with reference basin ----
    ref_idx = BASIN_NAMES.index(a.ref_basin)
    ref_ts = arr_h[:, ref_idx]  # (member,)
    corr = np.full(nreg, np.nan)
    for r in range(nreg):
        basin_ts = arr_h[:, r]
        ok = np.isfinite(ref_ts) & np.isfinite(basin_ts)
        if ok.sum() >= 3:
            corr[r] = np.corrcoef(ref_ts[ok], basin_ts[ok])[0, 1]

    # Print summary
    print(f"SSP585: {n_mem} members, horizon yr{a.horizon:.0f} (cal {hy:.0f})")
    print("\nPer-basin sigma and variance fraction:")
    sorted_idx = np.argsort(fraction)[::-1]
    cumsum = 0
    for r in sorted_idx:
        cumsum += fraction[r]
        print(f"  {BASIN_NAMES[r]:20s}  σ={sig[r]:6.2f} mm  σ²={100*fraction[r]:5.1f}%  (cum: {100*cumsum:5.1f}%)")

    print(f"\nCorrelation with {a.ref_basin}:")
    for r in range(nreg):
        tag = " ★" if r == ref_idx else ""
        print(f"  {BASIN_NAMES[r]:20s}  r={corr[r]:+6.3f}{tag}")

    # ---- Load mesh and masks ----
    dmesh = xr.open_dataset(a.mesh, decode_times=False)
    xC = np.asarray(dmesh["xCell"].values) / 1e3  # km
    yC = np.asarray(dmesh["yCell"].values) / 1e3
    masks = np.asarray(xr.open_dataset(a.mask, decode_times=False)["regionCellMasks"].values)
    if masks.shape[0] != xC.size and masks.shape[1] == xC.size:
        masks = masks.T
    cell_basin = np.argmax(masks, axis=1)
    in_any = masks.sum(axis=1) > 0

    # Paint cells
    cell_frac = np.where(in_any, fraction[cell_basin], np.nan)
    cell_corr = np.where(in_any, corr[cell_basin], np.nan)

    # ---- Figure ----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): variance fraction map
    ax_a.scatter(xC[~in_any], yC[~in_any], s=0.5, c="0.88", marker=".", linewidths=0)
    vmax_f = max(0.3, float(np.nanpercentile(cell_frac[in_any], 99)))
    sc_a = ax_a.scatter(xC[in_any], yC[in_any], s=1.2, c=cell_frac[in_any],
                        cmap="YlOrRd", vmin=0, vmax=vmax_f, marker=".", linewidths=0)
    fig.colorbar(sc_a, ax=ax_a, fraction=0.046, label="fraction of total σ²")
    ax_a.set_aspect("equal"); ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_title(f"(a) Variance budget on mesh — yr{a.horizon:.0f}\n"
                   f"Thwaites/PIG = {100*fraction[ref_idx]:.0f}% of total σ²")

    # Label top 3 basins at centroids
    top3 = sorted_idx[:3]
    for r in top3:
        sel = (cell_basin == r) & in_any
        if sel.any():
            ax_a.annotate(BASIN_NAMES[r], (xC[sel].mean(), yC[sel].mean()), fontsize=8,
                          fontweight="bold", color="k", ha="center",
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    # Panel (b): correlation with reference basin
    ax_b.scatter(xC[~in_any], yC[~in_any], s=0.5, c="0.88", marker=".", linewidths=0)
    sc_b = ax_b.scatter(xC[in_any], yC[in_any], s=1.2, c=cell_corr[in_any],
                        cmap="RdBu_r", vmin=-1, vmax=1, marker=".", linewidths=0)
    fig.colorbar(sc_b, ax=ax_b, fraction=0.046, label=f"Pearson r with {a.ref_basin}")
    ax_b.set_aspect("equal"); ax_b.set_xticks([]); ax_b.set_yticks([])
    ax_b.set_title(f"(b) Cross-basin correlation on mesh — yr{a.horizon:.0f}\n"
                   f"Reference: {a.ref_basin} (r=+1.00)")

    # Label all basins at centroids
    for r in range(nreg):
        sel = (cell_basin == r) & in_any
        if sel.any() and fraction[r] > 0.01:  # only label basins >1% of variance
            ax_b.annotate(BASIN_NAMES[r], (xC[sel].mean(), yC[sel].mean()), fontsize=7,
                          fontweight="bold", color="k", ha="center",
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    fig.suptitle(f"Spread budget and cross-basin coherence — SSP585 @ yr{a.horizon:.0f}",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(a.out_dir, "map_spread_budget_covariance.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
