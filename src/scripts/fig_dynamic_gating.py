#!/usr/bin/env python3
"""
fig_dynamic_gating.py — per-basin ΔVAF spread vs forcing low-frequency fraction.

Panel D1: scatter of sigma vs forcing low-freq fraction per basin.
Panel D2: map of per-basin sigma on the MALI mesh.
Shows spread is NOT explained by forcing spectral shape — it concentrates in MISI basins.
Data: connect_forcing_response + forcing per-sector CSV + mesh.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from connect_forcing_response import ensemble_region_sigma, load_forcing_lowfreq

MESH = ("data/MALI/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_"
        "Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_"
        "meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
MASK = "data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--members", default=r"^SSP585_\d+$")
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--forcing-csv", default="reports/spectrum_percell_generated0.csv")
    ap.add_argument("--mesh", default=MESH)
    ap.add_argument("--mask", default=MASK)
    ap.add_argument("--out", default="reports/fig_dynamic_gating.png")
    args = ap.parse_args()

    names, _, lf_dm = load_forcing_lowfreq(args.forcing_csv)          # (16,)  # [CHANGED 2026-07-22-opencode] use >8yr band
    sig, years, used = ensemble_region_sigma(args.root, args.ensemble, args.members,
                                             [args.horizon])
    if sig is None:
        sys.exit("no usable members")
    hy = list(sig)[-1]
    s = sig[hy]                                                        # (16,) sigma per basin
    print(f"{args.ensemble}: {used} members, sigma at yr {hy:.0f}")
    for r, nm in enumerate(names):
        print(f"  {nm:7s} lowfreq {100*lf_dm[r]:4.0f}%  sigma {s[r]:.3f} mm")
    ok = np.isfinite(s) & np.isfinite(lf_dm)
    pr, pp = pearsonr(lf_dm[ok], s[ok])
    sr, sp = spearmanr(lf_dm[ok], s[ok])
    # outlier sensitivity: drop the largest-sigma basin and recompute
    o2 = ok.copy(); o2[np.nanargmax(np.where(ok, s, np.nan))] = False
    pr2, _ = pearsonr(lf_dm[o2], s[o2])
    print(f"corr(lowfreq, sigma): Pearson {pr:+.2f} (p={pp:.2f})  Spearman {sr:+.2f} (p={sp:.2f})  "
          f"Pearson-drop-max {pr2:+.2f}")

    # mesh + basin-of-cell
    dmesh = xr.open_dataset(args.mesh, decode_times=False)
    xC = np.asarray(dmesh["xCell"].values) / 1e3   # km
    yC = np.asarray(dmesh["yCell"].values) / 1e3
    masks = np.asarray(xr.open_dataset(args.mask, decode_times=False)["regionCellMasks"].values)
    if masks.shape[0] != xC.size and masks.shape[1] == xC.size:
        masks = masks.T
    cell_basin = np.argmax(masks, axis=1)
    in_any = masks.sum(axis=1) > 0
    cell_sig = np.where(in_any, s[cell_basin], np.nan)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))

    # D1 scatter
    axL.scatter(100 * lf_dm, s, c="C3", zorder=3)
    for r, nm in enumerate(names):
        axL.annotate(nm, (100 * lf_dm[r], s[r]), fontsize=7, alpha=0.8)
    axL.set_xlabel("forcing low-frequency fraction  (>8 yr, decadal+multidecadal, %)")  # [CHANGED 2026-07-22-opencode]
    axL.set_ylabel(f"per-basin ΔVAF spread σ (mm) @ yr {hy:.0f}")
    axL.set_title(f"(D1) spread vs forcing low-freq FRACTION (amplitude-independent)\n"
                  f"Pearson r={pr:+.2f} (p={pp:.2f}); Spearman {sr:+.2f}; drop-max {pr2:+.2f}")
    axL.grid(alpha=0.2)

    # D2 map — background all cells grey, shelf/basin cells colored by sigma
    axR.scatter(xC[~in_any], yC[~in_any], s=0.5, c="0.88", marker=".", linewidths=0)
    vmax = float(np.nanpercentile(cell_sig, 99))
    sc = axR.scatter(xC[in_any], yC[in_any], s=1.2, c=cell_sig[in_any],
                     cmap="inferno", vmin=0, vmax=vmax, marker=".", linewidths=0)
    fig.colorbar(sc, ax=axR, fraction=0.046, label=f"σ ΔVAF (mm) @ yr {hy:.0f}")
    axR.set_aspect("equal"); axR.set_xticks([]); axR.set_yticks([])
    axR.set_title("(D2) spread lives in the MISI basins\n(Amundsen G-H; Filchner-Ronne J-K)")
    # label the two dominant basins at their cell centroids
    for tag in ["G-H", "J-K"]:
        r = names.index(tag)
        sel = (cell_basin == r) & in_any
        if sel.any():
            axR.annotate(tag, (xC[sel].mean(), yC[sel].mean()), fontsize=9,
                         fontweight="bold", color="white", ha="center")

    fig.suptitle("Fig D — per-basin spread does not track the forcing low-frequency fraction (>8 yr); "
                 "it concentrates in the MISI basins", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure -> {args.out}")


if __name__ == "__main__":
    main()
