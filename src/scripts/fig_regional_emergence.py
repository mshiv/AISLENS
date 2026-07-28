#!/usr/bin/env python3
"""
fig_regional_emergence.py — Spatial structure of the ensemble response.

(A) Per-basin time-of-emergence map: year mean ΔVAF exceeds k× internal spread.
(B) Mean-vs-spread scatter: basins carrying SLR also carry uncertainty (MISI basins).
Uses regionalStats.nc + mesh + regionCellMasks.
"""
from __future__ import annotations
import os, sys, csv as _csv, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from ismip6_regions import BASIN_NAMES, SHORT_LABELS

MESH = ("data/MALI/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_"
        "Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_"
        "meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
MASK = "data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"


def load_regional_sle(root, ensemble, include):
    members = eio.discover_members(os.path.join(root, ensemble),
                                   stats_filename="regionalStats.nc", include=include)
    stacks, nmin = [], None
    for name, path in members:
        try:
            ds = eio.to_year_dim(eio.load_member_regionalstats(path))
        except Exception:
            continue
        if "regionalVolumeAboveFloatation" not in ds:
            continue
        vaf = ds["regionalVolumeAboveFloatation"]; yr = ds["year"].values
        if yr[0] > 5.0 or len(yr) < 10:
            continue
        nreg = vaf.sizes["nRegions"]
        sle = np.column_stack([eio.vaf_to_sle_mm(vaf.isel(nRegions=r).values, reference="first")
                               for r in range(nreg)])
        stacks.append((yr, sle)); nmin = len(yr) if nmin is None else min(nmin, len(yr))
    if len(stacks) < 3:
        return None, None
    years = stacks[0][0][:nmin]
    arr = np.stack([s[:nmin] for _, s in stacks], axis=0)     # (member, year, nRegions)
    return years, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--members", default=r"^SSP585_\d+$")
    ap.add_argument("--k", type=float, default=1.0, help="S/N threshold for emergence")
    ap.add_argument("--start-year", type=float, default=2000.0)
    ap.add_argument("--forcing-csv", default="reports/spectrum_percell_generated0.csv")
    ap.add_argument("--out-prefix", default="reports/regional_emergence")
    args = ap.parse_args()

    names = BASIN_NAMES
    years, arr = load_regional_sle(args.root, args.ensemble, args.members)
    if arr is None:
        sys.exit("no usable members")
    cal = args.start_year + years
    mean_bt = np.nanmean(arr, axis=0)                 # (year, nRegions)
    std_bt = np.nanstd(arr, axis=0, ddof=1)
    nreg = mean_bt.shape[1]

    # ToE per basin: first year |mean| > k*std (and stays); NaN if never
    toe = np.full(nreg, np.nan)
    for r in range(nreg):
        emerged = np.abs(mean_bt[:, r]) > args.k * np.maximum(std_bt[:, r], 1e-9)
        # require sustained (emerged from here on)
        sustained = emerged & (np.cumprod(emerged[::-1])[::-1].astype(bool))
        idx = np.where(sustained)[0]
        if idx.size:
            toe[r] = cal[idx[0]]
    print(f"{args.ensemble}: {arr.shape[0]} members; per-basin ToE (S/N>{args.k}):")
    for r, nm in enumerate(names):
        print(f"  {nm:7s} ToE={('%.0f'%toe[r]) if np.isfinite(toe[r]) else 'none':>6s}"
              f"   mean_final={mean_bt[-1, r]:8.2f}  sigma_final={std_bt[-1, r]:6.3f}")

    dmesh = xr.open_dataset(MESH, decode_times=False)
    xC = np.asarray(dmesh["xCell"].values)/1e3; yC = np.asarray(dmesh["yCell"].values)/1e3
    masks = np.asarray(xr.open_dataset(MASK, decode_times=False)["regionCellMasks"].values)
    if masks.shape[0] != xC.size and masks.shape[1] == xC.size:
        masks = masks.T
    cell_basin = np.argmax(masks, axis=1); in_any = masks.sum(axis=1) > 0
    cell_toe = np.where(in_any, toe[cell_basin], np.nan)

    # ---- Fig A: ToE map ----
    figA, ax = plt.subplots(figsize=(6.6, 6))
    ax.scatter(xC[~in_any], yC[~in_any], s=0.5, c="0.9", marker=".", linewidths=0)
    never = in_any & ~np.isfinite(cell_toe)
    ax.scatter(xC[never], yC[never], s=1.2, c="0.55", marker=".", linewidths=0)
    m = in_any & np.isfinite(cell_toe)
    sc = ax.scatter(xC[m], yC[m], s=1.4, c=cell_toe[m], cmap="viridis", marker=".", linewidths=0)
    figA.colorbar(sc, ax=ax, fraction=0.046, label="time-of-emergence (year)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Per-basin time-of-emergence — {args.ensemble}\n"
                 f"(year mean ΔVAF > {args.k}σ internal; grey = never emerges)")
    figA.tight_layout(); fA = f"{args.out_prefix}_toe_map.png"
    figA.savefig(fA, dpi=150, bbox_inches="tight"); plt.close(figA); print(f"Figure -> {fA}")

    # ---- Fig B: mean vs spread ----
    figB, ax = plt.subplots(figsize=(7.5, 5.6))
    mf, sf = mean_bt[-1], std_bt[-1]
    ax.scatter(mf, sf, c="C3", s=32, zorder=3)
    for r, nm in enumerate(names):
        ax.annotate(nm, (mf[r], sf[r]), fontsize=7, alpha=0.85)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel(f"per-basin mean ΔVAF contribution (mm SLE) @ {cal[-1]:.0f}  "
                  "(>0 = sea-level rise)")
    ax.set_ylabel("per-basin ensemble spread σ (mm)")
    ax.set_title(f"{args.ensemble}: mean loss and uncertainty COINCIDE in the marine/MISI basins\n"
                 "(σ scales with loss — MISI amplification; exceptions: D-Dp steady, A-Ap small-but-uncertain)")
    ax.grid(alpha=0.2); figB.tight_layout()
    fB = f"{args.out_prefix}_mean_vs_spread.png"
    figB.savefig(fB, dpi=150, bbox_inches="tight"); plt.close(figB); print(f"Figure -> {fB}")


if __name__ == "__main__":
    main()
