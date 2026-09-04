#!/usr/bin/env python3
"""
fig_drift_basin_map.py — noise-induced mean displacement, mapped and summed by basin.

Per-cell VAF(10x) - VAF(1x) at one horizon, plus the same field summed into the 16 ISMIP6
basins. Positive = the louder ensemble loses more ice there. Numbers reproduce
drift_basin_horizons.py; sign pattern is transferable, magnitude is a 10x-amplitude result.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slidestyle as ds  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPAT = f"{ROOT}/reports/dissertation/figures/spatial/stats_sample"
MESH = (f"{ROOT}/data/MALI/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m"
        "_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu"
        "_meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
MASK = f"{ROOT}/data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"

RHO_I, RHO_O, A_O = 910.0, 1028.0, 3.625e14

LETTERS = ["A-Ap", "Ap-B", "B-C", "C-Cp", "Cp-D", "D-Dp", "Dp-E", "E-F",
           "F-G", "G-H", "H-Hp", "Hp-I", "I-Ipp", "Ipp-J", "J-K", "K-A"]
NAMES = {"A-Ap": "Dronning Maud", "Ap-B": "Enderby", "B-C": "Amery", "C-Cp": "Denman",
         "Cp-D": "Totten", "D-Dp": "Mertz", "Dp-E": "Victoria", "E-F": "Ross",
         "F-G": "Getz", "G-H": "Thwaites / Pine Is.", "H-Hp": "Bellingshausen",
         "Hp-I": "George VI", "I-Ipp": "Larsen A–C", "Ipp-J": "Larsen E",
         "J-K": "Filchner–Ronne", "K-A": "Brunt"}
# only these get a name on the slide; the rest stay unlabelled to keep it readable
LABEL_IF_ABOVE = 9.0  # mm SLE


def rd(path, var):
    d = netCDF4.Dataset(path)
    a = np.ma.filled(np.asarray(d[var][:], dtype=float), np.nan)
    d.close()
    return np.ravel(a) if a.ndim > 1 else a


def vaf(h, bed):
    return np.maximum(0.0, h - (RHO_O / RHO_I) * np.maximum(0.0, -bed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2300)
    ap.add_argument("--out", default=f"{ROOT}/reports/dissertation/figures/slides/"
                                     "fig_drift_basin_map.png")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ds.apply()

    bed = rd(MESH, "bedTopography")
    area = rd(MESH, "areaCell")
    x = rd(MESH, "xCell") / 1e3
    y = rd(MESH, "yCell") / 1e3
    m = netCDF4.Dataset(MASK)
    masks = np.asarray(m["regionCellMasks"][:])
    m.close()

    h1 = rd(f"{SPAT}/SSP585_{a.year}.nc", "thickness_mean")
    h10 = rd(f"{SPAT}/SSP585_varScaled10x_{a.year}.nc", "thickness_mean")

    dv = vaf(h10, bed) - vaf(h1, bed)
    dv = np.where(np.isfinite(dv), dv, 0.0)
    to_mm = RHO_I / (RHO_O * A_O) * 1000.0
    sle = -(dv * area) * to_mm                    # mm SLE per cell, + = more sea level
    drift_m = -dv                                 # metres of VAF lost under 10x, + = more loss

    basin = {L: float(np.nansum(sle[masks[:, i] > 0])) for i, L in enumerate(LETTERS)}
    vals = np.array([basin[L] for L in LETTERS])
    pos, neg, net = vals[vals > 0].sum(), vals[vals < 0].sum(), vals.sum()
    gross = np.abs(vals).sum()

    fig = plt.figure(figsize=(12.4, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.04,
                          left=0.01, right=0.985, top=0.90, bottom=0.10)

    # ---------------------------------------------------------------- map
    axm = fig.add_subplot(gs[0, 0])
    axm.set_aspect("equal")
    axm.axis("off")

    ice = np.isfinite(h1) & (h1 > 1.0)
    axm.scatter(x[ice], y[ice], s=0.55, c="#E6E1D6", marker=".",
                linewidths=0, rasterized=True)

    # crop to the ice sheet, not the mesh
    pad = 120.0
    axm.set_xlim(x[ice].min() - pad, x[ice].max() + pad)
    axm.set_ylim(y[ice].min() - pad, y[ice].max() + pad)

    # colour limit from the responding cells only, so the coastal ring reads
    active = np.abs(drift_m) > 1.0
    lim = float(np.nanpercentile(np.abs(drift_m[active]), 88))
    sel = np.abs(drift_m) > 0.5
    sc: PathCollection = axm.scatter(
        x[sel], y[sel], c=np.clip(drift_m[sel], -lim, lim), s=1.5, marker=".",
        cmap=ds.DIVERGING, vmin=-lim, vmax=lim, linewidths=0, rasterized=True)

    cax = axm.inset_axes([0.02, 0.045, 0.30, 0.024])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    cb.outline.set_visible(False)
    cb.set_ticks([-lim, 0, lim])
    cb.set_ticklabels([f"−{lim:.0f}", "0", f"+{lim:.0f}"])
    cb.ax.set_facecolor(ds.PAPER)
    cb.ax.tick_params(labelsize=9, length=2, colors=ds.INK_SOFT)
    cb.set_label("ice above flotation lost under 10× forcing  (m)",
                 fontsize=9, color=ds.INK_SOFT, labelpad=5)

    axm.text(0.02, 0.175, "more ice lost under louder forcing", transform=axm.transAxes,
             fontsize=10, color=ds.POS)
    axm.text(0.02, 0.132, "less ice lost", transform=axm.transAxes,
             fontsize=10, color=ds.NEG)

    # ---------------------------------------------------------------- bars
    axb = fig.add_subplot(gs[0, 1])
    order = sorted(LETTERS, key=lambda L: basin[L])
    ypos = np.arange(len(order))
    bv = [basin[L] for L in order]
    axb.barh(ypos, bv, height=0.66,
             color=[ds.POS if v > 0 else ds.NEG for v in bv], linewidth=0)
    axb.axvline(0, color=ds.INK, lw=0.9)
    axb.set_yticks([])
    ds.strip(axb, keep=("bottom",))
    axb.set_xlabel("basin contribution to the mean displacement  (mm SLE)", labelpad=6)
    axb.tick_params(axis="x", length=3)
    axb.set_ylim(-0.8, len(order) - 0.2)

    span = max(abs(min(bv)), abs(max(bv)))
    axb.set_xlim(-span * 1.30, span * 1.30)
    axb.set_xticks([-50, -25, 0, 25, 50])
    for yi, L in zip(ypos, order):
        v = basin[L]
        if abs(v) < LABEL_IF_ABOVE:
            continue
        ha = "left" if v > 0 else "right"
        off = span * 0.035 * (1 if v > 0 else -1)
        num = f"+{v:.0f}" if v > 0 else f"\u2212{abs(v):.0f}"
        axb.text(v + off, yi, f"{NAMES[L]}  {num}", va="center", ha=ha,
                 fontsize=10, color=ds.INK)

    # the arithmetic, spelled out
    parts = [(f"+{pos:.0f}", 26, ds.POS), ("  and  ", 12, ds.INK_SOFT),
             (f"−{abs(neg):.0f}", 26, ds.NEG), ("  report  ", 12, ds.INK_SOFT),
             (f"+{net:.0f} mm", 26, ds.INK)]
    xcur = 0.0
    for txt, size, col in parts:
        t = axb.text(xcur, 1.10, txt, transform=axb.transAxes, fontsize=size,
                     color=col, ha="left", va="bottom")
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
        xcur += bb.width / axb.get_window_extent().width
    axb.text(0.0, 1.035,
             f"{100*abs(net)/gross:.0f}% of the gross regional signal survives aggregation"
             f"     ·     model year {a.year-2000}",
             transform=axb.transAxes, fontsize=10.5, color=ds.INK_SOFT,
             ha="left", va="bottom")

    fig.text(0.01, 0.012,
             "10× amplitude experiment — the sign pattern is transferable, the magnitude is not",
             fontsize=9, color=ds.INK_SOFT, ha="left", va="bottom", style="italic")

    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.14)
    print(f"wrote {a.out}")
    print(f"  yr {a.year-2000}: pos {pos:+.2f}  neg {neg:+.2f}  net {net:+.2f}  "
          f"gross {gross:.2f}  surviving {100*abs(net)/gross:.1f}%")


if __name__ == "__main__":
    main()
