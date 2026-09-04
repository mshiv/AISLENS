#!/usr/bin/env python3
"""
fig_mali_mesh.py — what the ice-sheet model is, for people who have not seen one.

Two panels from the MALI mesh file itself: the variable-resolution grid, and the observed
surface speed it is refined to resolve. The point is that the mesh is fine exactly where the
ice moves fast and the grounding line lives, and coarse over the slow interior.

Cell spacing is sqrt(areaCell), which is the effective spacing of a Voronoi cell to within a
few percent -- close enough to label a mesh, and it is what the file actually stores.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import fig_gl_transect as glt    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(glt.ROOT,
                                                     "reports/dissertation/figures/slides"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ds.apply()

    d = netCDF4.Dataset(glt.MESH)
    x = np.asarray(d["xCell"][:]).ravel() / 1e3
    y = np.asarray(d["yCell"][:]).ravel() / 1e3
    area = np.asarray(d["areaCell"][:]).ravel()
    h = np.asarray(d["thickness"][:]).ravel()
    vx = np.asarray(d["observedSurfaceVelocityX"][:]).ravel()
    vy = np.asarray(d["observedSurfaceVelocityY"][:]).ravel()
    d.close()

    spacing = np.sqrt(area) / 1e3                     # km
    speed = np.hypot(vx, vy) * 31557600.0             # m/s -> m/yr
    ice = h > 1.0

    fig = plt.figure(figsize=(13.2, 6.6))
    axm = fig.add_axes([0.030, 0.075, 0.430, 0.815])
    axv = fig.add_axes([0.520, 0.075, 0.430, 0.815])

    sm = axm.scatter(x[ice], y[ice], c=spacing[ice], s=0.35, cmap="cividis",
                     vmin=np.percentile(spacing[ice], 1),
                     vmax=np.percentile(spacing[ice], 99),
                     linewidths=0, rasterized=True)
    cb = fig.colorbar(sm, ax=axm, fraction=0.036, pad=0.02)
    cb.set_label("cell spacing  (km)", fontsize=10)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=9)

    v = np.where(speed > 1.0, speed, 1.0)
    sv = axv.scatter(x[ice], y[ice], c=v[ice], s=0.35, cmap="magma_r",
                     norm=LogNorm(vmin=1.0, vmax=3000.0),
                     linewidths=0, rasterized=True)
    cb2 = fig.colorbar(sv, ax=axv, fraction=0.036, pad=0.02)
    cb2.set_label("observed surface speed  (m per year)", fontsize=10)
    cb2.outline.set_visible(False)
    cb2.ax.tick_params(length=2, labelsize=9)

    # area-weighted, because the coarse cells are few but cover most of the sheet
    ar = area[ice]
    fast = speed[ice] > 100.0
    o = np.argsort(spacing[ice][~fast])
    cw = np.cumsum(ar[~fast][o])
    slow_km = spacing[ice][~fast][o][np.searchsorted(cw, cw[-1] / 2)]

    for ax, title, sub in (
            (axm, "a variable-resolution mesh",
             f"{ice.sum():,} cells · {np.median(spacing[ice][fast]):.1f} km at the "
             f"margins, {slow_km:.0f} km over the interior"),
            (axv, "cells concentrate where the ice moves",
             f"fast ice is {100 * ar[fast].sum() / ar.sum():.0f}% of the area "
             f"and {100 * fast.mean():.0f}% of the cells")):
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.5, 1.055, title, transform=ax.transAxes, fontsize=14,
                color=ds.INK, ha="center", va="bottom")
        ax.text(0.5, 1.005, sub, transform=ax.transAxes, fontsize=10,
                color=ds.INK_SOFT, ha="center", va="bottom")

    out = f"{a.outdir}/fig_mali_mesh.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.14, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  {ice.sum():,} ice cells   spacing {spacing[ice].min():.1f}–"
          f"{spacing[ice].max():.1f} km, median {np.median(spacing[ice]):.1f}")
    fast = speed[ice] > 100.0
    print(f"  fast ice (>100 m/yr): {100 * fast.mean():.1f}% of cells, "
          f"median spacing there {np.median(spacing[ice][fast]):.1f} km "
          f"vs {np.median(spacing[ice][~fast]):.1f} km elsewhere")


if __name__ == "__main__":
    main()
