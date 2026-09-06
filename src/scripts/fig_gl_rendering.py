#!/usr/bin/env python3
"""
fig_gl_rendering.py -- two ways to draw a grounding line, on the same data.

MPAS-Tools contours the cellMask grounding-line flag as a 0/1 field at level 0.9999. A
binary field has no gradient inside a cell, so the contour can only run along cell
boundaries: at 4-20 km that is a visible staircase, and it is the reason the old animation
frames look ragged.

The alternative contours the flotation height itself,

    f = thickness - (rho_o / rho_i) * max(0, -bed)

which is continuous, changes sign exactly where the ice starts to float, and lets the
contour cross cells. Same definition, same data, sub-cell placement.

The model's own flag stays the reference: it is drawn as points so the smooth line can be
checked against it rather than assumed to agree.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import oceancolors as oc         # noqa: E402
import fig_gl_transect as glt    # noqa: E402

# MPAS-Albany cellMask bit values, from src/MPAS-Tools/plot_output_maps_masked_animation.py
INITIAL_EXTENT, DYNAMIC, FLOATING, GROUNDING_LINE = 1, 2, 4, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-km", type=float, nargs=4,
                    default=[-1750, -1150, -750, -150], metavar=("X0", "X1", "Y0", "Y1"),
                    help="Amundsen sector by default, where the mesh is finest")
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=os.path.join(
        glt.ROOT, "reports/dissertation/figures/slides"))
    a = ap.parse_args()
    out = a.out or f"{a.outdir}/fig_gl_rendering.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    import netCDF4
    mesh = os.path.join(glt.ROOT, "data/MALI", os.path.basename(glt.MESH))
    d = netCDF4.Dataset(mesh)
    cov = np.asarray(d["cellsOnVertex"][:]) - 1
    d.close()
    x, y = glt.rd(mesh, "xCell"), glt.rd(mesh, "yCell")
    bed, thk = glt.rd(mesh, "bedTopography"), glt.rd(mesh, "thickness")
    T = mtri.Triangulation(x, y, cov[(cov >= 0).all(axis=1)])

    ice = thk > 1.0
    flot = thk - (glt.RHO_O / glt.RHO_I) * np.maximum(0.0, -bed)   # continuous
    grounded = ice & (flot > 0)
    binary = grounded.astype(float)                                # what MPAS-Tools contours

    x0, x1, y0, y1 = [v * 1e3 for v in a.window_km]
    fig = plt.figure(figsize=(14.6, 7.4))
    panels = [("contouring the 0/1 mask, as MPAS-Tools does", binary, [0.9999]),
              ("contouring the flotation height at zero", np.where(ice, flot, np.nan), [0.0])]
    for i, (title, field, levels) in enumerate(panels):
        ax = fig.add_axes([0.035 + 0.495 * i, 0.075, 0.445, 0.815])
        T.set_mask(~ice[T.triangles].all(axis=1))
        ax.tripcolor(T, np.where(ice, thk, np.nan), cmap=oc.cmap("thickness", "cmocean"),
                     shading="gouraud", rasterized=True, alpha=.55, zorder=1)
        ax.tricontour(T, field, levels=levels, colors=[ds.INK], linewidths=1.8, zorder=4)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.0, 1.012, title, transform=ax.transAxes, fontsize=13,
                color=ds.INK, ha="left", va="bottom")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.12, dpi=190)
    plt.close(fig)
    n = int((ice & (np.abs(flot) < 50)).sum())
    print(f"wrote {out}")
    print(f"  {n} cells within 50 m of flotation in the whole domain")


if __name__ == "__main__":
    main()
