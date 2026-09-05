#!/usr/bin/env python3
"""
fig_bedrock.py -- the bed beneath the ice sheet, from the MALI mesh.

Bed elevation over the whole continent, with the present-day grounding line drawn on top.
Most of West Antarctica sits on bed well below sea level, and that bed deepens towards the
interior -- the geometric setting for marine ice-sheet instability.

Triangles come from cellsOnVertex, so the native mesh is drawn rather than an interpolation.
Year 0 only. An explainer, not a result.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import fig_gl_transect as glt    # noqa: E402
import oceancolors as oc         # noqa: E402

RHO_I, RHO_O = glt.RHO_I, glt.RHO_O

# deep ocean -> shelf -> sea level -> land.  TwoSlopeNorm sends sea level to 0.5,
# so the neutral tone sits there and "below sea level" reads as one colour family.
# kept as the legacy fallback; oceancolors resolves the palette
BEDCOL = LinearSegmentedColormap.from_list("bed", [
    (0.00, "#0B2545"), (0.16, "#17456E"), (0.32, "#2E76A8"),
    (0.44, "#7FB2D4"), (0.50, "#EDE7D8"), (0.62, "#CBB68B"),
    (0.80, "#9C8A63"), (1.00, "#6B5F49")])

# labels are placed at the centroid of each region in the mask file, not by hand
LABELS = ["Thwaites", "Pine_Island", "Ross", "Filchner-Ronne", "Amery", "Getz"]
NUDGE = {  # small offsets so a label clears its own grounding line, in metres
    "Thwaites":       (-190e3, -110e3),
    "Pine_Island":    (-150e3,  110e3),
    "Getz":           (-120e3, -150e3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        glt.ROOT, "reports/dissertation/figures/slides"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--palette", default=None, choices=["legacy", "cmocean"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out = a.out or f"{a.outdir}/fig_bedrock.png"
    ds.apply()

    import netCDF4
    d = netCDF4.Dataset(glt.MESH)
    cov = np.asarray(d["cellsOnVertex"][:]) - 1        # MPAS is 1-based
    d.close()
    x, y = glt.rd(glt.MESH, "xCell"), glt.rd(glt.MESH, "yCell")
    bed = glt.rd(glt.MESH, "bedTopography")
    thk = glt.rd(glt.MESH, "thickness")

    # a triangle is real only where all three cells exist
    tri = cov[(cov >= 0).all(axis=1)]
    T = mtri.Triangulation(x, y, tri)

    ice = thk > 1.0
    hflot = (RHO_O / RHO_I) * np.maximum(0.0, -bed)
    grounded = ice & (thk > hflot)

    fig = plt.figure(figsize=(12.6, 10.4))
    ax = fig.add_axes([0.02, 0.045, 0.96, 0.875])

    norm = TwoSlopeNorm(vmin=-2500.0, vcenter=0.0, vmax=2500.0)
    tp = ax.tripcolor(T, bed, cmap=oc.cmap("topography", a.palette), norm=norm,
                      shading="gouraud", rasterized=True)

    # present-day grounding line and ice edge, from the same mesh
    ax.tricontour(T, grounded.astype(float), levels=[0.5],
                  colors=[ds.INK], linewidths=1.5, zorder=4)
    ax.tricontour(T, ice.astype(float), levels=[0.5],
                  colors=[ds.INK], linewidths=0.7, alpha=.45, zorder=4)

    names, masks = glt.region_names(glt.SHELF_MASK)
    masks = np.asarray(masks)                      # (nCells, nRegions)
    for nm in LABELS:
        if nm not in names:
            continue
        m = np.asarray(masks[:, names.index(nm)], bool)
        if m.sum() < 50:
            continue
        dx, dy = NUDGE.get(nm, (0.0, 0.0))
        ax.text(x[m].mean() + dx, y[m].mean() + dy, nm.replace("_", " "),
                fontsize=12.5, color=ds.INK, ha="center", va="center", zorder=6,
                bbox=dict(fc="white", ec="none", alpha=.74, pad=2.4))

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    cax = fig.add_axes([0.115, 0.095, 0.245, 0.020])
    cb = fig.colorbar(tp, cax=cax, orientation="horizontal",
                      ticks=[-2500, -1000, 0, 1000, 2500])
    cb.set_label("bed elevation  (m relative to sea level)", fontsize=11, labelpad=6)
    cb.ax.tick_params(length=3, labelsize=10)
    cb.outline.set_visible(False)

    # header sits above the axes, so nothing is written over the continent
    below = 100.0 * (bed[grounded] < 0).sum() / grounded.sum()
    fig.text(0.02, 0.982, "the bed beneath the ice sheet",
             fontsize=15, color=ds.INK, ha="left", va="top")
    fig.text(0.02, 0.950,
             f"{below:.0f}% of grounded ice rests on bed below sea level."
             f"  Heavy line is the present-day grounding line, thin line the ice edge.",
             fontsize=11.5, color=ds.INK_SOFT, ha="left", va="top")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.10, dpi=190)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  {grounded.sum():,} grounded cells, {ice.sum():,} with ice")
    print(f"  grounded ice on bed below sea level: {below:.1f}%")
    print(f"  bed range {bed.min():.0f} to {bed.max():.0f} m; "
          f"deepest grounded bed {bed[grounded].min():.0f} m")


if __name__ == "__main__":
    main()
