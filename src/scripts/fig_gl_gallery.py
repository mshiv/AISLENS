#!/usr/bin/env python3
"""
fig_gl_gallery.py -- the grounding line drawn as small panels, one per sector.

The line is the zero contour of the flotation height, h - (rho_o/rho_i) * max(0, -bed),
rather than a contour of the 0/1 grounded mask. A binary field has no gradient inside a
cell so its contour can only follow cell edges, which is where the staircase in the old
frames came from.

Contouring that field over the whole sheet gives 425 pieces. Most of them are not error:
111 are closed loops with a median length of 20 km, which are ice rises and pinning
points, and only the pieces under about 5 km are noise -- 71 of them, 0.4 percent of the
total length. So the pieces are drawn by what they are: the long line heavy, the closed
loops lighter because they are real, and the short fragments dropped.

Pass --coast for the same idea at Pin Point, where the zero contour of the DEM is the
waterline instead of the grounding line.
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

REGIONS = ["Thwaites", "Pine_Island", "Getz", "Filchner-Ronne", "Ross", "Amery"]
MIN_KM = 5.0        # below this a piece is noise
LOOP_KM = 8.0       # a closed piece longer than this is an ice rise worth drawing


def pieces(T, field, level=0.0):
    """Contour pieces with their length in km and whether they close on themselves."""
    fig = plt.figure(); ax = fig.add_subplot(111)
    cs = ax.tricontour(T, field, levels=[level])
    out = []
    for coll in cs.allsegs:
        for s in coll:
            if len(s) < 2:
                continue
            L = float(np.hypot(*np.diff(s, axis=0).T).sum()) / 1e3
            out.append((s, L, bool(np.allclose(s[0], s[-1]))))
    plt.close(fig)
    return out


def draw(ax, segs):
    for s, L, closed in segs:
        if L < MIN_KM:
            continue
        if closed and L < LOOP_KM:
            continue
        ax.plot(s[:, 0], s[:, 1], "-", lw=1.0 if closed else 2.0,
                color=ds.INK_SOFT if closed else ds.INK,
                alpha=.85 if closed else 1.0, zorder=5)


def antarctic_panels(a):
    import netCDF4
    mesh = os.path.join(glt.ROOT, "data/MALI", os.path.basename(glt.MESH))
    d = netCDF4.Dataset(mesh); cov = np.asarray(d["cellsOnVertex"][:]) - 1; d.close()
    x, y = glt.rd(mesh, "xCell"), glt.rd(mesh, "yCell")
    bed, thk = glt.rd(mesh, "bedTopography"), glt.rd(mesh, "thickness")
    T = mtri.Triangulation(x, y, cov[(cov >= 0).all(axis=1)])
    ice = thk > 1.0
    flot = thk - (glt.RHO_O / glt.RHO_I) * np.maximum(0.0, -bed)
    T.set_mask(~ice[T.triangles].all(axis=1))
    segs = pieces(T, np.where(ice, flot, np.nan))

    names, masks = glt.region_names(glt.SHELF_MASK)
    masks = np.asarray(masks)
    out = []
    for r in a.regions:
        if r not in names:
            print(f"  ! no region {r}"); continue
        m = np.asarray(masks[:, names.index(r)], bool)
        cx, cy = x[m].mean(), y[m].mean()
        half = a.window_km * 1e3 / 2
        out.append((r.replace("_", " "), T, np.where(ice, thk, np.nan), segs,
                    (cx - half, cx + half, cy - half, cy + half), "thickness"))
    return out


def coastal_panel(a):
    """Pin Point, where the same zero contour is the waterline."""
    import rasterio
    dem = "/Users/smurugan9/research/coral/data/raw/ga_cudem_30m_savext.tif"
    if not os.path.exists(dem):
        print("  ! no CUDEM locally, skipping the coastal panel"); return []
    with rasterio.open(dem) as r:
        z = r.read(1)[::3, ::3].astype(np.float32)
        z[z == r.nodata] = np.nan
        ext = (r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top)
    return [("Pin Point", None, z, None, ext, "topography")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    ap.add_argument("--window-km", type=float, default=560.0)
    ap.add_argument("--coast", action="store_true", help="add the Pin Point panel")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=os.path.join(
        glt.ROOT, "reports/dissertation/figures/slides"))
    a = ap.parse_args()
    out = a.out or f"{a.outdir}/fig_gl_gallery.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    panels = antarctic_panels(a) + (coastal_panel(a) if a.coast else [])
    cols = a.cols
    rows = (len(panels) + cols - 1) // cols
    fig = plt.figure(figsize=(4.6 * cols, 4.9 * rows))

    for i, (name, T, field, segs, ext, role) in enumerate(panels):
        ax = fig.add_axes([(i % cols) / cols + 0.006, 1 - (i // cols + 1) / rows + 0.030,
                           1 / cols - 0.012, 1 / rows - 0.052])
        if T is None:                                   # the raster panel
            from matplotlib.colors import TwoSlopeNorm
            ax.imshow(field, origin="upper", extent=ext, cmap=oc.cmap(role, "cmocean"),
                      norm=TwoSlopeNorm(vmin=float(np.nanpercentile(field, 1)),
                                        vcenter=0.0, vmax=float(np.nanpercentile(field, 99))),
                      interpolation="bilinear", zorder=1)
            ax.contour(np.flipud(field), levels=[0.0], extent=ext, colors=[ds.INK],
                       linewidths=1.6, zorder=5)
        else:
            ax.tripcolor(T, field, cmap=oc.cmap(role, "cmocean"), shading="gouraud",
                         rasterized=True, alpha=.65, zorder=1)
            draw(ax, segs)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal" if T is not None else 1.0 / np.cos(np.radians(32.0)))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.03, 0.965, name, transform=ax.transAxes, fontsize=13.5, color=ds.INK,
                ha="left", va="top", zorder=8,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=.78))

    fig.savefig(out, bbox_inches="tight", pad_inches=0.10, dpi=185)
    plt.close(fig)
    print(f"wrote {out}   {len(panels)} panels")


if __name__ == "__main__":
    main()
