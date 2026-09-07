#!/usr/bin/env python3
"""
fig_section_marks.py -- engraved section marks for the two halves of the deck.

The proposal deck used an outline of Antarctica with light diagonal hatching. Same idea
here, but the outlines come from the model mesh and the Pin Point DEM rather than from a
generic shapefile, and there are four fills to choose between:

  hatch     diagonal ruling, closest to the original
  contour   nested isolines, the look of an engraved chart
  stipple   dots thinning with elevation
  hachure   short strokes down the slope, denser where it is steep

Each is drawn twice, once large for a section divider and once small enough to sit in a
corner as a running marker. Everything is line work on white, so it drops onto a slide
without a box around it.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.patches import PathPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import fig_gl_transect as glt    # noqa: E402

CUDEM = "/Users/smurugan9/research/coral/data/raw/ga_cudem_30m_savext.tif"


def paths_from(cs):
    """One compound Path from every closed piece of a contour set."""
    verts, codes = [], []
    for coll in cs.allsegs:
        for s in coll:
            if len(s) < 3:
                continue
            verts.extend(list(s) + [s[0]])
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(s) - 1) + [Path.CLOSEPOLY])
    return Path(np.asarray(verts), codes) if verts else None


def hatch(ax, filled, color="#9BB7D4", dense="////"):
    """filled is a contourf collection covering the region to rule."""
    plt.rcParams["hatch.linewidth"] = 0.55
    for c in (filled.collections if hasattr(filled, "collections") else [filled]):
        c.set_facecolor("none")
        c.set_edgecolor(color)
        c.set_linewidth(0.0)
        c.set_hatch(dense)
        c.set_zorder(2)


def stipple(ax, clip, field, T, color="#8FA9C4"):
    """Dots thinned by the field, so high ground stays open."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    n = 26000
    rng = np.random.default_rng(4)
    px = rng.uniform(x0, x1, n); py = rng.uniform(y0, y1, n)
    f = mtri.LinearTriInterpolator(T, field)(px, py).filled(np.nan)
    ok = np.isfinite(f)
    q = (f[ok] - np.nanmin(f)) / max(np.nanmax(f) - np.nanmin(f), 1e-9)
    keep = rng.uniform(size=ok.sum()) > q ** 0.7
    sc = ax.plot(px[ok][keep], py[ok][keep], ".", ms=1.05, color=color,
                 linestyle="none", zorder=2)[0]
    sc.set_clip_path(clip)


def hachure(ax, clip, field, T, color="#8FA9C4"):
    """Short strokes down the gradient, denser where the slope is steep."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    g = np.linspace(x0, x1, 150), np.linspace(y0, y1, 150)
    GX, GY = np.meshgrid(*g)
    Z = mtri.LinearTriInterpolator(T, field)(GX, GY).filled(np.nan)
    gy, gx = np.gradient(np.nan_to_num(Z))
    mag = np.hypot(gx, gy)
    thresh = np.nanpercentile(mag[np.isfinite(Z)], 45)
    L = (x1 - x0) / 110.0
    segs = []
    for j in range(0, Z.shape[0], 2):
        for i in range(0, Z.shape[1], 2):
            if not np.isfinite(Z[j, i]) or mag[j, i] < thresh:
                continue
            d = np.hypot(gx[j, i], gy[j, i]) or 1.0
            ux, uy = gx[j, i] / d, gy[j, i] / d
            segs.append(([GX[j, i] - ux * L, GX[j, i] + ux * L],
                         [GY[j, i] - uy * L, GY[j, i] + uy * L]))
    for xs, ys in segs:
        ln = ax.plot(xs, ys, color=color, lw=0.5, zorder=2)[0]
        ln.set_clip_path(clip)


def antarctic(ax, style):
    import netCDF4
    mesh = os.path.join(glt.ROOT, "data/MALI", os.path.basename(glt.MESH))
    d = netCDF4.Dataset(mesh); cov = np.asarray(d["cellsOnVertex"][:]) - 1; d.close()
    x, y = glt.rd(mesh, "xCell"), glt.rd(mesh, "yCell")
    thk, bed = glt.rd(mesh, "thickness"), glt.rd(mesh, "bedTopography")
    T = mtri.Triangulation(x, y, cov[(cov >= 0).all(axis=1)])
    ice = (thk > 1.0).astype(float)
    surf = np.where(thk > 1.0, bed + thk, np.nan)

    ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max())
    cs = ax.tricontour(T, ice, levels=[0.5], colors=[ds.INK], linewidths=1.5, zorder=4)
    clip = paths_from(cs)
    patch = PathPatch(clip, fc="none", ec="none")
    ax.add_patch(patch)

    if style == "hatch":
        hatch(ax, ax.tricontourf(T, ice, levels=[0.5, 1.5], colors="none", zorder=2))
    elif style == "contour":
        T.set_mask(~(thk > 1.0)[T.triangles].all(axis=1))
        ax.tricontour(T, np.nan_to_num(surf), levels=np.arange(0, 4200, 350),
                      colors=["#9BB7D4"], linewidths=0.5, zorder=2)
        T.set_mask(None)
    elif style == "stipple":
        stipple(ax, patch, np.nan_to_num(surf), T)
    elif style == "hachure":
        hachure(ax, patch, np.nan_to_num(surf), T)

    # the grounding line, heavier, as the second mark
    flot = thk - (glt.RHO_O / glt.RHO_I) * np.maximum(0.0, -bed)
    T.set_mask(~(thk > 1.0)[T.triangles].all(axis=1))
    ax.tricontour(T, np.where(thk > 1.0, flot, np.nan), levels=[0.0],
                  colors=[ds.INK], linewidths=0.9, zorder=5)
    T.set_mask(None)
    ax.set_aspect("equal")


def savannah(ax, style):
    import rasterio
    with rasterio.open(CUDEM) as r:
        z = r.read(1)[::4, ::4].astype(np.float32)
        z[z == r.nodata] = np.nan
        ext = (r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top)
    nan = ~np.isfinite(z)
    right = nan[:, int(0.70 * z.shape[1]):].mean(axis=1) > 0.5
    r1 = int(np.flatnonzero(right)[0]) if right.any() else z.shape[0]
    z = z[:r1]
    ny, nx = z.shape
    gx = np.linspace(ext[0], ext[1], nx)
    gy = np.linspace(ext[3], ext[3] - (ext[3] - ext[2]) * r1 / (r1 + nan.shape[0] - r1), ny)
    GX, GY = np.meshgrid(gx, gy)
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[-1], gy[0])

    cs = ax.contour(GX, GY, np.nan_to_num(z, nan=-99), levels=[0.0],
                    colors=[ds.INK], linewidths=1.3, zorder=4)
    clip = paths_from(cs)
    patch = PathPatch(clip, fc="none", ec="none")
    ax.add_patch(patch)

    if style == "hatch":
        hatch(ax, ax.contourf(GX, GY, np.nan_to_num(z, nan=-99),
                              levels=[0.0, 1e4], colors="none", zorder=2))
    elif style == "contour":
        # a coastal DEM has almost no relief, so a handful of levels is plenty
        ax.contour(GX, GY, np.nan_to_num(z), levels=[2, 5, 9],
                   colors=["#9BB7D4"], linewidths=0.45, zorder=2)
    elif style in ("stipple", "hachure"):
        Tt = mtri.Triangulation(GX.ravel(), GY.ravel())
        f = np.nan_to_num(z).ravel()
        (stipple if style == "stipple" else hachure)(ax, patch, f, Tt)
    ax.set_aspect(1.0 / np.cos(np.radians(32.0)))


def single(region, style, out, size=5.0, lw_scale=1.0):
    """One mark on its own transparent canvas, for a divider or a corner."""
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0, 0, 1, 1])
    (antarctic if region == "antarctic" else savannah)(ax, style)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=200, transparent=True)
    plt.close(fig)
    print(f"  wrote {os.path.basename(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles", action="store_true",
                    help="write each mark on its own transparent canvas")
    ap.add_argument("--styles", nargs="+",
                    default=["hatch", "contour", "stipple", "hachure"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=os.path.join(
        glt.ROOT, "reports/dissertation/figures/marks"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out = a.out or f"{a.outdir}/fig_section_marks.png"
    ds.apply()

    if a.singles:
        for reg in ("antarctic", "savannah"):
            for st in a.styles:
                single(reg, st, f"{a.outdir}/mark_{reg}_{st}.png")
        return

    rows = [("Antarctica", antarctic), ("Savannah coast", savannah)]
    fig = plt.figure(figsize=(4.4 * len(a.styles), 4.9 * len(rows)))
    for ri, (name, fn) in enumerate(rows):
        for ci, st in enumerate(a.styles):
            ax = fig.add_axes([ci / len(a.styles) + 0.008,
                               1 - (ri + 1) / len(rows) + 0.030,
                               1 / len(a.styles) - 0.016, 1 / len(rows) - 0.075])
            fn(ax, st)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ri == 0:
                ax.text(0.5, 1.02, st, transform=ax.transAxes, fontsize=14,
                        color=ds.INK, ha="center", va="bottom")
            if ci == 0:
                ax.text(-0.02, 0.5, name, transform=ax.transAxes, fontsize=13,
                        color=ds.INK_SOFT, ha="right", va="center", rotation=90)
            print(f"  {name:16s} {st}")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.14, dpi=170,
                facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
