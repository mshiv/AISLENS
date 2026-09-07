#!/usr/bin/env python3
"""
fig_coast_marks.py -- the coastal section mark at four extents.

The Pin Point DEM is too finely dissected to work as a small mark: at that zoom the tidal
creeks read as noise. These are the same engraved treatment at wider extents, using the
Natural Earth coastline that already ships with the project rather than the DEM.

  site        the DEM extent, for reference
  estuary     Ossabaw and Wassaw sounds, the ground the model actually covers
  bight       the Georgia and South Carolina coast
  southeast   far enough out to carry Matthew's track, which is the event scale

The track is drawn on the widest one because that is the extent it justifies. Coastline
reading is the dependency-free reader from coral.viz.fig_chapter_domains.
"""
from __future__ import annotations

import os, sys, struct, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import fig_gl_transect as glt    # noqa: E402

NE = os.path.expanduser("~/.local/share/cartopy/shapefiles/natural_earth/physical")
BDECK = ("/Users/smurugan9/research/coral/reports/chapter4_hpc_v3/"
         "source_package/context/bal142016.dat.gz")
PIN_POINT = (-81.09, 31.95)

CUDEM_GA = "/Users/smurugan9/research/coral/data/raw/ga_cudem_30m.tif"

EXTENTS = {
    "site":      (-81.30, -80.85, 31.78, 32.20),
    "estuary":   (-81.75, -80.60, 31.40, 32.45),
    "bight":     (-82.60, -78.60, 30.20, 34.00),
    "southeast": (-84.50, -74.00, 24.00, 37.00),
    "cudem":     (-81.80, -80.55, 30.70, 32.60),   # the Georgia CUDEM, its own coastline
}


def polylines(path, box):
    """WGS84 polyline parts that intersect the box, without a GIS dependency."""
    out = []
    with open(path, "rb") as f:
        if struct.unpack(">i", f.read(100)[:4])[0] != 9994:
            raise ValueError("not a shapefile")
        while rec := f.read(8):
            if len(rec) != 8:
                break
            _, words = struct.unpack(">ii", rec)
            content = f.read(words * 2)
            kind = struct.unpack("<i", content[:4])[0]
            if kind == 0:
                continue
            if kind not in (3, 13, 23, 5, 15, 25):     # polyline or polygon
                continue
            n_parts, n_pts = struct.unpack("<ii", content[36:44])
            starts = np.frombuffer(content, "<i4", n_parts, 44)
            xy = np.frombuffer(content, "<f8", 2 * n_pts, 44 + 4 * n_parts).reshape(-1, 2)
            for b, e in zip(starts, np.r_[starts[1:], n_pts]):
                seg = xy[b:e]
                if seg.size and ((seg[:, 0] > box[0]) & (seg[:, 0] < box[1]) &
                                 (seg[:, 1] > box[2]) & (seg[:, 1] < box[3])).any():
                    out.append(seg)
    return out


def track():
    import gzip
    lon, lat = [], []
    seen = set()
    with gzip.open(BDECK, "rt") as f:
        for line in f:
            c = [x.strip() for x in line.split(",")]
            if len(c) < 8 or c[4] != "BEST":
                continue
            if c[2] in seen:
                continue
            seen.add(c[2])
            la, lo = c[6], c[7]
            lat.append(int(la[:-1]) / 10.0 * (1 if la[-1] == "N" else -1))
            lon.append(int(lo[:-1]) / 10.0 * (-1 if lo[-1] == "W" else 1))
    return np.asarray(lon), np.asarray(lat)


def _blur(a, r):
    """Boxcar blur that ignores NaN, so nodata does not bleed into the coast."""
    if r < 1:
        return a
    k = np.ones(2 * r + 1) / (2 * r + 1)
    m = np.isfinite(a).astype(float)
    b = np.where(np.isfinite(a), a, 0.0)
    for ax_ in (0, 1):
        b = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), ax_, b)
        m = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), ax_, m)
    return np.where(m > 0.35, b / np.maximum(m, 1e-9), np.nan)


def cudem_coast(ax, box, min_km=3.0, stride=3, smooth=0):
    """Coastline from the Georgia CUDEM itself, generalised by dropping short pieces.

    Every tidal creek is resolved at 30 m, so the raw zero contour is thousands of
    fragments. Keeping only pieces longer than min_km leaves the shape of the coast
    and the major sounds without the hairline detail that turns a small mark to mush.
    """
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(CUDEM_GA) as r:
        w = from_bounds(box[0], box[2], box[1], box[3], r.transform)
        z = r.read(1, window=w)[::stride, ::stride].astype(np.float32)
        z[z == r.nodata] = np.nan
        t = r.window_transform(w)
    ny, nx = z.shape
    gx = t.c + t.a * stride * (np.arange(nx) + 0.5)
    gy = t.f + t.e * stride * (np.arange(ny) + 0.5)
    GX, GY = np.meshgrid(gx, gy)
    z = _blur(z, smooth)
    cs = ax.contour(GX, GY, z, levels=[0.0], colors="none", zorder=0)
    kept = 0
    deg_km = 111.0
    for coll in cs.allsegs:
        for seg in coll:
            if len(seg) < 4:
                continue
            d = np.diff(seg, axis=0)
            L = float(np.hypot(d[:, 0] * np.cos(np.radians(31.6)), d[:, 1]).sum()) * deg_km
            if L < min_km:
                continue
            ax.plot(seg[:, 0], seg[:, 1], "-", color=ds.INK, lw=0.85, zorder=4)
            kept += 1
    print(f"      cudem: kept {kept} pieces longer than {min_km:g} km")


def draw(ax, name, show_track=False, rule=False, min_km=3.0, smooth=0):
    from matplotlib.patches import Polygon as MplPolygon
    box = EXTENTS[name]

    # Ruling fills the whole box, which makes the mark read as a panel rather than a
    # shape. Antarctica works because it is a closed outline on empty ground, so the
    # coastal mark is left unruled by default to match it.
    if rule:
        step = (box[3] - box[2]) / 30.0
        for yv in np.arange(box[2], box[3] + step, step):
            ax.plot([box[0], box[1]], [yv, yv], "-", color="#C6D6E6", lw=0.45, zorder=1)

    if name != "cudem":
        for part in polylines(os.path.join(NE, "ne_10m_land.shp"), box):
            if len(part) > 2:
                ax.add_patch(MplPolygon(part, closed=True, fc="white", ec="none", zorder=2))

    if name == "cudem":
        cudem_coast(ax, box, min_km=min_km, smooth=smooth)
    else:
        for seg in polylines(os.path.join(NE, "ne_10m_coastline.shp"), box):
            ax.plot(seg[:, 0], seg[:, 1], "-", color=ds.INK, lw=1.1, zorder=4)

    if show_track:
        lo, la = track()
        m = (lo > box[0] - 6) & (lo < box[1] + 6) & (la > box[2] - 6) & (la < box[3] + 6)
        ax.plot(lo[m], la[m], "-", color=ds.MARSH_DEEP, lw=1.6, zorder=6)

    ax.plot(*PIN_POINT, "o", ms=5.5, mfc="white", mec=ds.MARSH_DEEP, mew=1.6, zorder=8)
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    ax.set_aspect(1.0 / np.cos(np.radians(0.5 * (box[2] + box[3]))))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extents", nargs="+", default=list(EXTENTS))
    ap.add_argument("--singles", action="store_true")
    ap.add_argument("--smooth", type=float, nargs="+", default=[0, 3, 6])
    ap.add_argument("--outdir", default=os.path.join(
        glt.ROOT, "reports/dissertation/figures/marks"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ds.apply()

    if a.singles:
        for n in a.extents:
            fig = plt.figure(figsize=(5, 5)); ax = fig.add_axes([0, 0, 1, 1])
            draw(ax, n, show_track=(n == "southeast"), min_km=8.0, smooth=4)
            fig.savefig(f"{a.outdir}/mark_coast_{n}.png", bbox_inches="tight",
                        pad_inches=0.02, dpi=200, transparent=True)
            plt.close(fig); print(f"  wrote mark_coast_{n}.png")
        return

    combos = [(n, k) for n in a.extents for k in (a.smooth if n == "cudem" else [0])]
    fig = plt.figure(figsize=(4.5 * len(combos), 5.2))
    for i, (n, k) in enumerate(combos):
        ax = fig.add_axes([i / len(combos) + 0.008, 0.045,
                           1 / len(combos) - 0.016, 0.885])
        draw(ax, n, show_track=(n == "southeast"), min_km=8.0, smooth=int(k))
        lab = f"{n}" if n != "cudem" else f"smoothed {int(k)} px"
        ax.text(0.5, 1.01, lab, transform=ax.transAxes, fontsize=14,
                color=ds.INK, ha="center", va="bottom")
    out = f"{a.outdir}/fig_coast_marks.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.14, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
