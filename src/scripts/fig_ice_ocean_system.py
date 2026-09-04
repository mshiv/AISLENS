#!/usr/bin/env python3
"""
fig_ice_ocean_system.py — how the ocean reaches grounded ice, on a real section.

An explainer for people who do not work on ice sheets, drawn on the actual Thwaites bed and
initial ice geometry rather than a cartoon: grounded sheet, floating shelf, grounding line,
cavity, and the bed that deepens inland. The lower panel measures that inland deepening,
which is the geometric condition behind marine ice-sheet instability.

Year 0 only -- no ensemble, no result. The transect machinery is shared with fig_gl_transect.py.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import fig_gl_transect as glt    # noqa: E402

RHO_I, RHO_O = glt.RHO_I, glt.RHO_O
INLAND_KM = 200.0                # how far inland the bed-slope panel looks
BED = "#7C6F58"
BEDFILL = "#DCD3C2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shelf", default="Thwaites")
    ap.add_argument("--outdir", default=os.path.join(glt.ROOT,
                                                     "reports/dissertation/figures/slides"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ds.apply()

    names, masks = glt.region_names(glt.SHELF_MASK)
    import netCDF4
    bm = netCDF4.Dataset(glt.BASIN_MASK)
    basins = np.asarray(bm["regionCellMasks"][:])
    bm.close()
    x, y = glt.rd(glt.MESH, "xCell"), glt.rd(glt.MESH, "yCell")
    bed, h0 = glt.rd(glt.MESH, "bedTopography"), glt.rd(glt.MESH, "thickness")
    tree = cKDTree(np.column_stack([x, y]))

    def sample(field, pts):
        d, idx = tree.query(pts, k=3)
        w = 1.0 / np.maximum(d, 1.0)
        w /= w.sum(axis=1, keepdims=True)
        return np.nansum(field[idx] * w, axis=1)

    sel = masks[:, names.index(a.shelf)] > 0
    home = np.bincount(np.argmax(basins[np.where(sel)[0]], axis=1),
                       minlength=basins.shape[1]).argmax()
    s, pts, _, _ = glt.build_transect(x, y, sel, bed, h0, basins[:, home] > 0)
    _, idx_all = tree.query(pts)
    in_basin = basins[idx_all, home] > 0
    if in_basin.any():
        j0, j1 = np.where(in_basin)[0][[0, -1]]
        s, pts = s[j0:j1 + 1], pts[j0:j1 + 1]

    b = sample(bed, pts)
    h = sample(h0, pts)
    hflot = (RHO_O / RHO_I) * np.maximum(0.0, -b)
    s_gl, _ = glt.gl_position(s, h, hflot)
    s = s - s_gl
    sk = s / 1e3

    floating = h < hflot - 1e-6
    base = np.where(floating, -(RHO_I / RHO_O) * h, b)
    surf = base + h
    ice = h > 1.0
    grounded = ice & ~floating
    base, surf = np.where(ice, base, np.nan), np.where(ice, surf, np.nan)

    xlo, xhi = max(sk.min(), -95.0), min(sk.max(), INLAND_KM)
    win = (sk >= xlo) & (sk <= xhi)

    fig = plt.figure(figsize=(15.4, 6.5))      # matches the deck's figure area
    ax = fig.add_axes([0.055, 0.375, 0.925, 0.545])
    axb = fig.add_axes([0.055, 0.088, 0.925, 0.185], sharex=ax)

    # ---------------------------------------------------------------- anatomy
    # water reaches sea level in the open ocean and the shelf base under the shelf;
    # inland of the grounding line there is no water at all
    otop = np.where(ice & floating, np.nan_to_num(base, nan=0.0), 0.0)
    ax.fill_between(sk, b, otop, where=(b < otop) & ~grounded, color=ds.ICE,
                    alpha=.40, linewidth=0, zorder=1)
    ax.fill_between(sk, b, b.min() - 900, color=BEDFILL, linewidth=0, zorder=2)
    ax.plot(sk, b, color=BED, lw=2.0, zorder=3)
    ax.axhline(0, color=ds.INK_SOFT, lw=.8, ls=(0, (4, 3)), zorder=3)

    # ice reads white against blue water -- the two must not share a tone
    ax.fill_between(sk, base, surf, where=grounded, color=ds.FIELD, zorder=4)
    ax.fill_between(sk, base, surf, where=floating & ice, color=ds.FIELD, zorder=4)
    ax.plot(sk, surf, color=ds.INK, lw=2.2, zorder=5)
    ax.plot(sk, base, color=ds.INK, lw=2.2, zorder=5)
    ax.plot([0], [np.interp(0.0, sk, b)], "v", ms=13, color=ds.MARSH_DEEP, zorder=8,
            clip_on=False)
    ax.axvline(0, color=ds.MARSH_DEEP, lw=1.1, alpha=.55, zorder=3)

    lo = float(np.nanmin(b[win]))
    hi = float(np.nanmax(surf[win]))
    ax.set_ylim(lo - 240, hi + 330)
    ax.set_xlim(xlo, xhi)
    span = (hi + 330) - (lo - 240)          # arrows sized off the axis, not the data

    # warm water in along the cavity floor, melt out of the shelf base
    cav = win & floating & ice & (sk < -6)
    if cav.any():
        xs = sk[cav]
        ybed = np.interp(xs, sk, b)
        ax.annotate("", xy=(xs.max() - 1, ybed[-1] + .035 * span),
                    xytext=(xs.min() + 3, ybed[0] + .035 * span),
                    arrowprops=dict(arrowstyle="-|>,head_width=.28,head_length=.6",
                                    color=ds.MARSH, lw=2.6,
                                    connectionstyle="arc3,rad=0.05"), zorder=7)
        for xm in np.linspace(xs.min() + 14, xs.max() - 5, 5):
            yb = float(np.interp(xm, sk, np.nan_to_num(base, nan=0.0)))
            ax.annotate("", xy=(xm, yb - .012 * span), xytext=(xm, yb - .10 * span),
                        arrowprops=dict(arrowstyle="-|>,head_width=.22,head_length=.5",
                                        color=ds.MARSH, lw=1.8, alpha=.9), zorder=7)
        # the cavity is too thin to letter -- the note goes in the air above the shelf,
        # where orange is used nowhere else
        ax.text(xlo + 3, hi - .10 * span,
                "warm ocean water enters the cavity\nand melts the shelf from below",
                fontsize=11.5, color=ds.MARSH_DEEP, ha="left", va="top",
                linespacing=1.45, zorder=8)

    # ice flows seaward out of the grounded interior
    inland = win & grounded & (sk > 40)
    if inland.any():
        xi = float(np.percentile(sk[inland], 72))
        yi = float(np.interp(xi, sk, surf)) - .085 * span
        ax.annotate("", xy=(xi - 46, yi), xytext=(xi, yi),
                    arrowprops=dict(arrowstyle="-|>,head_width=.28,head_length=.6",
                                    color=ds.ICE, lw=2.6), zorder=7)
        ax.text(xi + 5, yi, "ice flows seaward", fontsize=11, color=ds.ICE,
                ha="left", va="center", zorder=8)

    lab = dict(fontsize=12.5, color=ds.INK, ha="center", va="bottom", zorder=9)
    if inland.any():
        xg = float(np.percentile(sk[inland], 50))
        ax.text(xg, float(np.interp(xg, sk, surf)) + .02 * span,
                "grounded ice sheet", **lab)
    if cav.any():
        ax.text(float(np.mean(xs)), float(np.nanmax(surf[cav])) + .02 * span,
                "floating ice shelf", **lab)
    ax.text(0, hi + .075 * span, "grounding line", fontsize=12.5, color=ds.MARSH_DEEP,
            ha="center", va="bottom", zorder=9)
    ax.text(0, hi + .022 * span, "where ice stops resting on the bed and starts to float",
            fontsize=10, color=ds.INK_SOFT, ha="center", va="bottom", zorder=9)

    ds.strip(ax, keep=("left",))
    ax.set_ylabel("elevation  (m)", labelpad=6)
    ax.tick_params(labelbottom=False, length=3)

    # ---------------------------------------------------------------- bed slope inland
    # retrograde = the bed drops as you move inland, so a retreating grounding line
    # finds deeper water and thicker ice, which drives more flux out
    dbds = np.gradient(b, s)
    ahead = win & (sk >= 0)
    retro = ahead & (dbds < 0)
    frac = 100.0 * retro[ahead].sum() / max(ahead.sum(), 1)

    axb.axhline(0, color=ds.INK_SOFT, lw=.8, zorder=2)
    axb.fill_between(sk, 0, dbds * 1e3, where=(dbds < 0) & ahead, color=ds.MARSH,
                     alpha=.45, linewidth=0, zorder=3)
    axb.fill_between(sk, 0, dbds * 1e3, where=(dbds >= 0) & ahead, color=ds.ICE,
                     alpha=.30, linewidth=0, zorder=3)
    # seaward of the grounding line the bed slope says nothing about retreat
    axb.plot(sk, np.where(sk >= 0, dbds * 1e3, np.nan), color=ds.INK, lw=1.1, zorder=4)
    axb.plot(sk, np.where(sk < 0, dbds * 1e3, np.nan), color=ds.RULE, lw=1.0, zorder=4)
    axb.axvline(0, color=ds.MARSH_DEEP, lw=1.1, alpha=.55, zorder=3)
    q = float(np.nanpercentile(np.abs(dbds[win] * 1e3), 98))
    axb.set_ylim(-q, q)
    ds.strip(axb)
    axb.set_xlabel("distance from the present grounding line, seaward → inland  (km)",
                   labelpad=7)
    axb.set_ylabel("bed slope\n(m per km)", labelpad=6, fontsize=10)
    axb.tick_params(length=3)
    axb.text(0.995, 0.90, f"orange: the bed deepens inland — {frac:.0f}% of the "
             f"{INLAND_KM:.0f} km ahead of the grounding line",
             transform=axb.transAxes, fontsize=10, color=ds.MARSH_DEEP,
             ha="right", va="top")

    ax.text(0.004, 1.045, f"{a.shelf.replace('_', ' ')} · present-day section",
            transform=ax.transAxes, fontsize=15, color=ds.INK, ha="left", va="bottom")

    out = f"{a.outdir}/fig_ice_ocean_system.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  grounding line at bed {np.interp(0.0, sk, b):.0f} m")
    print(f"  bed deepens inland over {frac:.0f}% of the next {INLAND_KM:.0f} km")
    print(f"  bed at +{INLAND_KM:.0f} km: {np.interp(INLAND_KM, sk, b):.0f} m")


if __name__ == "__main__":
    main()
