#!/usr/bin/env python3
"""
fig_gl_backdrop_options.py -- what to put under the grounding lines, side by side.

One region, one year, the same member lines drawn over six different fields, so the choice
can be made by looking rather than by argument. Colour limits are taken over the whole
record, not the frame on show, which is what an animation needs anyway: limits that move
frame to frame make a still field look like it is changing.

The six:

  bed             does not change, and is what decides where the line can go
  thickness       the state, but most of the range sits in the interior and the margin
                  where the line actually is covers a small part of the scale
  change          thickness now minus thickness at the start, so it accumulates
  dhdt            the rate, which is noisier but shows where things are happening now
  flotation       the field being contoured, so the line is its zero
  spread          standard deviation across members, which is what the figure is about
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import oceancolors as oc         # noqa: E402
import fig_gl_transect as glt    # noqa: E402
from anim_gl_envelope import load, window, STATE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default="SSP585_varScaled10x")
    ap.add_argument("--region", default="Thwaites")
    ap.add_argument("--window-km", type=float, default=600.0)
    ap.add_argument("--year", type=int, default=2275)
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=os.path.join(
        glt.ROOT, "reports/dissertation/figures/slides"))
    a = ap.parse_args()
    out = a.out or f"{a.outdir}/fig_gl_backdrop_options.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    import netCDF4
    mesh = os.path.join(glt.ROOT, "data/MALI", os.path.basename(glt.MESH))
    d = netCDF4.Dataset(mesh); cov = np.asarray(d["cellsOnVertex"][:]) - 1; d.close()
    x, y = glt.rd(mesh, "xCell"), glt.rd(mesh, "yCell")
    bed_all = glt.rd(mesh, "bedTopography")

    names, masks = glt.region_names(glt.SHELF_MASK)
    m = np.asarray(np.asarray(masks)[:, names.index(a.region)], bool)
    cx, cy = x[m].mean() / 1e3, y[m].mean() / 1e3
    h = a.window_km / 2
    box = (cx - h, cx + h, cy - h, cy + h)

    z = load(a.ensemble)
    T, inbox, sub = window(x, y, cov, z["cells"], box)
    thk = z["thk"][:, :, inbox]                      # (member, year, cell)
    bed = bed_all[sub]
    yrs = np.asarray(z["years"])
    k = int(np.argmin(np.abs(yrs - a.year)))
    dt = float(np.median(np.diff(yrs)))
    print(f"  {a.ensemble}: {thk.shape[0]} members, year {yrs[k]-2000:.0f}, "
          f"{sub.size} cells in a {a.window_km:.0f} km box on {a.region}")

    flot = thk - (glt.RHO_O / glt.RHO_I) * np.maximum(0.0, -bed)[None, None, :]
    ice = thk > 1.0
    mean_h = np.where(ice, thk, np.nan).mean(axis=0)             # (year, cell)
    change = mean_h - mean_h[0]
    dhdt = np.gradient(mean_h, dt, axis=0)
    spread = np.where(ice, thk, np.nan).std(axis=0, ddof=1)

    # limits over the whole record, so a frame never rescales
    def lim(arr, pct=99.0):
        v = arr[np.isfinite(arr)]
        return float(np.percentile(np.abs(v), pct))

    panels = [
        ("bed elevation  (m)", np.where(np.isfinite(bed), bed, np.nan), "topography",
         TwoSlopeNorm(vmin=-1800, vcenter=0, vmax=1800)),
        ("ice thickness  (m)", mean_h[k], "thickness",
         Normalize(0, float(np.nanpercentile(mean_h, 99)))),
        ("thickness change since year 0  (m)", change[k], "tendency",
         TwoSlopeNorm(vmin=-lim(change), vcenter=0, vmax=lim(change))),
        ("dh/dt  (m yr$^{-1}$)", dhdt[k], "tendency",
         TwoSlopeNorm(vmin=-lim(dhdt, 98), vcenter=0, vmax=lim(dhdt, 98))),
        ("flotation height  (m)", np.where(ice.any(axis=0)[k], flot.mean(axis=0)[k], np.nan),
         "melt", TwoSlopeNorm(vmin=-600, vcenter=0, vmax=600)),
        ("spread across members, sigma h  (m)", spread[k], "magnitude",
         Normalize(0, float(np.nanpercentile(spread, 99)))),
    ]

    cols, rows = 3, 2
    fig = plt.figure(figsize=(6.2 * cols, 6.0 * rows))
    for i, (lab, field, role, norm) in enumerate(panels):
        ax = fig.add_axes([(i % cols) / cols + 0.010, 1 - (i // cols + 1) / rows + 0.055,
                           1 / cols - 0.020, 1 / rows - 0.105])
        tp = ax.tripcolor(T, field, cmap=oc.cmap(role, "cmocean"), norm=norm,
                          shading="gouraud", rasterized=True, zorder=1)
        ax.tricontour(T, flot[0, 0], levels=[0.0], colors=[ds.INK_SOFT],
                      linewidths=1.2, linestyles="dotted", zorder=3)
        for mi in range(thk.shape[0]):
            ax.tricontour(T, flot[mi, k], levels=[0.0], colors=[ds.INK],
                          linewidths=0.8, alpha=.55, zorder=5)
        cb = fig.colorbar(tp, ax=ax, fraction=0.036, pad=0.012)
        cb.ax.tick_params(labelsize=10); cb.outline.set_visible(False)
        ax.set_xlim(box[0] * 1e3, box[1] * 1e3); ax.set_ylim(box[2] * 1e3, box[3] * 1e3)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.02, 1.015, lab, transform=ax.transAxes, fontsize=13,
                color=ds.INK, ha="left", va="bottom")

    fig.text(0.01, 0.995, f"{a.region.replace('_',' ')} · {a.ensemble} · "
             f"model year {yrs[k]-2000:.0f} · every member's line drawn · "
             f"colour limits fixed over the whole record",
             fontsize=13, color=ds.INK_SOFT, ha="left", va="top")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.12, dpi=175)
    plt.close(fig)
    print(f"wrote {out}")
    for lab, field, _, _ in panels:
        v = field[np.isfinite(field)]
        print(f"  {lab:38s} {v.min():10.4g} .. {v.max():10.4g}")


if __name__ == "__main__":
    main()
