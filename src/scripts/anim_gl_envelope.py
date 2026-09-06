#!/usr/bin/env python3
"""
anim_gl_envelope.py -- every member's grounding line, 1x beside 10x, over three centuries.

The slide says the scenario decides how far the grounding line goes and the realisation
only decides when. That is currently a number, an interquartile range in kilometres. This
is the same result as a picture: draw all members in a frame and the 1x lines sit on top of
each other while the 10x lines come apart and close again.

The line is the zero contour of the flotation height rather than a contour of the grounded
mask, so it crosses cells instead of following their edges. cellMask is carried through the
extract and used to check it, printed at the start. The two do not sit on top of each other
and should not: MALI flags the last grounded cell, so its cells are inland of the crossing
by a median of about 50 m of flotation height, which at this mesh is roughly half a cell.
What matters is that the offset stays one-sided and small, not that it is zero.

Frames come from the state extract, so run the HPC job first. Members without a complete
record are dropped rather than drawn as gaps.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import oceancolors as oc         # noqa: E402
import fig_gl_transect as glt    # noqa: E402

STATE = f"{glt.ROOT}/reports/dissertation/figures/spatial/members_state/anim"
GROUNDING_LINE = 256          # MPAS-Albany cellMask bit value
# label and colour per ensemble; anything not listed falls back to the ink colour
SHOWN = {"SSP585": ("SSP5-8.5, 1x", ds.ICE),
         "SSP585_varScaled10x": ("SSP5-8.5, 10x", ds.MARSH),
         "CTRL": ("control", ds.INK),
         "SSP126": ("SSP1-2.6", ds.MARSH_DEEP),
         "SSP585-3X": ("SSP5-8.5, 3x trend", ds.MARSH_TINT)}
WINDOW_KM = (-1750, -1150, -750, -150)   # Amundsen, where the mesh is finest


def load(ens):
    z = np.load(f"{STATE}/member_thickness_{ens}.npz", allow_pickle=True)
    keep = (z["year_got"] >= 0).all(axis=1)
    return dict(cells=z["cells"], years=np.asarray(z["years"]),
                thk=z["thickness"][keep], mask=z["cellMask"][keep],
                members=z["members"][keep])


def window(x, y, cov, cells, box):
    """Triangulation of just the cells inside the box, with indices remapped."""
    x0, x1, y0, y1 = [v * 1e3 for v in box]
    inbox = (x[cells] >= x0) & (x[cells] <= x1) & (y[cells] >= y0) & (y[cells] <= y1)
    sub = cells[inbox]
    lut = -np.ones(x.size, np.int64)
    lut[sub] = np.arange(sub.size)
    tri = cov[(cov >= 0).all(axis=1)]
    tri = tri[(lut[tri] >= 0).all(axis=1)]
    return mtri.Triangulation(x[sub], y[sub], lut[tri]), inbox, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensembles", nargs="+",
                    default=["SSP585", "SSP585_varScaled10x"])
    ap.add_argument("--region", default=None,
                    help="centre the window on this shelf instead of using --box")
    ap.add_argument("--window-km", type=float, default=600.0)
    ap.add_argument("--box", type=float, nargs=4, default=WINDOW_KM,
                    metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--suffix", default="")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=f"{glt.ROOT}/reports/dissertation/figures/slides")
    a = ap.parse_args()
    tag = a.suffix or (f"_{a.region}" if a.region else "")
    out = a.out or f"{a.outdir}/anim_gl_envelope{tag}.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    import netCDF4
    mesh = os.path.join(glt.ROOT, "data/MALI", os.path.basename(glt.MESH))
    d = netCDF4.Dataset(mesh); cov = np.asarray(d["cellsOnVertex"][:]) - 1; d.close()
    x, y = glt.rd(mesh, "xCell"), glt.rd(mesh, "yCell")
    bed = glt.rd(mesh, "bedTopography")

    if a.region:
        names, masks = glt.region_names(glt.SHELF_MASK)
        if a.region not in names:
            sys.exit(f"no region {a.region}")
        m = np.asarray(np.asarray(masks)[:, names.index(a.region)], bool)
        cx, cy = x[m].mean() / 1e3, y[m].mean() / 1e3
        h = a.window_km / 2
        a.box = (cx - h, cx + h, cy - h, cy + h)
        print(f"  {a.region}: window {a.box[0]:.0f}..{a.box[1]:.0f} km, "
              f"{a.box[2]:.0f}..{a.box[3]:.0f} km")

    runs = []
    for ens in a.ensembles:
        lab, col = SHOWN.get(ens, (ens, ds.INK))
        z = load(ens)
        T, inbox, sub = window(x, y, cov, z["cells"], a.box)
        thk = z["thk"][:, :, inbox]
        msk = z["mask"][:, :, inbox]
        flot = thk - (glt.RHO_O / glt.RHO_I) * np.maximum(0.0, -bed[sub])[None, None, :]
        runs.append(dict(lab=lab, col=col, ens=ens, T=T, flot=flot, thk=thk, mask=msk,
                         years=z["years"], n=len(z["members"]), bed=bed[sub]))
        print(f"  {ens:22s} {len(z['members'])} complete members, "
              f"{sub.size} cells in the window")

    # does the smooth contour agree with the model's own flag?
    r = runs[0]
    gl_cells = (r["mask"][0, -1] & GROUNDING_LINE) == GROUNDING_LINE
    if gl_cells.any():
        v = r["flot"][0, -1][gl_cells]
        print(f"  check against cellMask: flagged cells sit {np.median(v):+.0f} m above "
              f"flotation (median), {100*np.mean(np.abs(v) < 200):.0f}% within 200 m")

    yrs = runs[0]["years"]
    np_ = len(runs)
    fig = plt.figure(figsize=(6.9 * np_, 7.4))
    axes = []
    for i, r in enumerate(runs):
        ax = fig.add_axes([0.02 + i / np_, 0.055, 1 / np_ - 0.025, 0.845])
        ax.tripcolor(r["T"], np.where(r["thk"][0, 0] > 1, r["thk"][0, 0], np.nan),
                     cmap=oc.cmap("thickness", "cmocean"), shading="gouraud",
                     rasterized=True, alpha=.45, zorder=1)
        # the starting line, kept faint so retreat reads as displacement
        ax.tricontour(r["T"], r["flot"][0, 0], levels=[0.0], colors=[ds.INK_SOFT],
                      linewidths=1.4, linestyles="dotted", zorder=3)
        ax.set_xlim(a.box[0] * 1e3, a.box[1] * 1e3)
        ax.set_ylim(a.box[2] * 1e3, a.box[3] * 1e3)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.03, 0.97, f"{r['lab']}  ·  {r['n']} members", transform=ax.transAxes,
                fontsize=14, color=ds.INK, ha="left", va="top", zorder=9,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="none", alpha=.8))
        axes.append(ax)
    clock = fig.text(0.5, 0.965, "", fontsize=16, color=ds.INK, ha="center", va="top")
    fig.text(0.5, 0.020, "dotted line is where the grounding line started",
             fontsize=11.5, color=ds.INK_SOFT, ha="center", va="bottom")

    drawn = [[] for _ in runs]

    def update(k):
        for ax, r, keep in zip(axes, runs, drawn):
            for c in keep:
                try:
                    c.remove()
                except Exception:
                    pass
            keep.clear()
            for m in range(r["flot"].shape[0]):
                cs = ax.tricontour(r["T"], r["flot"][m, k], levels=[0.0],
                                   colors=[r["col"]], linewidths=0.9, alpha=.55, zorder=5)
                keep.append(cs)
        clock.set_text(f"model year {yrs[k]-2000:.0f}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(yrs),
                                   interval=1000 / a.fps, blit=False)
    anim.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=3600),
              savefig_kwargs=dict(facecolor=ds.PAPER))
    plt.close(fig)
    print(f"wrote {out}   {len(yrs)} frames")


if __name__ == "__main__":
    main()
