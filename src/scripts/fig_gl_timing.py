#!/usr/bin/env python3
"""
fig_gl_timing.py — when each realisation's grounding line passes a given point.

Five-yearly output could not answer this: it quantises the retreat into steps, so realisation
disagreement showed up as tens of kilometres of position at a shared year and could not be
separated from the sampling. Annual fields turn the same disagreement into what it physically
is, a difference in timing.

Left: annual trajectories through the retreat. Right: the year each member crosses a set of
distances, one dot per member.

Needs the annual extract in reports/dissertation/figures/spatial/members_annual/.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds        # noqa: E402
import fig_gl_members as M     # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PAIR = [("SSP585", "SSP5-8.5, 1×", ds.ICE),
        ("SSP585_varScaled10x", "SSP5-8.5, 10×", ds.MARSH)]
THRESHOLDS = [100, 140, 170, 200, 240]


def crossings(gl, years, thr):
    """Year each member first reaches thr, NaN if it never does."""
    out = np.full(gl.shape[0], np.nan)
    for m in range(gl.shape[0]):
        k = np.flatnonzero(np.isfinite(gl[m]) & (gl[m] >= thr))
        if k.size:
            out[m] = years[k[0]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shelf", default="Thwaites")
    ap.add_argument("--members", default=f"{ROOT}/reports/dissertation/figures/"
                                         "spatial/members_annual")
    ap.add_argument("--xlim", nargs=2, type=float, default=[170, 250])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"{ROOT}/reports/dissertation/figures/slides/fig_gl_timing_{a.shelf}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    g = M.geometry(a.shelf)
    fig = plt.figure(figsize=(15.0, 6.2))
    axl = fig.add_axes([0.055, 0.130, 0.455, 0.770])
    axr = fig.add_axes([0.605, 0.130, 0.320, 0.770])

    rows = []
    for ens, label, colr in PAIR:
        d = M.series(ens, g, a.members)
        if d is None:
            print(f"  ! no annual extract for {ens}"); continue
        gl = d["gl"][d["complete"]]
        yrs = d["years"] - 2000
        for m in range(gl.shape[0]):
            axl.plot(yrs, gl[m], color=colr, lw=0.9, alpha=.55, zorder=3)
        rows.append((ens, label, colr, gl, yrs))

    ds.strip(axl)
    axl.set_xlim(*a.xlim)
    axr_pad = True
    axl.set_xlabel("model year", labelpad=7)
    axl.set_ylabel("grounding-line retreat  (km inland of year 0)", labelpad=7)
    axl.tick_params(length=3)
    axl.text(0.0, 1.045, "annual fields, every realisation drawn",
             transform=axl.transAxes, fontsize=11, color=ds.INK_SOFT,
             ha="left", va="bottom")
    for k, (_, label, colr, _, _) in enumerate(rows):
        axl.text(0.02, 0.94 - 0.065 * k, label, transform=axl.transAxes,
                 fontsize=11.5, color=colr, ha="left", va="top")

    # ---- crossing years, one dot per member
    ypos = np.arange(len(THRESHOLDS))
    rng = np.random.default_rng(0)
    stats = []
    for k, thr in enumerate(THRESHOLDS):
        axr.axhline(k, color=ds.RULE, lw=.7, zorder=1)
        for j, (_, label, colr, gl, yrs) in enumerate(rows):
            c = crossings(gl, yrs, thr)
            c = c[np.isfinite(c)]
            if c.size < 3:
                continue
            off = 0.17 * (1 if j else -1)
            axr.plot(c + rng.uniform(-.12, .12, c.size), np.full(c.size, k + off), "o",
                     ms=5.5, color=colr, alpha=.75, mew=0, zorder=3)
            axr.plot([c.min(), c.max()], [k + off, k + off], color=colr, lw=1.4,
                     alpha=.45, zorder=2)
            # every 1x member crosses in the same year, so the dots coincide and the
            # cluster reads as one member unless the count is stated
            axr.text(c.max() + 1.4, k + off,
                     f"N={c.size}, {'same year' if c.max() == c.min() else f'{c.max()-c.min():.0f} yr'}",
                     fontsize=9, color=colr, va="center", ha="left", zorder=4)
            stats.append((thr, label, c.size, np.median(c), c.max() - c.min(),
                          np.percentile(c, 75) - np.percentile(c, 25)))

    ds.strip(axr, keep=("bottom",))
    axr.set_yticks(ypos)
    axr.set_yticklabels([f"{t} km" for t in THRESHOLDS], fontsize=11)
    axr.tick_params(axis="y", length=0)
    axr.tick_params(axis="x", length=3)
    axr.set_ylim(-0.6, len(THRESHOLDS) - 0.4)
    axr.set_xlim(axr.get_xlim()[0], axr.get_xlim()[1] + 11)
    axr.set_xlabel("year the grounding line reaches that distance", labelpad=7)
    axr.text(0.0, 1.045, "one dot per realisation, jittered vertically",
             transform=axr.transAxes, fontsize=11, color=ds.INK_SOFT,
             ha="left", va="bottom")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote {out}")
    for thr, label, n, med, sp, iqr in stats:
        print(f"  {thr:3d} km  {label:14s} N={n:2d}  median yr {med:6.1f}  "
              f"full spread {sp:4.0f} yr  IQR {iqr:4.1f} yr")


if __name__ == "__main__":
    main()
