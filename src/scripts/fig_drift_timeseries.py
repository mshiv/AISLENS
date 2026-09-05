#!/usr/bin/env python3
"""
fig_drift_timeseries.py — D(t) = mean_10x(t) - mean_1x(t) with a +/-1 SE ribbon.

The two ensembles share trend, generator, initial state and numerics and differ only in
variability amplitude, so their difference in ensemble mean is a rectification signal that
needs no matched deterministic twin. Reproduces the frozen results table section 3b.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds        # noqa: E402
import ensemble_io as eio         # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ENS = f"{ROOT}/data/MALI/diagnostics/ENSEMBLES"

ANNOTATE = [100, 300]   # model years called out on the slide


def ensemble_sle(name: str):
    """Return (year grid, members x year array of SLE in mm, relative to year 0)."""
    ds_ = eio.load_ensemble_globalstats(os.path.join(ENS, name))
    vaf = ds_["volumeAboveFloatation"]                    # (member, year)
    sle = eio.vaf_to_sle_mm(np.asarray(vaf.values, dtype=float), reference="first")
    return np.asarray(ds_["year"].values, dtype=float), sle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/reports/dissertation/figures/slides/"
                                     "fig_drift_timeseries.png")
    ap.add_argument("--standalone", action="store_true",
                    help="draw the figure's own headline; off for deck use, "
                         "where the slide title is the caption")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ds.apply()

    y1, s1 = ensemble_sle("SSP585")
    y10, s10 = ensemble_sle("SSP585_varScaled10x")

    # common year grid
    hi = min(np.nanmax(y1), np.nanmax(y10))
    grid = np.arange(0.0, np.floor(hi) + 0.5, 1.0)

    def regrid(y, s):
        out = np.full((s.shape[0], grid.size), np.nan)
        for i in range(s.shape[0]):
            ok = np.isfinite(y) & np.isfinite(s[i])
            if ok.sum() > 10:
                out[i] = np.interp(grid, y[ok], s[i][ok], left=np.nan, right=np.nan)
        return out

    a1, a10 = regrid(y1, s1), regrid(y10, s10)
    n1 = np.sum(np.isfinite(a1), axis=0)
    n10 = np.sum(np.isfinite(a10), axis=0)
    keep = (n1 >= 8) & (n10 >= 10)

    m1 = np.nanmean(a1, axis=0)
    m10 = np.nanmean(a10, axis=0)
    sd1 = np.nanstd(a1, axis=0, ddof=1)
    sd10 = np.nanstd(a10, axis=0, ddof=1)

    D = m10 - m1
    SE = np.sqrt(sd1**2 / np.maximum(n1, 1) + sd10**2 / np.maximum(n10, 1))
    g, D, SE = grid[keep], D[keep], SE[keep]

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    fig.subplots_adjust(left=0.075, right=0.925, top=0.86, bottom=0.17)

    ax.axhline(0, color=ds.RULE, lw=1.1, zorder=1)
    ax.fill_between(g, D - SE, D + SE, color=ds.MARSH_TINT, alpha=0.55,
                    linewidth=0, zorder=2)
    ax.plot(g, D, color=ds.MARSH_DEEP, lw=2.4, zorder=3, solid_capstyle="round")

    ds.strip(ax)
    ax.set_xlim(0, g.max())
    ax.set_xlabel("model year", labelpad=6)
    ax.set_ylabel("drift in the ensemble mean, 10× minus 1×  (mm SLE)", labelpad=8)
    ax.tick_params(length=3)

    ax.set_ylim(min(0, float(np.nanmin(D - SE))) - 1.0, float(np.nanmax(D + SE)) * 1.16)

    for yr, ha, dx in zip(ANNOTATE, ("left", "right"), (8, -8)):
        i = int(np.argmin(np.abs(g - yr)))
        ax.plot([g[i]], [D[i]], "o", ms=6.5, color=ds.MARSH_DEEP, zorder=4)
        ax.annotate(f"+{D[i]:.0f} mm\nyear {int(g[i])}",
                    xy=(g[i], D[i]), xytext=(dx, 14), textcoords="offset points",
                    fontsize=11.5, color=ds.INK, linespacing=1.35, ha=ha)

    peak = int(np.argmax(D))
    ax.annotate(f"peak  +{D[peak]:.0f}", xy=(g[peak], D[peak] + SE[peak]),
                xytext=(0, 9), textcoords="offset points",
                fontsize=10.5, color=ds.INK_SOFT, ha="center")

    if a.standalone:
        ax.text(0.0, 1.085, "more variability, more ice lost",
                transform=ax.transAxes, fontsize=17, color=ds.MARSH_DEEP,
                ha="left", va="bottom")
    ax.text(0.0, 1.015,
            "ensemble mean of the 10× ensemble minus the 1× ensemble,  ±1 standard error"
            "     ·     identical trend, generator, initial state and numerics",
            transform=ax.transAxes, fontsize=10, color=ds.INK_SOFT,
            ha="left", va="bottom")

    fig.text(0.008, -0.005,
             "10× amplitude experiment — the sign is the result, the magnitude is not "
             "a claim about realistic variability",
             fontsize=9, color=ds.INK_SOFT, ha="left", va="bottom", style="italic")

    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.14)
    print(f"wrote {a.out}")
    for yr in (25, 50, 75, 100, 150, 200, 250, 300):
        i = int(np.argmin(np.abs(g - yr)))
        if abs(g[i] - yr) < 1.5:
            print(f"  yr {yr:3d}   D {D[i]:+7.2f}   SE {SE[i]:5.3f}   "
                  f"D/SE {D[i]/SE[i]:+6.1f}")


if __name__ == "__main__":
    main()
