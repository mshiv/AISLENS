#!/usr/bin/env python3
"""
fig_effect_hierarchy.py -- every effect Chapter 3 measures, ranked on one log axis.

Scenario change, 3x melt trend, gross regional reorganisation, net continental drift, and
ensemble sigma at 1x and 10x. Position carries the value; the top axis reads the same
positions as a share of the scenario effect. Open markers are measured at 10x amplitude.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# label, value (mm SLE), family, footnote, measured-at-10x
# the last flag marks quantities measured under the exaggerated 10x forcing
BARS = [
    ("changing the emissions scenario", 1619.1, "forced",
     "SSP5-8.5 − SSP1-2.6 ensemble mean · year 300", False),
    ("gross regional reorganisation\nfrom ocean variability", 177.0, "var",
     "sum of |basin drift| · year 300", True),
    ("tripling the melt trend", 349.8, "forced",
     "SSP585-3X − SSP585 · year 200 · N = 8", False),
    ("what that reorganisation leaves\nin the continental total", 12.1, "var",
     "net drift · year 300", True),
    ("spread across realisations, 10×", 6.6, "var",
     "σ of SSP585_varScaled10x · year 300 · N = 15", True),
    ("spread across realisations, 1×", 2.1, "var",
     "σ of SSP585 · year 300 · N = 10", False),
]
REF = 1619.1        # the scenario effect, for the "share of" axis



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/reports/dissertation/figures/slides/"
                                     "fig_effect_hierarchy.png")
    ap.add_argument("--standalone", action="store_true",
                    help="draw the figure's own headline; off for deck use, "
                         "where the slide title is the caption")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ds.apply()

    bars = sorted(BARS, key=lambda b: b[1])
    ypos = np.arange(len(bars))
    vals = np.array([b[1] for b in bars])
    cols = [ds.ICE if b[2] == "forced" else ds.MARSH for b in bars]

    ax_lo = 1.0
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    fig.subplots_adjust(left=0.285, right=0.965, top=0.715, bottom=0.145)

    # dots, not bars: on a log axis bar length depends on where the axis starts
    for y, v, c, b in zip(ypos, vals, cols, bars):
        ax.plot([ax_lo, v], [y, y], color=ds.RULE, lw=0.8, zorder=2)
        if b[4]:
            ax.plot([v], [y], "o", ms=11, mfc="white", mec=c, mew=2.2, zorder=4)
        else:
            ax.plot([v], [y], "o", ms=11, color=c, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(ax_lo, 4200)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("effect on Antarctic sea-level contribution  (mm)", labelpad=8)
    ax.tick_params(axis="x", length=3)

    # the same positions read as a share of the scenario effect
    sec = ax.secondary_xaxis("top", functions=(lambda x: 100 * x / REF,
                                               lambda x: x * REF / 100))
    sec.set_xscale("log")
    sec.set_xticks([0.1, 1, 10, 100])
    sec.set_xticklabels(["0.1%", "1%", "10%", "100%"])
    sec.minorticks_off()
    sec.tick_params(length=3, labelsize=10, colors=ds.INK_SOFT)
    sec.set_xlabel("share of the scenario effect", labelpad=7, fontsize=10,
                   color=ds.INK_SOFT)

    ax.set_yticks(ypos)
    ax.set_yticklabels([b[0] for b in bars], fontsize=11.5, linespacing=1.35)
    ax.tick_params(axis="y", length=0)
    ds.strip(ax, keep=("bottom",))
    ax.set_ylim(-0.7, len(bars) - 0.25)

    for y, b in zip(ypos, bars):
        v = b[1]
        lab = f"{v:,.0f}" if v >= 100 else f"{v:,.1f}"
        ax.text(v * 1.22, y + 0.10, lab,
                va="center", ha="left", fontsize=12, color=ds.INK, zorder=5)
        ax.text(v * 1.22, y - 0.24, b[3], va="center", ha="left",
                fontsize=8.8, color=ds.INK_SOFT, zorder=5)

    # legend as coloured words, not a box
    ax.text(0.0, 1.255, "prescribed forcing", transform=ax.transAxes, fontsize=11.5,
            color=ds.ICE, ha="left", va="bottom")
    ax.text(0.185, 1.255, "ocean variability", transform=ax.transAxes, fontsize=11.5,
            color=ds.MARSH, ha="left", va="bottom")
    ax.plot([0.375], [1.283], "o", ms=9, mfc="white", mec=ds.INK_SOFT, mew=2.0,
            transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0.392, 1.255, "open = measured at 10× amplitude, a diagnostic rather "
            "than a scenario", transform=ax.transAxes, fontsize=10,
            color=ds.INK_SOFT, ha="left", va="bottom")

    if a.standalone:
        ax.text(0.0, 1.365, "what actually moves Antarctic sea-level contribution",
                transform=ax.transAxes, fontsize=15.5, color=ds.INK,
                ha="left", va="bottom")

    fig.text(0.008, 0.008,
             "logarithmic axis, so position carries the value and spacing carries ratios · "
             "the 3× melt term is at year 200, where eight of its members remain",
             fontsize=8.8, color=ds.INK_SOFT, ha="left", va="bottom", style="italic")

    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.14)
    print(f"wrote {a.out}")
    for b in reversed(bars):
        print(f"  {b[1]:8.1f} mm   {b[0].splitlines()[0]}")


if __name__ == "__main__":
    main()
