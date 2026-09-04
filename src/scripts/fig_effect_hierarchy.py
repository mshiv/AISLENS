#!/usr/bin/env python3
"""
fig_effect_hierarchy.py — every effect Chapter 3 measures, ranked on one log axis in mm SLE.

Scenario change, 3x melt trend, gross regional reorganisation, net continental drift, and
ensemble sigma at 1x and 10x. The ordering is the result: gross reorganisation is the
second-largest term, and almost none of it survives the continental sum.
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

# label, value (mm SLE), family, footnote
BARS = [
    ("changing the emissions scenario", 1619.1, "forced",
     "SSP5-8.5 − SSP1-2.6 ensemble mean · year 300"),
    ("gross regional reorganisation\nfrom ocean variability", 177.0, "var",
     "sum of |basin drift| under 10× variability · year 300"),
    ("tripling the melt trend", 292.9, "forced",
     "SSP585-3X − SSP585 · year 178, the deepest common horizon"),
    ("what that reorganisation leaves\nin the continental total", 12.1, "var",
     "net drift under 10× variability · year 300"),
    ("spread across realisations, 10×", 6.6, "var",
     "σ of SSP585_varScaled10x · year 300 · N = 15"),
    ("spread across realisations, 1×", 2.1, "var",
     "σ of SSP585 · year 300 · N = 10"),
]



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

    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    fig.subplots_adjust(left=0.285, right=0.965, top=0.79, bottom=0.145)

    ax.barh(ypos, vals, height=0.58, color=cols, linewidth=0, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 4200)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("effect on Antarctic sea-level contribution  (mm)", labelpad=8)
    ax.tick_params(axis="x", length=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels([b[0] for b in bars], fontsize=11.5, linespacing=1.35)
    ax.tick_params(axis="y", length=0)
    ds.strip(ax, keep=("bottom",))
    ax.set_ylim(-0.7, len(bars) - 0.25)

    for x in (10, 100, 1000):
        ax.axvline(x, color=ds.RULE, lw=0.7, zorder=0)

    for y, b in zip(ypos, bars):
        v = b[1]
        lab = f"{v:,.0f}" if v >= 100 else f"{v:,.1f}"
        ax.text(v * 1.13, y + 0.10, lab,
                va="center", ha="left", fontsize=12, color=ds.INK, zorder=4)
        ax.text(v * 1.13, y - 0.24, b[3], va="center", ha="left",
                fontsize=8.8, color=ds.INK_SOFT, zorder=4)

    # legend as two coloured words, not a box
    ax.text(0.0, 1.055, "prescribed forcing", transform=ax.transAxes, fontsize=11.5,
            color=ds.ICE, ha="left", va="bottom")
    ax.text(0.185, 1.055, "ocean variability", transform=ax.transAxes, fontsize=11.5,
            color=ds.MARSH, ha="left", va="bottom")

    if a.standalone:
        ax.text(0.0, 1.135, "what actually moves Antarctic sea-level contribution",
                transform=ax.transAxes, fontsize=15.5, color=ds.INK,
                ha="left", va="bottom")

    fig.text(0.008, 0.008,
             "logarithmic axis · variability terms are measured at the exaggerated 10× amplitude, "
             "so their ordering is the result and their magnitude is not a claim about realistic "
             "variability",
             fontsize=8.8, color=ds.INK_SOFT, ha="left", va="bottom", style="italic")

    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.14)
    print(f"wrote {a.out}")
    for b in reversed(bars):
        print(f"  {b[1]:8.1f} mm   {b[0].splitlines()[0]}")


if __name__ == "__main__":
    main()
