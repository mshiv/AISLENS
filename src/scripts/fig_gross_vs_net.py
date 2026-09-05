#!/usr/bin/env python3
"""
fig_gross_vs_net.py — opposing local changes and the residual that survives aggregation.

Two blocks, one grammar: the gross signed activity split into its opposing components,
normalised to its own total, with the reported net drawn beneath at the same scale.
Antarctic basins and one Pin Point floodwall. Units differ; the comparison is of shape.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

BLOCKS = [
    dict(
        title="Antarctica",
        sub="16 basins · drift under 10× variability, 10× minus 1× · model year 300",
        warm=94.56, warm_lab="more ice lost",
        cool=82.45, cool_lab="less ice lost",
        net=12.11, net_tail="all that a continental total reports",
        fmt="{:,.0f} mm",
        y_title=0.95, y_sub=0.885, y_gross=0.735, y_net=0.590,
    ),
    dict(
        title="Pin Point",
        sub="one floodwall · matched intervention minus control · Int2050 sea level",
        warm=27491.4, warm_lab="depth increased",
        cool=52529.4, cool_lab="depth reduced",
        net=52529.4 - 27491.4, net_tail="all that a headline benefit reports",
        fmt="{:,.0f} m³",
        y_title=0.400, y_sub=0.335, y_gross=0.185, y_net=0.040,
    ),
]

BAR_W = 0.74          # width of the bar field, in axes fraction
H_GROSS = 0.085
H_NET = 0.055


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/reports/dissertation/figures/slides/"
                                     "fig_gross_vs_net.png")
    ap.add_argument("--standalone", action="store_true",
                    help="draw the figure's own headline; off for deck use, "
                         "where the slide title is the caption")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ds.apply()

    fig = plt.figure(figsize=(12.2, 5.6))
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.80])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def bar(x0, y, w, h, color):
        ax.add_patch(Rectangle((x0, y - h / 2), w, h, facecolor=color,
                               edgecolor="none", zorder=2))

    for b in BLOCKS:
        gross = b["warm"] + b["cool"]
        fc, fw, fn = b["cool"] / gross, b["warm"] / gross, abs(b["net"]) / gross

        ax.text(0.0, b["y_title"], b["title"], fontsize=16, color=ds.INK,
                ha="left", va="center")
        ax.text(0.0, b["y_sub"], b["sub"], fontsize=10, color=ds.INK_SOFT,
                ha="left", va="center")

        # gross: the two opposing components, end to end
        bar(0.0, b["y_gross"], fc * BAR_W, H_GROSS, ds.ICE)
        bar(fc * BAR_W, b["y_gross"], fw * BAR_W, H_GROSS, ds.MARSH)
        ax.text(fc * BAR_W / 2, b["y_gross"],
                f"{b['cool_lab']}   {b['fmt'].format(b['cool'])}",
                fontsize=11, color=ds.PAPER, ha="center", va="center", zorder=3)
        ax.text((fc + fw / 2) * BAR_W, b["y_gross"],
                f"{b['warm_lab']}   {b['fmt'].format(b['warm'])}",
                fontsize=11, color=ds.PAPER, ha="center", va="center", zorder=3)
        ax.text(BAR_W + 0.016, b["y_gross"], "everything that\nactually happened",
                fontsize=10, color=ds.INK_SOFT, ha="left", va="center", linespacing=1.5)

        # net: what a single number reports, at the same scale
        bar(0.0, b["y_net"], fn * BAR_W, H_NET, ds.INK)
        ax.text(fn * BAR_W + 0.014, b["y_net"],
                f"{b['fmt'].format(abs(b['net']))}   ·   {100*fn:.0f}% of the gross change"
                f"   ·   {b['net_tail']}",
                fontsize=11, color=ds.INK, ha="left", va="center")

    if a.standalone:
        fig.text(0.03, 0.935,
                 "opposing local changes, and the residual that survives aggregation",
                 fontsize=15.5, color=ds.INK, ha="left", va="bottom")
    fig.text(0.03, 0.888,
             "each block is normalised to its own gross change — the two systems are compared "
             "in shape, not in units",
             fontsize=10, color=ds.INK_SOFT, ha="left", va="bottom")

    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.16)
    print(f"wrote {a.out}")
    for b in BLOCKS:
        gross = b["warm"] + b["cool"]
        print(f"  {b['title']:12s} gross {gross:12,.1f}  net {b['net']:12,.1f}  "
              f"surviving {100*abs(b['net'])/gross:5.1f}%")


if __name__ == "__main__":
    main()
