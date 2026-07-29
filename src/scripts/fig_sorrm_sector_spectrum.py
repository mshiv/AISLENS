#!/usr/bin/env python3
"""
fig_sorrm_sector_spectrum.py — SORRM internal ocean variability (F_v) spectral bands per ISMIP6 basin.

Horizontal stacked bars per basin ordered by low-frequency fraction. ALL-shelf summary bar.
Uses per-cell band fractions from spectrum_percell_generated{0,1}.csv.
"""
from __future__ import annotations
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BANDS = ["seasonal", "interannual", "decadal", "multidecadal"]
# high-freq -> low-freq, colorblind-safe (orange -> teal -> blue)
COLORS = {"seasonal": "#E69F00", "interannual": "#F0E442", "decadal": "#009E73", "multidecadal": "#0072B2"}
# the dynamically critical sectors (MISI-prone): Amundsen G-H, Filchner-Ronne J-K
KEY = {"ISMIP6 Basin G-H": "Amundsen", "ISMIP6 Basin J-K": "Filchner-Ronne"}


def main():
    d0 = pd.read_csv(os.path.join(REPO, "reports/spectrum_percell_generated0.csv"))
    d1 = pd.read_csv(os.path.join(REPO, "reports/spectrum_percell_generated1.csv"))
    df = d0.copy()
    for b in BANDS:                                  # average the two members
        df[b] = 0.5 * (d0[b].values + d1[b].values)
    df["lowfreq"] = df[["interannual", "decadal", "multidecadal"]].sum(axis=1)

    allshelf = df[df.sector == "ALL-shelf"]
    sect = df[df.sector != "ALL-shelf"].sort_values("seasonal")   # most low-freq at top
    order = pd.concat([sect, allshelf], ignore_index=True)

    labels = [s.replace("ISMIP6 Basin ", "") + (f"  ({KEY[s]})" if s in KEY else "") for s in order.sector]
    labels = [("ALL-SHELF" if s == "ALL-shelf" else l) for s, l in zip(order.sector, labels)]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10, 7.5))
    left = np.zeros(len(order))
    for b in BANDS:
        ax.barh(y, order[b], left=left, color=COLORS[b], edgecolor="white", linewidth=0.5, height=0.72)
        left += order[b].values
    # separator + emphasis for the ALL-shelf summary row
    ax.axhline(len(order) - 1.5, color="0.4", lw=0.8, ls=":")
    ax.axvline(0.5, color="0.25", lw=1.0, ls="--")   # 50% seasonal | low-freq divider
    ax.text(0.5, len(order) - 0.2, "50%", color="0.25", fontsize=8, ha="center", va="bottom")

    for i, s in enumerate(order.sector):             # bold the key sectors + ALL-shelf
        if s in KEY or s == "ALL-shelf":
            ax.get_yticklabels()  # placeholder; styled below

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    for tl, s in zip(ax.get_yticklabels(), order.sector):
        if s in KEY or s == "ALL-shelf":
            tl.set_fontweight("bold")
    ax.set_xlim(0, 1); ax.set_xlabel("fraction of F$_v$ power", fontsize=11)
    ax.invert_yaxis()
    ax.legend(handles=[Patch(facecolor=COLORS[b], label=b) for b in BANDS],
              ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.11), frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.14)
    out = os.path.join(REPO, "reports/figures/sorrm_sector_spectrum.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", out)
    # print the headline table
    print("\nsector                 seasonal  low-freq")
    for _, r in order.iterrows():
        print(f"  {r.sector:22s} {100*r.seasonal:5.0f}%   {100*r.lowfreq:5.0f}%")


if __name__ == "__main__":
    main()
