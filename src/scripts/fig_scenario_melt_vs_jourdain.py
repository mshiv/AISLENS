#!/usr/bin/env python3
"""Ensemble-mean sub-shelf melt per ISMIP6 basin under SSP585 and SSP126, Jourdain Fig-10 sectors highlighted.
Averages regionalAvgSubshelfMelt (m/yr) across ensemble members (annual-binned via xtime). Companion to
Jourdain 2020 Fig 10/13 (RCP8.5 projections). --xmax limits the year axis (e.g. 2100 to match Fig 10)."""
from __future__ import annotations
import argparse, glob, os
import numpy as np
from netCDF4 import Dataset, chartostring
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASK = os.path.join(REPO, "data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc")
ENS = {  # the current 10-member ensembles (SSP585_00..09, SSP126_00..09); other subdirs are older runs
    "SSP585": os.path.join(REPO, "data/MALI/diagnostics/ENSEMBLES/SSP585/SSP585_0[0-9]/regionalStats.nc"),
    "SSP126": os.path.join(REPO, "data/MALI/diagnostics/ENSEMBLES/SSP126/SSP126_0[0-9]/regionalStats.nc"),
}
JOURDAIN = {"J-K": ("Ronne-Filchner", "#0072B2"), "G-H": ("Pine Is.-Thwaites", "#D55E00"),
            "D-Dp": ("Cook-Ninnis", "#009E73"), "Cp-D": ("Totten-Moscow U.", "#CC79A7")}


def ensemble_mean(pat, y0=2000, y1=2300, min_members=3):
    files = sorted(glob.glob(pat))
    years = np.arange(y0, y1 + 1)
    stack = np.full((len(files), len(years), 16), np.nan)
    for k, f in enumerate(files):
        d = Dataset(f)
        melt = np.asarray(d.variables["regionalAvgSubshelfMelt"][:], float)
        yint = np.array([int(str(s).strip()[:4]) if str(s).strip()[:4].isdigit() else -1
                         for s in chartostring(d.variables["xtime"][:])])
        for iy, y in enumerate(years):
            m = yint == y
            if m.any():
                stack[k, iy] = np.nanmean(melt[m], axis=0)
    em = np.nanmean(stack, axis=0)
    n = np.sum(~np.isnan(stack[:, :, 0]), axis=0)
    keep = n >= min_members
    return years[keep], em[keep], len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmax", type=int, default=2300)
    ap.add_argument("--yscale", choices=["log", "linear"], default="log")
    ap.add_argument("--ymax", type=float, default=None, help="cap y-axis (linear) to see low sectors + refreezing")
    a = ap.parse_args()
    names = [str(s).strip().replace("ISMIP6 Basin ", "") for s in
             chartostring(Dataset(MASK).variables["regionNames"][:])]
    jidx = {b: names.index(b) for b in JOURDAIN}

    clip = (lambda v: np.clip(v, 1e-3, None)) if a.yscale == "log" else (lambda v: v)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (scen, pat) in zip(axes, ENS.items()):
        yr, em, nfiles = ensemble_mean(pat, y1=a.xmax)
        for j in range(16):
            if names[j] not in JOURDAIN:
                ax.plot(yr, clip(em[:, j]), color="0.8", lw=0.8)
        for b, (sec, c) in JOURDAIN.items():
            ax.plot(yr, clip(em[:, jidx[b]]), color=c, lw=2.2, label=f"{sec} ({b})")
        ax.set_yscale(a.yscale); ax.set_xlabel("year"); ax.set_title(scen)
        ax.grid(alpha=0.25, which="both")
        if a.yscale == "linear":
            ax.axhline(0, color="0.5", lw=0.7)                 # zero line: below = refreezing
            if a.ymax:
                ax.set_ylim(top=a.ymax)
    axes[0].set_ylabel("sub-shelf melt (m/yr)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    tag = f"_{a.yscale}" if a.yscale != "log" else ""
    tag += f"_ymax{int(a.ymax)}" if a.ymax else ""
    tag += f"_to{a.xmax}" if a.xmax != 2300 else ""
    out = os.path.join(REPO, f"reports/figures/scenario_melt_vs_jourdain{tag}.png")
    fig.savefig(out, dpi=150); print("wrote", out)
    for scen, pat in ENS.items():
        yr, em, nf = ensemble_mean(pat, y1=a.xmax)
        print(f"\n{scen} ({nf} members) — Jourdain sectors, {yr[0]}->{yr[-1]}:")
        for b in JOURDAIN:
            print(f"  {b:5s} {JOURDAIN[b][0]:20s} {em[0,jidx[b]]:6.2f} -> {em[-1,jidx[b]]:6.2f} m/yr")


if __name__ == "__main__":
    main()
