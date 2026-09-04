#!/usr/bin/env python3
"""
fig_grounded_area.py — grounded area of each shelf catchment through time.

Grounded where thickness exceeds flotation on the fixed bed; catchments are a Voronoi
partition of grounded ice among the named shelves in the 133-region mask. No transect and no
crossing rule, so it is comparable across shelves. Band recomputes the area at h_mean -/+
sigma_h. Computes f(mean(h)); regenerate from members once the per-member extract exists.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SPAT = f"{ROOT}/reports/dissertation/figures/spatial/stats_sample"
MESH = (f"{ROOT}/data/MALI/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m"
        "_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu"
        "_meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
SHELF_MASK = f"{ROOT}/data/MALI/aislens_draftDepen_regionMasks.nc"

RHO_I, RHO_O = 910.0, 1028.0
YEARS = [0, 100, 200, 300]

# the shelves worth naming on a slide: warm-cavity Amundsen, plus cold-cavity controls
SHELVES = ["Thwaites", "Pine_Island", "Crosson", "Dotson", "Getz",
           "Totten", "Filchner", "Ronne", "Ross" if False else "Western_Ross"]
LABEL = {"Pine_Island": "Pine Island", "Western_Ross": "Ross (west)"}


def rd(path, var):
    d = netCDF4.Dataset(path)
    a = np.ma.filled(np.asarray(d[var][:], dtype=float), np.nan)
    d.close()
    return np.ravel(a) if a.ndim > 1 else a


def region_names(path):
    d = netCDF4.Dataset(path)
    raw = d["regionNames"][:]
    names = ["".join(c.decode() if isinstance(c, bytes) else str(c) for c in r)
             .replace("\x00", "").strip("-").strip() for r in raw]
    masks = np.asarray(d["regionCellMasks"][:])
    d.close()
    return names, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or (f"{ROOT}/reports/dissertation/figures/slides/"
                    f"fig_grounded_area_{a.ensemble}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    names, masks = region_names(SHELF_MASK)
    shelves = [s for s in SHELVES if s in names]
    x, y = rd(MESH, "xCell"), rd(MESH, "yCell")
    bed, h0, area = rd(MESH, "bedTopography"), rd(MESH, "thickness"), rd(MESH, "areaCell")
    hflot = (RHO_O / RHO_I) * np.maximum(0.0, -bed)

    # Voronoi partition of grounded ice among the named shelves
    seeds, seed_of = [], []
    for si, sh in enumerate(shelves):
        cells = np.where(masks[:, names.index(sh)] > 0)[0]
        seeds.append(np.column_stack([x[cells], y[cells]]))
        seed_of.append(np.full(cells.size, si))
    seeds = np.vstack(seeds)
    seed_of = np.concatenate(seed_of)
    _, nearest = cKDTree(seeds).query(np.column_stack([x, y]))
    catchment = seed_of[nearest]

    def grounded_area(h):
        gr = (h > hflot + 1.0) & (h > 1.0)
        return np.array([np.nansum(area[gr & (catchment == si)])
                         for si in range(len(shelves))])

    A = {0: grounded_area(h0)}
    Alo, Ahi = {0: A[0]}, {0: A[0]}
    for yr in YEARS[1:]:
        f = f"{SPAT}/{a.ensemble}_{yr + 2000}.nc"
        if not os.path.exists(f):
            continue
        hm, sd = rd(f, "thickness_mean"), rd(f, "thickness_std")
        A[yr] = grounded_area(hm)
        Alo[yr] = grounded_area(hm - sd)
        Ahi[yr] = grounded_area(hm + sd)

    yrs = sorted(A)
    frac = np.array([[100 * A[t][i] / A[0][i] for t in yrs] for i in range(len(shelves))])
    flo = np.array([[100 * Alo[t][i] / A[0][i] for t in yrs] for i in range(len(shelves))])
    fhi = np.array([[100 * Ahi[t][i] / A[0][i] for t in yrs] for i in range(len(shelves))])

    order = np.argsort(frac[:, -1])
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    fig.subplots_adjust(left=0.062, right=0.745, top=0.80, bottom=0.145)

    ylo = max(0.0, float(np.floor(frac.min() / 10) * 10) - 6)
    # de-collide the end labels: nudge each up until it clears the one below
    gap = (102 - ylo) * 0.052
    label_y, prev = {}, -1e9
    for i in order:
        yv = max(frac[i, -1], prev + gap)
        label_y[i] = yv
        prev = yv

    for rank, i in enumerate(order):
        loss = 100 - frac[i, -1]
        col = ds.MARSH if loss > 25 else ds.ICE if loss > 5 else ds.INK_SOFT
        lw = 2.6 if loss > 25 else 1.8
        ax.fill_between(yrs, flo[i], fhi[i], color=col, alpha=.18, linewidth=0, zorder=2)
        ax.plot(yrs, frac[i], color=col, lw=lw, zorder=3, solid_capstyle="round")
        ax.plot([yrs[-1]], [frac[i, -1]], "o", ms=5.5, color=col, zorder=4)
        nm = LABEL.get(shelves[i], shelves[i].replace("_", " "))
        ly = label_y[i]
        if abs(ly - frac[i, -1]) > 0.4:      # leader line when the label was nudged
            ax.plot([yrs[-1], yrs[-1] + 6], [frac[i, -1], ly], color=col, lw=.8,
                    alpha=.6, clip_on=False, zorder=3)
        ax.text(yrs[-1] + 9, ly, f"{nm}   {frac[i,-1]:.0f}%",
                color=col, fontsize=11, va="center", ha="left", clip_on=False)

    ds.strip(ax)
    ax.set_xlim(0, yrs[-1])
    ax.set_ylim(ylo, 103)
    ax.set_xticks(yrs)
    ticks = [t for t in (0, 25, 40, 50, 60, 75, 90, 100) if t >= ylo]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t}%" if t == 100 else str(t) for t in ticks])
    ax.set_xlabel("model year", labelpad=7)
    ax.set_ylabel("grounded area retained", labelpad=8)
    ax.tick_params(length=3)
    ax.grid(axis="y", zorder=0)

    ax.text(0.0, 1.115, "the same forcing ungrounds some catchments and leaves others intact",
            transform=ax.transAxes, fontsize=15, color=ds.INK, ha="left", va="bottom")
    ax.text(0.0, 1.04,
            f"{a.ensemble} · grounded area of each shelf's catchment, relative to year 0 · "
            "band = ±1σ of the ensemble thickness field",
            transform=ax.transAxes, fontsize=10, color=ds.INK_SOFT, ha="left", va="bottom")

    fig.text(0.008, 0.008,
             "catchments are a Voronoi partition of grounded ice among the named shelves · "
             "grounded where thickness exceeds flotation on the fixed bed",
             fontsize=8.8, color=ds.INK_SOFT, ha="left", va="bottom", style="italic")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
    print(f"wrote {out}")
    for i in order:
        nm = LABEL.get(shelves[i], shelves[i])
        env = fhi[i, -1] - flo[i, -1]
        print(f"  {nm:14s} yr0 {A[0][i]/1e9:7.1f}e3 km²   "
              + "  ".join(f"y{t}:{frac[i,k]:5.1f}%" for k, t in enumerate(yrs))
              + f"   envelope {env:4.1f} pt")


if __name__ == "__main__":
    main()
