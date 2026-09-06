#!/usr/bin/env python3
"""
fig_amplification_decomposition.py -- why tenfold forcing gives only threefold spread.

The slide says the widening is sub-proportional and the notes explain it with two causes at
two times. Neither the claim nor the explanation appears in any figure. This is that figure.

Write the continental spread as a sum over cells. If every cell moved in lockstep the
continental sigma would be the area-weighted sum of the local sigmas, T = sum_c w_c sigma_c.
It does not, so define a coherence factor phi = sigma_global / T, between zero and one.
The amplitude ratio then factors:

    sigma_global(10x)     T(10x)       phi(10x)
    ----------------  =  --------  x  ---------
    sigma_global(1x)      T(1x)        phi(1x)

    (what is measured)   (local gain)  (how much survives aggregation)

phi needs a thickness-to-sea-level calibration that is only approximate, but the ratio does
not: the calibration divides out. So the split is trustworthy even though phi alone is not.

Reading it: a coherence ratio below one means cells cancel more at greater amplitude, and a
falling local gain means individual cells stop responding. spatial_coherence_and_basins.py
reports the median of the coherence ratio, which averages across both regimes and hides the
handover between them; this keeps the time axis.
"""
from __future__ import annotations

import os, sys, argparse, glob, re
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import ensemble_io as eio        # noqa: E402
import fig_gl_transect as glt    # noqa: E402
from fig_std_vs_mean import load_var  # noqa: E402

SPAT = f"{glt.ROOT}/reports/dissertation/figures/spatial/stats_sample"
PAIR = [("SSP585_varScaled10x", "SSP585")]


def rd(path, var):
    d = netCDF4.Dataset(path)
    a = np.ma.filled(np.asarray(d[var][:], dtype=float), np.nan)
    d.close()
    return np.ravel(a)


def local_T(ens, year, area):
    """Area-weighted sum of per-cell sigma -- the perfectly coherent limit."""
    f = f"{SPAT}/{ens}_{year}.nc"
    if not os.path.exists(f):
        return None
    return float(np.nansum(rd(f, "thickness_std") * area))


def global_sigma(root, ens, include):
    """sigma of VAF across members, mm SLE, on the same processed axis as the deck.

    MALI writes globalStats at irregular intervals, so stacking members by index compares
    them at different actual times and inflates sigma -- and inflates it more for the
    noisier 10x ensemble, which would corrupt the very ratio this figure is about.
    load_var regrids every member onto a common annual grid first.
    """
    da, years = load_var(root, ens, include, "volumeAboveFloatation",
                         min_years=50.0, align="intersection", regrid_dt=1.0,
                         min_members=3, despike_thresh=0.0)
    v = np.asarray(da.values)
    sle = np.stack([eio.vaf_to_sle_mm(v[m], reference="first") for m in range(v.shape[0])])
    return np.asarray(years), np.nanstd(sle, axis=0, ddof=1), v.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=os.path.join(
        glt.ROOT, "reports/dissertation/figures/slides"))
    a = ap.parse_args()
    out = a.out or f"{a.outdir}/fig_amplification_decomposition.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    mesh = os.path.join(glt.ROOT, "data/MALI", os.path.basename(glt.MESH))
    area = rd(mesh, "areaCell")

    num, den = PAIR[0]
    yN, sN, nN = global_sigma(a.root, num, r".*")
    yD, sD, nD = global_sigma(a.root, den, r"^SSP585_\d+$")
    print(f"  {num} n={nN}, {den} n={nD}")

    years = sorted({int(re.search(r"_(\d{4})\.nc$", f).group(1))
                    for f in glob.glob(f"{SPAT}/{den}_*.nc")}
                   & {int(re.search(r"_(\d{4})\.nc$", f).group(1))
                      for f in glob.glob(f"{SPAT}/{num}_*.nc")})
    rows = []
    for y in years:
        tN, tD = local_T(num, y, area), local_T(den, y, area)
        if not tN or not tD:
            continue
        my = y - 2000
        gN = float(np.interp(my, yN, sN)); gD = float(np.interp(my, yD, sD))
        if gD <= 0:
            continue
        loc, glo = tN / tD, gN / gD
        rows.append((my, loc, glo, glo / loc))
    R = np.array(rows)

    print(f"  {'yr':>5} {'local gain':>11} {'global ratio':>13} {'coherence ratio':>16}")
    for r in R:
        print(f"  {r[0]:5.0f} {r[1]:11.2f} {r[2]:13.2f} {r[3]:16.3f}")

    fig = plt.figure(figsize=(13.6, 6.8))
    ax = fig.add_axes([0.078, 0.135, 0.885, 0.750])

    ax.axhline(1.0, color=ds.RULE, lw=1.0, zorder=2)
    ax.axhline(10.0, color=ds.RULE, lw=1.0, ls=":", zorder=2)
    ax.text(R[-1, 0], 10.0, " proportional response", fontsize=11.5, color=ds.INK_SOFT,
            va="center", ha="left")

    ax.plot(R[:, 0], R[:, 1], "-o", lw=2.6, ms=8, color=ds.ICE, zorder=4)
    ax.plot(R[:, 0], R[:, 2], "-o", lw=2.6, ms=8, color=ds.INK, zorder=5)
    ax.plot(R[:, 0], R[:, 3], "-o", lw=2.6, ms=8, color=ds.MARSH, zorder=4)
    lab = [("local gain, cell by cell", ds.ICE, R[0, 1]),
           ("what the continent actually shows", ds.INK, R[0, 2]),
           ("fraction surviving aggregation", ds.MARSH, R[0, 3])]
    for txt, c, y0 in lab:
        ax.annotate(txt, (R[0, 0], y0), fontsize=12.5, color=c,
                    xytext=(10, 8), textcoords="offset points", zorder=6)

    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 3, 5, 10])
    ax.set_yticklabels(["0.5", "1", "2", "3", "5", "10"])
    ds.strip(ax)
    ax.set_xlabel("model year", labelpad=8)
    ax.set_ylabel("ratio, 10× forcing over 1×", labelpad=8)
    ax.text(0.0, 1.035,
            "early, cells cancel more at greater amplitude · later, cells stop responding "
            "and the two ratios meet",
            transform=ax.transAxes, fontsize=12.5, color=ds.INK_SOFT, ha="left", va="bottom")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
