#!/usr/bin/env python3
"""
fig_basin_covariance.py — #4: is the internal SLR uncertainty a COHERENT multi-basin swing or
INDEPENDENT basins? Correlation matrix of per-basin ΔVAF *internal residuals* (member deviation
from the ensemble mean) across the 16 ISMIP6 basins. Q: within a given ocean realization, do
Amundsen and Filchner-Ronne move together, or independently?

Residual_mr(t) = sle_mr(t) - ensemblemean_r(t); pooled over (member, year) samples.
(Time autocorrelation inflates effective N, so read the correlation VALUES, not the p-values.)
"""
from __future__ import annotations
import os, sys, csv as _csv, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from fig_regional_emergence import load_regional_sle
from ismip6_regions import BASIN_NAMES, SHORT_LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--members", default=r"^SSP585_\d+$")
    ap.add_argument("--min-year-frac", type=float, default=0.1,
                    help="drop the first fraction of years (spin-up, ~zero spread)")
    ap.add_argument("--out", default="reports/basin_covariance.png")
    args = ap.parse_args()

    names = BASIN_NAMES
    years, arr = load_regional_sle(args.root, args.ensemble, args.members)  # (member,year,region)
    if arr is None:
        sys.exit("no usable members")
    y0 = int(args.min_year_frac * arr.shape[1])
    arr = arr[:, y0:, :]
    resid = arr - np.nanmean(arr, axis=0, keepdims=True)          # internal deviations
    nreg = resid.shape[2]
    R = resid.reshape(-1, nreg)                                   # (member*year, region)
    ok = np.isfinite(R).all(axis=1)
    C = np.corrcoef(R[ok].T)                                      # (region, region)

    # Short labels for tight axis ticks
    short = [SHORT_LABELS.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(nreg)); ax.set_yticks(range(nreg))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label="corr of internal ΔVAF residuals")
    ax.set_title(f"{args.ensemble}: cross-basin coherence of internal variability\n"
                 "(red block = basins that swing together within an ocean realization)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {args.out}")

    # report the strongest off-diagonal couplings among the high-spread basins
    key = ["Thwaites/PIG", "Getz", "FRIS", "Ross", "Dronning Maud Land"]
    print("\ncross-basin residual correlations among the high-spread basins:")
    print("        " + "  ".join(f"{SHORT_LABELS.get(b,b):>12s}" for b in key))
    for bi in key:
        i = names.index(bi)
        print(f"  {SHORT_LABELS.get(bi,bi):12s} " +
              "  ".join(f"{C[i, names.index(bj)]:+5.2f}" for bj in key))


if __name__ == "__main__":
    main()
