#!/usr/bin/env python3
"""
fig_ar1_timescale.py — AR(1) decorrelation timescale of annual SLE increments.
For each member: first-difference the SLE time series (removes trend), fit AR(1),
compute τ = −1 / log(ρ₁). Histogram across the ensemble.

Answers: how long does the ice sheet remember its past year-to-year fluctuations?
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from amplitude_response import load_sle

SCEN = [("CTRL", r"^CTRL_\d+$", "#888888"),
        ("SSP126", r"^SSP126_\d+$", "#0072B2"),
        ("SSP585", r"^SSP585_\d+$", "#D55E00")]


def ar1_tau(x):
    """Fit AR(1) to detrended series, return decorrelation timescale τ (years).
    Uses lag-1 autocorrelation of the first-differenced (i.e. prewhitened) series
    to isolate the stochastic component from any residual trend."""
    dx = np.diff(x)
    dx = dx[~np.isnan(dx)]
    if len(dx) < 3:
        return np.nan
    r1 = np.corrcoef(dx[:-1], dx[1:])[0, 1]
    if not np.isfinite(r1) or abs(r1) >= 1:
        return np.nan
    return -1.0 / np.log(max(r1, 1e-10))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--min-years", type=float, default=50.0)
    ap.add_argument("--align", default="union")
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--no-member-counts", action="store_true",
                    help="omit '(n=N)' from legend labels (publication style)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bin_width = 0.5

    for ens, inc, col in SCEN:
        try:
            sle, yr = load_sle(a.root, ens, inc, a.min_years, a.align)
        except Exception as e:
            print(f"{ens}: skip ({e})"); continue

        taus = []
        for m in range(sle.sizes["member"]):
            t = ar1_tau(sle.isel(member=m).values)
            if np.isfinite(t):
                taus.append(t)

        taus = np.array(taus)
        bins = np.arange(0, max(taus) + bin_width, bin_width)
        lbl = ens if a.no_member_counts else f"{ens} (n={len(taus)})"
        ax.hist(taus, bins=bins, alpha=0.5, color=col, label=lbl)
        ax.axvline(np.median(taus), color=col, ls="--", lw=1)
        print(f"{ens}: τ median={np.median(taus):.2f} yr, range=[{taus.min():.2f}, {taus.max():.2f}] yr")

    ax.set_xlabel("AR(1) decorrelation timescale τ (yr)")
    ax.set_ylabel("count")
    ax.set_title("Persistence of annual SLE fluctuations")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(a.out_dir, "ar1_timescale.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
