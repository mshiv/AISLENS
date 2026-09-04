#!/usr/bin/env python3
"""
fig_rate_of_change.py — annual SLE rate d(SLE)/dt (mm/yr) from ensemble mean.
Shows how fast sea level is rising and whether the rate accelerates.

Left panel: SSP126, Right panel: SSP585 (both with ±1σ envelope).
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from amplitude_response import load_sle

SCEN = [("SSP126", r"^SSP126_\d+$", "#0072B2"),
        ("SSP585", r"^SSP585_\d+$", "#D55E00")]


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

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

    for idx, (ens, inc, col) in enumerate(SCEN):
        try:
            sle, yr = load_sle(a.root, ens, inc, a.min_years, a.align)
        except Exception as e:
            print(f"{ens}: skip ({e})"); continue

        mean_sle = sle.mean("member").values
        sig_sle = sle.std("member", ddof=1).values
        n = sle.sizes["member"]

        dt = np.diff(yr)
        dmean = np.diff(mean_sle) / dt
        dsig = np.diff(sig_sle) / dt
        yr_mid = (yr[:-1] + yr[1:]) / 2.0

        ax = axL if idx == 0 else axR
        lbl = f"{ens} mean" if a.no_member_counts else f"{ens} mean (n={n})"
        ax.plot(yr_mid, dmean, col, lw=2, label=lbl)
        ax.fill_between(yr_mid, dmean - dsig, dmean + dsig,
                        color=col, alpha=0.2, label="±1σ")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel("year")
        ax.set_ylabel("d(SLE)/dt (mm/yr)")
        ax.set_title(f"{ens} — annual SLE rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        print(f"{ens}: mean d(SLE)/dt@yr{int(yr_mid[-1]):d}={dmean[-1]:.3g} mm/yr")

    fig.suptitle("Ensemble-mean sea-level rise rate")
    fig.tight_layout()
    out = os.path.join(a.out_dir, "rate_of_change.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
