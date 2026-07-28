#!/usr/bin/env python3
"""
fig_percentile_band.py — Ensemble uncertainty band (median + IQR + 5–95%) over time.

Percentiles are robust to skew (unlike mean ± σ), showing honest uncertainty for
asymmetric ensemble distributions. Uses ensemble_io loader (restart segments dropped).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from amplitude_response import load_sle

SCEN = [("CTRL", r"^CTRL_\d+$", "C2"), ("SSP126", r"^SSP126_\d+$", "C0"),
        ("SSP585", r"^SSP585_\d+$", "C3")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--min-years", type=float, default=50.0)
    ap.add_argument("--align", default="intersection",
                    help="intersection = full-ensemble years only (recommended)")
    ap.add_argument("--out-dir", default="reports")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    fig, (ax, axw) = plt.subplots(1, 2, figsize=(14, 5.4))
    for ens, inc, col in SCEN:
        try:
            sle, yr = load_sle(a.root, ens, inc, a.min_years, a.align)
        except Exception as e:
            print(f"{ens}: skip ({e})"); continue
        v = sle.values                                   # (member, year)
        med = np.nanmedian(v, 0)
        q25, q75 = np.nanpercentile(v, 25, 0), np.nanpercentile(v, 75, 0)
        q05, q95 = np.nanpercentile(v, 5, 0), np.nanpercentile(v, 95, 0)
        n = sle.sizes["member"]
        ax.fill_between(yr, q05, q95, color=col, alpha=0.12)
        ax.fill_between(yr, q25, q75, color=col, alpha=0.28)
        ax.plot(yr, med, col, lw=2, label=f"{ens} (n={n})")
        # (b) band WIDTH over time = division-free "uncertainty magnitude" (mm); no |mean| in it
        w90, wiqr = q95 - q05, q75 - q25
        axw.plot(yr, w90, col, lw=2, label=f"{ens} 90% (Q95−Q5)")
        axw.plot(yr, wiqr, col, lw=1.2, ls="--")
        print(f"{ens}: median@end={med[-1]:.1f} mm; IQR=[{q25[-1]:.1f},{q75[-1]:.1f}]; "
              f"5-95=[{q05[-1]:.1f},{q95[-1]:.1f}]; 90%-width@end={w90[-1]:.1f} mm")

    axw.set_xlabel("year"); axw.set_ylabel("band width (mm SLE)")
    axw.set_title("(b) uncertainty magnitude = band width\n"
                  "solid = 90% (Q95−Q5), dashed = IQR; no division → no zero-crossing artifact")
    axw.legend(fontsize=8); axw.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("year"); ax.set_ylabel("VAF→SLE (mm, rise +)")
    ax.set_title("Ensemble sea-level contribution — median + IQR (25–75%) + 5–95% range\n"
                 "(percentiles: robust to skew; dark band = IQR, light band = 5–95%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(a.out_dir, "percentile_band.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
