#!/usr/bin/env python3
"""
fig_distribution_snapshots.py — histogram of ensemble members at yr0, yr150, yr300.
One panel per scenario (CTRL, SSP126, SSP585). Shows how spread grows over time.
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
SNAPSHOT_YEARS = [0, 150, 300]
SNAPSHOT_COLORS = ["#B0B0B0", "#5697CE", "#C34040"]
SNAPSHOT_LABELS = ["yr 0", "yr 150", "yr 300"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--min-years", type=float, default=50.0)
    ap.add_argument("--align", default="union")
    ap.add_argument("--out-dir", default="reports/figures")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    for idx, (ens, inc, col) in enumerate(SCEN):
        try:
            sle, yr = load_sle(a.root, ens, inc, a.min_years, a.align)
        except Exception as e:
            print(f"{ens}: skip ({e})"); continue

        ax = axes[idx]
        vals_at_year = {}
        for sy in SNAPSHOT_YEARS:
            if sy > yr[-1]:
                continue
            i = np.argmin(np.abs(yr - sy))
            vals = sle.isel(year=i).dropna("member").values
            vals_at_year[sy] = vals
            ax.hist(vals, bins="auto", alpha=0.6, color=SNAPSHOT_COLORS[SNAPSHOT_YEARS.index(sy)],
                    label=SNAPSHOT_LABELS[SNAPSHOT_YEARS.index(sy)])
            actual_yr = yr[i]
            print(f"{ens} @ yr={actual_yr:.0f} (requested {sy}): "
                  f"n={len(vals)}, μ={np.mean(vals):.3g} mm, σ={np.std(vals, ddof=1):.3g} mm")

        ax.set_xlabel("SLE (mm)")
        ax.set_title(ens)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis="y")

    axes[0].set_ylabel("count")
    fig.suptitle("Ensemble distribution snapshots")
    fig.tight_layout()
    out = os.path.join(a.out_dir, "distribution_snapshots.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
