#!/usr/bin/env python3
"""
fig_spread_budget.py — Decomposes ensemble spread by ISMIP6 basin.

Panel (a): stacked area of per-basin variance fraction over time.
Panel (b): per-basin σ at yr50/100/200/300.
Panel (c): SSP585 vs SSP126 σ at final year.
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


# Color cycle for 16 basins
BASIN_COLORS = plt.cm.tab20(np.linspace(0, 1, 16))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--start-year", type=float, default=2000.0)
    ap.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    names = BASIN_NAMES
    short = [SHORT_LABELS.get(n, n) for n in names]
    nreg = len(names)

    years, arr = load_regional_sle(a.root, "SSP585", r"^SSP585_\d+$")
    if arr is None:
        sys.exit("SSP585: no usable members")
    cal = a.start_year + years
    n = arr.shape[1]

    # Per-basin variance over time: sigma^2 per basin
    sig2 = np.nanvar(arr, axis=0)  # (year, region)
    total_sig2 = sig2.sum(axis=1)  # total variance = sum of per-basin variances
    fraction = sig2 / np.maximum(total_sig2[:, None], 1e-9)  # fraction of total

    # Per-basin sigma
    sig = np.nanstd(arr, axis=0, ddof=1)

    # ---- Figure ----
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(18, 6),
                                            gridspec_kw={"width_ratios": [2, 1.2, 1.2]})

    # Panel (a): stacked area chart of variance fraction
    ax_a.stackplot(cal[:n], fraction[:n].T, labels=names,
                   colors=BASIN_COLORS[:nreg], alpha=0.85)
    ax_a.set_xlabel("year")
    ax_a.set_ylabel("fraction of total σ²")
    ax_a.set_title("(a) Variance budget: which basins dominate uncertainty?")
    ax_a.set_ylim(0, 1)
    ax_a.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, ncol=1)
    ax_a.grid(alpha=0.2)

    # Panel (b): per-basin σ at selected horizons
    horizons_yr = [50, 100, 200, min(298, n-1)]
    bar_height = 0.18
    y_pos = np.arange(nreg)
    for j, h_yr in enumerate(horizons_yr):
        i = min(h_yr, n-1)
        ax_b.barh(y_pos + j*bar_height, sig[i], height=bar_height,
                  label=f"yr{h_yr}", alpha=0.8)
    ax_b.set_yticks(y_pos + 1.5*bar_height)
    ax_b.set_yticklabels(short, fontsize=7)
    ax_b.set_xlabel("ensemble spread σ (mm SLE)")
    ax_b.set_title(f"(b) Per-basin σ at horizons")
    ax_b.legend(fontsize=8)
    ax_b.grid(axis="x", alpha=0.2)

    # Panel (c): SSP585 vs SSP126 at final year
    years_126, arr_126 = load_regional_sle(a.root, "SSP126", r"^SSP126_\d+$")
    if arr_126 is not None:
        sig_126 = np.nanstd(arr_126, axis=0, ddof=1)
        n_126 = sig_126.shape[0]
        # Final year values
        sig_585_final = sig[-1]
        sig_126_final = sig_126[-1]
        y_pos_c = np.arange(nreg)
        ax_c.barh(y_pos_c - 0.2, sig_585_final, height=0.35, color="C3",
                  label="SSP585", alpha=0.8)
        ax_c.barh(y_pos_c + 0.2, sig_126_final, height=0.35, color="C0",
                  label="SSP126", alpha=0.8)
        ax_c.set_yticks(y_pos_c)
        ax_c.set_yticklabels(short, fontsize=7)
        ax_c.set_xlabel("ensemble spread σ (mm SLE)")
        ax_c.set_title(f"(c) SSP585 vs SSP126 @ yr{cal[-1]:.0f}")
        ax_c.legend(fontsize=8)
        ax_c.grid(axis="x", alpha=0.2)
    else:
        ax_c.text(0.5, 0.5, "SSP126 not available", ha="center", va="center",
                  transform=ax_c.transAxes)
        ax_c.set_title("(c) SSP126 unavailable")

    fig.suptitle("Spread budget: spatial decomposition of ensemble uncertainty",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = os.path.join(a.out_dir, "spread_budget.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")

    print("\nVariance fraction by basin at key horizons:")
    for h_yr in [50, 100, 200, n-1]:
        i = min(h_yr, n-1)
        print(f"\n  yr{cal[i]:.0f}:")
        sorted_idx = np.argsort(fraction[i])[::-1]
        cumsum = 0
        for r in sorted_idx[:5]:
            cumsum += fraction[i, r]
            print(f"    {names[r]:20s}  {100*fraction[i,r]:5.1f}%  (cumulative: {100*cumsum:.1f}%)")


if __name__ == "__main__":
    main()
