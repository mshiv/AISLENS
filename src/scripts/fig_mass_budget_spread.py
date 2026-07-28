#!/usr/bin/env python3
"""
fig_mass_budget_spread.py — Which flux term carries the most ensemble spread?

Grouped bar chart: ensemble std per flux term (sfcMassBal, floatingBasalMassBal, calvingFlux,
groundingLineFlux) averaged over the last 100 years. SSP585 vs varScaled10x.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio

FLUX_VARS = ["totalSfcMassBal", "totalFloatingBasalMassBal", "totalCalvingFlux",
             "groundingLineFlux"]
FLUX_LABELS = ["Surface\nMass Bal", "Sub-shelf\nBasal Melt", "Calving\nFlux",
               "Grounding\nLine Flux"]
VAR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    ap.add_argument("--last-n-years", type=float, default=100.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    variables = FLUX_VARS + ["daysSinceStart"]

    print("Loading SSP585...")
    ssp585 = eio.load_ensemble_globalstats(
        os.path.join(args.root, "SSP585"),
        variables=variables, include=r"^SSP585_\d+$", min_years=50, align="union")
    print("Loading SSP585_varScaled10x...")
    var10x = eio.load_ensemble_globalstats(
        os.path.join(args.root, "SSP585_varScaled10x"),
        variables=variables, include=r"^SSP585_\d+$", min_years=50, align="union")

    # Determine the last-N-years window for each ensemble
    std_585_vals = []
    std_10x_vals = []
    for var in FLUX_VARS:
        yr = ssp585["year"].values
        yr_end = yr[-1]
        yr_start = yr_end - args.last_n_years
        std_585 = float(ssp585[var].sel(year=slice(yr_start, yr_end)).std("member").mean("year").item())
        std_585_vals.append(std_585)

        yr2 = var10x["year"].values
        yr_end2 = yr2[-1]
        yr_start2 = yr_end2 - args.last_n_years
        std_10x = float(var10x[var].sel(year=slice(yr_start2, yr_end2)).std("member").mean("year").item())
        std_10x_vals.append(std_10x)

    print(f"\nEnsemble std averaged over last {int(args.last_n_years)} yr:")
    print(f"{'Flux term':<25s} {'SSP585':>12s} {'10x var':>12s} {'ratio':>8s}")
    for label, v585, v10x in zip(FLUX_LABELS, std_585_vals, std_10x_vals):
        lab = label.replace("\n", " ")
        ratio = v10x / v585 if v585 > 0 else float("inf")
        print(f"  {lab:<25s} {v585:12.4f} {v10x:12.4f} {ratio:8.2f}x")

    # --- Grouped bar chart ---
    n_terms = len(FLUX_VARS)
    x = np.arange(n_terms)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars1 = ax.bar(x - bar_width / 2, std_585_vals, bar_width,
                   color=VAR_COLORS, alpha=0.85, edgecolor="white", linewidth=0.8,
                   label="SSP585")
    bars2 = ax.bar(x + bar_width / 2, std_10x_vals, bar_width,
                   color=VAR_COLORS, alpha=0.5, edgecolor="white", linewidth=0.8,
                   hatch="//", label="SSP585_varScaled10x")

    # Annotate values
    for bar, val in zip(bars1, std_585_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, std_10x_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Ratio annotations
    for i, (v585, v10x) in enumerate(zip(std_585_vals, std_10x_vals)):
        ratio = v10x / v585 if v585 > 0 else float("inf")
        max_val = max(v585, v10x)
        ax.text(x[i], max_val + 0.025, f"{ratio:.1f}x", ha="center", va="bottom",
                fontsize=8, color="0.3", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(FLUX_LABELS, fontsize=10)
    ax.set_ylabel("ensemble std (kg m⁻² s⁻¹, last 100 yr mean)")
    ax.set_title("Which flux term carries the most ensemble spread?")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "fig_mass_budget_spread.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure -> {out_path}")


if __name__ == "__main__":
    main()
