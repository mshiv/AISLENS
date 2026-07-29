#!/usr/bin/env python3
"""
fig_exceedance_probability.py — probability of exceeding SLE thresholds over time.

P(SLE > X mm) at each year for SSP585, SSP126, varScaled10x, plus
P(SLE > X) vs X curves at selected horizons (yr50, 100, 200, 300).

Author: Shivaprakash Muruganandham (2026-07-22)
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio


def load_sle(root, ensemble, include, min_years=50):
    ds = eio.load_ensemble_globalstats(
        os.path.join(root, ensemble),
        variables=["volumeAboveFloatation", "daysSinceStart"],
        include=include, min_years=min_years, align="union")
    sle = xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, reference="first"),
                         ds["volumeAboveFloatation"])
    return ds["year"].values, sle  # (member, year)


def exceedance_prob(sle_values, thresholds):
    """P(SLE > threshold) for each threshold, given a set of member values."""
    n = len(sle_values)
    return np.array([np.sum(sle_values > t) / n for t in thresholds])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--start-year", type=float, default=2000.0)
    ap.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # Load ensembles
    ensembles = {}
    for ens, inc in [("SSP585", r"^SSP585_\d+$"), ("SSP126", r"^SSP126_\d+$"),
                     ("varScaled10x", r"^SSP585_\d+$")]:
        try:
            yr, sle = load_sle(a.root, ens, inc)
            ensembles[ens] = {"yr": yr, "sle": np.asarray(sle), "n": sle.sizes["member"]}
            print(f"{ens}: {sle.sizes['member']} members, yr {yr[0]:.0f}..{yr[-1]:.0f}")
        except Exception as e:
            print(f"{ens}: skip ({e})")

    colors = {"SSP585": "C3", "SSP126": "C0", "varScaled10x": "C4"}

    # ---- Figure 1: P(SLE > X) evolution ----
    thresholds_mm = [5, 10, 20, 50]
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))

    for ens, d in ensembles.items():
        cal = a.start_year + d["yr"]
        sle = d["sle"]  # (member, year)
        for t in thresholds_mm:
            p_exceed = np.array([np.sum(sle[:, i] > t) / d["n"]
                                 for i in range(sle.shape[1])])
            ls = {5: "-", 10: "--", 20: ":", 50: "-."}[t]
            lw = {5: 1.2, 10: 1.5, 20: 1.8, 50: 2.0}[t]
            ax1.plot(cal, 100*p_exceed, color=colors[ens], ls=ls, lw=lw,
                     label=f"{ens} ({t}mm)" if t == 5 else None, alpha=0.8)
            # Annotate the 50% crossing for SSP585
            if ens == "SSP585":
                crossed = np.where(p_exceed >= 0.5)[0]
                if crossed.size > 0:
                    yr_cross = cal[crossed[0]]
                    ax1.annotate(f"~{yr_cross:.0f}", (yr_cross, 50),
                                fontsize=7, color=colors[ens], ha="left",
                                xytext=(5, 2), textcoords="offset points")

    ax1.set_xlabel("year")
    ax1.set_ylabel("P(SLE > X mm)")
    ax1.set_title("Probability of exceeding sea-level thresholds over time")
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.2)
    # Custom legend: line styles for thresholds, colors for ensembles
    from matplotlib.lines import Line2D
    legend_elements = []
    for ens in ensembles:
        legend_elements.append(Line2D([0], [0], color=colors[ens], lw=2, label=ens))
    legend_elements.append(Line2D([0], [0], color="0.5", ls="-", lw=1.2, label="5 mm"))
    legend_elements.append(Line2D([0], [0], color="0.5", ls="--", lw=1.5, label="10 mm"))
    legend_elements.append(Line2D([0], [0], color="0.5", ls=":", lw=1.8, label="20 mm"))
    legend_elements.append(Line2D([0], [0], color="0.5", ls="-.", lw=2.0, label="50 mm"))
    ax1.legend(handles=legend_elements, fontsize=8, loc="upper left")
    fig1.tight_layout()
    out1 = os.path.join(a.out_dir, "exceedance_probability_evolution.png")
    fig1.savefig(out1, dpi=200, bbox_inches="tight"); plt.close(fig1)
    print(f"Figure -> {out1}")

    # ---- Figure 2: P(SLE > X) vs X at selected horizons ----
    horizons_yr = [50, 100, 200, 300]
    thresholds = np.arange(0, 200, 1)
    fig2, axs = plt.subplots(1, len(horizons_yr), figsize=(4*len(horizons_yr), 4.5),
                             sharey=True)

    for j, h_yr in enumerate(horizons_yr):
        ax = axs[j]
        for ens, d in ensembles.items():
            i = int(np.argmin(np.abs(d["yr"] - h_yr)))
            p = exceedance_prob(d["sle"][:, i], thresholds)
            ax.plot(thresholds, 100*p, color=colors[ens], lw=2, label=ens)
        ax.axvline(0, color="0.7", lw=0.5)
        ax.set_xlabel("SLE threshold (mm)")
        ax.set_title(f"yr{h_yr} ({a.start_year + h_yr:.0f})", fontsize=10)
        ax.grid(alpha=0.2)
        if j == 0:
            ax.set_ylabel("P(SLE > X mm)")
            ax.legend(fontsize=8)
        ax.set_ylim(0, 105)

    fig2.suptitle("Exceedance probability curves at selected horizons", fontsize=11, y=1.02)
    fig2.tight_layout()
    out2 = os.path.join(a.out_dir, "exceedance_probability_curves.png")
    fig2.savefig(out2, dpi=200, bbox_inches="tight"); plt.close(fig2)
    print(f"Figure -> {out2}")

    # Print summary
    print("\nExceedance probabilities at yr200:")
    for ens, d in ensembles.items():
        i = int(np.argmin(np.abs(d["yr"] - 200)))
        for t in [5, 10, 20, 50]:
            p = np.sum(d["sle"][:, i] > t) / d["n"]
            print(f"  {ens:20s}  P(SLE>{t:2d}mm) = {100*p:5.1f}%")


if __name__ == "__main__":
    main()
