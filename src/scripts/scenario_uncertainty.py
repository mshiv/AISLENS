#!/usr/bin/env python3
"""
scenario_uncertainty.py — scenario separability and uncertainty partition (SSP126 vs SSP585).

Fig 1: overlapping 5-95% envelopes with separability year. Fig 2: Hawkins-Sutton style
fraction of sea-level uncertainty from internal variability vs scenario choice.

Author: Shivaprakash Muruganandham
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


def stats(sle):
    return dict(mean=np.nanmean(sle, 0), std=np.nanstd(sle, 0, ddof=1),
               p05=np.nanpercentile(sle, 5, 0), p95=np.nanpercentile(sle, 95, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--start-year", type=float, default=2000.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    y126, s126 = load_sle(args.root, "SSP126", r"^SSP126_\d+$")
    y585, s585 = load_sle(args.root, "SSP585", r"^SSP585_\d+$")
    try:
        y10, s10 = load_sle(args.root, "SSP585_varScaled10x", r"^SSP585_\d+$")
    except Exception:
        y10, s10 = None, None

    n = min(len(y126), len(y585))
    yr = y126[:n]; cal = args.start_year + yr
    a126, a585 = np.asarray(s126)[:, :n], np.asarray(s585)[:, :n]
    st126, st585 = stats(a126), stats(a585)

    # ---------- Fig 1: separability ----------
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for st, c, lab in [(st126, "C0", "SSP1-2.6"), (st585, "C3", "SSP5-8.5")]:
        ax.fill_between(cal, st["p05"], st["p95"], color=c, alpha=0.25)
        ax.plot(cal, st["mean"], c, lw=2, label=f"{lab} (mean, 5-95%)")
    # scenario becomes distinguishable when SSP585 5th pct > SSP126 95th pct
    sep = st585["p05"] > st126["p95"]
    sep_yr = cal[np.argmax(sep)] if sep.any() else None
    if sep_yr is not None:
        ax.axvline(sep_yr, color="0.4", ls="--", lw=1)
        ax.annotate(f"scenarios separate\n~{sep_yr:.0f}", (sep_yr, ax.get_ylim()[1]*0.6),
                    fontsize=9, ha="left")
    ax.set_xlabel("year"); ax.set_ylabel("AIS contribution, VAF->SLE (mm, rise +)")
    ax.set_title("Scenario separability vs internal variability")
    ax.legend(loc="upper left"); ax.grid(alpha=0.2); fig.tight_layout()
    f1 = os.path.join(args.out_dir, "scenario_separability.png")
    fig.savefig(f1, dpi=150); plt.close(fig); print(f"Figure -> {f1}"
          f"   (scenarios separate ~{sep_yr:.0f})" if sep_yr else f"Figure -> {f1}")

    # ---------- Fig 2: uncertainty partition ----------
    internal_var = 0.5 * (st126["std"]**2 + st585["std"]**2)          # avg internal variance
    scen_var = ((st585["mean"] - st126["mean"]) / 2.0)**2             # across-scenario-mean var
    total = internal_var + scen_var
    with np.errstate(invalid="ignore", divide="ignore"):
        f_int = internal_var / total
        f_scn = scen_var / total

    fig, axs = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)
    # absolute (sqrt = sigma-equivalent, mm), log
    axs[0].plot(cal, np.sqrt(internal_var), "C2", lw=2, label="internal variability")
    axs[0].plot(cal, np.sqrt(scen_var), "C1", lw=2, label="scenario choice")
    axs[0].set_yscale("log"); axs[0].set_ylabel("uncertainty (mm, σ-equiv)")
    axs[0].set_title("AIS sea-level uncertainty: internal variability vs scenario choice")
    axs[0].legend(loc="upper left"); axs[0].grid(alpha=0.2, which="both")
    if s10 is not None:
        n10 = min(len(y10), n)
        st10 = stats(np.asarray(s10)[:, :n10])
        axs[0].plot(cal[:n10], st10["std"], "C4", lw=1.2, ls=":", label="internal (10x forcing)")
        axs[0].legend(loc="upper left")
    # fractional stacked
    axs[1].stackplot(cal, 100*f_int, 100*f_scn, colors=["C2", "C1"], alpha=0.8,
                     labels=["internal variability", "scenario choice"])
    axs[1].set_ylim(0, 100); axs[1].set_ylabel("% of total uncertainty")
    axs[1].set_xlabel("year"); axs[1].legend(loc="center right")
    # mark where scenario choice overtakes internal variability (f_int drops below 50%)
    below = np.where(f_int < 0.5)[0]
    cross_yr = cal[below[0]] if below.size else None
    for a in axs:
        if cross_yr is not None:
            a.axvline(cross_yr, color="0.4", ls="--", lw=1)
    if cross_yr is not None:
        axs[1].annotate(f"scenario overtakes\ninternal ~{cross_yr:.0f}", (cross_yr + 1, 8), fontsize=9)
        print(f"  internal variability dominates until ~{cross_yr:.0f} (f_int crosses 50%)")
    fig.tight_layout()
    f2 = os.path.join(args.out_dir, "uncertainty_partition.png")
    fig.savefig(f2, dpi=150); plt.close(fig); print(f"Figure -> {f2}")

    # print the numbers at key horizons
    print("\nyear   internal_sigma  scenario_sigma  internal_%")
    for h in [30, 65, 100, 200, n-1]:
        i = min(h, n-1)
        print(f"  {cal[i]:.0f}   {np.sqrt(internal_var[i]):8.2f}     "
              f"{np.sqrt(scen_var[i]):10.1f}      {100*f_int[i]:5.1f}%")


if __name__ == "__main__":
    main()
