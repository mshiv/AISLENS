#!/usr/bin/env python3
"""
fig_lowpass_filter.py — demonstrates the ice sheet acts as a low-pass filter.

Panel (a): VAF time series for SSP585 and CTRL. Panel (b): VAF power spectral density
showing peak at long periods, drop at short periods.
Data: globalStats from ensemble runs.

Author: Shivaprakash Muruganandham (2026-07-22)
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio


def load_sle_array(root, ensemble, include, min_years=50):
    """Load ensemble and return (year_array, member_array) in SLE mm."""
    ds = eio.load_ensemble_globalstats(
        os.path.join(root, ensemble),
        variables=["volumeAboveFloatation", "daysSinceStart"],
        include=include, min_years=min_years, align="union")
    sle = eio.vaf_to_sle_mm(ds["volumeAboveFloatation"].values, reference="first")
    return ds["year"].values, sle  # (year,), (member, year)


def compute_psd(timeseries, fs=12.0):
    """Welch PSD. timeseries is 1D, fs=12 samples/yr (monthly)."""
    clean = timeseries - np.nanmean(timeseries)
    nperseg = min(len(clean), 2048)
    f, P = welch(clean, fs=fs, nperseg=nperseg)
    mask = f > 0
    return 1.0 / f[mask], P[mask]  # period (yr), power


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    yr_ctrl, sle_ctrl = load_sle_array(args.root, "CTRL", r"^CTRL_\d+$", min_years=50)
    yr_ssp, sle_ssp = load_sle_array(args.root, "SSP585", r"^SSP585_\d+$", min_years=50)

    n_ctrl = sle_ctrl.shape[0]
    n_ssp = sle_ssp.shape[0]

    # --- Panel (a): raw VAF time series ---
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(8, 8),
                                     gridspec_kw={"height_ratios": [1, 1.2]})

    for i in range(n_ctrl):
        ax_a.plot(yr_ctrl, sle_ctrl[i], color="C0", alpha=0.15, lw=0.6)
    ax_a.plot(yr_ctrl, np.nanmean(sle_ctrl, axis=0), color="C0", lw=2.2, label="CTRL mean")

    for i in range(n_ssp):
        ax_a.plot(yr_ssp, sle_ssp[i], color="C3", alpha=0.15, lw=0.6)
    ax_a.plot(yr_ssp, np.nanmean(sle_ssp, axis=0), color="C3", lw=2.2, label="SSP585 mean")

    ax_a.set_xlabel("year since run start")
    ax_a.set_ylabel("VAF change (mm SLE)")
    ax_a.set_title("(a) VAF time series: individual members + ensemble mean")
    ax_a.legend(loc="best", fontsize=9)
    ax_a.grid(alpha=0.2)

    # --- Panel (b): PSD ---
    # Compute PSD for each member, then the mean PSD
    periods_all, psd_all = [], []

    for i in range(n_ctrl):
        p, psd = compute_psd(sle_ctrl[i])
        periods_all.append(p)
        psd_all.append(psd)
    # Interpolate to common period grid (use finest grid among members)
    common_period = np.logspace(np.log10(1.5), np.log10(yr_ctrl[-1] / 2), 512)
    psd_ctrl = np.array([np.interp(common_period, periods_all[j], psd_all[j])
                         for j in range(n_ctrl)])

    periods_all_ssp, psd_all_ssp = [], []
    for i in range(n_ssp):
        p, psd = compute_psd(sle_ssp[i])
        periods_all_ssp.append(p)
        psd_all_ssp.append(psd)
    common_period_ssp = np.logspace(np.log10(1.5), np.log10(yr_ssp[-1] / 2), 512)
    psd_ssp = np.array([np.interp(common_period_ssp, periods_all_ssp[j], psd_all_ssp[j])
                        for j in range(n_ssp)])

    period_lo, period_hi = 1.5, 200.0
    common_period_unified = np.logspace(np.log10(period_lo), np.log10(period_hi), 512)

    psd_ctrl_u = np.array([np.interp(common_period_unified, periods_all[j], psd_all[j])
                           for j in range(n_ctrl)])
    psd_ssp_u = np.array([np.interp(common_period_unified, periods_all_ssp[j], psd_all_ssp[j])
                          for j in range(n_ssp)])

    for i in range(n_ctrl):
        ax_b.plot(common_period_unified, psd_ctrl_u[i], color="C0", alpha=0.15, lw=0.6)
    ax_b.plot(common_period_unified, np.mean(psd_ctrl_u, axis=0), color="C0", lw=2.2,
              label="CTRL mean PSD")

    for i in range(n_ssp):
        ax_b.plot(common_period_unified, psd_ssp_u[i], color="C3", alpha=0.15, lw=0.6)
    ax_b.plot(common_period_unified, np.mean(psd_ssp_u, axis=0), color="C3", lw=2.2,
              label="SSP585 mean PSD")

    ylims = ax_b.get_ylim()
    ax_b.axvspan(period_lo, 2.0, color="C1", alpha=0.08, zorder=0)
    ax_b.axvspan(2.0, 20.0, color="C2", alpha=0.08, zorder=0)
    ax_b.axvspan(20.0, period_hi, color="C4", alpha=0.08, zorder=0)

    ax_b.text(1.6, ylims[1] * 0.85 if ylims[1] > 0 else 1e-4, "inter-\nannual",
              fontsize=7, color="C1", ha="left", va="top", fontweight="bold")
    ax_b.text(7.0, ylims[1] * 0.85 if ylims[1] > 0 else 1e-4, "decadal",
              fontsize=7, color="C2", ha="center", va="top", fontweight="bold")
    ax_b.text(60.0, ylims[1] * 0.85 if ylims[1] > 0 else 1e-4, "multi-\ndecadal",
              fontsize=7, color="C4", ha="center", va="top", fontweight="bold")

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("period (years)")
    ax_b.set_ylabel("PSD (mm² yr)")
    ax_b.set_title("(b) VAF power spectral density — ice sheet is a low-pass filter")
    ax_b.legend(loc="lower left", fontsize=9)
    ax_b.grid(alpha=0.2, which="both")
    ax_b.set_xlim(period_lo, period_hi)

    fig.suptitle("The Antarctic Ice Sheet as a Low-Pass Filter", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(args.out_dir, "fig_lowpass_filter.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {out_path}")


if __name__ == "__main__":
    main()
