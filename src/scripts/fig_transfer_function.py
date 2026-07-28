#!/usr/bin/env python3
"""
fig_transfer_function.py — Ice-sheet memory timescale via VAF autocorrelation.

Panel (a): autocorrelation function per ensemble. Panel (b): e-folding timescale bar chart.
Panel (c): VAF power spectrum. Internal variability isolated by subtracting ensemble mean.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=eio.default_ensembles_root())
    p.add_argument("--min-years", type=float, default=50.0)
    p.add_argument("--max-lag", type=int, default=100,
                   help="Maximum lag in years for autocorrelation")
    p.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    return p.parse_args()


def autocorrelation(timeseries, max_lag=100):
    x = timeseries - np.nanmean(timeseries)
    # Detrend (remove linear trend) so ACF captures variability, not the forced signal
    t = np.arange(len(x))
    ok = np.isfinite(x)
    if ok.sum() > 2:
        coeffs = np.polyfit(t[ok], x[ok], 1)
        x = x - np.polyval(coeffs, t)
    n = len(x)
    c0 = np.var(x)
    if c0 < 1e-30:
        return np.ones(max_lag + 1)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (c0 * n)
    return acf[:max_lag + 1]


def efolding_timescale(acf, dt=1.0, max_lag=None):
    threshold = 1.0 / np.e
    if max_lag is None:
        max_lag = len(acf) - 1
    idx = np.where(acf < threshold)[0]
    if idx.size == 0:
        return np.nan  # ACF never drops below 1/e (trend-dominated)
    i = idx[0]
    if i == 0:
        return 0.0
    slope = acf[i] - acf[i - 1]
    if abs(slope) < 1e-30:
        return float(i * dt)
    frac = (acf[i - 1] - threshold) / (acf[i - 1] - acf[i])
    return float((i - 1 + frac) * dt)


def load_sle_timeseries(ensemble_dir, include, min_years=50):
    ds = eio.load_ensemble_globalstats(
        ensemble_dir,
        variables=["volumeAboveFloatation", "daysSinceStart"],
        include=include, min_years=min_years, align="union",
    )
    vaf = ds["volumeAboveFloatation"]
    sle = xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, reference="first"), vaf)
    sle.name = "sle_mm"
    yr = ds["year"].values
    return sle, yr


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ensembles = {
        "CTRL":           (os.path.join(args.root, "CTRL"),                     r"^CTRL_\d+$"),
        "SSP585":         (os.path.join(args.root, "SSP585"),                   r"^SSP585_\d+$"),
        "SSP126":         (os.path.join(args.root, "SSP126"),                   r"^SSP126_\d+$"),
        "varScaled10x":   (os.path.join(args.root, "SSP585_varScaled10x"),     r"^SSP585_\d+$"),
    }

    colors = {"CTRL": "C2", "SSP585": "C0", "SSP126": "C1", "varScaled10x": "C3"}
    styles = {"CTRL": "-", "SSP585": "-", "SSP126": "--", "varScaled10x": "-."}

    acf_ensemble = {}   # label -> (max_lag+1,) ensemble-mean ACF
    efolding = {}       # label -> list of per-member e-folding timescales
    psd_data = {}       # label -> (freq, psd_mean)

    for label, (ens_dir, inc) in ensembles.items():
        print(f"\n  Loading {label}...")
        sle, yr = load_sle_timeseries(ens_dir, inc, args.min_years)
        n_mem = sle.sizes["member"]
        # isolate INTERNAL VARIABILITY: subtract the ensemble mean (the forced response).
        # ACF/PSD of the raw VAF is dominated by the forced trend, not the ice-sheet memory.
        ens_mean = sle.mean(dim="member")
        dt = np.median(np.diff(yr))
        max_lag = min(args.max_lag, len(yr) // 2)
        print(f"    {n_mem} members, year {yr[0]:.1f}..{yr[-1]:.1f}, dt={dt:.2f} yr")

        # ---- autocorrelation ----
        acfs = []
        efolds = []
        for m in range(n_mem):
            ts = (sle.isel(member=m) - ens_mean).values   # anomaly = internal variability
            ok = np.isfinite(ts)
            if ok.sum() < 20:
                continue
            ts_clean = ts[ok]
            acf = autocorrelation(ts_clean, max_lag)
            acfs.append(acf)
            efolds.append(efolding_timescale(acf, dt))
        acf_ensemble[label] = np.mean(acfs, axis=0)
        efolding[label] = efolds
        print(f"    ACF computed for {len(acfs)} members; "
              f"e-folding = {np.nanmedian(efolds):.1f} ± {np.nanstd(efolds):.1f} yr")

        # ---- PSD via Welch ----
        psds = []
        for m in range(n_mem):
            ts = (sle.isel(member=m) - ens_mean).values   # anomaly = internal variability
            ok = np.isfinite(ts)
            if ok.sum() < 50:
                continue
            ts_clean = ts[ok] - np.nanmean(ts[ok])
            nperseg = min(len(ts_clean) // 4, 128)
            f, pxx = welch(ts_clean, fs=1.0 / dt, nperseg=nperseg, noverlap=nperseg // 2)
            psds.append(pxx)
        if psds:
            freq = f
            psd_data[label] = (freq, np.mean(psds, axis=0))

    lag_years = np.arange(len(list(acf_ensemble.values())[0])) * dt

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): autocorrelation
    ax = axes[0]
    for label, acf in acf_ensemble.items():
        ax.plot(lag_years, acf, color=colors[label], ls=styles[label], lw=1.8, label=label)
    ax.axhline(1.0 / np.e, color="0.5", lw=0.8, ls=":", label="1/e threshold")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lag (years)")
    ax.set_ylabel("VAF autocorrelation")
    ax.set_title("(a) VAF autocorrelation\n(ensemble mean)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, args.max_lag)
    ax.grid(alpha=0.2)

    # Panel (b): e-folding timescale bar chart
    ax = axes[1]
    labels_list = list(efolding.keys())
    medians = [np.nanmedian(efolding[l]) for l in labels_list]
    spreads = [np.nanstd(efolding[l]) for l in labels_list]
    bars = ax.bar(labels_list, medians, yerr=spreads, capsize=4,
                  color=[colors[l] for l in labels_list], edgecolor="k", linewidth=0.5)
    ax.axhline(50, color="0.5", lw=0.8, ls="--", label="50 yr threshold")
    ax.set_ylabel("e-folding timescale (years)")
    ax.set_title("(b) Ice-sheet memory timescale\n(VAF ACF → 1/e)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    for i, (m, s) in enumerate(zip(medians, spreads)):
        ax.text(i, m + s + 2, f"{m:.0f} yr", ha="center", fontsize=8)

    # Panel (c): PSD
    ax = axes[2]
    for label, (freq, psd) in psd_data.items():
        mask = freq > 0
        ax.loglog(1.0 / freq[mask], psd[mask], color=colors[label], ls=styles[label],
                  lw=1.5, label=label)
    ax.set_xlabel("period (years)")
    ax.set_ylabel("PSD (mm² · yr)")
    ax.set_title("(c) VAF power spectrum\n(all ensembles)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, which="both")

    fig.suptitle("Ice-sheet transfer function: memory timescale and frequency response",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "transfer_function.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
