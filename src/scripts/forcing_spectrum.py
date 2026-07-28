#!/usr/bin/env python3
"""
forcing_spectrum.py — spectral content of the AISLENS variability forcing.

Computes Welch PSD for each per-realization time series, averages across realizations,
and integrates variance into seasonal (<1.5 yr), interannual (1.5-8 yr), decadal
(8-30 yr), and multidecadal (>30 yr) bands.

Author: Shivaprakash Muruganandham
"""
from __future__ import annotations

import os
import sys
import glob
import argparse

import numpy as np
import xarray as xr
from scipy import signal as scipy_signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio  # noqa: F401 (kept for path/consistency; not strictly needed)

DEFAULT_BANDS = {          # name: (min_period_yr, max_period_yr)
    "seasonal": (0.0, 1.5),
    "interannual": (1.5, 8.0),
    "decadal": (8.0, 30.0),
    "multidecadal": (30.0, np.inf),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forcing-dir", required=True,
                   help="Directory of per-realization *_ts.nc series")
    p.add_argument("--pattern", default="*_ts.nc",
                   help="Glob for the series files (default *_ts.nc)")
    p.add_argument("--var", default="floatingBasalMassBalAdjustment")
    p.add_argument("--dt-months", type=float, default=1.0,
                   help="Sampling interval in months (default 1 = monthly)")
    p.add_argument("--detrend", default="constant", choices=["constant", "linear"],
                   help="Welch detrending (default constant = remove mean only)")
    p.add_argument("--out-fig-dir", default=None)
    return p.parse_args()


def load_series(path, var):
    ds = xr.open_dataset(path, decode_times=False)
    if var not in ds:
        # single-variable file fallback
        var = [v for v in ds.data_vars if ds[v].ndim == 1][0]
    x = np.asarray(ds[var].values, dtype=float).squeeze()
    return x[np.isfinite(x)]


def band_variance(freqs_per_yr, psd, bands):
    """Integrate PSD over each period band; return {name: variance, ...} and total.

    Uses interpolation on the cumulative integral so the bands PARTITION the spectrum
    exactly (they sum to the total) -- a plain trapz over masked sub-arrays drops the
    trapezoid segment straddling each band edge and undercounts by ~10%.
    """
    order = np.argsort(freqs_per_yr)
    f = np.asarray(freqs_per_yr)[order]
    p = np.asarray(psd)[order]
    # cumulative trapezoidal integral of the PSD, cum[i] = int_{f0}^{f_i} psd df
    cum = np.concatenate([[0.0], np.cumsum(np.diff(f) * (p[1:] + p[:-1]) / 2.0)])
    total = float(cum[-1])

    def C(x):  # cumulative integral up to frequency x (clamped to the resolved range)
        return float(np.interp(np.clip(x, f[0], f[-1]), f, cum))

    out = {}
    for name, (pmin, pmax) in bands.items():
        # period in [pmin, pmax)  <=>  freq in (1/pmax, 1/pmin]
        fmin = 1.0 / pmax if np.isfinite(pmax) and pmax > 0 else 0.0
        fmax = 1.0 / pmin if pmin > 0 else np.inf
        out[name] = max(0.0, C(fmax) - C(fmin))
    return out, total


def main():
    args = parse_args()
    fig_dir = args.out_fig_dir or os.path.join(os.getcwd(), "reports")
    os.makedirs(fig_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.forcing_dir, args.pattern)))
    if not files:
        sys.exit(f"No files matching {args.pattern} in {args.forcing_dir}")
    print(f"Found {len(files)} realization series.")

    fs_per_yr = 12.0 / args.dt_months  # samples per year (monthly -> 12)
    psds, freqs = [], None
    for fpath in files:
        x = load_series(fpath, args.var)
        if x.size < 64:
            print(f"  skip (too short): {os.path.basename(fpath)}")
            continue
        nperseg = min(len(x), 1024)
        f, pxx = scipy_signal.welch(x, fs=fs_per_yr, nperseg=nperseg,
                                    detrend=args.detrend)
        freqs = f
        psds.append(pxx)
    if not psds:
        sys.exit("No usable series.")
    psd_mean = np.mean(np.vstack(psds), axis=0)

    # drop the zero-frequency bin for period conversion
    nz = freqs > 0
    freqs_nz, psd_nz = freqs[nz], psd_mean[nz]
    periods = 1.0 / freqs_nz

    bands_var, total = band_variance(freqs_nz, psd_nz, DEFAULT_BANDS)
    print("\nVariance fraction by band (of resolved, f>0 variance):")
    fracs = {}
    for name in DEFAULT_BANDS:
        frac = bands_var[name] / total if total > 0 else np.nan
        fracs[name] = frac
        print(f"  {name:13s}: {100*frac:5.1f}%")

    # ---- Figure: PSD vs period + band-fraction bar ----
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].loglog(periods, psd_nz, "C0")
    for name, (pmin, pmax) in DEFAULT_BANDS.items():
        lo = pmin if pmin > 0 else periods.min()
        hi = pmax if np.isfinite(pmax) else periods.max()
        axs[0].axvspan(lo, hi, alpha=0.08)
    axs[0].set_xlabel("period (years)"); axs[0].set_ylabel("PSD")
    axs[0].set_title(f"Mean PSD of forcing '{args.var}'\n({len(psds)} realizations, "
                     f"{os.path.basename(args.forcing_dir.rstrip('/'))})")
    axs[0].invert_xaxis()  # long periods on the right

    axs[1].bar(list(fracs.keys()), [100 * v for v in fracs.values()], color="C1")
    axs[1].set_ylabel("% of resolved variance")
    axs[1].set_title("Variance fraction by band")
    for i, v in enumerate(fracs.values()):
        axs[1].text(i, 100 * v + 1, f"{100*v:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    # tag from the last two path components so ".../vargen_realizations-ssn/ts"
    # doesn't collapse to just "ts" and clobber other runs
    parts = [p for p in args.forcing_dir.rstrip("/").split(os.sep) if p][-2:]
    tag = "_".join(parts) + ("_" + args.detrend if args.detrend != "constant" else "")
    out = os.path.join(fig_dir, f"forcing_spectrum_{tag}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nFigure -> {out}")

    # ---- verdict ----
    seas = fracs.get("seasonal", 0)
    low = fracs.get("decadal", 0) + fracs.get("multidecadal", 0)
    print("\nVERDICT:")
    if seas > 0.6:
        print(f"  Seasonal band holds {100*seas:.0f}% of the variance -> the low-frequency")
        print("  content the ice sheet responds to is weak. This explains the small")
        print("  centennial spread, and means REPEATING a 300-yr block would inject a")
        print("  spurious 300-yr period into an already-weak low-frequency band.")
        print("  -> Regenerating a genuine long realization gains little on multidecadal")
        print("     power; consider leaning on the amplitude/threshold framing instead.")
    elif low > 0.2:
        print(f"  Decadal+multidecadal bands hold {100*low:.0f}% of the variance -> there IS")
        print("  low-frequency power worth resolving. For the 2300-3000 extension, prefer")
        print("  regenerating a genuine long realization (Option 3) over repeating blocks.")
    else:
        print("  Mixed spectrum; inspect the PSD figure before deciding the extension route.")


if __name__ == "__main__":
    main()
