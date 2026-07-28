#!/usr/bin/env python3
"""
plot_forcing_diagnostics.py — time-series and power-spectrum comparison of forcing files.

Auto-discovers *_ts.nc files in a diagnostics folder, derives scenario/length labels,
and produces time series (annual + zoom panels) and Welch PSD plots with band-power tables.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

BANDS = [(2, 10, "interann"), (10, 30, "decadal"), (30, 100, "multidec"), (100, 1e9, "centennial+")]


def annual(a):
    n = len(a) // 12 * 12
    return a[:n].reshape(-1, 12).mean(1)


def parse_label(path):
    """Filename prefix -> (key, scenario, colour, is_offset)."""
    stem = os.path.basename(path).split("__")[0]
    key = stem.replace("vargen_realizations-", "")
    low = key.lower()
    if "ssp585" in low:
        scen, color = "SSP585", "C3"
    elif "ssp126" in low:
        scen, color = "SSP126", "C0"
    else:
        scen, color = "CTRL", "0.4"
    return key, scen, color, ("offset" in low)


def load_series(path, var):
    ds = xr.open_dataset(path, decode_times=False)
    if var not in ds:
        ds.close()
        return None
    a = np.asarray(ds[var].values, dtype=float)
    ds.close()
    return a  # monthly


def _detrend_smooth(x, win_yr):
    """Subtract a centered running-mean trend (window in years, annual data) to isolate
    variability. Needed for the SSP scenarios, whose non-linear melt ramp swamps a plain
    linear detrend and shows up as spurious low-frequency 'oscillation' power."""
    w = int(max(3, win_yr))
    if w >= len(x):
        return x - np.nanmean(x)
    k = np.ones(w) / w
    trend = np.convolve(x, k, mode="same")
    # fix convolution edge roll-off with nearest valid trend value
    h = w // 2
    trend[:h] = trend[h]; trend[-h:] = trend[-h - 1]
    return x - trend


def psd(x, fs=1.0, detrend_window=0):
    if detrend_window and detrend_window > 0:
        x = _detrend_smooth(x, detrend_window)
        f, p = welch(x, fs=fs, nperseg=min(len(x), max(64, len(x) // 3)), detrend="constant")
    else:
        x = x - np.nanmean(x)
        f, p = welch(x, fs=fs, nperseg=min(len(x), max(64, len(x) // 3)), detrend="linear")
    return f, p


def band_fracs(x, detrend_window=0):
    f, p = psd(x, detrend_window=detrend_window)
    per = 1.0 / np.where(f > 0, f, np.nan)
    tot = np.nansum(p[f > 0])
    out = {}
    for lo, hi, nm in BANDS:
        m = (per >= lo) & (per < hi)
        out[nm] = 100.0 * np.nansum(p[m]) / tot if tot else np.nan
    return out


def discover(folder, var):
    """Return list of dicts sorted CTRL, SSP126, SSP585 then by length."""
    items = []
    for f in sorted(glob.glob(os.path.join(folder, "*_ts.nc"))):
        a = load_series(f, var)
        if a is None or len(a) < 24:
            print(f"  [skip] {os.path.basename(f)} (no '{var}' or too short)")
            continue
        key, scen, color, off = parse_label(f)
        ya = annual(a)
        yrs = len(ya)
        ls = ":" if off else ("-" if yrs <= 350 else "--")
        lab = f"{scen} {yrs}yr{' -OFF' if off else ''}"
        items.append(dict(key=key, scen=scen, color=color, ls=ls, label=lab,
                          monthly=a, annual=ya, yrs=yrs))
    order = {"CTRL": 0, "SSP126": 1, "SSP585": 2}
    items.sort(key=lambda d: (order.get(d["scen"], 9), d["yrs"], "off" in d["key"].lower()))
    return items


def fig_timeseries(items, var, zoom, out):
    nz = len(zoom)
    fig, axes = plt.subplots(1 + (nz + 1) // 2 if nz else 1, 2 if nz else 1,
                             figsize=(13, 4.2 * (1 + (nz + 1) // 2)), squeeze=False)
    axfull = axes[0, 0]
    for it in items:
        axfull.plot(np.arange(it["yrs"]), it["annual"], color=it["color"],
                    ls=it["ls"], lw=1.2, label=it["label"])
    axfull.axhline(0, color="k", lw=.5)
    axfull.set_title(f"{var} — annual mean, full extent")
    axfull.set_xlabel("year"); axfull.set_ylabel(var); axfull.grid(alpha=.3)
    axfull.legend(fontsize=8)
    if nz:
        axes[0, 1].axis("off")  # keep the full-extent panel on its own row
        flat = [axes[r, c] for r in range(1, axes.shape[0]) for c in range(2)]
        for ax, zy in zip(flat, zoom):
            for it in items:
                t = np.arange(len(it["monthly"])) / 12.0
                m = t <= zy
                ax.plot(t[m], it["monthly"][m], color=it["color"], ls=it["ls"], lw=1.0, label=it["label"])
            ax.axhline(0, color="k", lw=.5)
            ax.set_title(f"zoom: first {zy} yr (monthly)")
            ax.set_xlabel("year"); ax.set_ylabel(var); ax.grid(alpha=.3); ax.legend(fontsize=7)
        for ax in flat[nz:]:
            ax.axis("off")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"saved {out}")


def fig_spectra(items, var, out, detrend_window=0):
    fig, ax = plt.subplots(figsize=(9, 6))
    for it in items:
        f, p = psd(it["annual"], detrend_window=detrend_window); m = f > 0
        ax.loglog(1 / f[m], p[m], color=it["color"], ls=it["ls"], lw=1.5, label=it["label"])
    for lo, hi, nm in BANDS[:-1]:
        ax.axvspan(lo, hi, alpha=0.05, color="k")
    dt = f"{detrend_window}-yr running-mean detrend (variability only)" if detrend_window else "linear-detrended"
    ax.set_xlabel("period (yr)"); ax.set_ylabel(f"PSD (annual, {dt})")
    ax.set_title(f"{var} — power spectra"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"saved {out}")
    print(f"\nband power % (of >2yr variance){' [smooth-detrended]' if detrend_window else ''}:")
    for it in items:
        bf = band_fracs(it["annual"], detrend_window=detrend_window)
        print(f"  {it['label']:18s} " + "  ".join(f"{k}={v:4.1f}" for k, v in bf.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/processed/diagnostics", help="folder of *_ts.nc files")
    ap.add_argument("--var", default="floatingBasalMassBalAdjustment")
    ap.add_argument("--mode", default="both", choices=["both", "timeseries", "spectra"])
    ap.add_argument("--zoom", default="15,50", help="comma list of zoom windows (yr) for the time series")
    ap.add_argument("--detrend-window", type=int, default=0,
                    help="spectra: if >0, subtract an N-yr running-mean trend before the PSD to isolate "
                         "variability (recommended for SSP scenarios, e.g. 30; 0=linear detrend)")
    ap.add_argument("--out-prefix", default="reports/figures/forcing_diag")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f"no such dir: {args.dir}\n(run forcing_to_ts_array.sbatch and scp the diagnostics folder here)")
    items = discover(args.dir, args.var)
    if not items:
        sys.exit(f"no usable *_ts.nc with '{args.var}' in {args.dir}")
    print(f"{len(items)} file(s): " + ", ".join(it["label"] + f" [{it['key']}]" for it in items))
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    zoom = [float(z) for z in args.zoom.split(",") if z.strip()]
    if args.mode in ("both", "timeseries"):
        fig_timeseries(items, args.var, zoom, args.out_prefix + "_timeseries.png")
    if args.mode in ("both", "spectra"):
        fig_spectra(items, args.var, args.out_prefix + "_spectra.png", detrend_window=args.detrend_window)


if __name__ == "__main__":
    main()
