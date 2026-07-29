#!/usr/bin/env python3
"""
fig_cv_timeseries.py — how big is the internal-variability spread, absolute and relative?
  (a) ensemble spread sigma (mm SLE) vs year for CTRL, SSP126, SSP585 — the absolute
      noise floor (CTRL = variability with no forced trend).
  (b) coefficient of variation CV = sigma / |ensemble mean| (%) for the forced scenarios
      (SSP126, SSP585). CTRL is omitted: its mean signal ~ 0, so CV is undefined.

Headline: the ensemble spread is only a few percent (or less) of the mean sea-level signal.

Author: Shivaprakash Muruganandham (2026-07-22)
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from amplitude_response import load_sle

SCEN = [("CTRL", r"^CTRL_\d+$", "C2"),
        ("SSP126", r"^SSP126_\d+$", "C0"),
        ("SSP585", r"^SSP585_\d+$", "C3")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--min-years", type=float, default=50.0)
    ap.add_argument("--align", default="union")
    ap.add_argument("--out-dir", default="reports")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ens, inc, col in SCEN:
        try:
            sle, yr = load_sle(a.root, ens, inc, a.min_years, a.align)
        except Exception as e:
            print(f"{ens}: skip ({e})"); continue
        mean = sle.mean("member").values
        sig = sle.std("member", ddof=1).values
        n = sle.sizes["member"]
        axA.plot(yr, sig, col, lw=2, label=f"{ens} (n={n})")
        if ens != "CTRL":
            ok = np.abs(mean) > 1.0                       # only where a real signal exists
            cv = np.full_like(sig, np.nan)
            cv[ok] = 100.0 * sig[ok] / np.abs(mean[ok])
            axB.plot(yr, cv, col, lw=2, label=ens)
            j = np.where(np.isfinite(cv))[0]
            if j.size:
                print(f"{ens}: sigma@end={sig[j[-1]]:.2f} mm, mean@end={mean[j[-1]]:.0f} mm, "
                      f"CV@end={cv[j[-1]]:.2f}%")

    axA.set_xlabel("year"); axA.set_ylabel("ensemble spread σ (mm SLE)")
    axA.set_title("(a) absolute spread — the noise floor"); axA.legend(); axA.grid(alpha=0.3)
    axB.set_xlabel("year"); axB.set_ylabel("CV = σ / |mean|  (%)")
    axB.set_title("(b) spread as a fraction of the signal"); axB.legend(); axB.grid(alpha=0.3)
    fig.suptitle("Internal variability: absolute spread and spread relative to the mean signal")
    fig.tight_layout()
    out = os.path.join(a.out_dir, "cv_timeseries.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
