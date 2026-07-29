#!/usr/bin/env python3
"""
amplitude_response.py — compares SSP585 ensemble spread under 1x vs 10x forcing variability.

Plots mean and 5-95% bands for both ensembles (top) and ensemble spread sigma(t) (bottom).
Prints sigma at the common evaluation horizon.

Author: Shivaprakash Muruganandham
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=eio.default_ensembles_root(),
                   help="ENSEMBLES root dir")
    p.add_argument("--realistic-ensemble", default="SSP585",
                   help="1x (realistic-variability) ensemble sub-dir")
    p.add_argument("--boosted-ensemble", default="SSP585_varScaled10x",
                   help="10x (boosted-variability) ensemble sub-dir")
    p.add_argument("--members", default=r"^SSP585_\d+$",
                   help="Regex to select a clean member subset for BOTH ensembles")
    p.add_argument("--min-years", type=float, default=50.0,
                   help="Drop members whose record spans fewer than this many years")
    p.add_argument("--align", default="union", choices=["union", "intersection"],
                   help="'union' shows the full available span per ensemble "
                        "(members NaN-drop after they end); 'intersection' truncates "
                        "to the shortest member")
    p.add_argument("--out-dir", default=None,
                   help="Figure output dir (default: <root>/figures)")
    return p.parse_args()


def _drop_restart_segments(ensemble_dir, include, max_start_year=5.0):
    """Some ensembles mix full 0..N-year runs with restart-only continuation
    segments whose globalStats.nc starts partway through (e.g. year 200..300).
    ensemble_io's start_at_zero=True would silently relabel such a segment's
    year axis to start at 0, making a late-simulation state look like an early
    one and badly corrupting ensemble statistics. Detect and exclude those
    members up front (by their RAW, un-shifted year0) and return a regex that
    additionally excludes them."""
    members = eio.discover_members(ensemble_dir, include=include)
    bad = []
    for name, path in members:
        ds = eio.to_year_dim(eio.load_member_globalstats(path))
        y0 = float(ds["year"].values[0])
        if y0 > max_start_year:
            bad.append(name)
    if bad:
        print(f"  [amplitude_response] excluding {len(bad)} restart-segment member(s) "
              f"whose globalStats.nc starts at year>{max_start_year:.0f} "
              f"(would be mislabeled as year 0 by start_at_zero): {bad}")
    return bad


def load_sle(root, ensemble, include, min_years, align):
    ens_dir = os.path.join(root, ensemble)
    bad = _drop_restart_segments(ens_dir, include)
    if bad:
        # extend the include regex with a negative lookahead per bad member name
        exclude = "".join(f"(?!^{name}$)" for name in bad)
        include = f"{exclude}{include}"
    ds = eio.load_ensemble_globalstats(
        ens_dir, variables=["volumeAboveFloatation", "daysSinceStart"],
        include=include, min_years=min_years, align=align,
    )
    vaf = ds["volumeAboveFloatation"]
    sle = xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, reference="first"), vaf)
    sle.name = "sle_mm"
    return sle, ds["year"].values


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    sle_1x, yr_1x = load_sle(args.root, args.realistic_ensemble, args.members,
                              args.min_years, args.align)
    sle_10x, yr_10x = load_sle(args.root, args.boosted_ensemble, args.members,
                                args.min_years, args.align)

    n_1x = sle_1x.sizes["member"]
    n_10x = sle_10x.sizes["member"]
    print(f"Realistic (1x)  {args.realistic_ensemble}: {n_1x} members, "
          f"year {yr_1x[0]:.1f}..{yr_1x[-1]:.1f}")
    print(f"Boosted  (10x)  {args.boosted_ensemble}: {n_10x} members, "
          f"year {yr_10x[0]:.1f}..{yr_10x[-1]:.1f}")

    stats_1x = eio.ensemble_stats(sle_1x)
    stats_10x = eio.ensemble_stats(sle_10x)

    # ---- common evaluation horizon: latest year both ensembles still have their
    # full member count (i.e. before any NaN-dropout under 'union' alignment) ----
    count_1x = sle_1x.notnull().sum("member").values
    count_10x = sle_10x.notnull().sum("member").values
    full_1x_years = yr_1x[count_1x == count_1x.max()]
    full_10x_years = yr_10x[count_10x == count_10x.max()]
    horizon = min(full_1x_years[-1], full_10x_years[-1])
    print(f"Common evaluation horizon (both ensembles at full member count): "
          f"year {horizon:.1f}")

    i1 = int(np.argmin(np.abs(yr_1x - horizon)))
    i10 = int(np.argmin(np.abs(yr_10x - horizon)))
    sigma_1x_h = float(stats_1x["std"].values[i1])
    sigma_10x_h = float(stats_10x["std"].values[i10])
    mean_1x_h = float(stats_1x["mean"].values[i1])
    mean_10x_h = float(stats_10x["mean"].values[i10])
    ratio = sigma_10x_h / sigma_1x_h if sigma_1x_h != 0 else np.nan
    print(f"At year {horizon:.1f}: sigma_1x = {sigma_1x_h:.4g} mm SLE "
          f"(mean {mean_1x_h:.4g} mm), sigma_10x = {sigma_10x_h:.4g} mm SLE "
          f"(mean {mean_10x_h:.4g} mm)  ->  sigma ratio (10x/1x) = {ratio:.2f}")

    # ---- figure ----
    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax = axs[0]
    ax.plot(yr_1x, stats_1x["mean"], color="C0", lw=2, label=f"1x mean ({n_1x} mem)")
    ax.fill_between(yr_1x, stats_1x["p05"], stats_1x["p95"], color="C0", alpha=0.2,
                     label="1x 5-95%")
    ax.plot(yr_1x, stats_1x["median"], color="C0", lw=1, ls=":", alpha=0.7)
    ax.plot(yr_10x, stats_10x["mean"], color="C3", lw=2, label=f"10x mean ({n_10x} mem)")
    ax.fill_between(yr_10x, stats_10x["p05"], stats_10x["p95"], color="C3", alpha=0.2,
                     label="10x 5-95%")
    ax.plot(yr_10x, stats_10x["median"], color="C3", lw=1, ls=":", alpha=0.7)
    ax.axvline(horizon, color="0.5", lw=0.8, ls="--")
    ax.set_ylabel("VAF -> SLE (mm, rise positive)")
    ax.set_title("Amplitude response: realistic (1x) vs boosted (10x) SSP585 forcing variability")
    ax.legend(loc="upper left", fontsize=9)

    ax2 = axs[1]
    ax2.plot(yr_1x, stats_1x["std"], color="C0", lw=2, label="sigma 1x")
    ax2.plot(yr_10x, stats_10x["std"], color="C3", lw=2, label="sigma 10x")
    ax2.axvline(horizon, color="0.5", lw=0.8, ls="--",
                label=f"common horizon (yr {horizon:.0f})")
    ax2.set_xlabel("year")
    ax2.set_ylabel("ensemble spread sigma (mm SLE)")
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "amplitude_response_1x_vs_10x.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure -> {out_path}")


if __name__ == "__main__":
    main()
