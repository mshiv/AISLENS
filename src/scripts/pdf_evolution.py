#!/usr/bin/env python3
"""
pdf_evolution.py — ridgeline plot of ensemble SLE distribution over time.

Stacked kernel-density curves of VAF->SLE (mm) at multiple years, with mean/skewness
printed per year. Detects widening or skewing as the ice sheet approaches nonlinear
regimes (Robel et al. 2019).

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
from scipy.stats import gaussian_kde, skew

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=eio.default_ensembles_root(),
                   help="ENSEMBLES root dir")
    p.add_argument("--ensemble", default="SSP585_varScaled10x",
                   help="Ensemble sub-directory name")
    p.add_argument("--members", default=r"^SSP585_\d+$",
                   help="Regex to select a clean member subset")
    p.add_argument("--min-years", type=float, default=50.0,
                   help="Drop members whose record spans fewer than this many years")
    p.add_argument("--align", default="union", choices=["union", "intersection"],
                   help="'union' keeps the full span (members NaN beyond their own "
                        "end); 'intersection' truncates to the shortest member")
    p.add_argument("--years", default="25,50,75,100,150",
                   help="Comma-separated years at which to show the distribution "
                        "(clamped to the available range)")
    p.add_argument("--year-step", type=float, default=None,
                   help="If set, show distributions every N years up to the available "
                        "max (overrides --years), e.g. --year-step 10")
    p.add_argument("--out-dir", default=None,
                   help="Figure output dir (default: <root>/figures)")
    return p.parse_args()


def _restart_segment_members(ensemble_dir, include, max_start_year=5.0):
    """Flag members whose globalStats.nc is a restart-only continuation segment
    (raw year axis starts well after 0) -- these would be mislabeled as
    'early simulation' by start_at_zero and corrupt the per-year distribution."""
    members = eio.discover_members(ensemble_dir, include=include)
    bad = []
    for name, path in members:
        ds = eio.to_year_dim(eio.load_member_globalstats(path))
        if float(ds["year"].values[0]) > max_start_year:
            bad.append(name)
    return bad


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    ens_dir = os.path.join(args.root, args.ensemble)
    bad = _restart_segment_members(ens_dir, args.members)
    include = args.members
    if bad:
        print(f"  [pdf_evolution] excluding {len(bad)} restart-segment member(s): {bad}")
        include = "".join(f"(?!^{n}$)" for n in bad) + include

    ds = eio.load_ensemble_globalstats(
        ens_dir, variables=["volumeAboveFloatation", "daysSinceStart"],
        include=include, min_years=args.min_years, align=args.align,
    )
    vaf = ds["volumeAboveFloatation"]
    sle = xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, reference="first"), vaf)
    year = ds["year"].values
    n_members = sle.sizes["member"]
    print(f"Loaded {n_members} members for {args.ensemble}; year "
          f"{year[0]:.1f}..{year[-1]:.1f}")

    avail_max = year[-1]
    if args.year_step:
        req_years = list(np.arange(args.year_step, avail_max + 1e-6, args.year_step))
    else:
        req_years = [float(y) for y in args.years.split(",")]
    plot_years = sorted(set(min(max(y, year[0]), avail_max) for y in req_years))
    if any(y != r for y, r in zip(plot_years, sorted(req_years))):
        print(f"  [pdf_evolution] requested years {req_years} clamped to available "
              f"range -> using {plot_years}")

    # ---- gather per-year member samples + mean/skew ----
    samples_by_year = []
    for y in plot_years:
        i = int(np.argmin(np.abs(year - y)))
        vals = sle.isel(year=i).values
        vals = vals[np.isfinite(vals)]
        samples_by_year.append((year[i], vals))

    print("Ensemble MEAN and skewness of VAF->SLE distribution by year:")
    for yr_actual, vals in samples_by_year:
        m = np.mean(vals)
        sk = skew(vals) if len(vals) > 2 else np.nan
        print(f"  year {yr_actual:6.1f}: n={len(vals):3d}  mean={m:8.3f} mm SLE  "
              f"std={np.std(vals):7.3f}  skewness={sk:7.3f}")

    # ---- ridgeline figure ----
    n_rows = len(samples_by_year)
    fig, ax = plt.subplots(figsize=(9, max(6.0, 0.8 * n_rows + 1.5)))
    cmap = plt.get_cmap("viridis")
    row_h = 1.0
    overlap = 0.55  # ridge vertical offset fraction of its own peak height

    y_ticks, y_labels = [], []
    # global x-range across all years' samples, plus small padding
    all_vals = np.concatenate([v for _, v in samples_by_year])
    xpad = 0.05 * (all_vals.max() - all_vals.min() + 1e-9)
    x_grid = np.linspace(all_vals.min() - xpad, all_vals.max() + xpad, 400)

    for row, (yr_actual, vals) in enumerate(samples_by_year):
        baseline = row * row_h
        color = cmap(row / max(1, n_rows - 1))
        if len(vals) > 1 and np.std(vals) > 0:
            kde = gaussian_kde(vals)
            dens = kde(x_grid)
            dens = dens / dens.max() * row_h * overlap * 3  # scale for visibility
        else:
            dens = np.zeros_like(x_grid)
        ax.fill_between(x_grid, baseline, baseline + dens, color=color, alpha=0.6,
                         zorder=n_rows - row)
        ax.plot(x_grid, baseline + dens, color=color, lw=1.2, zorder=n_rows - row)
        # member rug + mean marker
        ax.plot(vals, np.full_like(vals, baseline), "|", color="k", ms=6, alpha=0.5,
                 zorder=n_rows - row + 0.5)
        m = np.mean(vals)
        ax.plot([m], [baseline], marker="o", color="k", ms=6, zorder=n_rows + 1)
        y_ticks.append(baseline)
        y_labels.append(f"yr {yr_actual:.0f}\nn={len(vals)}, mean={m:.0f}")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("VAF -> SLE (mm, rise positive)")
    ax.set_title(f"{args.ensemble}: ensemble distribution of SLE over time "
                 f"({n_members} members)\nblack dot = ensemble mean, | = individual members")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{args.ensemble}_pdf_evolution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure -> {out_path}")


if __name__ == "__main__":
    main()
