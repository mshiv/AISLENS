#!/usr/bin/env python3
"""
fig_full_ensemble_topline.py — Topline figures for the re-stitched SSP126-FULL /
SSP585-FULL ensembles (base run 2000-2300 + restart extension 2300+, concatenated
by concat_restart_ensembles.py).

Produces, into reports/figures/full-ensembles/:
  1. full_mean_spread_VAF.png    — 2 panels: (a) ensemble sigma vs year,
                                    (b) ensemble mean vs year, for SSP126-FULL and
                                    SSP585-FULL over their full span (VAF -> mm SLE).
  2. full_member_trajectories.png — per-member VAF (mm SLE) trajectories for both
                                    scenarios, with the 2300 seam marked by a
                                    vertical dashed line.

METHOD: members have irregular, differing output cadence (e.g. sub-day spacing
near some restarts, ~monthly elsewhere). Computing mean/sigma on the raw merged
time axis produces spurious spikes wherever only 1-2 members happen to have a
timestamp. So every member is first interpolated onto a COMMON ANNUAL GRID
(1 yr steps, no extrapolation past a member's own record), and the ensemble
mean/sigma are masked to NaN wherever fewer than --min-members (default 3)
members have data at that year.

CAVEAT (see docs/aislens-forcing-seam-2300 memory / aislens-setup-assessment.md):
post-2300 extensions were generated with a DIFFERENT (1000-yr) stochastic forcing
generator whose variance is ~3.3x the pre-2300 (300-yr) generator's. The forcing
baseline was offset-corrected so the MEAN is continuous across the 2300 seam, but
the VARIANCE is not. Any change in ensemble SPREAD after 2300 is therefore
confounded by greater forcing and must NOT be read as an ice-dynamical result.
This script marks the seam on every plot but does not attempt to correct for the
variance discontinuity.
"""
from __future__ import annotations
import os
import sys
import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio

SEAM_YEAR = 300.0  # run starts at year 0 = 2000; base ends 2299-12, ext begins 2300 -> ~yr 300

SCEN = [
    ("SSP126-FULL", r"^SSP126-FULL_\d+$", "#0072B2"),
    ("SSP585-FULL", r"^SSP585-FULL_\d+$", "#D55E00"),
]

CAVEAT = (
    "CAVEAT: post-2300 extensions use a DIFFERENT (1000-yr) stochastic forcing generator\n"
    "with ~3.3x the pre-2300 forcing variance. Mean is offset-corrected & continuous across\n"
    "the seam; ensemble SPREAD is NOT -- any post-2300 change in spread is confounded by\n"
    "greater forcing, not an ice-dynamical result."
)


def load_regridded(root, ensemble, include, variable="volumeAboveFloatation",
                   regrid_dt=1.0, min_members=3):
    """Load an ensemble variable via ensemble_io, then regrid every member onto a
    common annual grid before any ensemble statistic is taken (see module docstring).

    Returns (grid_years, vals) where vals has shape (member, len(grid_years)), and
    also (member_names, per_member_raw) where per_member_raw is a list of
    (years, values) tuples on each member's OWN native (irregular) axis, for the
    trajectory plot.
    """
    ens_dir = os.path.join(root, ensemble)
    ds = eio.load_ensemble_globalstats(
        ens_dir, variables=[variable, "daysSinceStart"],
        include=include, align="union",
    )
    names = list(ds["member"].values)
    years_union = ds["year"].values
    vals_union = np.asarray(ds[variable].values, dtype=float)  # (member, time) on union axis
    ds.close()

    n_members = len(names)

    # Per-member native trajectories (for the trajectory plot): reload individually
    # so each member keeps its own (non-interpolated) native cadence.
    members = eio.discover_members(ens_dir, stats_filename="globalStats.nc", include=include)
    per_member_raw = []
    for name, path in members:
        d = eio.to_year_dim(eio.load_member_globalstats(path))
        if variable not in d:
            continue
        per_member_raw.append((name, d["year"].values, np.asarray(d[variable].values, dtype=float)))

    # Regrid the union-aligned array onto a regular annual grid.
    grid = np.arange(0.0, np.nanmax(years_union) + 1e-9, regrid_dt)
    out = np.full((n_members, grid.size), np.nan)
    for m in range(n_members):
        ok = np.isfinite(vals_union[m]) & np.isfinite(years_union)
        if ok.sum() < 2:
            continue
        out[m] = np.interp(grid, years_union[ok], vals_union[m][ok], left=np.nan, right=np.nan)

    counts = np.sum(np.isfinite(out), axis=0)
    out[:, counts < min_members] = np.nan

    return grid, out, names, per_member_raw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--out-dir",
                    default="/Users/smurugan9/research/aislens/AISLENS/reports/figures/full-ensembles")
    ap.add_argument("--regrid-dt", type=float, default=1.0,
                    help="common annual grid spacing in years (default: 1.0)")
    ap.add_argument("--min-members", type=int, default=3,
                    help="minimum members required at a grid year for mean/sigma to be plotted")
    ap.add_argument("--wiki-dest",
                    default="/Users/smurugan9/Documents/vaults/shadowfax-wiki/raw/assets/aislens-chapter",
                    help="destination dir to copy PNGs into (set to '' to skip)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(CAVEAT)
    print()

    # ---- gather per-scenario data ----
    scen_data = {}
    for ens, inc, col in SCEN:
        grid, vals, names, per_member_raw = load_regridded(
            args.root, ens, inc, variable="volumeAboveFloatation",
            regrid_dt=args.regrid_dt, min_members=args.min_members,
        )
        sle = eio.vaf_to_sle_mm(vals, reference="first")  # (member, grid) mm SLE, loss -> +
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(sle, axis=0)
            sigma = np.nanstd(sle, axis=0, ddof=1)
        scen_data[ens] = dict(grid=grid, sle=sle, mean=mean, sigma=sigma,
                              names=names, col=col, per_member_raw=per_member_raw)
        n = len(names)
        end_yr = grid[np.isfinite(mean)][-1] if np.any(np.isfinite(mean)) else np.nan
        print(f"{ens}: n_members={n}  span=0-{end_yr:.0f}yr  members={names}")

        # report mean/sigma at the 2300 seam and at the end of the run
        def _at(year_target):
            idx = np.argmin(np.abs(grid - year_target))
            return grid[idx], mean[idx], sigma[idx]
        gy, gm, gs = _at(SEAM_YEAR)
        print(f"    at seam  (yr~{gy:.0f}, 2300): mean={gm:.1f} mm SLE, sigma={gs:.2f} mm SLE")
        finite_idx = np.where(np.isfinite(mean))[0]
        if finite_idx.size:
            last = finite_idx[-1]
            print(f"    at end   (yr={grid[last]:.0f}):      mean={mean[last]:.1f} mm SLE, "
                  f"sigma={sigma[last]:.2f} mm SLE")
        print()

    # ---- Figure 1: mean_spread ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ens, inc, col in SCEN:
        d = scen_data[ens]
        axL.plot(d["grid"], d["sigma"], color=col, lw=2, label=ens)
        axR.plot(d["grid"], d["mean"], color=col, lw=2, label=ens)

    for ax in (axL, axR):
        ax.axvline(SEAM_YEAR, color="gray", lw=1.2, ls="--")
        ax.grid(alpha=0.3)
        ax.set_xlabel("year (0 = 2000)")

    axL.set_ylabel("ensemble sigma (mm SLE)")
    axL.set_title("(a) ensemble spread of VAF")
    axL.legend(fontsize=9)

    axR.set_ylabel("ensemble mean (mm SLE)")
    axR.set_title("(b) ensemble mean of VAF")
    axR.legend(fontsize=9)

    fig.suptitle("Full ensembles (base 2000-2300 + restart extension), "
                 "regridded to a common annual axis", fontsize=11, y=1.03)
    fig.text(0.5, -0.06, CAVEAT, ha="center", va="top", fontsize=7.5, color="#8B0000",
             wrap=True)
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, "full_mean_spread_VAF.png")
    fig.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {out1}")

    # ---- Figure 2: member trajectories ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, (ens, inc, col) in zip(axes, SCEN):
        d = scen_data[ens]
        first_line = True
        for name, yr, vaf in d["per_member_raw"]:
            sle = eio.vaf_to_sle_mm(vaf, reference="first")
            ax.plot(yr, sle, color=col, lw=0.8, alpha=0.6,
                    label=ens if first_line else None)
            first_line = False
        ax.axvline(SEAM_YEAR, color="gray", lw=1.2, ls="--")
        ax.set_xlabel("year (0 = 2000)")
        ax.set_ylabel("VAF-derived SLE change (mm)")
        ax.set_title(ens, fontsize=11, fontweight="bold", color=col)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("Full-ensemble member trajectories (2300 seam marked)", fontsize=12, y=1.03)
    fig.text(0.5, -0.08, CAVEAT, ha="center", va="top", fontsize=7.5, color="#8B0000",
             wrap=True)
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, "full_member_trajectories.png")
    fig.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {out2}")

    # ---- copy to wiki vault ----
    if args.wiki_dest:
        os.makedirs(args.wiki_dest, exist_ok=True)
        for p in (out1, out2):
            dest = os.path.join(args.wiki_dest, os.path.basename(p))
            os.system(f"/bin/cp -f {p!r} {dest!r}")
            print(f"Copied -> {dest}")


if __name__ == "__main__":
    main()
