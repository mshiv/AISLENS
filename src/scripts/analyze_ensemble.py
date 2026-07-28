#!/usr/bin/env python3
"""
analyze_ensemble.py — ensemble diagnostics, drift, and spread-convergence.

Computes VAF->SLE per member with median/IQR/5-95% bands, dispersion sigma(t),
skewness(t), noise-induced drift vs deterministic baseline (Robel et al. 2024),
and bootstrap spread-convergence.

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
                   help="ENSEMBLES root dir (default: config DIR_MALI/diagnostics/ENSEMBLES)")
    p.add_argument("--ensemble", required=True,
                   help="Ensemble sub-directory name, e.g. SSP585_varScaled10x")
    p.add_argument("--det-baseline", default=None,
                   help="Deterministic-baseline ensemble/member dir (e.g. DET-SSP585) "
                        "for the noise-induced-drift diagnostic. Single-member dir is fine.")
    p.add_argument("--var", default="volumeAboveFloatation",
                   help="globalStats variable to analyze (default VAF)")
    p.add_argument("--members", default=None,
                   help=r"Regex to select a clean member subset, e.g. '^SSP585_\d+$' "
                        "(skips short -EM / _V2 / CHANGEPOINT test runs)")
    p.add_argument("--min-years", type=float, default=None,
                   help="Drop members whose record spans fewer than this many years")
    p.add_argument("--full-period", action="store_true",
                   help="DEPRECATED (now the default). Ignored.")
    p.add_argument("--as-sle", action="store_true", default=True,
                   help="Convert VAF to sea-level equivalent (mm). On by default for VAF.")
    p.add_argument("--out-fig-dir", default=None,
                   help="Where to save figures (default: <root>/figures)")
    p.add_argument("--out-stats", default=None,
                   help="Path for the cached stats NetCDF "
                        "(default: <root>/ensemble_stats/<ensemble>_stats.nc)")
    p.add_argument("--boot-iters", type=int, default=500,
                   help="Bootstrap iterations for spread-convergence")
    return p.parse_args()


def load_series(root, ensemble, var, include=None, min_years=None, align="union"):
    ens_dir = os.path.join(root, ensemble)
    ds = eio.load_ensemble_globalstats(ens_dir, variables=[var, "daysSinceStart"],
                                       include=include, min_years=min_years, align=align)
    da = ds[var]  # dims (member, year)
    return da, ds["year"].values


def to_sle_if_vaf(da, var, as_sle):
    if as_sle and var == "volumeAboveFloatation":
        sle = xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, reference="first"), da)
        sle.name = "sle_mm"
        sle.attrs["units"] = "mm SLE (rise positive, relative to t0)"
        return sle, "sea-level equivalent (mm)"
    # otherwise report change from t0 in native units
    d = da - da.isel(year=0)
    d.name = var + "_change"
    return d, f"{var} change ({da.attrs.get('units','')})"


def load_baseline(root, det_baseline, var):
    """Load a deterministic baseline series aligned to the ensemble year axis.
    Accepts either an ensemble dir (uses first member) or a direct member dir."""
    if det_baseline is None:
        return None
    cand = os.path.join(root, det_baseline)
    # direct member dir?
    gp = os.path.join(cand, "globalStats.nc")
    if os.path.isfile(gp):
        ds = eio.load_member_globalstats(gp)
    else:
        members = eio.discover_members(cand)
        if not members:
            print(f"WARNING: no baseline globalStats under {cand}; skipping drift.")
            return None
        ds = eio.load_member_globalstats(members[0][1])
    return eio.to_year_dim(ds)[var]


def bootstrap_convergence(da, iters=500, rng=None):
    """For N=2..M members, resample-with-replacement and record the spread of the
    sigma estimate at the LAST year where all members are still present (in
    --full-period mode the true final year may have only one member, giving sigma~0).
    Returns (Ns, mean_sigma, sigma_of_sigma, eval_year)."""
    rng = rng or np.random.default_rng(0)
    M = da.sizes["member"]
    count = da.notnull().sum("member").values          # members per year
    full_years = np.where(count == count.max())[0]
    yi = int(full_years[-1])                            # last year with the most members
    eval_year = float(da["year"].values[yi])
    final = da.isel(year=yi).values
    final = final[np.isfinite(final)]
    Ns = list(range(2, M + 1))
    mean_sig, spread_sig = [], []
    for N in Ns:
        sigs = []
        for _ in range(iters):
            samp = rng.choice(final, size=N, replace=True)
            sigs.append(np.std(samp, ddof=1))
        mean_sig.append(np.mean(sigs))
        spread_sig.append(np.std(sigs))
    return np.array(Ns), np.array(mean_sig), np.array(spread_sig), eval_year


def main():
    args = parse_args()
    fig_dir = args.out_fig_dir or os.path.join(args.root, "figures")
    stats_dir = os.path.dirname(args.out_stats) if args.out_stats else \
        os.path.join(args.root, "ensemble_stats")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    out_stats = args.out_stats or os.path.join(stats_dir, f"{args.ensemble}_stats.nc")

    da_raw, year = load_series(args.root, args.ensemble, args.var,
                               include=args.members, min_years=args.min_years,
                               align="union")
    da, ylabel = to_sle_if_vaf(da_raw, args.var, args.as_sle)
    n_members = da.sizes["member"]
    print(f"Loaded {n_members} members for {args.ensemble}; year "
          f"{year[0]:.1f}..{year[-1]:.1f}")

    stats = eio.ensemble_stats(da)
    stats.to_netcdf(out_stats)
    print(f"Wrote stats -> {out_stats}")

    # ---- Figure 1: ensemble spread ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in range(n_members):
        ax.plot(year, da.isel(member=m), color="0.7", lw=0.6)
    ax.plot(year, stats["mean"], "k", lw=2, label="ensemble mean")
    ax.plot(year, stats["median"], color="0.45", lw=1, ls="--", label="median")
    ax.fill_between(year, stats["p05"], stats["p95"], color="C0", alpha=0.2, label="5-95%")
    ax.fill_between(year, stats["p25"], stats["p75"], color="C0", alpha=0.35, label="IQR")
    base = load_baseline(args.root, args.det_baseline, args.var)
    if base is not None:
        b, _ = to_sle_if_vaf(base.interp(year=year), args.var, args.as_sle)
        ax.plot(year, b, "r--", lw=1.5, label="deterministic")
    # how many members contribute at each year (matters in --full-period mode)
    count = da.notnull().sum("member")
    axc = ax.twinx()
    axc.step(year, count, color="0.55", lw=0.9, alpha=0.6, where="mid")
    axc.set_ylabel("# members contributing", color="0.55")
    axc.set_ylim(0, n_members * 1.05); axc.tick_params(axis="y", labelcolor="0.55")
    ax.set_xlabel("year"); ax.set_ylabel(ylabel)
    ax.set_title(f"{args.ensemble}: ensemble spread "
                 f"({n_members} members, {'full-period' if args.full_period else 'all-members'})")
    ax.legend(loc="upper left"); fig.tight_layout()
    f1 = os.path.join(fig_dir, f"{args.ensemble}_ensemble_spread.png")
    fig.savefig(f1, dpi=150); plt.close(fig)

    # ---- Figure 2: noise-induced drift ----
    if base is not None:
        b, _ = to_sle_if_vaf(base.interp(year=year), args.var, args.as_sle)
        drift = stats["mean"] - b
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(year, drift, "C3", lw=2)
        ax.set_xlabel("year"); ax.set_ylabel(f"Delta_mean ({ylabel})")
        ax.set_title(f"{args.ensemble}: noise-induced drift (ensemble mean - deterministic)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{args.ensemble}_drift.png"), dpi=150)
        plt.close(fig)
    else:
        print("No baseline -> skipped drift figure (pass --det-baseline).")

    # ---- Figure 3: dispersion + relative uncertainty ----
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(year, stats["std"], "C0"); axs[0].set_ylabel(f"sigma ({ylabel})")
    axs[1].plot(year, stats["rel_uncertainty"], "C1")
    axs[1].set_ylabel("sigma / |mean|"); axs[1].set_yscale("log")
    axs[1].set_xlabel("year")
    axs[0].set_title(f"{args.ensemble}: dispersion")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{args.ensemble}_dispersion.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 4: skewness ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(year, stats["skewness"], "C4", lw=1.5)
    ax.set_xlabel("year"); ax.set_ylabel("skewness across members")
    ax.set_title(f"{args.ensemble}: ensemble skewness "
                 "(negative = heavy tail toward more loss)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{args.ensemble}_skewness.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 5: spread convergence ----
    Ns, msig, ssig, eval_year = bootstrap_convergence(da, iters=args.boot_iters)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(Ns, msig, yerr=ssig, marker="o", capsize=3)
    ax.set_xlabel("ensemble size N"); ax.set_ylabel(f"sigma at year {eval_year:.0f} ({ylabel})")
    ax.set_title(f"{args.ensemble}: spread-convergence (bootstrap, year {eval_year:.0f})")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{args.ensemble}_convergence.png"), dpi=150)
    plt.close(fig)

    print(f"Figures -> {fig_dir}")
    print(f"Spread converges? sigma at year {eval_year:.0f}, N={Ns[-1]} = {msig[-1]:.3g} "
          f"+/- {ssig[-1]:.2g} (bootstrap). Compare to smaller N in the figure.")


if __name__ == "__main__":
    main()
