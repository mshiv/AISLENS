#!/usr/bin/env python3
"""
fig_melt3x_calibration.py — Calibrate the realized total-melt ratio of SSP585-3X.

SSP585-3X (5 members, currently ~yr 165-179) was built as F_3x = F_1x + 2*T, where
T is the deterministic SSP585 melt TREND. This TRIPLES the trend/adjustment term
only -- it does NOT triple total melt, because total melt = background
draft-dependent parameterized melt B (untouched) + adjustment (scaled). A nominal
3x adjustment was PREDICTED (not measured) to yield roughly ~1.9x realized total
melt at yr300 (see verify_3x_forcing.py CAVEAT 1). This script MEASURES the actual
realized ratio from MALI output -- the number itself is a scientific result: it
quantifies the same melt-saturation ceiling that makes the ensemble's amplitude
response sub-linear (see the 10x-variability ensemble, which only realized
~6.3-7.6x, not 10x).

Compares SSP585-3X (n=5) against SSP585 (n=10, complete 300 yr) over their COMMON
window only (~yr 0-165, set by the shortest SSP585-3X member). Both ensembles are
interpolated onto a shared annual grid before any ensemble statistics are taken
(per-member output cadence is irregular), and members starting more than 5 yr from
year 0 (restart-continuation segments) are dropped.

Quantities:
  1. Realized mean-melt ratio (ensemble-mean avgSubshelfMelt, 3X/1X) over yr windows
     0-50, 50-100, 100-<common_end>, and the full common span. THE headline number.
  2. Realized stochastic amplitude A = mean_over_members(SD_t[m_i(t)-ensmean(t)])
     for avgSubshelfMelt -- should be ~unchanged (~1.0), since only the trend was
     scaled. A ratio far from 1 means the forcing construction did something
     unintended and is flagged loudly.
  3. Sea-level response: ensemble-mean SLE (vaf_to_sle_mm, reference="first") and
     the ratio of cumulative loss at the common end year.
  4. Ensemble spread sigma(t) of SLE for both ensembles, with the ±35% (n=5) /
     ±24% (n=10) member-count sampling caveat.

Figure -> reports/figures/melt3x_calibration.png (4 panels).
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

INCLUDE_1X = r"^SSP585_\d+$"
INCLUDE_3X = r"^SSP585-3X_\d+$"
EXCLUDE_MEMBERS = {"SSP585-3X_00"}
MIN_YEARS = 5.0
START_TOL = 5.0        # GUARD: drop members starting more than 5 yr from year 0
NOMINAL_TREND_FACTOR = 3.0
PREDICTED_RATIO = 1.9   # prior (unmeasured) estimate from verify_3x_forcing.py

COLOR_1X = "#D55E00"
COLOR_3X = "#0072B2"

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "savefig.dpi": 300,
})

VARS = ["avgSubshelfMelt", "volumeAboveFloatation"]


# ----------------------------------------------------------------------------
# Loading: interpolate every member onto a SHARED annual grid (not each
# ensemble's own union grid) so 3X (n=5, irregular cadence, ends ~yr165-179)
# and 1X (n=10, complete 300 yr) are directly comparable year-by-year.
# ----------------------------------------------------------------------------
def load_ensemble_annual(ensemble_dir, include, variables, annual_grid,
                          min_years=MIN_YEARS, start_tol=START_TOL):
    members = eio.discover_members(ensemble_dir, include=include)
    if not members:
        raise RuntimeError(f"No members found under {ensemble_dir} matching /{include}/")

    per_member, names, end_years = [], [], []
    dropped_offaxis, dropped_short = [], []
    for name, path in members:
        if name in EXCLUDE_MEMBERS:
            print(f"  [load] excluded quality-control member {name}")
            continue
        ds = eio.load_member_globalstats(path)
        keep = [v for v in variables if v in ds]
        ds = ds[keep]
        d = eio.to_year_dim(ds)
        raw_year0 = float(d["year"].values[0])
        if abs(raw_year0) > start_tol:
            dropped_offaxis.append((name, round(raw_year0, 1)))
            continue
        if raw_year0 != 0.0:
            d = d.assign_coords(year=d["year"] - raw_year0)
        end_year = float(d["year"].values[-1])
        span = end_year - float(d["year"].values[0])
        if span < min_years:
            dropped_short.append((name, round(span, 1)))
            continue
        # Interpolate onto the shared annual grid; xarray.interp does not
        # extrapolate by default -> NaN beyond this member's own last year.
        d_annual = d.interp(year=annual_grid)
        per_member.append(d_annual)
        names.append(name)
        end_years.append(end_year)

    if dropped_offaxis:
        print(f"  [load] dropped {len(dropped_offaxis)} off-axis member(s) under "
              f"{ensemble_dir}: {dropped_offaxis}")
    if dropped_short:
        print(f"  [load] dropped {len(dropped_short)} short member(s) (<{min_years} yr) "
              f"under {ensemble_dir}: {dropped_short}")
    if not per_member:
        raise RuntimeError(f"All members dropped under {ensemble_dir}")

    out = xr.concat(per_member, dim="member")
    out = out.assign_coords(member=("member", names))
    return out, dict(zip(names, end_years))


def window_member_means(da, y0, y1):
    """Per-member time-mean of da (dims member, year) over [y0, y1]. Returns a
    1-D numpy array (length n_members), NaN-dropped."""
    sub = da.sel(year=slice(y0, y1))
    vals = sub.mean("year", skipna=True).values.astype(float)
    return vals[~np.isnan(vals)]


def mean_and_se(vals):
    n = len(vals)
    m = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return m, se, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--out", default="reports/figures/melt3x_calibration.png")
    ap.add_argument("--wiki-assets-dir", default=None,
                    help="Optional explicit copy destination; omitted by default.")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    annual_grid = np.arange(0.0, 301.0, 1.0)

    print(f"Loading SSP585 (1X) from {args.root}/SSP585 ...")
    ds_1x, end_1x = load_ensemble_annual(
        os.path.join(args.root, "SSP585"), INCLUDE_1X, VARS, annual_grid)
    print(f"Loading SSP585-3X from {args.root}/SSP585-3X ...")
    ds_3x, end_3x = load_ensemble_annual(
        os.path.join(args.root, "SSP585-3X"), INCLUDE_3X, VARS, annual_grid)

    n_1x, n_3x = ds_1x.sizes["member"], ds_3x.sizes["member"]
    print(f"\nMember counts: SSP585 (1X) n={n_1x}  {sorted(ds_1x['member'].values.tolist())}")
    print(f"                SSP585-3X   n={n_3x}  {sorted(ds_3x['member'].values.tolist())}")
    print(f"Per-member end years: 1X {end_1x}")
    print(f"                      3X {end_3x}")

    # Common window: bounded by the SHORTEST member across BOTH ensembles
    # (in practice, the shortest SSP585-3X member sets this).
    common_end_raw = min(min(end_1x.values()), min(end_3x.values()))
    common_end = float(np.floor(common_end_raw))
    print(f"\nCommon comparison window: yr 0-{common_end:.0f} "
          f"(bounded by shortest member, raw end = {common_end_raw:.2f} yr)")

    melt_1x = ds_1x["avgSubshelfMelt"]
    melt_3x = ds_3x["avgSubshelfMelt"]
    vaf_1x = ds_1x["volumeAboveFloatation"]
    vaf_3x = ds_3x["volumeAboveFloatation"]

    # ------------------------------------------------------------------
    # (1) Realized mean-melt ratio per window
    # ------------------------------------------------------------------
    windows = [
        ("0-50", 0.0, 50.0),
        ("50-100", 50.0, 100.0),
        (f"100-{common_end:.0f}", 100.0, common_end),
        (f"0-{common_end:.0f} (full common span)", 0.0, common_end),
    ]

    print("\n" + "=" * 78)
    print("(1) REALIZED MEAN-MELT RATIO (avgSubshelfMelt, ensemble mean, 3X/1X)")
    print("=" * 78)
    ratio_rows = []
    for label, y0, y1 in windows:
        v1 = window_member_means(melt_1x, y0, y1)
        v3 = window_member_means(melt_3x, y0, y1)
        m1, se1, k1 = mean_and_se(v1)
        m3, se3, k3 = mean_and_se(v3)
        ratio = m3 / m1 if m1 != 0 else float("nan")
        se_ratio = (ratio * np.sqrt((se3 / m3) ** 2 + (se1 / m1) ** 2)
                    if m1 != 0 and m3 != 0 else float("nan"))
        z_diff = ((m3 - m1) / np.sqrt(se1 ** 2 + se3 ** 2)
                  if np.isfinite(se1) and np.isfinite(se3) else float("nan"))
        ratio_rows.append((label, y0, y1, m1, se1, k1, m3, se3, k3, ratio, se_ratio, z_diff))
        print(f"  yr {label:>22s}: 1X = {m1:7.4f} ± {se1:.4f} (n={k1})   "
              f"3X = {m3:7.4f} ± {se3:.4f} (n={k3})   "
              f"ratio = {ratio:5.2f} ± {se_ratio:.2f}   "
              f"z(3X-1X) = {z_diff:6.2f}")

    headline = ratio_rows[-1]  # full common span
    headline_ratio = headline[9]
    headline_se = headline[10]

    # ------------------------------------------------------------------
    # (2) Realized stochastic amplitude (variability should be ~unchanged)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("(2) REALIZED STOCHASTIC AMPLITUDE (avgSubshelfMelt anomaly about ensemble "
          "mean)")
    print("=" * 78)

    def amplitude(da, y0, y1):
        sub = da.sel(year=slice(y0, y1))
        ensmean = sub.mean("member", skipna=True)
        anom = sub - ensmean
        sd_t = anom.std("year", skipna=True, ddof=1)  # per member
        vals = sd_t.values.astype(float)
        vals = vals[~np.isnan(vals)]
        return float(np.mean(vals)), vals

    print(f"  {'window':>28s}   {'A(1X)':>9s}   {'A(3X)':>9s}   {'ratio':>7s}")
    amp_rows = []
    for label, y0, y1 in windows:
        A1, v1 = amplitude(melt_1x, y0, y1)
        A3, v3 = amplitude(melt_3x, y0, y1)
        r = A3 / A1 if A1 != 0 else float("nan")
        amp_rows.append((label, y0, y1, A1, A3, r))
        print(f"  {label:>28s}   {A1:9.4f}   {A3:9.4f}   {r:7.3f}")

    # Headline amplitude ratio = full common span (last row); ALSO check whether
    # the early windows (where trend divergence is small) show ratio ~1 while a
    # late window does not -- that pattern points to a saturation/truncation
    # nonlinearity in the MALI melt response at high melt rates, not a bug in the
    # forcing construction (verify_3x_forcing.py already confirms the prescribed
    # per-cell forcing satisfies scaled=orig+(w-1)*T exactly).
    amp_ratio = amp_rows[-1][5]
    early_ratios = [r[5] for r in amp_rows[:-1] if r[2] <= 100.0]
    variability_preserved_early = all(abs(r - 1.0) <= 0.25 for r in early_ratios)
    variability_preserved = abs(amp_ratio - 1.0) <= 0.25
    print(f"\n  Full-common-span amplitude ratio (3X/1X) = {amp_ratio:.3f}  "
          f"(expected ~1.0 -- only the trend was scaled)")
    if not variability_preserved:
        if variability_preserved_early:
            print("  *** FLAG: amplitude ratio is ~1.0 in the early windows (yr<=100) but "
                  "grows sharply in the late window -- consistent with a melt-saturation / "
                  "truncation nonlinearity kicking in as mean melt rates climb, NOT a bug "
                  "in the additive forcing construction (per-cell forcing was independently "
                  "verified unchanged by verify_3x_forcing.py). The realized OUTPUT "
                  "variability of avgSubshelfMelt is state-dependent and grows with the "
                  "mean forcing even though the prescribed variability perturbation did not. "
                  "***")
        else:
            print("  *** FLAG: variability amplitude ratio deviates from 1.0 by >25% and is "
                  "NOT confined to the late window. Investigate the forcing construction "
                  "before trusting SSP585-3X as a pure trend-scaling experiment. ***")

    # ------------------------------------------------------------------
    # (3) Sea-level response
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("(3) SEA-LEVEL RESPONSE (SLE, mm, reference=first)")
    print("=" * 78)
    sle_1x = xr.apply_ufunc(eio.vaf_to_sle_mm, vaf_1x, kwargs=dict(reference="first"))
    sle_3x = xr.apply_ufunc(eio.vaf_to_sle_mm, vaf_3x, kwargs=dict(reference="first"))
    ensmean_sle_1x = sle_1x.mean("member", skipna=True)
    ensmean_sle_3x = sle_3x.mean("member", skipna=True)

    sle_1x_end = float(ensmean_sle_1x.sel(year=common_end, method="nearest").item())
    sle_3x_end = float(ensmean_sle_3x.sel(year=common_end, method="nearest").item())
    sle_ratio = sle_3x_end / sle_1x_end if sle_1x_end != 0 else float("nan")
    print(f"  ensemble-mean SLE @ yr {common_end:.0f}: 1X = {sle_1x_end:.2f} mm   "
          f"3X = {sle_3x_end:.2f} mm   ratio = {sle_ratio:.2f}")

    # ------------------------------------------------------------------
    # (4) Ensemble spread sigma(t) of SLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("(4) ENSEMBLE SPREAD sigma(t) of SLE")
    print("=" * 78)
    sigma_1x = sle_1x.std("member", skipna=True, ddof=1)
    sigma_3x = sle_3x.std("member", skipna=True, ddof=1)
    sigma_1x_end = float(sigma_1x.sel(year=common_end, method="nearest").item())
    sigma_3x_end = float(sigma_3x.sel(year=common_end, method="nearest").item())
    sigma_ratio = sigma_3x_end / sigma_1x_end if sigma_1x_end != 0 else float("nan")
    print(f"  sigma(SLE) @ yr {common_end:.0f}: 1X (n={n_1x}) = {sigma_1x_end:.2f} mm "
          f"(±24% sampling)   3X (n={n_3x}) = {sigma_3x_end:.2f} mm (±35% sampling)   "
          f"ratio = {sigma_ratio:.2f}")
    print("  NOTE: with only n=5 (3X) / n=10 (1X) members, sigma itself carries "
          "±35% / ±24% sampling uncertainty (1/sqrt(2(n-1)) for a std-dev estimate) "
          "-- a sigma ratio within roughly that combined envelope is not distinguishable "
          "from 'spread unchanged'.")

    # ------------------------------------------------------------------
    # Solve for the weight w needed for a TRUE 3x total-melt ratio, from the
    # two measured points (w=1 -> ratio=1 by construction; w=3 -> ratio=headline_ratio).
    # Linear-in-(w-1) assumption -- almost certainly optimistic given melt
    # saturation, but is the only extrapolation the data supports.
    # ------------------------------------------------------------------
    slope = (headline_ratio - 1.0) / (NOMINAL_TREND_FACTOR - 1.0)
    w_needed = 1.0 + (NOMINAL_TREND_FACTOR - 1.0) / slope if slope != 0 else float("nan")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  Nominal trend factor: {NOMINAL_TREND_FACTOR:.1f}")
    print(f"  Realized total-melt ratio (full common span, yr 0-{common_end:.0f}): "
          f"{headline_ratio:.2f} ± {headline_se:.2f}")
    print(f"  (prior unmeasured prediction was ~{PREDICTED_RATIO:.1f}x; measured value "
          f"{'is consistent with' if abs(headline_ratio - PREDICTED_RATIO) < 0.3 else 'DIFFERS from'} "
          f"that prediction)")
    if variability_preserved:
        preserved_str = "YES"
    elif variability_preserved_early:
        preserved_str = "YES through yr~100, NO in the late window -- SEE FLAG ABOVE (saturation, not a forcing bug)"
    else:
        preserved_str = "NO -- SEE FLAG ABOVE"
    print(f"  Variability preserved (amplitude ratio ~1.0): {preserved_str} "
          f"(full-span measured {amp_ratio:.3f})")
    print(f"  Sea-level response ratio @ yr {common_end:.0f}: {sle_ratio:.2f}")
    print(f"  Ensemble spread ratio @ yr {common_end:.0f}: {sigma_ratio:.2f} "
          f"(within n=5/n=10 sampling noise: {'plausibly unchanged' if 0.5 < sigma_ratio < 2.0 else 'notably different'})")
    print(f"  Melt saturates sub-linearly in the trend weight w: assuming the SAME "
          f"linear-in-(w-1) relationship extrapolated from (w=1 -> ratio=1) and "
          f"(w={NOMINAL_TREND_FACTOR:.0f} -> ratio={headline_ratio:.2f}), reaching a TRUE "
          f"3.0x total-melt ratio would require w ≈ {w_needed:.2f} "
          f"(NOT a measured/calibrated value -- linear extrapolation of a saturating "
          f"process is optimistic; treat as a lower bound on the weight actually needed).")

    # ==================================================================
    # Figure
    # ==================================================================
    # Two panels: the forcing that was actually delivered, and the response to it. The
    # realized-ratio-vs-nominal curve and the spread panel were diagnostics used to verify
    # the experiment; neither carries an argument the chapter makes.
    fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax_b = ax_d = None

    yr = annual_grid

    # (a) ensemble-mean avgSubshelfMelt vs year
    ensmean_melt_1x = melt_1x.mean("member", skipna=True)
    ensmean_melt_3x = melt_3x.mean("member", skipna=True)
    ax_a.plot(yr, ensmean_melt_1x.values, color=COLOR_1X, lw=1.8, label="SSP585")
    ax_a.plot(yr, ensmean_melt_3x.values, color=COLOR_3X, lw=1.8, label="SSP585-3X")
    ax_a.set_xlim(0, 300)
    ax_a.set_xlabel("Model year")
    ax_a.set_ylabel("Mean sub-shelf melt (m ice yr$^{-1}$)")
    ax_a.text(0.98, 0.98, "(a)", transform=ax_a.transAxes, ha="right", va="top",
              fontsize=15, fontweight="bold")
    ax_a.legend(loc="upper left")
    ax_a.grid(alpha=0.2)

    # (b) ratio 3X/1X vs year
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_t = (ensmean_melt_3x / ensmean_melt_1x).values
    mask = yr <= common_end
    _ = None and ax_b.plot(yr[mask], ratio_t[mask], color="0.15", lw=1.6)
    _ = None and ax_b.axhline(NOMINAL_TREND_FACTOR, color="0.4", ls="--", lw=1.2,
                 label=f"nominal {NOMINAL_TREND_FACTOR:.1f}$\\times$")
    _ = None and ax_b.axhline(PREDICTED_RATIO, color="#CC79A7", ls="--", lw=1.2,
                 label=f"predicted ~{PREDICTED_RATIO:.1f}$\\times$")
    _ = None and ax_b.axvline(common_end, color="0.5", ls=":", lw=1)
    _ = None and ax_b.set_xlim(0, common_end + 5)
    _ = None and ax_b.set_ylim(0, max(3.5, np.nanmax(ratio_t[mask]) * 1.15 if np.any(mask) else 3.5))
    _ = None and ax_b.set_xlabel("year")
    _ = None and ax_b.set_ylabel("ratio 3X / 1X (avgSubshelfMelt)")
    _ = None and ax_b.set_title("(b) Realized melt ratio vs. nominal trend factor")
    _ = None and ax_b.legend(loc="best", fontsize=9)
    _ = None and ax_b.grid(alpha=0.2)

    # (c) ensemble-mean SLE vs year
    ax_c.plot(yr, ensmean_sle_1x.values, color=COLOR_1X, lw=1.8, label="SSP585")
    ax_c.plot(yr, ensmean_sle_3x.values, color=COLOR_3X, lw=1.8, label="SSP585-3X")
    ax_c.set_xlim(0, 300)
    ax_c.set_xlabel("Model year")
    ax_c.set_ylabel("Sea-level contribution (mm SLE)")
    ax_c.text(0.98, 0.98, "(b)", transform=ax_c.transAxes, ha="right", va="top",
              fontsize=15, fontweight="bold")
    ax_c.legend(loc="upper left")
    ax_c.grid(alpha=0.2)

    # (d) sigma(t) for both
    _ = None and ax_d.plot(yr, sigma_1x.values, color=COLOR_1X, lw=1.8, label="SSP585")
    _ = None and ax_d.plot(yr, sigma_3x.values, color=COLOR_3X, lw=1.8, label="SSP585-3X")
    _ = None and ax_d.axvline(common_end, color="0.5", ls=":", lw=1)
    _ = None and ax_d.set_xlim(0, common_end + 20)
    _ = None and ax_d.set_xlabel("year")
    _ = None and ax_d.set_ylabel("ensemble spread $\\sigma$(SLE) (mm)")
    # the 3X spread past ~yr 175 is members diverging as they approach termination, and
    # then the sample itself shrinking; it is not an ensemble property
    _ = None and ax_d.set_title(f"(d) Ensemble spread  --  ratio {sigma_ratio:.2f}$\\times$ "
                    f"@ yr {common_end:.0f}\n"
                    "3X spread past yr 175 reflects members terminating, not ensemble width",
                    fontsize=10)
    _ = None and ax_d.legend(loc="upper left", fontsize=9)
    _ = None and ax_d.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure -> {args.out}")

    if args.wiki_assets_dir and os.path.isdir(args.wiki_assets_dir):
        import shutil
        dest = os.path.join(args.wiki_assets_dir, os.path.basename(args.out))
        shutil.copy2(args.out, dest)
        print(f"Copied -> {dest}")
    elif args.wiki_assets_dir:
        print(f"NOTE: wiki assets dir not found ({args.wiki_assets_dir}); skipped copy.")


if __name__ == "__main__":
    main()
