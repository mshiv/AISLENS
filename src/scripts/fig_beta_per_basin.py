#!/usr/bin/env python3
"""
fig_beta_per_basin.py — per-basin amplitude exponent beta_b: sigma_response ~ A_realized^beta_b.

The GLOBAL amplitude exponent is sigma_response ~ A^beta with beta~0.62 (sub-linear),
measured against the REALIZED forcing amplitude (the prescribed 10x scale factor only
delivers ~6-7.6x once melt saturates -- floating ice runs out to melt). beta=1 is the
LINEAR expectation; beta<1 quantifies the nonlinearity/damping.

HYPOTHESIS tested here: beta_b is near 1 in ice-rich basins that never run out of
floating ice, and well below 1 in the MISI basins (G-H Amundsen/Thwaites-PIG idx 9,
J-K FRIS idx 14) -- i.e. the sub-linearity is driven by exactly the basins that
dominate ensemble spread (fig_dynamic_gating.py), because those are the ones that
saturate.

Data: SSP585 (1x) vs SSP585_varScaled10x (10x), both regionalStats.nc
(regionalVolumeAboveFloatation, regionalAvgSubshelfMelt, regionalFloatingIceArea,
nRegions=16, ISMIP6 basins) and globalStats.nc (volumeAboveFloatation,
avgSubshelfMelt, floatingIceArea; used for a whole-domain beta computed with the
IDENTICAL methodology, as a sanity comparison against the ~0.62 headline number).

Method, per basin b (see std_sle_at_horizon / realized_amplitude / beta_point_and_boot):
  1. sigma_b(ens) = std across members of per-basin VAF->SLE (mm), at a horizon T.
  2. REALIZED forcing amplitude A_b(ens) = mean over members of the temporal SD (over
     [0,T]) of (member melt(t) - ensemble-mean melt(t)) -- subtracting the ensemble
     mean removes the forced signal exactly, leaving the STOCHASTIC part actually
     delivered. This is NOT the nominal 10x prescribed scale factor.
  3. r_b = A_b(10x) / A_b(1x)  (realized amplitude ratio, basin-specific).
     beta_b = ln( sigma_b(10x) / sigma_b(1x) ) / ln( r_b ).
  4. Member-bootstrap (>=1000 draws, resampling each ensemble independently WITH
     replacement and recomputing sigma/A/beta end-to-end on the resample) for a 95%
     CI. n=10 (1x) / n=15 (10x) -> CIs are reported honestly, i.e. wide.
  5. Remaining floating-ice fraction at the horizon = ensemble-mean (1x)
     regionalFloatingIceArea(T) / regionalFloatingIceArea(0).

GUARDS: members whose record starts after yr5 are dropped (restart-continuation
segments); every variable is interpolated onto a COMMON ANNUAL GRID per member
BEFORE any ensemble statistic (native output cadence is irregular, ~0.15-0.45 yr
and differs member to member -- statistics on the raw merged axis produce spurious
spikes); regionalAvgSubshelfMelt is masked to NaN wherever regionalFloatingIceArea
has collapsed below AREA_FRAC_FLOOR of that member/basin's year-0 area ("no ice",
which is not the same as "no melt").

CAVEATS (also printed at runtime):
  - n=10/15 members -> sigma carries ~+-24%/+-19% sampling uncertainty (1/sqrt(2(n-1))
    for a std-dev estimate), and beta_b inherits it via error propagation through two
    log-ratios -- hence the wide bootstrap CIs.
  - Basins whose sigma_1x is ~0 or whose realized ratio r_b<=1 (forcing did not
    actually amplify in that basin) give an unstable or undefined beta_b and are
    SKIPPED, not plotted as a spurious number.
  - beta_b is measured under SSP585 forcing only; it is not necessarily transferable
    to CTRL (unforced) noise or to other scenarios.

Author: Shivaprakash Muruganandham (2026-08-09)
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from ismip6_regions import BASIN_NAMES, short_label, letter_from_index

MISI_INDICES = (9, 14)     # G-H (Amundsen/Thwaites-PIG), J-K (FRIS) -- see fig_dynamic_gating.py
AREA_FRAC_FLOOR = 0.02     # "no ice": floating-ice area < 2% of that member/basin's yr-0 area
RESTART_TOL = 5.0          # drop members whose record starts after this many years
SIGMA_FLOOR_MM = 0.01      # sigma_1x below this (mm SLE) is treated as a near-zero-signal
                           # divide-by-near-zero artefact, not a stable amplitude measurement


# ----------------------------------------------------------------------------
# Loading: per-member arrays on a COMMON ANNUAL GRID, shape (member, year, nRegions)
# ----------------------------------------------------------------------------
def load_regional_on_grid(root, ensemble, max_horizon, restart_tol=RESTART_TOL):
    """Discover members, drop restart-continuation segments and members that don't
    reach max_horizon, interpolate each survivor's regionalStats variables onto a
    common annual grid year=0..max_horizon (step 1 yr).

    Returns (vaf, melt_masked, area, names, grid): vaf/melt_masked/area are numpy
    arrays shaped (n_members, max_horizon+1, nRegions); melt_masked has NaN wherever
    regionalFloatingIceArea < AREA_FRAC_FLOOR * that member/basin's year-0 area.
    """
    ens_dir = os.path.join(root, ensemble)
    members = eio.discover_members(ens_dir, stats_filename="regionalStats.nc")
    grid = np.arange(0.0, max_horizon + 1.0, 1.0)
    vaf_list, melt_list, area_list, names = [], [], [], []
    dropped, tooshort = [], []
    for name, path in members:
        ds = eio.to_year_dim(eio.load_member_regionalstats(path))
        yr = ds["year"].values
        if yr[0] > restart_tol:
            dropped.append((name, round(float(yr[0]), 1)))
            continue
        if yr[-1] < max_horizon:
            tooshort.append((name, round(float(yr[-1]), 1)))
            continue
        vaf_i = ds["regionalVolumeAboveFloatation"].interp(year=grid).values
        melt_i = ds["regionalAvgSubshelfMelt"].interp(year=grid).values
        area_i = ds["regionalFloatingIceArea"].interp(year=grid).values
        vaf_list.append(vaf_i); melt_list.append(melt_i); area_list.append(area_i)
        names.append(name)
    if dropped:
        print(f"  [{ensemble}] dropped {len(dropped)} restart-continuation member(s) "
              f"(start > yr{restart_tol}): {dropped}")
    if tooshort:
        print(f"  [{ensemble}] dropped {len(tooshort)} member(s) not reaching yr{max_horizon:.0f}: {tooshort}")
    if not vaf_list:
        raise RuntimeError(f"No usable members for {ensemble} at horizon {max_horizon}")
    vaf = np.stack(vaf_list, axis=0)
    melt = np.stack(melt_list, axis=0)
    area = np.stack(area_list, axis=0)
    area0 = area[:, 0:1, :]
    with np.errstate(invalid="ignore"):
        no_ice = area < AREA_FRAC_FLOOR * area0
    melt_masked = np.where(no_ice, np.nan, melt)
    print(f"  [{ensemble}] {len(names)} usable members (common annual grid, "
          f"yr0-{max_horizon:.0f}); masked {100 * np.mean(no_ice):.1f}% of "
          f"(member,year,basin) melt cells as 'no ice' (< {100*AREA_FRAC_FLOOR:.0f}% "
          f"of yr-0 floating area)")
    return vaf, melt_masked, area, names, grid


def load_global_on_grid(root, ensemble, max_horizon, restart_tol=RESTART_TOL):
    """Same as load_regional_on_grid but for globalStats.nc (no nRegions dim) --
    used only for the whole-domain beta comparison. Returns arrays reshaped with a
    trailing singleton 'basin' axis so they can reuse the same beta machinery."""
    ens_dir = os.path.join(root, ensemble)
    members = eio.discover_members(ens_dir, stats_filename="globalStats.nc")
    grid = np.arange(0.0, max_horizon + 1.0, 1.0)
    vaf_list, melt_list, area_list, names = [], [], [], []
    dropped, tooshort = [], []
    for name, path in members:
        ds = eio.to_year_dim(eio.load_member_globalstats(path))
        yr = ds["year"].values
        if yr[0] > restart_tol:
            dropped.append((name, round(float(yr[0]), 1)))
            continue
        if yr[-1] < max_horizon:
            tooshort.append((name, round(float(yr[-1]), 1)))
            continue
        vaf_i = ds["volumeAboveFloatation"].interp(year=grid).values
        melt_i = ds["avgSubshelfMelt"].interp(year=grid).values
        area_i = ds["floatingIceArea"].interp(year=grid).values
        vaf_list.append(vaf_i); melt_list.append(melt_i); area_list.append(area_i)
        names.append(name)
    if not vaf_list:
        raise RuntimeError(f"No usable members for {ensemble} globalStats at horizon {max_horizon}")
    vaf = np.stack(vaf_list, axis=0)[:, :, None]
    melt = np.stack(melt_list, axis=0)[:, :, None]
    area = np.stack(area_list, axis=0)[:, :, None]
    area0 = area[:, 0:1, :]
    with np.errstate(invalid="ignore"):
        no_ice = area < AREA_FRAC_FLOOR * area0
    melt_masked = np.where(no_ice, np.nan, melt)
    return vaf, melt_masked, area, names, grid


# ----------------------------------------------------------------------------
# Core statistics
# ----------------------------------------------------------------------------
def std_sle_at_horizon(vaf, horizon, grid):
    """vaf: (member, year, nBasin) on `grid` (grid[0]=0). Returns (sigma, sle_h):
    sigma = std across members (ddof=1) of VAF->SLE(mm) at `horizon`, referenced to
    year 0; sle_h = per-member SLE(mm) at horizon, shape (member, nBasin)."""
    idx_h = int(np.argmin(np.abs(grid - horizon)))
    vaf_t = np.transpose(vaf, (0, 2, 1))                    # (member, nBasin, year)
    sle_t = eio.vaf_to_sle_mm(vaf_t, reference="first")      # (member, nBasin, year)
    sle_h = sle_t[:, :, idx_h]                                # (member, nBasin)
    return np.nanstd(sle_h, axis=0, ddof=1), sle_h


def realized_amplitude(melt_masked, horizon, grid):
    """melt_masked: (member, year, nBasin). Returns A_b = mean over members of the
    temporal SD (over [0,horizon]) of member melt minus the ensemble-mean melt at
    each timestep -- the realized STOCHASTIC forcing amplitude actually delivered."""
    idx_h = int(np.argmin(np.abs(grid - horizon)))
    win = melt_masked[:, :idx_h + 1, :]                       # (member, year, nBasin)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        ensmean = np.nanmean(win, axis=0, keepdims=True)      # (1, year, nBasin)
        dev = win - ensmean
        member_sd = np.nanstd(dev, axis=1, ddof=1)            # (member, nBasin)
        return np.nanmean(member_sd, axis=0)                   # (nBasin,)


def beta_point_and_boot(vaf_1x, melt_1x, vaf_10x, melt_10x, horizon, grid,
                         n_boot=1000, seed=42, ci=0.95):
    """Point estimate + member-bootstrap 95% CI of beta_b for every basin."""
    n1, n10 = vaf_1x.shape[0], vaf_10x.shape[0]
    nreg = vaf_1x.shape[2]

    sig1, _ = std_sle_at_horizon(vaf_1x, horizon, grid)
    sig10, _ = std_sle_at_horizon(vaf_10x, horizon, grid)
    A1 = realized_amplitude(melt_1x, horizon, grid)
    A10 = realized_amplitude(melt_10x, horizon, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_b = A10 / A1
        beta_b = np.log(sig10 / sig1) / np.log(r_b)

    rng = np.random.RandomState(seed)
    beta_boot = np.full((n_boot, nreg), np.nan)
    for k in range(n_boot):
        idx1 = rng.choice(n1, n1, replace=True)
        idx10 = rng.choice(n10, n10, replace=True)
        s1, _ = std_sle_at_horizon(vaf_1x[idx1], horizon, grid)
        s10, _ = std_sle_at_horizon(vaf_10x[idx10], horizon, grid)
        a1 = realized_amplitude(melt_1x[idx1], horizon, grid)
        a10 = realized_amplitude(melt_10x[idx10], horizon, grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = a10 / a1
            beta_boot[k] = np.log(s10 / s1) / np.log(rb)

    alpha = (1.0 - ci) / 2.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        beta_lo = np.nanpercentile(beta_boot, 100 * alpha, axis=0)
        beta_hi = np.nanpercentile(beta_boot, 100 * (1 - alpha), axis=0)
    return dict(sig1=sig1, sig10=sig10, A1=A1, A10=A10, r_b=r_b, beta_b=beta_b,
                beta_lo=beta_lo, beta_hi=beta_hi, beta_boot=beta_boot)


def floating_ice_fraction(area_1x, horizon, grid):
    """Ensemble-mean (1x) regionalFloatingIceArea(T)/regionalFloatingIceArea(0)."""
    idx_h = int(np.argmin(np.abs(grid - horizon)))
    area_mean_t = np.nanmean(area_1x, axis=0)      # (year, nBasin)
    with np.errstate(divide="ignore", invalid="ignore"):
        return area_mean_t[idx_h, :] / area_mean_t[0, :]


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--ensemble-1x", default="SSP585")
    ap.add_argument("--ensemble-10x", default="SSP585_varScaled10x")
    ap.add_argument("--horizon", type=float, default=300.0,
                     help="primary horizon (yr) for the figure and headline table; "
                          "100 and 200 yr are also reported")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/figures/beta_per_basin.png")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    report_horizons = sorted(set([100.0, 200.0, a.horizon]))
    max_horizon = max(report_horizons)

    print(f"ENSEMBLES root: {a.root}")
    print(f"1x  ensemble: {a.ensemble_1x}")
    print(f"10x ensemble: {a.ensemble_10x}")
    print(f"Report horizons: {report_horizons} yr (figure uses {a.horizon:.0f} yr)")

    # --- Load regional data on a common annual grid (0..max_horizon) ---
    vaf_1x, melt_1x, area_1x, names_1x, grid = load_regional_on_grid(a.root, a.ensemble_1x, max_horizon)
    vaf_10x, melt_10x, area_10x, names_10x, grid10 = load_regional_on_grid(a.root, a.ensemble_10x, max_horizon)
    assert np.array_equal(grid, grid10)
    n1, n10 = vaf_1x.shape[0], vaf_10x.shape[0]
    nreg = vaf_1x.shape[2]
    print(f"  {a.ensemble_1x}: n={n1} members  |  {a.ensemble_10x}: n={n10} members")

    # --- Global (whole-domain) beta, identical methodology, for comparison ---
    try:
        gvaf_1x, gmelt_1x, garea_1x, gnames_1x, ggrid = load_global_on_grid(a.root, a.ensemble_1x, max_horizon)
        gvaf_10x, gmelt_10x, garea_10x, gnames_10x, _ = load_global_on_grid(a.root, a.ensemble_10x, max_horizon)
        global_res = {h: beta_point_and_boot(gvaf_1x, gmelt_1x, gvaf_10x, gmelt_10x, h, ggrid,
                                             n_boot=a.n_boot, seed=a.seed) for h in report_horizons}
    except Exception as e:
        print(f"  [WARN] global beta comparison unavailable: {e}")
        global_res = None

    # --- Per-basin beta at each report horizon ---
    per_horizon = {}
    for h in report_horizons:
        per_horizon[h] = beta_point_and_boot(vaf_1x, melt_1x, vaf_10x, melt_10x, h, grid,
                                             n_boot=a.n_boot, seed=a.seed)

    res = per_horizon[a.horizon]     # headline horizon, used for the figure
    ice_frac = floating_ice_fraction(area_1x, a.horizon, grid)   # (nreg,)

    # --- Skip guard: non-physical basins ---
    skipped = []
    valid = np.ones(nreg, dtype=bool)
    for b in range(nreg):
        reason = None
        if not np.isfinite(res["sig1"][b]) or res["sig1"][b] < SIGMA_FLOOR_MM:
            reason = (f"sigma_1x near-zero/undefined ({res['sig1'][b]:.4g} mm < "
                      f"floor {SIGMA_FLOOR_MM} mm)")
        elif not np.isfinite(res["r_b"][b]) or res["r_b"][b] <= 1.0:
            reason = f"realized ratio r_b non-physical ({res['r_b'][b]:.4g} <= 1)"
        elif not np.isfinite(res["beta_b"][b]):
            reason = "beta_b non-finite"
        if reason is not None:
            valid[b] = False
            skipped.append((letter_from_index(b), BASIN_NAMES[b], reason))
    if skipped:
        print(f"\n[SKIPPED {len(skipped)} basin(s) -- non-physical, excluded from table/figure]")
        for letter, nm, reason in skipped:
            print(f"  {letter:6s} {nm:16s} {reason}")

    # =========================================================================
    # Printed table (headline horizon)
    # =========================================================================
    print(f"\n=== Per-basin beta table (horizon = {a.horizon:.0f} yr) ===")
    hdr = (f"  {'basin':7s} {'name':16s} {'sigma_1x':>9s} {'sigma_10x':>10s} "
           f"{'r_b':>6s} {'beta_b':>7s} {'95% CI':>16s} {'ice_frac':>9s}")
    print(hdr)
    order = np.argsort(np.where(valid, res["beta_b"], np.inf))
    for b in order:
        if not valid[b]:
            continue
        misi = " *MISI*" if b in MISI_INDICES else ""
        print(f"  {letter_from_index(b):7s} {BASIN_NAMES[b]:16s} {res['sig1'][b]:9.2f} "
              f"{res['sig10'][b]:10.2f} {res['r_b'][b]:6.2f} {res['beta_b'][b]:7.3f} "
              f"[{res['beta_lo'][b]:5.2f},{res['beta_hi'][b]:5.2f}] {ice_frac[b]:9.3f}{misi}")

    print(f"\n=== Report horizons comparison (beta_b [95% CI]) ===")
    print(f"  {'basin':7s} " + " ".join(f"{'yr'+str(int(h)):>18s}" for h in report_horizons))
    for b in order:
        if not valid[b]:
            continue
        row = " ".join(f"{per_horizon[h]['beta_b'][b]:6.3f}[{per_horizon[h]['beta_lo'][b]:5.2f},"
                       f"{per_horizon[h]['beta_hi'][b]:5.2f}]".rjust(18) for h in report_horizons)
        print(f"  {letter_from_index(b):7s} {row}")

    # --- Global vs per-basin comparison ---
    print(f"\n=== Global (whole-domain) beta, identical methodology ===")
    if global_res is not None:
        for h in report_horizons:
            g = global_res[h]
            print(f"  yr{h:4.0f}: beta_global = {g['beta_b'][0]:.3f} "
                  f"[95% CI {g['beta_lo'][0]:.2f},{g['beta_hi'][0]:.2f}]  "
                  f"(r_global={g['r_b'][0]:.2f}, cf. literature reference beta~0.62 "
                  f"against ~6-7.6x realized amplitude)")
        beta_global_headline = global_res[a.horizon]["beta_b"][0]
    else:
        print("  unavailable")
        beta_global_headline = np.nan

    valid_idx = np.where(valid)[0]
    misi_valid = [b for b in valid_idx if b in MISI_INDICES]
    nonmisi_valid = [b for b in valid_idx if b not in MISI_INDICES]
    if misi_valid and nonmisi_valid:
        misi_beta = res["beta_b"][misi_valid]
        nonmisi_beta = res["beta_b"][nonmisi_valid]
        print(f"\n  MISI basins (G-H, J-K) beta_b: {dict(zip([letter_from_index(b) for b in misi_valid], np.round(misi_beta,3)))}")
        print(f"  MISI mean beta_b = {np.mean(misi_beta):.3f}  |  non-MISI mean beta_b = {np.mean(nonmisi_beta):.3f}")
        ranked = valid_idx[np.argsort(res["beta_b"][valid_idx])]
        lowest_two = set(ranked[:2].tolist())
        misi_are_lowest = lowest_two == set(misi_valid) if len(misi_valid) == 2 else set(misi_valid).issubset(
            set(ranked[:max(2, len(misi_valid))].tolist()))
        print(f"  Lowest-beta basin(s): {[letter_from_index(b) for b in ranked[:3]]}")
        print(f"  MISI-basins-have-lowest-beta: {misi_are_lowest}")
    else:
        print("  MISI/non-MISI comparison unavailable (one or both MISI basins skipped)")

    # =========================================================================
    # Panel (b): beta_b vs remaining floating-ice fraction
    # =========================================================================
    bvals = res["beta_b"][valid_idx]
    fvals = ice_frac[valid_idx]
    ok = np.isfinite(bvals) & np.isfinite(fvals)
    if ok.sum() >= 3:
        pr, pp = pearsonr(fvals[ok], bvals[ok])
        slope, intercept = np.polyfit(fvals[ok], bvals[ok], 1)
    else:
        pr, pp, slope, intercept = np.nan, np.nan, np.nan, np.nan
    print(f"\n=== Panel (b): beta_b vs remaining floating-ice fraction ===")
    print(f"  Pearson r = {pr:+.3f} (p={pp:.3g}), n={int(ok.sum())}")
    print(f"  {'A negative' if (np.isfinite(pr) and pr < 0) else 'No clear negative'} relationship "
          f"{'supports' if (np.isfinite(pr) and pr < 0) else 'does not clearly support'} "
          f"the saturation mechanism.")

    # =========================================================================
    # CAVEATS
    # =========================================================================
    se_sigma_pct_1x = 100.0 / np.sqrt(2 * (n1 - 1))
    se_sigma_pct_10x = 100.0 / np.sqrt(2 * (n10 - 1))
    print("\n=== CAVEATS ===")
    print(f"1. n={n1} (1x) / n={n10} (10x) members -> sigma carries ~+-{se_sigma_pct_1x:.0f}% / "
          f"+-{se_sigma_pct_10x:.0f}% sampling uncertainty (1/sqrt(2(n-1)) for a std-dev "
          f"estimate); beta_b inherits this via two nested log-ratios, hence the wide "
          f"bootstrap CIs above.")
    if skipped:
        print(f"2. {len(skipped)} basin(s) skipped as non-physical (sigma_1x near/at zero, or "
              f"r_b<=1 i.e. forcing did not actually amplify in that basin): "
              + ", ".join(f"{letter} ({nm})" for letter, nm, _ in skipped))
    else:
        print("2. No basins were skipped -- all had sigma_1x>0 and r_b>1.")
    print("3. beta_b is measured under SSP585 forcing only; transfer to other scenarios "
          "or to unforced (CTRL) noise is an untested assumption.")

    # =========================================================================
    # Figure: 3 panels
    # =========================================================================
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(19, 6.2))

    # --- Panel (a): horizontal bar chart of beta_b with 95% CI, sorted ---
    plot_idx = valid_idx[np.argsort(res["beta_b"][valid_idx])]
    y = np.arange(len(plot_idx))
    bvals_p = res["beta_b"][plot_idx]
    lo_p = bvals_p - res["beta_lo"][plot_idx]
    hi_p = res["beta_hi"][plot_idx] - bvals_p
    colors = ["crimson" if b in MISI_INDICES else "C0" for b in plot_idx]
    ax_a.barh(y, bvals_p, xerr=[np.clip(lo_p, 0, None), np.clip(hi_p, 0, None)],
              color=colors, capsize=3, height=0.65,
              error_kw=dict(ecolor="0.3", lw=1))
    ax_a.axvline(1.0, color="0.3", ls="--", lw=1.2)
    ax_a.text(1.0, len(y) - 0.3, " linear expectation ($\\beta$=1)", fontsize=8,
              va="bottom", ha="left", color="0.3")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"{letter_from_index(b)} {short_label(b)}" for b in plot_idx], fontsize=8)
    ax_a.set_xlabel(r"$\beta_b$ (per-basin amplitude exponent)")
    ax_a.set_title(f"(a) Per-basin $\\beta_b$ [95% bootstrap CI], yr{a.horizon:.0f}\n"
                   f"red = MISI basins (G-H, J-K)")
    ax_a.grid(alpha=0.2, axis="x")

    # --- Panel (b): beta_b vs remaining floating-ice fraction ---
    for b in valid_idx:
        c = "crimson" if b in MISI_INDICES else "C0"
        ax_b.scatter(ice_frac[b], res["beta_b"][b], color=c, zorder=3)
        ax_b.annotate(short_label(b), (ice_frac[b], res["beta_b"][b]), fontsize=7,
                      xytext=(3, 3), textcoords="offset points", alpha=0.85)
    if np.isfinite(slope):
        xf = np.linspace(np.nanmin(fvals[ok]), np.nanmax(fvals[ok]), 50)
        ax_b.plot(xf, slope * xf + intercept, "0.3", lw=1.2, ls="--")
    ax_b.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax_b.set_xlabel(f"remaining floating-ice fraction at yr{a.horizon:.0f} (1x ensemble mean)")
    ax_b.set_ylabel(r"$\beta_b$")
    ax_b.set_title(f"(b) $\\beta_b$ vs ice-shelf survival\nPearson r={pr:+.2f} (p={pp:.3g})")
    ax_b.grid(alpha=0.2)

    # --- Panel (c): realized ratio r_b vs nominal 10x ---
    for b in valid_idx:
        c = "crimson" if b in MISI_INDICES else "C0"
        ax_c.scatter(letter_from_index(b), res["r_b"][b], color=c, zorder=3, s=40)
    ax_c.axhline(10.0, color="0.3", ls="--", lw=1.2, label="nominal 10x")
    ax_c.set_ylabel(r"realized forcing-amplitude ratio $r_b = A_b(10x)/A_b(1x)$")
    ax_c.set_xlabel("basin")
    ax_c.set_title(f"(c) Where is the 10x forcing actually delivered?\nyr{a.horizon:.0f}")
    ax_c.tick_params(axis="x", rotation=90, labelsize=8)
    ax_c.legend(loc="best", fontsize=8)
    ax_c.grid(alpha=0.2, axis="y")

    fig.suptitle("Per-basin amplitude exponent $\\beta_b$: does sub-linear scaling concentrate "
                 "in the MISI (saturating) basins?", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure -> {a.out}")


if __name__ == "__main__":
    main()
