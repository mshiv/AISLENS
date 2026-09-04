#!/usr/bin/env python3
"""
fig_amplitude_vs_color.py — separates AMPLITUDE from SPECTRAL COLOR in ensemble spread.

Uses SSP585 (1x) vs SSP585_varScaled10x (10x) to calibrate amplitude sensitivity,
then decomposes the legacy 1000yr-generator no-offset run (CTRL-1000GEN-NOOFF) spread
into amplitude-explained and color-residual components.
Four panels: calibration, rectification, color decomposition, summary.
Data: globalStats from ensemble runs.

Author: Shivaprakash Muruganandham (2026-07-27)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio

# sigma ratios above this are treated as divide-by-near-zero artefacts, not physics
RATIO_SANITY_MAX = 50.0


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_ensemble(root, ensemble, include, variables, min_years=20, align="intersection"):
    """Load an ensemble's globalStats variables with a 'member' dim.

    Returns None (with a printed message) if the ensemble directory is missing, has
    no members matching `include`, or every member is dropped by min_years -- callers
    must handle None and skip the panel(s) that depend on it.
    """
    ens_dir = os.path.join(root, ensemble)
    if not os.path.isdir(ens_dir):
        print(f"  [amplitude_vs_color] ensemble '{ensemble}' NOT FOUND under {root} -- skipping")
        return None
    try:
        ds = eio.load_ensemble_globalstats(
            ens_dir, variables=variables, include=include,
            min_years=min_years, align=align)
    except Exception as e:
        print(f"  [amplitude_vs_color] failed to load ensemble '{ensemble}': {e} -- skipping")
        return None
    print(f"  [amplitude_vs_color] {ensemble}: {ds.sizes['member']} members, "
          f"year 0-{float(ds['year'].values[-1]):.1f}")
    return ds


def sle_da(ds):
    return xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, reference="first"),
                          ds["volumeAboveFloatation"])


# ----------------------------------------------------------------------------
# Bootstrap helpers (member resampling)
# ----------------------------------------------------------------------------
def _common_years(da_num, da_den, yr_lo=None, yr_hi=None):
    yr_a = da_num["year"].values
    yr_b = da_den["year"].values
    lo = max(float(yr_a[0]), float(yr_b[0]))
    hi = min(float(yr_a[-1]), float(yr_b[-1]))
    if yr_lo is not None:
        lo = max(lo, yr_lo)
    if yr_hi is not None:
        hi = min(hi, yr_hi)
    yr = yr_a[(yr_a >= lo) & (yr_a <= hi)]
    if len(yr) == 0:
        return np.array([])
    return yr


def bootstrap_ratio_sigma(num, den, n_boot=2000, ci=0.95, seed=42, yr_lo=None, yr_hi=None):
    """Bootstrap sigma_num(t)/sigma_den(t) by resampling members (with replacement)
    independently in each ensemble. Returns (year, ratio_mean, ratio_lo, ratio_hi)."""
    yr = _common_years(num, den, yr_lo, yr_hi)
    if len(yr) == 0:
        return yr, np.array([]), np.array([]), np.array([])
    num_c = num.interp(year=yr)
    den_c = den.interp(year=yr)
    n_num = num_c.sizes["member"]
    n_den = den_c.sizes["member"]
    rng = np.random.RandomState(seed)
    ratio_boot = np.full((n_boot, len(yr)), np.nan)
    for b in range(n_boot):
        idx_n = rng.choice(n_num, n_num, replace=True)
        idx_d = rng.choice(n_den, n_den, replace=True)
        s_num = num_c.isel(member=idx_n).std("member").values
        s_den = den_c.isel(member=idx_d).std("member").values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_boot[b] = np.where(s_den > 0, s_num / s_den, np.nan)
    alpha = (1.0 - ci) / 2.0
    # sigma is trivially 0 at year 0 (SLE is referenced to the first timestep), giving a
    # 0/0 -> NaN column there by construction, not a real problem -- suppress the noisy
    # "Mean of empty slice" / "All-NaN slice" warnings that produces.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        ratio_mean = np.nanmean(ratio_boot, axis=0)
        ratio_lo = np.nanpercentile(ratio_boot, 100 * alpha, axis=0)
        ratio_hi = np.nanpercentile(ratio_boot, 100 * (1 - alpha), axis=0)
    return yr, ratio_mean, ratio_lo, ratio_hi


def bootstrap_beta(sle_1x, sle_10x, amp_ratio=10.0, yr_lo=50.0, yr_hi=300.0,
                    n_boot=2000, seed=42, ci=0.95):
    """Member-bootstrap of beta(t) = log(sigma_10x/sigma_1x)/log(amp_ratio) and of a
    single representative beta = median_t beta(t) over [yr_lo, yr_hi].

    Returns (year, beta_t_mean, beta_rep, beta_rep_lo, beta_rep_hi).
    """
    yr = _common_years(sle_1x, sle_10x, yr_lo, yr_hi)
    if len(yr) == 0:
        return yr, np.array([]), np.nan, np.nan, np.nan
    x1 = sle_1x.interp(year=yr)
    x10 = sle_10x.interp(year=yr)
    n1 = x1.sizes["member"]
    n10 = x10.sizes["member"]
    rng = np.random.RandomState(seed)
    log_amp = np.log(amp_ratio)
    beta_t_boot = np.full((n_boot, len(yr)), np.nan)
    beta_rep_boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx1 = rng.choice(n1, n1, replace=True)
        idx10 = rng.choice(n10, n10, replace=True)
        s1 = x1.isel(member=idx1).std("member").values
        s10 = x10.isel(member=idx10).std("member").values
        with np.errstate(divide="ignore", invalid="ignore"):
            beta_t = np.where((s1 > 0) & (s10 > 0), np.log(s10 / s1) / log_amp, np.nan)
        beta_t_boot[b] = beta_t
        beta_rep_boot[b] = np.nanmedian(beta_t)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        beta_t_mean = np.nanmean(beta_t_boot, axis=0)
        alpha = (1.0 - ci) / 2.0
        beta_rep = float(np.nanmedian(beta_rep_boot))
        beta_rep_lo, beta_rep_hi = np.nanpercentile(beta_rep_boot, [100 * alpha, 100 * (1 - alpha)])
    return yr, beta_t_mean, beta_rep, float(beta_rep_lo), float(beta_rep_hi)


def window_member_means(da, yr_lo, yr_hi):
    """Per-member scalar = time-mean of da over [yr_lo, yr_hi]. Collapsing to one
    scalar per member first (rather than averaging per-timestep SEs) gives a
    defensible member-independent SE even though the time series are autocorrelated."""
    yr = da["year"].values
    mask = (yr >= yr_lo) & (yr <= yr_hi)
    if not mask.any():
        return np.array([])
    return da.isel(year=np.where(mask)[0]).mean("year").values


def mean_diff_significance(da_1x, da_10x, yr_lo, yr_hi):
    """Window-mean difference (10x - 1x) of a per-member-collapsed quantity, its
    standard error (independent-samples), and a z-score. Returns a dict."""
    m1 = window_member_means(da_1x, yr_lo, yr_hi)
    m10 = window_member_means(da_10x, yr_lo, yr_hi)
    if len(m1) == 0 or len(m10) == 0:
        return None
    mean1, mean10 = float(np.mean(m1)), float(np.mean(m10))
    se1 = float(np.std(m1, ddof=1) / np.sqrt(len(m1)))
    se10 = float(np.std(m10, ddof=1) / np.sqrt(len(m10)))
    se_diff = float(np.sqrt(se1**2 + se10**2))
    diff = mean10 - mean1
    z = diff / se_diff if se_diff > 0 else np.nan
    pct = 100.0 * diff / mean1 if mean1 != 0 else np.nan
    return dict(mean1=mean1, mean10=mean10, diff=diff, se_diff=se_diff, z=z, pct=pct,
                n1=len(m1), n10=len(m10))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--amp-ensembles", default="SSP585,SSP585_varScaled10x",
                     help="comma-separated (1x_ensemble,10x_ensemble) pair for amplitude calibration")
    ap.add_argument("--color-ensembles", default="CTRL,CTRL-1000GEN-NOOFF",
                     help="comma-separated (CTRL,CTRL-1000GEN-NOOFF) pair for color decomposition; "
                          "second entry is the legacy 1000yr-generator no-offset run and may be "
                          "absent if that ensemble doesn't exist")
    ap.add_argument("--amplitude-ratio", type=float, default=1.82,
                     help="ratio of forcing STANDARD DEVIATIONS between the 1000yr and 300yr "
                          "melt-forcing generators (CTRL-1000GEN-NOOFF, the legacy 1000yr-generator "
                          "no-offset run, vs CTRL). Default 1.82 = "
                          "sqrt(3.3), from a measured forcing VARIANCE ratio of ~3.3x. This is "
                          "the amplitude-only part of the CTRL-1000GEN-NOOFF/CTRL difference; the "
                          "generators also differ in spectral color (~8x more multidecadal "
                          "power in the 1000yr generator), which this ratio does NOT capture.")
    ap.add_argument("--horizons", default="50,100,200",
                     help="comma-separated year horizons for panel D")
    ap.add_argument("--min-years", type=float, default=20.0,
                     help="drop members shorter than this many years")
    ap.add_argument("--out", default="reports/figures/amplitude_vs_color.png")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    amp_names = [s.strip() for s in a.amp_ensembles.split(",") if s.strip()]
    color_names = [s.strip() for s in a.color_ensembles.split(",") if s.strip()]
    horizons = [float(h) for h in a.horizons.split(",") if h.strip() != ""]

    if len(amp_names) != 2:
        sys.exit("--amp-ensembles must name exactly two ensembles: 1x,10x")
    name_1x, name_10x = amp_names
    name_ctrl = color_names[0] if len(color_names) >= 1 else None
    name_ctrlext = color_names[1] if len(color_names) >= 2 else None

    print(f"ENSEMBLES root: {a.root}")
    print(f"Amplitude pair : {name_1x} (1x) vs {name_10x} (10x)")
    print(f"Color pair     : {name_ctrl} vs {name_ctrlext}")
    print(f"amplitude_ratio (SD, 1000yr/300yr generator) = {a.amplitude_ratio}")

    # --- Load amplitude pair (required) ---
    ds_1x = load_ensemble(a.root, name_1x, r"^SSP585_\d+$",
                          ["volumeAboveFloatation", "avgSubshelfMelt"], min_years=a.min_years)
    ds_10x = load_ensemble(a.root, name_10x, r"^SSP585_\d+$",
                           ["volumeAboveFloatation", "avgSubshelfMelt"], min_years=a.min_years)
    if ds_1x is None or ds_10x is None:
        sys.exit("Amplitude-calibration pair could not be loaded -- nothing to plot. "
                 "Check --root / --amp-ensembles.")
    sle_1x, sle_10x = sle_da(ds_1x), sle_da(ds_10x)
    melt_1x, melt_10x = ds_1x["avgSubshelfMelt"], ds_10x["avgSubshelfMelt"]

    # --- Load color pair (optional; CTRL-EXT may not exist yet) ---
    ds_ctrl = load_ensemble(a.root, name_ctrl, r"^CTRL_?\d+$",
                            ["volumeAboveFloatation"], min_years=a.min_years) if name_ctrl else None
    ds_ctrlext = load_ensemble(a.root, name_ctrlext, r"^CTRL_?\d+$",
                               ["volumeAboveFloatation"], min_years=a.min_years) if name_ctrlext else None
    have_color = ds_ctrl is not None and ds_ctrlext is not None
    if not have_color:
        print("  [amplitude_vs_color] color-decomposition pair incomplete -- "
              "panels C and D will be skipped; panels A and B still produced.")
    sle_ctrl = sle_da(ds_ctrl) if ds_ctrl is not None else None
    sle_ctrlext = sle_da(ds_ctrlext) if ds_ctrlext is not None else None

    # =====================================================================
    # Panel A: amplitude calibration
    # =====================================================================
    yr_beta, beta_t, beta_rep, beta_lo, beta_hi = bootstrap_beta(
        sle_1x, sle_10x, amp_ratio=10.0, yr_lo=50.0, yr_hi=300.0, n_boot=2000)
    print(f"\n=== Panel A: amplitude exponent beta (sigma ~ A^beta, SSP585 1x vs 10x) ===")
    print(f"  beta(yr50-300 median) = {beta_rep:.3f}  [95% bootstrap CI {beta_lo:.3f}, {beta_hi:.3f}]")
    print(f"  (beta=1 -> linear; beta<1 -> sub-linear/damped response to amplitude)")

    sigma_1x_full = sle_1x.std("member")
    sigma_10x_full = sle_10x.std("member")

    # =====================================================================
    # Panel B: rectification diagnostic
    # =====================================================================
    mean_melt_1x = melt_1x.mean("member")
    mean_melt_10x = melt_10x.mean("member")
    mean_sle_1x = sle_1x.mean("member")
    mean_sle_10x = sle_10x.mean("member")

    yr_common_b = _common_years(mean_melt_1x, mean_melt_10x)
    melt_diff_t = (mean_melt_10x.interp(year=yr_common_b) - mean_melt_1x.interp(year=yr_common_b)).values

    stats_50 = mean_diff_significance(melt_1x, melt_10x, 0.0, 50.0)
    yr_hi_full = min(float(melt_1x["year"].values[-1]), float(melt_10x["year"].values[-1]))
    stats_300 = mean_diff_significance(melt_1x, melt_10x, 0.0, yr_hi_full)

    print(f"\n=== Panel B: rectification diagnostic (ensemble-mean avgSubshelfMelt, 10x - 1x) ===")
    for label, s in [("yr0-50", stats_50), (f"yr0-{yr_hi_full:.0f}", stats_300)]:
        if s is None:
            print(f"  {label}: insufficient overlap")
            continue
        print(f"  {label}: mean_1x={s['mean1']:+.4f} m/yr  mean_10x={s['mean10']:+.4f} m/yr  "
              f"diff={s['diff']:+.4f} m/yr ({s['pct']:+.1f}%)  "
              f"SE={s['se_diff']:.4f}  z={s['z']:+.2f} (n1={s['n1']}, n10={s['n10']})")

    # Verdict based on the early (yr0-50) window: closest to a pure forcing-only test,
    # before geometry/ice-state feedbacks confound the comparison (see caveat 2 analog).
    verdict_stats = stats_50 if stats_50 is not None else stats_300
    if verdict_stats is not None and np.isfinite(verdict_stats["z"]) and abs(verdict_stats["z"]) >= 1.96:
        verdict = (f"RECTIFICATION DETECTED (10x mean melt differs by "
                   f"{verdict_stats['pct']:+.1f}%, z={verdict_stats['z']:+.1f})")
    elif verdict_stats is not None:
        verdict = (f"NO significant rectification (10x mean melt differs by "
                   f"{verdict_stats['pct']:+.1f}%, z={verdict_stats['z']:+.1f}, |z|<1.96)")
    else:
        verdict = "rectification test inconclusive (insufficient overlap)"
    print(f"  VERDICT: {verdict}")

    # =====================================================================
    # Panel C & D: color decomposition (only if both CTRL and CTRL-EXT loaded)
    # =====================================================================
    if have_color:
        sigma_ctrl_full = sle_ctrl.std("member")
        sigma_ctrlext_full = sle_ctrlext.std("member")
        ctrl_max_yr = float(sle_ctrl["year"].values[-1])
        ctrlext_max_yr = float(sle_ctrlext["year"].values[-1])
        print(f"\n=== Panel C/D: color decomposition ({name_ctrl} vs {name_ctrlext}) ===")
        print(f"  {name_ctrl} usable span: 0-{ctrl_max_yr:.1f} yr (n={sle_ctrl.sizes['member']})")
        print(f"  {name_ctrlext} usable span: 0-{ctrlext_max_yr:.1f} yr (n={sle_ctrlext.sizes['member']})")
        if ctrl_max_yr < max(horizons):
            print(f"  NOTE: {name_ctrl} only reaches yr{ctrl_max_yr:.1f} -- horizons beyond that "
                  f"cannot be compared and will be dropped from panel D.")

        amp_factor = a.amplitude_ratio ** beta_rep
        print(f"  amplitude-explained factor = amplitude_ratio^beta = "
              f"{a.amplitude_ratio}^{beta_rep:.3f} = {amp_factor:.3f}")

        yr_ratio, ratio_mean, ratio_lo, ratio_hi = bootstrap_ratio_sigma(
            sle_ctrlext, sle_ctrl, n_boot=2000)

        # dashed reference: sigma_CTRL(t) * amp_factor, only where sigma_CTRL is defined
        yr_dashed = sigma_ctrl_full["year"].values
        dashed = sigma_ctrl_full.values * amp_factor

        # per-horizon decomposition (panel D), dropping horizons CTRL can't reach
        valid_horizons, obs_ratio_h, obs_lo_h, obs_hi_h, color_factor_h = [], [], [], [], []
        dropped_horizons = []
        for h in horizons:
            if h > ctrl_max_yr or h > ctrlext_max_yr:
                dropped_horizons.append(h)
                continue
            i = int(np.argmin(np.abs(yr_ratio - h)))
            # Guard against a near-zero denominator: sigma_CTRL passes close to zero at early
            # horizons (all members start from the same state), so sigma_EXT/sigma_CTRL can blow
            # up by many orders of magnitude and is meaningless there. Such a ratio also produces
            # a bootstrap CI whose lower bound sits below the mean, which matplotlib rejects as a
            # negative yerr. Drop the horizon instead of plotting a spurious number.
            r_i, lo_i, hi_i = ratio_mean[i], ratio_lo[i], ratio_hi[i]
            if (not np.isfinite(r_i)) or r_i <= 0 or r_i > RATIO_SANITY_MAX:
                dropped_horizons.append(h)
                print(f"  [WARN] horizon yr{h:.0f}: sigma ratio {r_i:.3g} is not physical "
                      f"(sigma_{name_ctrl} near zero) — dropped from panel D")
                continue
            # clip the CI so it brackets the mean (bootstrap tails can invert near zero)
            lo_i = min(lo_i, r_i) if np.isfinite(lo_i) else r_i
            hi_i = max(hi_i, r_i) if np.isfinite(hi_i) else r_i
            valid_horizons.append(h)
            obs_ratio_h.append(r_i)
            obs_lo_h.append(lo_i)
            obs_hi_h.append(hi_i)
            color_factor_h.append(r_i / amp_factor)
        if dropped_horizons:
            print(f"  Dropped horizon(s) {dropped_horizons} yr from panel D "
                  f"(exceed {name_ctrl} usable span of {ctrl_max_yr:.1f} yr)")
        print(f"  {'yr':>6s} {'obs_ratio':>10s} {'95% CI':>18s} {'amp_factor':>11s} {'color_factor(resid)':>20s}")
        for h, r, lo, hi, c in zip(valid_horizons, obs_ratio_h, obs_lo_h, obs_hi_h, color_factor_h):
            print(f"  {h:6.0f} {r:10.3f} [{lo:6.3f},{hi:6.3f}] {amp_factor:11.3f} {c:20.3f}")

        # mean-melt bias between CTRL and CTRL-EXT (caveat 2), reported for the reader
        # (avgSubshelfMelt not requested for CTRL loads above -> reload lightly if present)
        try:
            ds_ctrl_melt = load_ensemble(a.root, name_ctrl, r"^CTRL_?\d+$",
                                         ["avgSubshelfMelt"], min_years=a.min_years)
            ds_ctrlext_melt = load_ensemble(a.root, name_ctrlext, r"^CTRL_?\d+$",
                                            ["avgSubshelfMelt"], min_years=a.min_years)
            if ds_ctrl_melt is not None and ds_ctrlext_melt is not None:
                s_ctrl_bias = mean_diff_significance(
                    ds_ctrl_melt["avgSubshelfMelt"], ds_ctrlext_melt["avgSubshelfMelt"],
                    0.0, min(ctrl_max_yr, ctrlext_max_yr))
                if s_ctrl_bias is not None:
                    print(f"  CAVEAT 2 check: {name_ctrlext} mean melt - {name_ctrl} mean melt "
                          f"(no OFFSET correction confound) = {s_ctrl_bias['diff']:+.4f} m/yr "
                          f"({s_ctrl_bias['pct']:+.1f}% of {name_ctrl} mean), z={s_ctrl_bias['z']:+.2f}")
        except Exception as e:
            print(f"  CAVEAT 2 melt-bias check skipped: {e}")
    else:
        sigma_ctrl_full = sigma_ctrlext_full = None
        yr_dashed = dashed = None
        valid_horizons = obs_ratio_h = obs_lo_h = obs_hi_h = color_factor_h = []
        amp_factor = a.amplitude_ratio ** beta_rep

    # =====================================================================
    # Printed caveats (always)
    # =====================================================================
    print("\n=== CAVEATS ===")
    print("1. beta is measured under SSP585 (forced) noise and transferred to CTRL "
          "(unforced) noise -- an ASSUMPTION. Recommend a CTRL_varScaled10x run to "
          "measure beta directly in the unforced regime.")
    print(f"2. {name_ctrlext or 'the color-pair second ensemble'} was run WITHOUT the OFFSET "
          "baseline correction CTRL has, so it also carries a mean-melt bias that shifts "
          "the operating state -- a confound for spread independent of amplitude/color. "
          "Compare early years where feasible; see the CAVEAT 2 numeric check above "
          "(if color pair available).")
    print("3. The color factor C = observed_ratio / amplitude_factor is a RESIDUAL: it "
          "absorbs every unmodeled difference (including caveats 1 and 2), so it is "
          "NOT a clean, isolated measurement of spectral color alone.")
    print(f"4. Member counts: {name_1x} n={sle_1x.sizes['member']}, {name_10x} "
          f"n={sle_10x.sizes['member']}"
          + (f", {name_ctrl} n={sle_ctrl.sizes['member']}, {name_ctrlext} n={sle_ctrlext.sizes['member']}"
             if have_color else "")
          + " -- sigma carries substantial sampling uncertainty at these n; ratios above "
            "propagate it via member-bootstrap CIs.")

    # =====================================================================
    # Figure
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- Panel A ---
    ax_a2 = ax_a.twinx()
    ax_a.plot(sigma_1x_full["year"].values, sigma_1x_full.values, "C0", lw=1.8, label=f"{name_1x} (1x)")
    ax_a.plot(sigma_10x_full["year"].values, sigma_10x_full.values, "C3", lw=1.8, label=f"{name_10x} (10x)")
    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"$\sigma_{\mathrm{SLE}}$ (mm), log scale")
    ax_a2.plot(yr_beta, beta_t, "0.3", lw=1.2, ls="--", label=r"$\beta(t)$")
    ax_a2.axhspan(beta_lo, beta_hi, color="0.3", alpha=0.12)
    ax_a2.axhline(beta_rep, color="0.2", lw=0.8, ls=":")
    ax_a2.set_ylabel(r"$\beta(t) = \log(\sigma_{10x}/\sigma_{1x})/\log(10)$")
    ax_a.set_xlabel("year")
    ax_a.set_title(f"(A) Amplitude calibration: "
                   f"$\\beta_{{rep}}$={beta_rep:.2f} [{beta_lo:.2f},{beta_hi:.2f}] (yr50-300)")
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax_a.grid(alpha=0.2)

    # --- Panel B ---
    ax_b2 = ax_b.twinx()
    ax_b.plot(mean_melt_1x["year"].values, mean_melt_1x.values, "C0", lw=1.8, label=f"{name_1x} mean melt")
    ax_b.plot(mean_melt_10x["year"].values, mean_melt_10x.values, "C3", lw=1.8, ls="--",
              label=f"{name_10x} mean melt")
    ax_b.set_ylabel("ensemble-mean avgSubshelfMelt (m/yr)")
    ax_b2.plot(mean_sle_1x["year"].values, mean_sle_1x.values, "C0", lw=1.0, alpha=0.5,
               label=f"{name_1x} mean SLE")
    ax_b2.plot(mean_sle_10x["year"].values, mean_sle_10x.values, "C3", lw=1.0, alpha=0.5, ls="--",
               label=f"{name_10x} mean SLE")
    ax_b2.set_ylabel("ensemble-mean SLE (mm)")
    ax_b.set_xlabel("year")
    ax_b.set_title("(B) Rectification diagnostic: does the mean respond to zero-mean noise?")
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    ax_b.grid(alpha=0.2)

    ax_b_in = ax_b.inset_axes([0.55, 0.08, 0.42, 0.32])
    ax_b_in.plot(yr_common_b, melt_diff_t, "C4", lw=1.2)
    ax_b_in.axhline(0, color="0.5", ls="--", lw=0.7)
    ax_b_in.set_title("mean melt: 10x − 1x", fontsize=7)
    ax_b_in.tick_params(labelsize=6)
    ax_b_in.grid(alpha=0.15)

    verdict_color = "crimson" if "DETECTED" in verdict else "seagreen"
    ax_b.text(0.98, 0.98, verdict.replace(" (", "\n("), transform=ax_b.transAxes,
              fontsize=7.5, va="top", ha="right", color=verdict_color, weight="bold",
              bbox=dict(boxstyle="round", fc="white", ec=verdict_color, alpha=0.95))

    # --- Panel C ---
    if have_color:
        ax_c.plot(sigma_ctrl_full["year"].values, sigma_ctrl_full.values, "C0", lw=1.8,
                  label=f"{name_ctrl}")
        ax_c.plot(sigma_ctrlext_full["year"].values, sigma_ctrlext_full.values, "C3", lw=1.8,
                  label=f"{name_ctrlext} (observed)")
        ax_c.plot(yr_dashed, dashed, "0.3", lw=1.5, ls="--",
                  label=f"{name_ctrl} × amplitude_ratio$^\\beta$ (amplitude-only prediction)")
        ax_c.set_yscale("log")
        ax_c.set_xlabel("year")
        ax_c.set_ylabel(r"$\sigma_{\mathrm{SLE}}$ (mm), log scale")
        ax_c.set_title(f"(C) Color decomposition: {name_ctrlext} vs amplitude-only prediction")
        ax_c.legend(loc="upper left", fontsize=7.5)
        ax_c.grid(alpha=0.2)
    else:
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, f"Color pair incomplete\n({name_ctrl!r}, {name_ctrlext!r})\n"
                  "-- panel skipped", transform=ax_c.transAxes, ha="center", va="center",
                  fontsize=10, color="0.4")

    # --- Panel D ---
    if have_color and valid_horizons:
        x = np.arange(len(valid_horizons))
        w = 0.35
        obs_err = [np.array(obs_ratio_h) - np.array(obs_lo_h), np.array(obs_hi_h) - np.array(obs_ratio_h)]
        ax_d.bar(x - w/2, [amp_factor] * len(valid_horizons), width=w, color="C0",
                 label="amplitude-explained (r_A$^\\beta$)")
        ax_d.bar(x + w/2, obs_ratio_h, width=w, yerr=obs_err, color="C3", capsize=3,
                 label="observed ratio (bootstrap 95% CI)")
        ax_d.axhline(1.0, color="0.5", ls=":", lw=1)
        for xi, c in zip(x, color_factor_h):
            ax_d.text(xi, max(amp_factor, obs_ratio_h[x.tolist().index(xi)]) * 1.05,
                      f"C={c:.2f}", ha="center", fontsize=8)
        ax_d.set_xticks(x)
        ax_d.set_xticklabels([f"yr{h:.0f}" for h in valid_horizons])
        ax_d.set_ylabel(f"$\\sigma_{{{name_ctrlext}}} / \\sigma_{{{name_ctrl}}}$")
        ax_d.set_title("(D) Amplitude- vs color-explained spread ratio")
        ax_d.legend(loc="upper left", fontsize=8)
        ax_d.grid(alpha=0.2, axis="y")
    else:
        ax_d.axis("off")
        msg = ("Color pair incomplete -- panel skipped" if not have_color
               else "No horizon within usable overlap -- panel skipped")
        ax_d.text(0.5, 0.5, msg, transform=ax_d.transAxes, ha="center", va="center",
                  fontsize=10, color="0.4")

    fig.suptitle("Separating amplitude and spectral-color effects on ensemble spread",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure -> {a.out}")


if __name__ == "__main__":
    main()
