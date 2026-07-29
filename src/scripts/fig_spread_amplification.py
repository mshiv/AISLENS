#!/usr/bin/env python3
"""
fig_spread_amplification.py — Tests if forced warming amplifies internal variability.

Main: σ_SSP585(t) / σ_CTRL(t) with 95% CI from bootstrap. Inset: both sigmas on log scale.
Optional panel: σ_10x / σ_SSP585.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio


def load_sle(root, ensemble, include, min_years=50):
    """Return (year_values, sleDataMember, year) as xr.DataArray with member dim."""
    ds = eio.load_ensemble_globalstats(
        os.path.join(root, ensemble),
        variables=["volumeAboveFloatation", "daysSinceStart"],
        include=include, min_years=min_years, align="union")
    sle = xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, reference="first"),
                         ds["volumeAboveFloatation"])
    return sle  # DataArray(member, year)


def bootstrap_ratio_sigma(num, den, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap ratio of std across members at each timestep.

    num, den: DataArrays with (member, year). They may have different year grids.
    Interpolates both to the intersection of their year grids before computing.
    Returns ratio, lo, hi arrays along the common year axis.
    """
    rng = np.random.RandomState(seed)
    # Interpolate both to their overlapping year range
    yr_num = set(num["year"].values)
    yr_den = set(den["year"].values)
    yr_common = np.array(sorted(yr_num & yr_den))
    if len(yr_common) < 10:
        # Fall back to intersection via interp
        yr_lo = max(float(num["year"].values[0]), float(den["year"].values[0]))
        yr_hi = min(float(num["year"].values[-1]), float(den["year"].values[-1]))
        yr_common = num["year"].values[(num["year"].values >= yr_lo) & (num["year"].values <= yr_hi)]
    num_c = num.interp(year=yr_common)
    den_c = den.interp(year=yr_common)

    n_m = num_c.sizes["member"]
    ratio_boot = np.full((n_boot, len(yr_common)), np.nan)
    for b in range(n_boot):
        idx = rng.choice(n_m, size=n_m, replace=True)
        num_boot = num_c.isel(member=idx)
        den_boot = den_c.isel(member=idx)
        s_num = num_boot.std("member").values
        s_den = den_boot.std("member").values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_boot[b] = np.where(s_den > 0, s_num / s_den, np.nan)
    alpha = (1.0 - ci) / 2.0
    ratio_mean = np.nanmean(ratio_boot, axis=0)
    ratio_lo = np.nanpercentile(ratio_boot, 100 * alpha, axis=0)
    ratio_hi = np.nanpercentile(ratio_boot, 100 * (1 - alpha), axis=0)
    return yr_common, ratio_mean, ratio_lo, ratio_hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    ap.add_argument("--start-year", type=float, default=2000.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading CTRL...")
    ctrl_sle = load_sle(args.root, "CTRL", r"^CTRL_\d+$", min_years=50)
    print("Loading SSP585...")
    ssp_sle = load_sle(args.root, "SSP585", r"^SSP585_\d+$", min_years=50)
    try:
        print("Loading SSP585_varScaled10x...")
        var10x_sle = load_sle(args.root, "SSP585_varScaled10x", r"^SSP585_\d+$", min_years=50)
        has_var10x = True
    except Exception:
        has_var10x = False

    ctrl_sigma = ctrl_sle.std("member")
    ssp_sigma = ssp_sle.std("member")

    # Interpolate SSP to CTRL's year grid for the ratio
    year_common = ctrl_sigma["year"].values
    ssp_sigma_interp = ssp_sigma.interp(year=year_common)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ssp_sigma_interp / ctrl_sigma

    yr = args.start_year + year_common

    # Bootstrap CI for the ratio
    yr_boot, ratio_mean, ratio_lo, ratio_hi = bootstrap_ratio_sigma(ssp_sle, ctrl_sle, n_boot=2000)
    yr_boot_plot = args.start_year + yr_boot

    # --- Main figure ---
    fig, ax_main = plt.subplots(figsize=(8, 5))

    ax_main.fill_between(yr_boot_plot, ratio_lo, ratio_hi, color="C3", alpha=0.2, label="95% CI (bootstrap)")
    ax_main.plot(yr_boot_plot, ratio_mean, color="C3", lw=2, label=r"$\sigma_{\mathrm{SSP585}} / \sigma_{\mathrm{CTRL}}$")
    ax_main.axhline(1.0, color="0.4", ls="--", lw=1, label="no amplification")
    ax_main.set_xlabel("year")
    ax_main.set_ylabel(r"$\sigma_{\mathrm{SSP585}}(t)\, /\, \sigma_{\mathrm{CTRL}}(t)$")
    ax_main.set_title("Does forced warming amplify internal variability?")
    ax_main.legend(loc="best", fontsize=9)
    ax_main.grid(alpha=0.2)

    # Inset: sigma SSP585 and sigma CTRL separately (log scale)
    ax_in = ax_main.inset_axes([0.55, 0.55, 0.40, 0.38])
    ax_in.plot(yr, ctrl_sigma.values, "C0", lw=1.5, label="CTRL")
    ax_in.plot(yr, ssp_sigma_interp.values, "C3", lw=1.5, label="SSP585")
    if has_var10x:
        var10x_sigma = var10x_sle.std("member")
        yr_v10 = args.start_year + var10x_sigma["year"].values
        ax_in.plot(yr_v10, var10x_sigma.values, "C4", lw=1.2, ls=":", label="10x var")
    ax_in.set_yscale("log")
    ax_in.set_xlabel("year", fontsize=7)
    ax_in.set_ylabel(r"$\sigma$ (mm SLE)", fontsize=7)
    ax_in.tick_params(labelsize=6)
    ax_in.legend(fontsize=6, loc="upper left")
    ax_in.grid(alpha=0.15)

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "fig_spread_amplification.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {out_path}")

    # --- Optional second panel: 10x ratio ---
    if has_var10x:
        var10x_sigma_interp = var10x_sle.std("member").interp(year=year_common)
        ssp_sigma_on_common = ssp_sigma.interp(year=year_common)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_10x = var10x_sigma_interp / ssp_sigma_on_common

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.plot(yr, ratio_10x.values, color="C4", lw=2,
                 label=r"$\sigma_{\mathrm{10x}} / \sigma_{\mathrm{SSP585}}$")
        ax2.axhline(10.0, color="0.4", ls="--", lw=1, label="10x (target)")
        ax2.axhline(1.0, color="0.4", ls=":", lw=0.8, label="1x")
        ax2.set_xlabel("year")
        ax2.set_ylabel(r"$\sigma_{\mathrm{10x}} / \sigma_{\mathrm{SSP585}}$")
        ax2.set_title("10x forcing variability amplification ratio")
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(alpha=0.2)
        fig2.tight_layout()
        out2 = os.path.join(args.out_dir, "fig_spread_amplification_10x.png")
        fig2.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"Figure -> {out2}")


if __name__ == "__main__":
    main()
