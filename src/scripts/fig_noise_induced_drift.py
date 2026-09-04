#!/usr/bin/env python3
"""
fig_noise_induced_drift.py — Tests if stochastic forcing biases the ensemble mean.

Panel (a): SSP585 ensemble mean VAF→SLE vs DET-SSP585 over time.
Panel (b): difference (ensemble mean − deterministic).
Panel (c): signal-to-noise ratio of the drift.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio


def load_ensemble_mean(root, ensemble, include, min_years=50):
    ds = eio.load_ensemble_globalstats(
        os.path.join(root, ensemble),
        variables=["volumeAboveFloatation", "daysSinceStart"],
        include=include, min_years=min_years, align="union")
    sle = xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, reference="first"),
                         ds["volumeAboveFloatation"])
    return ds["year"].values, sle.mean("member").values, sle.std("member", ddof=1).values, sle.sizes["member"]


def load_deterministic(root, det_dir):
    """Load a deterministic run (globalStats.nc directly or one level nested)."""
    path = os.path.join(root, det_dir, "globalStats.nc")
    if not os.path.isfile(path):
        path = os.path.join(root, det_dir, det_dir, "globalStats.nc")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DET run not found in {root}/{det_dir}/")
    ds = xr.open_dataset(path, decode_times=False)
    if "daysSinceStart" in ds:
        yr = ds["daysSinceStart"].values / 365.0
        ds = ds.assign_coords(year=("Time", yr))
    ds = ds.swap_dims({"Time": "year"})
    yr = ds["year"].values
    _, first_idx = np.unique(yr, return_index=True)
    ds = ds.isel(year=np.sort(first_idx)).sortby("year")
    vaf = ds["volumeAboveFloatation"].values
    sle = eio.vaf_to_sle_mm(vaf, reference="first")
    return ds["year"].values, sle


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--start-year", type=float, default=2000.0)
    ap.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    ap.add_argument("--no-member-counts", action="store_true",
                    help="omit '(n=N)' from legend labels (publication style)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    yr_e, mean_e, std_e, n_e = load_ensemble_mean(a.root, "SSP585", r"^SSP585_\d+$")

    yr_d, sle_d = load_deterministic(a.root, "DET-SSP585")

    # Clip to overlap region (DET run may be shorter than ensemble)
    yr_lo = max(yr_e[0], yr_d[0])
    yr_hi = min(yr_e[-1], yr_d[-1])
    mask = (yr_e >= yr_lo) & (yr_e <= yr_hi)
    yr_plot = yr_e[mask]
    mean_e_plot = mean_e[mask]
    std_e_plot = std_e[mask]
    sle_d_plot = np.interp(yr_plot, yr_d, sle_d)
    cal = a.start_year + yr_plot

    # Difference = ensemble mean - deterministic
    diff = mean_e_plot - sle_d_plot
    # Standard error of the ensemble mean
    se_ens = std_e_plot / np.sqrt(n_e)
    # S/N of the drift relative to ensemble spread
    snr = np.abs(diff) / np.maximum(std_e_plot, 1e-9)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    drift_lbl = "SSP585 ensemble mean" if a.no_member_counts else f"SSP585 ensemble mean (n={n_e})"
    ax_a.plot(cal, mean_e_plot, "C3", lw=2, label=drift_lbl)
    ax_a.plot(cal, sle_d_plot, "0.3", lw=2, ls="--", label="DET-SSP585 (deterministic)")
    ax_a.fill_between(cal, mean_e_plot - 1.96*se_ens, mean_e_plot + 1.96*se_ens,
                      color="C3", alpha=0.2, label="95% CI of ensemble mean")
    ax_a.set_ylabel("VAF→SLE (mm)")
    ax_a.set_title("(a) Ensemble mean vs deterministic run")
    ax_a.legend(loc="upper left"); ax_a.grid(alpha=0.2)

    ax_b.plot(cal, diff, "C1", lw=2)
    ax_b.fill_between(cal, diff - 1.96*se_ens, diff + 1.96*se_ens,
                      color="C1", alpha=0.2, label="95% CI")
    ax_b.axhline(0, color="0.5", ls="--", lw=0.8)
    ax_b.set_ylabel("ensemble mean − deterministic (mm)")
    ax_b.set_title("(b) Noise-induced drift: stochastic forcing biases the mean?")
    ax_b.legend(loc="upper left"); ax_b.grid(alpha=0.2)

    ax_c.plot(cal, snr, "C4", lw=2)
    ax_c.axhline(1.0, color="0.5", ls="--", lw=0.8, label="S/N = 1 (drift = spread)")
    ax_c.set_xlabel("year")
    ax_c.set_ylabel("|drift| / σ_ensemble")
    ax_c.set_title("(c) Drift relative to ensemble spread")
    ax_c.legend(loc="upper left"); ax_c.grid(alpha=0.2)

    fig.suptitle("Noise-induced drift: does stochastic variability bias the mean?",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = os.path.join(a.out_dir, "noise_induced_drift.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")

    print(f"\nnoise-induced drift (SSP585 ensemble mean vs DET-SSP585), overlap yr{yr_lo:.0f}-{yr_hi:.0f}:")
    for h_yr in [50, 100, min(169, len(yr_plot)-1)]:
        i = min(h_yr, len(yr_plot)-1)
        print(f"  yr{yr_plot[i]:5.0f}  ens_mean={mean_e_plot[i]:+7.2f}  det={sle_d_plot[i]:+7.2f}  "
              f"drift={diff[i]:+7.2f}  σ={std_e_plot[i]:5.2f}  S/N={snr[i]:5.2f}")

    try:
        yr_c, mean_c, std_c, n_c = load_ensemble_mean(a.root, "CTRL", r"^CTRL_\d+$")
        yr_dc, sle_dc = load_deterministic(a.root, "DET-CTRL")
        yr_lo_c = max(yr_c[0], yr_dc[0])
        yr_hi_c = min(yr_c[-1], yr_dc[-1])
        mask_c = (yr_c >= yr_lo_c) & (yr_c <= yr_hi_c)
        yr_c_p = yr_c[mask_c]
        sle_dc_i = np.interp(yr_c_p, yr_dc, sle_dc)
        diff_c = mean_c[mask_c] - sle_dc_i
        print(f"\nnoise-induced drift (CTRL ensemble mean vs DET-CTRL), overlap yr{yr_lo_c:.0f}-{yr_hi_c:.0f}:")
        for h_yr in [50, 100, min(200, len(yr_c_p)-1)]:
            i = min(h_yr, len(yr_c_p)-1)
            print(f"  yr{yr_c_p[i]:5.0f}  ens_mean={mean_c[mask_c][i]:+7.2f}  det={sle_dc_i[i]:+7.2f}  "
                  f"drift={diff_c[i]:+7.2f}  σ={std_c[mask_c][i]:5.2f}")
    except Exception as e:
        print(f"\nCTRL drift test skipped: {e}")


if __name__ == "__main__":
    main()
