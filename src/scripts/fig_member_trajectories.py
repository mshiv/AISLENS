#!/usr/bin/env python3
"""
fig_member_trajectories.py — Ensemble member dVAF/dt trajectories (butterfly diagram).

Shows all members' dVAF/dt over time per scenario. Layout: 1×3 (SSP585, varScaled10x, SSP126).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio

SCENARIO_DIRS = {"SSP585": "SSP585", "varScaled10x": "SSP585_varScaled10x", "SSP126": "SSP126"}
SCENARIO_INCLUDE = {"SSP585": r"^SSP585_\d+$", "varScaled10x": r"^SSP585_\d+$", "SSP126": r"^SSP126_\d+$"}
SCENARIO_COLORS = {"SSP585": "#C62828", "varScaled10x": "#E65100", "SSP126": "#1565C0"}


def load_ensemble_sle(root, ensemble, include):
    members = eio.discover_members(
        os.path.join(root, ensemble), stats_filename="globalStats.nc", include=include
    )
    per_member, names = [], []
    for name, path in members:
        try:
            ds = eio.to_year_dim(eio.load_member_globalstats(path))
        except Exception:
            continue
        if "volumeAboveFloatation" not in ds:
            continue
        yr = ds["year"].values
        if yr[0] > 5.0 or len(yr) < 10:
            continue
        vaf = ds["volumeAboveFloatation"].values
        sle = eio.vaf_to_sle_mm(vaf, reference="first")
        per_member.append((yr, sle))
        names.append(name)
    return per_member, names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--out-dir", default="/Users/smurugan9/research/aislens/AISLENS/reports/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scenarios = list(SCENARIO_DIRS.keys())
    nsc = len(scenarios)
    fig, axes = plt.subplots(1, nsc, figsize=(5.5 * nsc, 4.5))
    if nsc == 1:
        axes = [axes]

    for col, sc in enumerate(scenarios):
        ax = axes[col]
        per_member, names = load_ensemble_sle(args.root, SCENARIO_DIRS[sc], SCENARIO_INCLUDE[sc])
        if not per_member:
            ax.set_visible(False)
            continue

        for i, (yr, sle) in enumerate(per_member):
            if len(yr) < 3:
                continue
            dvdt = np.gradient(sle, yr)
            kernel = np.ones(10) / 10
            dvdt_smooth = np.convolve(dvdt, kernel, mode="valid")
            yr_smooth = yr[9:]
            ax.plot(yr_smooth, dvdt_smooth, color=SCENARIO_COLORS.get(sc, "gray"),
                    lw=0.6, alpha=0.5)

        # Ensemble mean
        # Intersect to common grid for mean
        common_yr = per_member[0][0]
        all_dvdt = []
        for yr, sle in per_member:
            dvdt = np.gradient(sle, yr)
            all_dvdt.append(np.interp(common_yr, yr, dvdt))
        ens_mean = np.mean(all_dvdt, axis=0)
        kernel = np.ones(10) / 10
        ens_mean_smooth = np.convolve(ens_mean, kernel, mode="valid")
        yr_smooth = common_yr[9:]
        ax.plot(yr_smooth, ens_mean_smooth, color="k", lw=2.0, label="Ensemble mean")
        ax.axhline(0, color="gray", lw=0.5, ls="--")

        ax.set_xlabel("Year")
        ax.set_ylabel("dVAF/dt (mm/yr SLE)" if col == 0 else "")
        ax.set_title(sc, fontsize=11, fontweight="bold", color=SCENARIO_COLORS.get(sc, "k"))
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Ensemble Member Trajectories — dVAF/dt (butterfly diagram)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "member_trajectories.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out}")


if __name__ == "__main__":
    main()
