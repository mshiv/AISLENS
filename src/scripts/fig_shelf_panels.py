#!/usr/bin/env python3
"""
fig_shelf_panels.py -- per-shelf response, all five experiments, one 2x3 grid.

Three quantities, selected with --quantity:

  sle     sea-level contribution of the shelf's drainage, from regionalVolumeAboveFloatation
  volume  cumulative loss of total ice volume including floating ice, in 10^3 km^3. Signed
          so that UP is loss, matching the sea-level panels; this quantity responds to shelf
          thinning that never reaches sea level
  melt    applied sub-shelf melt

The melt quantity is regionalMeltFluxSum / regionalFloatingArea, i.e. an AREA-AVERAGED
rate in m ice/yr, not an integrated total. Shelf areas differ by more than an order of
magnitude, so the integral would rank shelves by size (Ronne largest, Thwaites quiet)
and would hide the thing this panel exists to show: the melt rate a cavity is running
at when a variability anomaly is applied to it.

The linear scale is honest but compresses the cold shelves onto the axis -- Filchner
and Ronne sit at ~0.1 m/yr in CTRL while Thwaites runs at ~19, and under SSP5-8.5 the
same two shelves climb past 20 after year 100. Use --yscale log to make the baseline
separation legible; --yscale linear to keep the late-time magnitudes comparable.

Usage
    python3 fig_shelf_panels.py --quantity melt --yscale log
    python3 fig_shelf_panels.py --quantity sle
"""
from __future__ import annotations
import os, csv, glob, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "data/MALI/diagnostics/COMMON_MASK"
RHO_I, RHO_O, A_O, SPY = 910.0, 1028.0, 3.625e14, 3.15576e7
PANELS = ["Thwaites", "Pine_Island", "Crosson", "Ronne", "Filchner", "Totten"]
ENS = [("CTRL", "#4D4D4D", "-"), ("SSP126", "#0072B2", "-"), ("SSP585", "#C1121F", "-"),
       ("SSP585_varScaled10x", "#E69F00", "--"), ("SSP585-3X", "#7B3FA0", "-")]
LABEL = {"SSP585_varScaled10x": "SSP585 10$\\times$ variability", "SSP585-3X": "SSP585 3$\\times$ melt"}
EXCLUDE_MEMBERS = {"SSP585-3X_00_shelfStats.nc"}

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "savefig.dpi": 300,
})


def region_index():
    names = [f"r{i}" for i in range(133)]
    for r in csv.DictReader(open("docs/region_mapping_133_to_ismip6.csv")):
        names[int(r["idx_133"])] = r["name_133"]
    return {n: i for i, n in enumerate(names)}


def series(ens, ri, grid, quantity):
    """Ensemble mean and member envelope of one shelf's response."""
    M = []
    for f in sorted(glob.glob(f"{ROOT}/{ens}/*_shelfStats.nc")):
        if os.path.basename(f) in EXCLUDE_MEMBERS:
            continue
        d = netCDF4.Dataset(f)
        yr = np.asarray(d["daysSinceStart"][:], float) / 365.0
        if quantity == "melt":
            flx = np.asarray(d["regionalMeltFluxSum"][:], float)[:, ri]
            are = np.asarray(d["regionalFloatingArea"][:], float)[:, ri]
            with np.errstate(invalid="ignore", divide="ignore"):
                q = np.where(are > 0, -flx / np.maximum(are, 1e-9) / RHO_I * SPY, np.nan)
        else:
            v = np.asarray(d[{"sle": "regionalVolumeAboveFloatation",
                              "volume": "regionalIceVolume"}[quantity]][:], float)[:, ri]
            # referenced to each member's own initial state, so spread is generated
            # during the integration rather than inherited
            q = (-(v - v[0]) * (RHO_I / RHO_O) / A_O * 1e3 if quantity == "sle"
                 else -(v - v[0]) / 1e12)     # 10^3 km^3 lost; up = loss, as for sle
        d.close()
        M.append(np.interp(grid, yr, q, left=np.nan, right=np.nan))
    if not M:
        return None, None, None
    M = np.array(M)
    mu, sd = np.nanmean(M, 0), np.nanstd(M, 0, ddof=1)
    if quantity == "melt":
        return mu, np.nanmin(M, 0), np.nanmax(M, 0)
    return mu, mu - sd, mu + sd          # +/- one ensemble standard deviation


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quantity", choices=["sle", "volume", "melt"], default="melt")
    ap.add_argument("--yscale", choices=["log", "linear"], default="log")
    ap.add_argument("--out-dir", default="reports/dissertation/figures/tierA")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    I = region_index()
    g = np.arange(0.0, 301.0)

    fig, axg = plt.subplots(2, 3, figsize=(15.0, 8.5), sharex=True)
    for k, sh in enumerate(PANELS):
        ax = axg.flat[k]
        for ens, col, ls in ENS:
            mu, lo, hi = series(ens, I[sh], g, a.quantity)
            if mu is None:
                continue
            if a.yscale == "log":   # a log axis cannot show refreezing; clip and say so
                mu, lo, hi = (np.where(v > 0, v, np.nan) for v in (mu, lo, hi))
            ax.fill_between(g, lo, hi, color=col, alpha=.18, lw=0)
            ax.plot(g, mu, color=col, lw=1.7, ls=ls,
                    label=LABEL.get(ens, ens) if k == 0 else None)
        if a.quantity == "melt":
            ax.set_yscale(a.yscale)
            if a.yscale == "log":
                ax.set_ylim(3e-2, 2e2)
        else:
            ax.axhline(0, color="k", lw=.6, ls=":")
        ax.set_title(f"({'abcdef'[k]}) {sh.replace('_', ' ')}", loc="left", pad=7)
        ax.grid(alpha=.3, which="both")
        ax.set_xlim(0, 300)
        if k >= 3:
            ax.set_xlabel("model year")
        if k % 3 == 0:
            ax.set_ylabel({"melt": "applied melt (m ice yr$^{-1}$)",
                           "sle": "sea-level contribution (mm SLE)",
                           "volume": "ice-volume loss (10$^3$ km$^3$)"}[a.quantity])
        for y, mu_ in [(50, None), (100, None), (200, None)]:
            ax.axvline(y, color="#BBBBBB", lw=.6, zorder=0)

        # Ronne and Filchner gain a few millimetres of grounded ice before the
        # scenario-driven losses begin. Those complete trajectories are present in
        # the data but are visually compressed by the hundreds-of-millimetres main
        # scale, so show the two low-response experiments in a compact linear inset.
        if a.quantity in {"sle", "volume"} and sh in {"Ronne", "Filchner"}:
            iax = ax.inset_axes([0.06, 0.56, 0.43, 0.34])
            for ens, col, _ in ENS[:2]:
                mu, _, _ = series(ens, I[sh], g, a.quantity)
                iax.plot(g, mu, color=col, lw=1.6)
            iax.axhline(0, color="k", lw=.5, ls=":")
            iax.set_xlim(0, 300)
            iax.set_xticks([0, 150, 300])
            iax.tick_params(labelsize=9, length=2.5)
            iax.grid(alpha=.2)
            iax.text(0.03, 0.06, "CTRL and SSP1-2.6", transform=iax.transAxes,
                     fontsize=9, va="bottom")

    fig.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .95))
    stem = {"melt": "F16_shelf_applied_melt_2x3" + ("_log" if a.yscale == "log" else ""),
            "sle": "F14_shelf_panel_2x3",
            "volume": "F15_shelf_panel_2x3_totalIceVolume"}[a.quantity]
    o = f"{a.out_dir}/{stem}.png"
    fig.savefig(o, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(o)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", o)


if __name__ == "__main__":
    main()
