#!/usr/bin/env python3
"""
fig_spatial_ensemble_maps.py -- laptop-side maps from spatial_ensemble_stats.py output.

Four figures, all from the small per-cell stats NetCDFs (no HPC access needed):

  spread   sigma and range of cumulative thickness change, at each target year.
           WHERE the ensemble spread lives, on the mesh.

  ratio    per-cell sigma_10x / sigma_1x. The spatial amplitude sensitivity: which
           shelves actually transmit an amplified forcing and which are indifferent
           to it. Both numerator and denominator are RESPONSE sigma, so unlike the
           global beta this carries no contaminated forcing denominator.

  rate     ensemble-mean secular thinning rate -- the like-for-like quantity to set
           beside satellite dh/dt (e.g. Smith et al. 2020). The only AISLENS figure
           that touches a direct observable.

  snr      |mean dH| / sigma_dH. Where the forced signal exceeds internal variability,
           and when. The map version of the chapter's central argument.

Colour conventions, applied consistently:
  * diverging + symmetric about zero for signed fields (dH, rate) so that the zero
    contour is the visual reference and thickening/thinning are not conflated;
  * sequential for strictly positive fields (sigma, range);
  * for `ratio`, log-scaled and centred on 1.0 -- a ratio of 2 and a ratio of 0.5 are
    equal and opposite departures, and a linear scale would misrepresent that.

Percentile clipping (2-98) sets the limits: a handful of grounding-line cells carry
values orders of magnitude above the field and would otherwise flatten the whole map.

Usage
    python3 fig_spatial_ensemble_maps.py --kind spread --stats stats_SSP585.nc \
        --mesh MESH.nc --out-dir reports/figures/spatial
    python3 fig_spatial_ensemble_maps.py --kind ratio \
        --stats stats_SSP585_varScaled10x.nc --ref-stats stats_SSP585.nc --mesh MESH.nc
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio


def years_in(ds, prefix="dH_mean_"):
    ys = [int(v.replace(prefix, "")) for v in ds.data_vars if v.startswith(prefix)]
    return sorted(ys)


def paint(ax, x, y, v, title, cmap, norm=None, vmin=None, vmax=None, size=0.6):
    m = np.isfinite(v)
    s = ax.scatter(x[m], y[m], c=v[m], s=size, cmap=cmap, norm=norm,
                   vmin=None if norm else vmin, vmax=None if norm else vmax,
                   linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    return s


def lim(v, lo=2, hi=98):
    v = v[np.isfinite(v)]
    return (np.percentile(v, lo), np.percentile(v, hi)) if v.size else (0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["spread", "ratio", "rate", "snr"])
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ref-stats", help="denominator ensemble, for --kind ratio")
    ap.add_argument("--mesh", required=True, help="mesh file with xCell/yCell")
    ap.add_argument("--out-dir", default="reports/figures/spatial")
    ap.add_argument("--min-members", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    ds = xr.open_dataset(a.stats)
    ens = ds.attrs.get("ensemble", "?")
    x, y = sio.load_mesh_coords(a.mesh)
    yrs = years_in(ds)
    if not yrs:
        sys.exit("no dH_mean_* variables in stats file")

    def get(dsx, name, yr):
        v = dsx[f"{name}_{yr}"].values.astype(float)
        n = dsx[f"{name.split('_')[0]}_n_{yr}"].values
        v[n < a.min_members] = np.nan          # never plot a cell below the member floor
        return v

    # ---------------------------------------------------------------- spread
    if a.kind == "spread":
        fig, axes = plt.subplots(2, len(yrs), figsize=(4.0 * len(yrs), 8.0), squeeze=False)
        for j, yr in enumerate(yrs):
            sg, rg = get(ds, "dH_sigma", yr), get(ds, "dH_range", yr)
            s = paint(axes[0][j], x, y, sg, f"$\\sigma(\\Delta H)$  yr {yr}",
                      "viridis", vmin=0, vmax=lim(sg)[1])
            plt.colorbar(s, ax=axes[0][j], shrink=.7, label="m")
            s = paint(axes[1][j], x, y, rg, f"range (max$-$min)  yr {yr}",
                      "magma", vmin=0, vmax=lim(rg)[1])
            plt.colorbar(s, ax=axes[1][j], shrink=.7, label="m")
        fig.suptitle(f"{ens} — ensemble spread of cumulative thickness change "
                     f"(N={ds.attrs.get('n_members','?')})", fontsize=11)
        out = f"{a.out_dir}/spatial_spread_{ens}.png"

    # ---------------------------------------------------------------- ratio
    elif a.kind == "ratio":
        if not a.ref_stats:
            sys.exit("--kind ratio needs --ref-stats (the 1x ensemble)")
        dr = xr.open_dataset(a.ref_stats)
        ref = dr.attrs.get("ensemble", "?")
        common = [t for t in yrs if t in years_in(dr)]
        fig, axes = plt.subplots(1, len(common), figsize=(4.4 * len(common), 4.6), squeeze=False)
        for j, yr in enumerate(common):
            num, den = get(ds, "dH_sigma", yr), get(dr, "dH_sigma", yr)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where((den > 0) & np.isfinite(num), num / den, np.nan)
            # a ratio field is multiplicative: log scale, centred on 1
            r = np.where(r > 0, r, np.nan)
            hi = max(np.nanpercentile(r, 98), 1.05)
            lo = min(np.nanpercentile(r, 2), 0.95)
            s = paint(axes[0][j], x, y, r, f"$\\sigma_{{10\\times}}/\\sigma_{{1\\times}}$  yr {yr}",
                      "RdBu_r", norm=LogNorm(vmin=lo, vmax=hi))
            cb = plt.colorbar(s, ax=axes[0][j], shrink=.75)
            cb.set_label("ratio (log scale; 1 = no amplification)")
            fin = r[np.isfinite(r)]
            if fin.size:
                axes[0][j].set_xlabel(f"median {np.median(fin):.2f}×   "
                                      f"cells >1: {100*np.mean(fin>1):.0f} %", fontsize=8)
        fig.suptitle(f"Spatial amplitude sensitivity — {ens} vs {ref}", fontsize=11)
        out = f"{a.out_dir}/spatial_sigma_ratio_{ens}_over_{ref}.png"

    # ---------------------------------------------------------------- rate
    elif a.kind == "rate":
        fig, axes = plt.subplots(1, len(yrs), figsize=(4.4 * len(yrs), 4.6), squeeze=False)
        allv = np.concatenate([get(ds, "rate_mean", t)[np.isfinite(get(ds, "rate_mean", t))]
                               for t in yrs])
        v = max(abs(np.percentile(allv, 2)), abs(np.percentile(allv, 98))) or 1.0
        for j, yr in enumerate(yrs):
            s = paint(axes[0][j], x, y, get(ds, "rate_mean", yr),
                      f"mean $\\partial H/\\partial t$  yr {yr}", "RdBu",
                      norm=TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v))
            plt.colorbar(s, ax=axes[0][j], shrink=.75, label="m yr$^{-1}$")
        fig.suptitle(f"{ens} — ensemble-mean secular thinning rate "
                     f"(centred ±{ds.attrs.get('window_yr','?')} yr window)", fontsize=11)
        out = f"{a.out_dir}/spatial_rate_{ens}.png"

    # ---------------------------------------------------------------- snr
    else:
        fig, axes = plt.subplots(1, len(yrs), figsize=(4.4 * len(yrs), 4.6), squeeze=False)
        for j, yr in enumerate(yrs):
            snr = get(ds, "dH_snr", yr)
            s = paint(axes[0][j], x, y, snr, f"$|\\overline{{\\Delta H}}|/\\sigma$  yr {yr}",
                      "cividis", vmin=0, vmax=max(np.nanpercentile(snr, 98), 2))
            cb = plt.colorbar(s, ax=axes[0][j], shrink=.75)
            cb.set_label("signal-to-noise")
            fin = snr[np.isfinite(snr)]
            if fin.size:
                axes[0][j].set_xlabel(f"{100*np.mean(fin>2):.0f} % of cells above S/N = 2",
                                      fontsize=8)
        fig.suptitle(f"{ens} — where the forced signal exceeds internal variability", fontsize=11)
        out = f"{a.out_dir}/spatial_snr_{ens}.png"

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
