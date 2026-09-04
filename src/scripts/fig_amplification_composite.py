#!/usr/bin/env python3
"""
fig_amplification_composite.py -- amplitude sensitivity in time and space, one figure.

Top:    sigma_10x(t)/sigma_1x(t) for the continental sea-level contribution, with the
        years shown below marked. This says WHEN the ensemble responds to a larger
        stochastic forcing.
Bottom: the same ratio per cell, mapped over a chosen sector at each marked year.
        This says WHERE.

The pairing is the point: the continental curve decays from about 6 to about 3 over
the integration, and the maps show whether that decay is a uniform weakening or the
loss of specific regions.

The ratio pivots on 1, not 0, so the map colour scale uses RECIPROCAL log bounds
(1/k .. k) which put white exactly at unity. Bounds that are not reciprocal place the
neutral colour somewhere other than 1 and make cells that doubled look unchanged.

Early years are excluded from the time series where sigma_1x is below --sigma-floor:
the ensemble has barely diverged there, so the ratio is dividing by noise.

Usage
    python3 fig_amplification_composite.py --years 2050,2100,2200,2300 \
        --regions amundsen,fris,wilkes --shelves Pine_Island,Ronne,Totten
"""
from __future__ import annotations
import os, glob, argparse, importlib.util
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("amaps", os.path.join(_d, "fig_amundsen_maps.py"))
A = importlib.util.module_from_spec(_s); _s.loader.exec_module(A)

RHO_I, RHO_O, A_O = 910.0, 1028.0, 3.625e14
ENS_ROOT = "data/MALI/diagnostics/ENSEMBLES"
PRETTY = {"amundsen": "Amundsen", "fris": "Filchner-Ronne", "wilkes": "Wilkes"}

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "savefig.dpi": 300,
})


def sle_members(ens, pat):
    out = []
    for f in sorted(glob.glob(os.path.join(ENS_ROOT, ens, pat, "globalStats.nc"))):
        d = netCDF4.Dataset(f)
        yr = np.asarray(d["daysSinceStart"][:], float) / 365.0
        v = np.asarray(d["volumeAboveFloatation"][:], float)
        d.close()
        k = np.isfinite(yr) & np.isfinite(v) & (v > 0)
        if k.sum() < 50 or yr[k][0] > 5:
            continue
        out.append((yr[k], -(v[k] - v[k][0]) * (RHO_I / RHO_O) / A_O * 1e3))
    return out


def shelf_ratio(shelf, grid):
    """sigma_10x/sigma_1x of a single shelf's sea-level contribution, in time.

    Individual shelves rather than sector sums: the 133-region mask contains nested
    aggregates, so summing its regions would double-count.
    """
    import csv as _csv
    names = [f"r{i}" for i in range(133)]
    for r in _csv.DictReader(open("docs/region_mapping_133_to_ismip6.csv")):
        names[int(r["idx_133"])] = r["name_133"]
    if shelf not in names:
        return None
    ri = names.index(shelf)
    out = {}
    for ens in ("SSP585", "SSP585_varScaled10x"):
        M = []
        for f in sorted(glob.glob(f"data/MALI/diagnostics/COMMON_MASK/{ens}/*_shelfStats.nc")):
            d = netCDF4.Dataset(f)
            yr = np.asarray(d["daysSinceStart"][:], float) / 365.0
            v = np.asarray(d["regionalVolumeAboveFloatation"][:], float)[:, ri]
            d.close()
            s_ = -(v - v[0]) * (RHO_I / RHO_O) / A_O * 1e3
            M.append(np.interp(grid, yr, s_, left=np.nan, right=np.nan))
        out[ens] = np.nanstd(np.array(M), 0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(out["SSP585"] > 1e-4,
                        out["SSP585_varScaled10x"] / out["SSP585"], np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2050,2100,2200,2300")
    ap.add_argument("--regions", default="amundsen,fris,wilkes",
                    help="comma-separated sectors; one map row each")
    ap.add_argument("--shelves", default="Pine_Island,Ronne,Totten",
                    help="shelves whose ratio time series overlay the continental curve")
    ap.add_argument("--sigma-floor", type=float, default=0.30,
                    help="mm; below this the continental ratio is denominator noise")
    ap.add_argument("--out-dir", default="reports/dissertation/figures/tierA")
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(",")]
    os.makedirs(a.out_dir, exist_ok=True)

    # ---- continental ratio in time
    g = np.arange(1.0, 301.0)
    M1 = np.array([np.interp(g, y, s, left=np.nan, right=np.nan)
                   for y, s in sle_members("SSP585", "SSP585_[0-9][0-9]")])
    M10 = np.array([np.interp(g, y, s, left=np.nan, right=np.nan)
                    for y, s in sle_members("SSP585_varScaled10x", "SSP585_[0-9][0-9]")])
    s1, s10 = np.nanstd(M1, 0, ddof=1), np.nanstd(M10, 0, ddof=1)
    ratio = np.where(s1 >= a.sigma_floor, s10 / s1, np.nan)

    bed = A.v(A.MESH, "bedTopography")
    area = A.v(A.MESH, "areaCell")
    hf = (RHO_O / RHO_I) * np.maximum(0.0, -bed)

    regions = [r.strip() for r in a.regions.split(",")]
    shelves_ts = [t.strip() for t in a.shelves.split(",") if t.strip()]
    n = len(years); nr = len(regions)
    fig = plt.figure(figsize=(4.7 * n, 3.8 + 4.3 * nr))
    gs = fig.add_gridspec(1 + nr, n, height_ratios=[1.15] + [1.5] * nr,
                          hspace=0.30, wspace=0.08)

    axT = fig.add_subplot(gs[0, :])
    axT.plot(g, ratio, color="#333333", lw=2)
    axT.axhline(1, color="k", lw=.6, ls=":")
    med = np.nanmedian(ratio)
    axT.axhline(med, color="#999999", lw=.8, ls="--")
    axT.text(g[-1] - 22, med, f"median {med:.1f}", ha="right", va="center",
             fontsize=9, color="#666666")
    for Y in years:
        axT.axvline(Y - 2000, color="#C1121F", lw=1.0, alpha=.65)
        i = int(np.argmin(abs(g - (Y - 2000))))
        if np.isfinite(ratio[i]):
            is_last = (Y == years[-1])
            axT.annotate(f"{ratio[i]:.1f}", (Y - 2000, ratio[i]), textcoords="offset points",
                         xytext=((-6 if is_last else 4), 6),
                         ha=("right" if is_last else "left"),
                         fontsize=9, color="#C1121F")
    for sh, col in zip(shelves_ts, ["#D55E00", "#0072B2", "#009E73", "#7B3FA0"]):
        rs = shelf_ratio(sh, g)
        if rs is None:
            continue
        axT.plot(g, np.where(np.isfinite(rs) & (rs > 0), rs, np.nan), color=col, lw=1.3,
                 alpha=.85, label=sh.replace("_", " "))
    axT.plot([], [], color="#333333", lw=2, label="continental")
    axT.set_xlabel("model year"); axT.set_ylabel("$\\sigma_{10\\times}/\\sigma_{1\\times}$")
    axT.text(0.01, 0.96, "(a)", transform=axT.transAxes, ha="left", va="top",
             fontsize=15, fontweight="bold")
    axT.grid(alpha=.3); axT.set_xlim(0, 300); axT.set_yscale("log")
    axT.legend(fontsize=8, ncol=4, loc="upper right")

    for ri_, reg in enumerate(regions):
        shelves, labels = A.REGIONS[reg]
        x, y, inbox, box, centres = A.load_domain(shelves=shelves, labels=labels)
        for j, Y in enumerate(years):
            ax = fig.add_subplot(gs[1 + ri_, j])
            f5, f10 = A.stats_file("SSP585", Y), A.stats_file("SSP585_varScaled10x", Y)
            if not (f5 and f10):
                ax.set_visible(False); continue
            d1 = np.where(inbox, A.v(f5, "thickness_std"), np.nan)
            d2 = np.where(inbox, A.v(f10, "thickness_std"), np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(d1 > 1e-3, d2 / d1, np.nan)
            ok = np.isfinite(r) & (r > 0)
            fin = r[ok]
            # AREA-weighted, not cell-counted: the mesh spans 4-20 km, so counting
            # cells over-weights the refined coastal margin relative to the interior
            w = area[ok]
            frac_lo = float(np.sum(w[fin < 1]) / np.sum(w))
            srt = np.argsort(fin); cw = np.cumsum(w[srt]) / np.sum(w)
            med = float(fin[srt][np.searchsorted(cw, 0.5)])
            k = float(np.percentile(np.maximum(fin, 1.0 / fin), 98))
            k = min(max(k, 2.0), 40.0)
            hm = A.v(f5, "thickness_mean")
            gl = np.zeros(x.size, bool)
            okh = np.isfinite(hm) & (hm > 1.0)
            gl[okh] = np.abs(hm[okh] - hf[okh]) < 40.0
            sc = A.paint(ax, x, y, r, box, "RdBu_r", norm=LogNorm(vmin=1.0 / k, vmax=k),
                         centres=centres if j == 0 else None, gl=gl)
            letter = f"({'bcdefghijklm'[ri_ * n + j]})"
            ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha="left", va="top",
                    fontsize=13, fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=.72, pad=1.2))
            ax.text(0.98, 0.98, f"yr {Y-2000}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=12,
                    bbox=dict(fc="white", ec="none", alpha=.72, pad=1.2))
            if j == 0:
                ax.text(0.02, 0.88, PRETTY.get(reg, reg), transform=ax.transAxes,
                        ha="left", va="top", fontsize=12, fontweight="bold",
                        bbox=dict(fc="white", ec="none", alpha=.72, pad=1.2))
            ax.text(0.02, 0.03,
                    f"median {med:.1f}; {100*frac_lo:.0f}% below 1",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5,
                    bbox=dict(fc="white", ec="none", alpha=.78, pad=1.2))
            if j == n - 1:
                plt.colorbar(sc, ax=ax, shrink=.8,
                             label="$\\sigma_{10\\times}/\\sigma_{1\\times}$ (white = 1)")
            print(f"  {reg:12s} yr {Y}: area-wtd median {med:.2f}, {100*frac_lo:.0f}% of area below 1")

    out = f"{a.out_dir}/F24_amplification_composite.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
