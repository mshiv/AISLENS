#!/usr/bin/env python3
"""
fig_amundsen_maps.py -- per-cell maps zoomed on the Amundsen sector.

The shelf-level analysis put the internal variability there: Thwaites carries 2.3x
the CTRL ensemble spread of any other shelf, and Thwaites/Crosson/Pine Island/Dotson
take four of the top seven. These maps show that concentration on the mesh, where no
regional aggregation is involved and the mask split is irrelevant.

PANELS
  A  sigma(dH) per ensemble        where the spread lives, one panel per ensemble
  B  sigma_10x / sigma_1x          local amplitude sensitivity
  C  drift, mean_10x - mean_1x     signed; the sector losing ice under louder forcing
  D  emergence, |mean dH| / sigma  where the forced signal exceeds internal variability

The domain is taken from the union of the named Amundsen shelves in the 133-region
mask rather than a hand-typed bounding box, so it follows the actual shelves and
cannot drift out of date if the mask is regenerated.

Grounding line: cells are coloured by whether the ENSEMBLE-MEAN thickness exceeds
flotation, so the contour is the mean grounding line rather than any one member's.

Usage
    python3 fig_amundsen_maps.py                    # all panels
    python3 fig_amundsen_maps.py --panels AB
"""
from __future__ import annotations
import os, csv, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm

RHO_I, RHO_O = 910.0, 1028.0
MESH = ("data/MALI/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m"
        "_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu"
        "_meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
MASK133 = "data/MALI/aislens_draftDepen_regionMasks.nc"
STATS = "reports/dissertation/figures/spatial/stats_sample"
SIGRATIO = "reports/dissertation/figures/spatial/sigma_ratio/SSP585_varScaled10x_over_SSP585"

AMUNDSEN = ["Thwaites", "Pine_Island", "Crosson", "Dotson", "Getz", "Abbot", "Cosgrove"]
LABEL_AT = ["Thwaites", "Pine_Island", "Crosson", "Dotson", "Getz"]
ENSEMBLES = [("CTRL", "CTRL_2150.nc"), ("SSP126", "SSP126_2150.nc"),
             ("SSP585", "SSP585_2150.nc"), ("SSP585_varScaled10x", "var10x_2150.nc"),
             ("SSP585-3X", "x3_2150.nc")]


def v(path, name):
    d = netCDF4.Dataset(path)
    a = np.ma.filled(np.asarray(d[name][:], dtype=float), np.nan).ravel()
    d.close()
    return a


def load_domain(pad=6.0e4):
    names = [f"r{i}" for i in range(133)]
    for r in csv.DictReader(open("docs/region_mapping_133_to_ismip6.csv")):
        names[int(r["idx_133"])] = r["name_133"]
    d = netCDF4.Dataset(MASK133)
    m = np.asarray(d["regionCellMasks"][:])
    d.close()
    if m.shape[0] > m.shape[1]:
        m = m.T
    x, y = v(MESH, "xCell"), v(MESH, "yCell")
    sel = np.zeros(x.size, bool)
    for s in AMUNDSEN:
        sel |= m[names.index(s)].astype(bool)
    box = (x[sel].min() - pad, x[sel].max() + pad, y[sel].min() - pad, y[sel].max() + pad)
    inbox = (x >= box[0]) & (x <= box[1]) & (y >= box[2]) & (y <= box[3])
    centres = {s: (float(np.mean(x[m[names.index(s)].astype(bool)])),
                   float(np.mean(y[m[names.index(s)].astype(bool)]))) for s in LABEL_AT}
    return x, y, inbox, box, centres


def paint(ax, x, y, val, box, cmap, norm=None, vmin=None, vmax=None, centres=None,
          gl=None, size=3.0):
    m = np.isfinite(val)
    sc = ax.scatter(x[m], y[m], c=val[m], s=size, cmap=cmap, norm=norm,
                    vmin=None if norm else vmin, vmax=None if norm else vmax,
                    linewidths=0, rasterized=True)
    if gl is not None:
        g = gl & np.isfinite(val)
        ax.scatter(x[g], y[g], s=0.6, c="k", linewidths=0, alpha=0.55, rasterized=True)
    if centres:
        for nm_, (cx, cy) in centres.items():
            ax.annotate(nm_.replace("_", " "), (cx, cy), fontsize=6.5, ha="center",
                        color="k",
                        bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.8))
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return sc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", default="ABCD")
    ap.add_argument("--out-dir", default="reports/dissertation/figures/tierB")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    x, y, inbox, box, centres = load_domain()
    bed = v(MESH, "bedTopography")
    hf = (RHO_O / RHO_I) * np.maximum(0.0, -bed)
    print(f"Amundsen domain: {inbox.sum()} cells")

    def sub(arr):
        out = np.full(arr.shape, np.nan)
        out[inbox] = arr[inbox]
        return out

    # grounding line from the SSP585 ensemble-mean thickness: cells within one
    # cell-width of the flotation threshold
    hm = v(f"{STATS}/SSP585_2150.nc", "thickness_mean")
    grounded = hm > hf
    gl = np.zeros(x.size, bool)
    ok = np.isfinite(hm) & (hm > 1.0)
    gl[ok] = np.abs(hm[ok] - hf[ok]) < 40.0

    # ---------------- A: sigma per ensemble
    if "A" in a.panels:
        fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
        vmax = 0.0
        S = {}
        for e, f in ENSEMBLES:
            p = os.path.join(STATS, f)
            if not os.path.exists(p):
                continue
            S[e] = sub(v(p, "thickness_std"))
            fin = S[e][np.isfinite(S[e])]
            if fin.size:
                vmax = max(vmax, np.percentile(fin, 98))
        for k, (e, _f) in enumerate(ENSEMBLES):
            ax = axes[k]
            if e not in S:
                ax.set_visible(False); continue
            sc = paint(ax, x, y, S[e], box, "viridis", vmin=0, vmax=vmax,
                       centres=centres if k == 0 else None, gl=gl)
            fin = S[e][np.isfinite(S[e])]
            ax.set_title(f"{e}\nmedian $\\sigma$={np.median(fin):.2f} m", fontsize=9)
        fig.colorbar(sc, ax=axes, shrink=.8, label="$\\sigma$(thickness) (m)")
        fig.suptitle("Amundsen sector: ensemble spread of ice thickness at model year 150 "
                     "(common colour scale; black = mean grounding line)", fontsize=12)
        o = f"{a.out_dir}/F17_amundsen_sigma_by_ensemble.png"
        fig.savefig(o, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", o)

    # ---------------- B, C, D
    if any(c in a.panels for c in "BCD"):
        s1 = sub(v(f"{SIGRATIO}/sigmaRatio_thickness_2150.nc", "thickness_std_den"))
        s2 = sub(v(f"{SIGRATIO}/sigmaRatio_thickness_2150.nc", "thickness_std_num"))
        h1 = sub(v(f"{STATS}/SSP585_2150.nc", "thickness_mean"))
        h2 = sub(v(f"{STATS}/var10x_2150.nc", "thickness_mean"))
        h0 = sub(v(MESH, "thickness"))

        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where((s1 > 1e-3) & np.isfinite(s2), s2 / s1, np.nan)
        fin = ratio[np.isfinite(ratio) & (ratio > 0)]
        sc = paint(axes[0], x, y, ratio, box, "RdBu_r",
                   norm=LogNorm(vmin=max(np.percentile(fin, 2), 0.2),
                                vmax=min(np.percentile(fin, 98), 40)),
                   centres=centres, gl=gl)
        plt.colorbar(sc, ax=axes[0], shrink=.8, label="$\\sigma_{10\\times}/\\sigma_{1\\times}$")
        axes[0].set_title(f"(a) amplitude sensitivity\nmedian {np.median(fin):.2f}"
                          f"  (continental 3.11)", fontsize=10)

        # drift in VAF terms: only ice above flotation reaches sea level
        vaf1 = np.maximum(0.0, h1 - hf); vaf2 = np.maximum(0.0, h2 - hf)
        drift = vaf2 - vaf1
        fin = drift[np.isfinite(drift)]
        lim = np.percentile(np.abs(fin), 98) or 1.0
        sc = paint(axes[1], x, y, drift, box, "RdBu",
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), gl=gl)
        plt.colorbar(sc, ax=axes[1], shrink=.8, label="$\\Delta$VAF, 10$\\times$ $-$ 1$\\times$ (m)")
        axes[1].set_title(f"(b) noise-induced drift\n"
                          f"{100*np.mean(fin<0):.0f} % of cells thinner under 10$\\times$",
                          fontsize=10)

        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.where(s1 > 1e-6, np.abs(h1 - h0) / s1, np.nan)
        fin = snr[np.isfinite(snr) & (snr > 0)]
        sc = paint(axes[2], x, y, snr, box, "cividis",
                   norm=LogNorm(vmin=max(np.percentile(fin, 2), 0.1),
                                vmax=np.percentile(fin, 98)), gl=gl)
        plt.colorbar(sc, ax=axes[2], shrink=.8, label="$|\\overline{\\Delta H}|/\\sigma$")
        axes[2].set_title(f"(c) emergence\n{100*np.mean(fin>1):.0f} % of cells above S/N = 1",
                          fontsize=10)

        fig.suptitle("Amundsen sector at model year 150 — SSP585 vs SSP585_varScaled10x "
                     "(black = mean grounding line)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        o = f"{a.out_dir}/F18_amundsen_ratio_drift_emergence.png"
        fig.savefig(o, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", o)


if __name__ == "__main__":
    main()
