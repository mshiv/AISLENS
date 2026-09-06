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
  C  drift, mean_10x - mean_1x     signed; the sector losing ice under greater forcing
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

# Named shelf groups. Each was chosen because it tops a DIFFERENT ranking -- no
# single sector dominates all three, so a figure built only on the Amundsen would
# illustrate the drift result with the sector that does not drive it.
REGIONS = {
    # largest internal variability: Thwaites is 2.3x the next shelf in CTRL sigma
    "amundsen":  (["Thwaites", "Pine_Island", "Crosson", "Dotson", "Getz", "Abbot",
                   "Cosgrove"],
                  ["Thwaites", "Pine_Island", "Crosson", "Dotson", "Getz"]),
    # largest noise-induced drift, and opposite in sign to the Amundsen
    "fris":      (["Ronne", "Filchner", "Riiser-Larsen", "Brunt_Stancomb", "Stancomb"],
                  ["Ronne", "Filchner", "Riiser-Larsen"]),
    # strongest amplitude sensitivity outside the Amundsen
    "wilkes":    (["Totten", "Cook", "Ninnis", "Mertz", "Moscow_University", "Holmes",
                   "Vincennes", "Frost", "Dibble"],
                  ["Totten", "Cook", "Ninnis", "Holmes"]),
    # largest absolute spread of any shelf (Borchgrevink) plus large positive drift
    "dronningmaud": (["Fimbul", "Baudouin", "Borchgrevink", "Nivl", "Vigrid", "Atka",
                      "Ekstrom", "Quar", "Jelbart"],
                     ["Fimbul", "Baudouin", "Borchgrevink", "Nivl"]),
    # the extreme amplitude ratio (West, 36x) and the Larsen system
    "peninsula": (["West", "George_VI", "Wilkins", "Stange", "Wordie", "Larsen_B",
                   "Larsen_C", "Larsen_D", "Larsen_E", "Larsen_F"],
                  ["West", "George_VI", "Wilkins", "Larsen_C"]),
    # large mean signal, very small variability -- the contrast case
    "ross":      (["Eastern_Ross", "Western_Ross", "Sulzberger", "Withrow", "Nickerson"],
                  ["Eastern_Ross", "Western_Ross"]),
}
AMUNDSEN, LABEL_AT = REGIONS["amundsen"]
ENS_NAMES = ["CTRL", "SSP126", "SSP585", "SSP585_varScaled10x", "SSP585-3X"]

# Files are named <ENSEMBLE>_<YEAR>.nc, with a few legacy aliases from earlier copies.
ALIAS = {("SSP585_varScaled10x", 2150): "var10x_2150.nc",
         ("SSP585-3X", 2150): "x3_2150.nc"}


def stats_file(ens, year):
    """Path for an ensemble/year, or None. Honours the legacy aliases."""
    for cand in (ALIAS.get((ens, year)), f"{ens}_{year}.nc"):
        if cand and os.path.exists(os.path.join(STATS, cand)):
            return os.path.join(STATS, cand)
    return None


def parse_overrides(text):
    """'SSP585-3X:2150,CTRL:2200' -> {'SSP585-3X': 2150, 'CTRL': 2200}"""
    out = {}
    for part in filter(None, (t.strip() for t in text.split(","))):
        k, _, v = part.partition(":")
        out[k.strip()] = int(v)
    return out


def v(path, name):
    d = netCDF4.Dataset(path)
    a = np.ma.filled(np.asarray(d[name][:], dtype=float), np.nan).ravel()
    d.close()
    return a


def load_domain(pad=6.0e4, shelves=None, labels=None):
    shelves = shelves or AMUNDSEN
    labels = labels or LABEL_AT
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
    present = [s for s in shelves if s in names]
    missing = [s for s in shelves if s not in names]
    if missing:
        print(f"  (not in mask, skipped: {', '.join(missing)})")
    for s in present:
        sel |= m[names.index(s)].astype(bool)
    if sel.sum() == 0:
        raise SystemExit("no cells for this region")
    box = (x[sel].min() - pad, x[sel].max() + pad, y[sel].min() - pad, y[sel].max() + pad)
    inbox = (x >= box[0]) & (x <= box[1]) & (y >= box[2]) & (y <= box[3])
    centres = {s: (float(np.mean(x[m[names.index(s)].astype(bool)])),
                   float(np.mean(y[m[names.index(s)].astype(bool)])))
               for s in labels if s in names}
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
    ap.add_argument("--rows", default="2150,2300",
                    help="comma-separated years, one map row each")
    ap.add_argument("--override", default="SSP585-3X:2150",
                    help="per-ensemble year substitution, e.g. 'SSP585-3X:2150'; "
                         "panels using a substituted year are labelled as such")
    ap.add_argument("--ratio-cmap", default="RdBu_r",
                    help="colormap for the ratio panel; RdBu_r is diverging about 1, "
                         "use e.g. viridis for a sequential single-hue version")
    ap.add_argument("--sigma-floor", type=float, default=0.02,
                    help="mask cells below this fraction of the domain max sigma")
    ap.add_argument("--region", default="amundsen", choices=sorted(REGIONS),
                    help="named shelf group defining the map domain")
    ap.add_argument("--out-dir", default="reports/dissertation/figures/tierB")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    shelves, labels = REGIONS[a.region]
    x, y, inbox, box, centres = load_domain(shelves=shelves, labels=labels)
    print(f"region '{a.region}': {len(shelves)} shelves")
    bed = v(MESH, "bedTopography")
    hf = (RHO_O / RHO_I) * np.maximum(0.0, -bed)
    print(f"Amundsen domain: {inbox.sum()} cells")

    def sub(arr):
        out = np.full(arr.shape, np.nan)
        out[inbox] = arr[inbox]
        return out

    def gl_from(thick_mean):
        """Grounding line for a GIVEN mean-thickness field: cells close to flotation.

        Must be recomputed per panel. A single reference line drawn on every panel
        would misplace it wherever that ensemble or year has actually retreated,
        which is precisely what these figures are about.
        """
        g_ = np.zeros(x.size, bool)
        ok = np.isfinite(thick_mean) & (thick_mean > 1.0)
        g_[ok] = np.abs(thick_mean[ok] - hf[ok]) < 40.0
        return g_

    # ---------------- A: sigma per ensemble, one row per year
    if "A" in a.panels:
        rows = [int(y) for y in a.rows.split(",")]
        over = parse_overrides(a.override)
        fig, axes = plt.subplots(len(rows), 5, figsize=(21, 4.6 * len(rows)),
                                 squeeze=False)
        # common colour scale across every panel so rows are comparable
        cache, vmax = {}, 0.0
        for ri, yr_ in enumerate(rows):
            for e in ENS_NAMES:
                use = over.get(e, yr_)
                f = stats_file(e, use) or stats_file(e, over.get(e, rows[0]))
                if f is None:
                    continue
                arr = sub(v(f, "thickness_std"))
                cache[(ri, e)] = (arr, use, os.path.basename(f),
                                  gl_from(v(f, "thickness_mean")))
                fin = arr[np.isfinite(arr) & (arr > 0)]
                if fin.size:
                    vmax = max(vmax, np.percentile(fin, 98))
        # mask the near-zero tail: most cells barely move, and leaving them in
        # compresses the colour scale onto the values that matter
        floor = a.sigma_floor * vmax
        for ri, yr_ in enumerate(rows):
            for k, e in enumerate(ENS_NAMES):
                ax = axes[ri][k]
                if (ri, e) not in cache:
                    ax.set_visible(False); continue
                arr, use, fn, gl_p = cache[(ri, e)]
                shown = np.where(arr >= floor, arr, np.nan)
                sc = paint(ax, x, y, shown, box, "inferno",
                           norm=LogNorm(vmin=max(floor, 1e-3), vmax=vmax),
                           centres=centres if (ri == 0 and k == 0) else None, gl=gl_p)
                fin = arr[np.isfinite(arr) & (arr > 0)]
                tag = f"yr {use-2000}" + ("  (substituted)" if use != yr_ else "")
                ax.set_title(f"{e}\n{tag}   median $\\sigma$={np.median(fin):.2f} m",
                             fontsize=8.5)
        fig.colorbar(sc, ax=axes, shrink=.75,
                     label="$\\sigma$(thickness) (m), log scale")
        fig.suptitle(f"{a.region.upper()} sector: ensemble spread of ice thickness\n"
                     f"common log colour scale; cells below {100*a.sigma_floor:.0f}% of the "
                     "domain maximum are masked; black = mean grounding line", fontsize=12)
        o = f"{a.out_dir}/F17_{a.region}_sigma_by_ensemble.png"
        fig.savefig(o, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", o)

    # ---------------- B, C, D
    if any(c in a.panels for c in "BCD"):
        rows = [int(yy) for yy in a.rows.split(",")]
        h0 = sub(v(MESH, "thickness"))
        fig, axg = plt.subplots(len(rows), 3, figsize=(16.5, 5.2 * len(rows)),
                                squeeze=False)
        for ri, YR in enumerate(rows):
            sr = f"{SIGRATIO}/sigmaRatio_thickness_{YR}.nc"
            f5, f10 = stats_file("SSP585", YR), stats_file("SSP585_varScaled10x", YR)
            if not (os.path.exists(sr) and f5 and f10):
                print(f"  row {YR}: missing inputs, skipped")
                for c in range(3):
                    axg[ri][c].set_visible(False)
                continue
            s1 = sub(v(sr, "thickness_std_den"))
            s2 = sub(v(sr, "thickness_std_num"))
            h1 = sub(v(f5, "thickness_mean"))
            h2 = sub(v(f10, "thickness_mean"))
            gl = gl_from(v(f5, "thickness_mean"))   # 1x reference line, this year
            axes = axg[ri]

            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where((s1 > 1e-3) & np.isfinite(s2), s2 / s1, np.nan)
            fin = ratio[np.isfinite(ratio) & (ratio > 0)]
            # A ratio pivots on 1, not 0. LogNorm places its midpoint halfway
            # between log(vmin) and log(vmax), so unless the bounds are RECIPROCAL
            # the neutral colour lands somewhere other than 1 and cells that
            # actually doubled read as unchanged. Symmetric bounds 1/k .. k fix that.
            k = float(np.percentile(np.maximum(fin, 1.0 / fin), 98))
            k = min(max(k, 2.0), 40.0)
            cmap = a.ratio_cmap
            sc = paint(axes[0], x, y, ratio, box, cmap,
                       norm=LogNorm(vmin=1.0 / k, vmax=k),
                       centres=centres if ri == 0 else None, gl=gl)
            plt.colorbar(sc, ax=axes[0], shrink=.8,
                         label="$\\sigma_{10\\times}/\\sigma_{1\\times}$"
                               + ("  (white = 1)" if a.ratio_cmap.startswith("RdBu") else ""))
            axes[0].set_title(f"(a) amplitude sensitivity — yr {YR-2000}\n"
                              f"median {np.median(fin):.2f}  (continental 3.11); "
                              f"{100*np.mean(fin<1):.0f} % below 1", fontsize=10)

            # drift in VAF terms: only ice above flotation reaches sea level
            drift = np.maximum(0.0, h2 - hf) - np.maximum(0.0, h1 - hf)
            fin = drift[np.isfinite(drift)]
            lim = np.percentile(np.abs(fin), 98) or 1.0
            sc = paint(axes[1], x, y, drift, box, "RdBu",
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), gl=gl)
            plt.colorbar(sc, ax=axes[1], shrink=.8,
                         label="$\\Delta$VAF, 10$\\times$ $-$ 1$\\times$ (m)")
            axes[1].set_title(f"(b) noise-induced drift — yr {YR-2000}\n"
                              f"{100*np.mean(fin<0):.0f} % of cells thinner under "
                              f"10$\\times$", fontsize=10)

            with np.errstate(invalid="ignore", divide="ignore"):
                snr = np.where(s1 > 1e-6, np.abs(h1 - h0) / s1, np.nan)
            fin = snr[np.isfinite(snr) & (snr > 0)]
            sc = paint(axes[2], x, y, snr, box, "cividis",
                       norm=LogNorm(vmin=max(np.percentile(fin, 2), 0.1),
                                    vmax=np.percentile(fin, 98)), gl=gl)
            plt.colorbar(sc, ax=axes[2], shrink=.8,
                         label="$|\\overline{\\Delta H}|/\\sigma$")
            axes[2].set_title(f"(c) emergence — yr {YR-2000}\n"
                              f"{100*np.mean(fin>1):.0f} % of cells above S/N = 1",
                              fontsize=10)

        fig.suptitle(f"{a.region.upper()} sector — SSP585 vs SSP585_varScaled10x "
                     "(black = mean grounding line)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        o = f"{a.out_dir}/F18_{a.region}_ratio_drift_emergence.png"
        fig.savefig(o, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", o)


if __name__ == "__main__":
    main()
