#!/usr/bin/env python3
"""
fig_per_region_panels.py -- ONE small figure per region: total ice volume | VAF.

Each figure is two panels side by side for a single region:
    left   total ice volume change (10^12 m3)  -- includes FLOATING ice
    right  VAF -> sea-level equivalent (mm)    -- grounded ice only
with every applicable ensemble overlaid as mean +/- 1 sigma.

The pair is the point: a region whose total volume falls while VAF holds flat is
thinning its shelf, not losing grounded ice. Only the right panel reaches the ocean.

TWO FAMILIES, BECAUSE THE MASKS DIFFER
    shelves   idx 33..132 of the 133-region mask -- 100 NAMED ICE SHELVES (Abbot ...
              Zubchatyy). Indices 0..32 are aggregates and IMBIE basins and are
              excluded. Only CTRL and SSP585-3X write this mask, so only those two
              ensembles can appear.
    basins    the 16 ISMIP6 basins. Only SSP126, SSP585 and varScaled10x write this
              mask, so only those three appear.

There is no way around the split at region level: a shelf and a drainage basin are
different sets of cells (see fig_regional_all_ensembles.py, where only Filchner-Ronne,
Ross and Amery pass a Jaccard test). Cross-ensemble comparison at region level is
available only for those three.

Output goes to per-family subdirectories so the ~116 figures stay trackable:
    <out-dir>/by_shelf/<Name>.png
    <out-dir>/by_ismip6_basin/<Name>.png

Usage
    python3 fig_per_region_panels.py --family shelves
    python3 fig_per_region_panels.py --family basins
    python3 fig_per_region_panels.py --family shelves --only Thwaites,Pine_Island
"""
from __future__ import annotations
import os, sys, glob, csv, re, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO_I, RHO_O, A_O = 910.0, 1028.0, 3.625e14
SHELF_START = 33          # idx 33 = Abbot; 0..32 are aggregates + IMBIE basins

FAMILIES = {
    "shelves": dict(nreg=133, sub="by_shelf",
                    ens=[("CTRL", "CTRL_[0-9][0-9]", "#444444"),
                         ("SSP585-3X", "SSP585-3X_[0-9][0-9]", "#8B0000")]),
    "basins":  dict(nreg=16, sub="by_ismip6_basin",
                    ens=[("SSP126", "SSP126_[0-9][0-9]", "#0072B2"),
                         ("SSP585", "SSP585_[0-9][0-9]", "#D55E00"),
                         ("SSP585_varScaled10x", "SSP585_[0-9][0-9]", "#7B3FA0")]),
}
KEYS = ["regionalIceVolume", "regionalVolumeAboveFloatation"]


def region_names(n):
    if n == 16:
        f = "data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"
        d = netCDF4.Dataset(f)
        nm = [b"".join(r).decode(errors="ignore").strip().replace("ISMIP6 Basin ", "")
              for r in np.asarray(d["regionNames"][:]).astype("S1")]
        d.close(); return nm
    nm = [f"region {i}" for i in range(n)]
    f = "docs/region_mapping_133_to_ismip6.csv"
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            i = int(r["idx_133"])
            if i < n:
                nm[i] = r["name_133"]
    return nm


def load(root, ens, pat, nreg):
    out = []
    for d in sorted(glob.glob(os.path.join(root, ens, pat))):
        f = os.path.join(d, "regionalStats.nc")
        if not os.path.exists(f):
            continue
        try:
            ds = netCDF4.Dataset(f)
            if len(ds.dimensions["nRegions"]) != nreg:
                ds.close(); continue
            yr = np.asarray(ds["daysSinceStart"][:], float) / 365.0
            V = {k: np.asarray(ds[k][:], float) for k in KEYS if k in ds.variables}
            ds.close()
        except Exception:
            continue
        if len(V) < 2 or len(yr) < 50 or yr[0] > 5.0:
            continue
        out.append((yr, V))
    return out


def stats(mem, key, r, grid, to_sle, min_members):
    M = np.full((len(mem), grid.size), np.nan)
    for i, (yr, V) in enumerate(mem):
        v = V[key][:, r]
        ok = np.isfinite(yr) & np.isfinite(v)
        if ok.sum() < 10:
            continue
        d = v[ok] - v[ok][0]
        d = -d * (RHO_I / RHO_O) / A_O * 1e3 if to_sle else d / 1e12
        M[i] = np.interp(grid, yr[ok], d, left=np.nan, right=np.nan)
    n = np.sum(np.isfinite(M), axis=0)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0, ddof=1)
    mu[n < min_members] = np.nan; sd[n < min_members] = np.nan
    return mu, sd, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/MALI/diagnostics/ENSEMBLES")
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--out-dir", default="reports/dissertation/figures/regions")
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--min-members", type=int, default=5)
    ap.add_argument("--only", default=None, help="comma-separated region names")
    a = ap.parse_args()

    F = FAMILIES[a.family]
    outdir = os.path.join(a.out_dir, F["sub"])
    os.makedirs(outdir, exist_ok=True)
    names = region_names(F["nreg"])
    grid = np.arange(1.0, a.horizon + 1e-9, 1.0)

    data = {}
    for ens, pat, col in F["ens"]:
        m = load(a.root, ens, pat, F["nreg"])
        if len(m) >= 3:
            data[ens] = (m, col)
            print(f"  {ens}: N={len(m)}")
        else:
            print(f"  {ens}: skipped ({len(m)} usable members)")
    if not data:
        sys.exit("no ensembles with this mask")

    idxs = range(SHELF_START, F["nreg"]) if a.family == "shelves" else range(F["nreg"])
    want = {s.strip() for s in a.only.split(",")} if a.only else None

    summary, made = [], 0
    for r in idxs:
        nm = names[r] if r < len(names) else f"region {r}"
        if want and nm not in want:
            continue
        fig, (axV, axS) = plt.subplots(1, 2, figsize=(9.2, 3.4))
        any_data = False
        for ens, (mem, col) in data.items():
            for ax, key, to_sle in [(axV, KEYS[0], False), (axS, KEYS[1], True)]:
                mu, sd, n = stats(mem, key, r, grid, to_sle, a.min_members)
                f = np.isfinite(mu)
                if not f.any():
                    continue
                any_data = True
                ax.plot(grid, mu, color=col, lw=1.7, label=f"{ens} (n={len(mem)})")
                ax.fill_between(grid, mu - sd, mu + sd, color=col, alpha=0.25, lw=0)
                if to_sle:
                    j = np.where(f)[0][-1]
                    summary.append((nm, ens, int(n[j]), mu[j], sd[j]))
        if not any_data:
            plt.close(fig); continue
        axV.set_title("total ice volume (incl. floating)", fontsize=9)
        axV.set_ylabel("change (10$^{12}$ m$^3$)", fontsize=8.5)
        axS.set_title("volume above flotation → sea level", fontsize=9)
        axS.set_ylabel("SLE (mm)", fontsize=8.5)
        for ax in (axV, axS):
            ax.axhline(0, color="k", lw=0.5, ls=":")
            ax.grid(alpha=.3); ax.set_xlabel("model year", fontsize=8.5)
            ax.tick_params(labelsize=8)
        axS.legend(fontsize=7.5, loc="best")
        fig.suptitle(f"{nm}   —   line = ensemble mean, shading = $\\pm1\\sigma$",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", nm)
        fig.savefig(os.path.join(outdir, f"{safe}.png"), dpi=140, bbox_inches="tight")
        plt.close(fig); made += 1

    print(f"\nwrote {made} figures to {outdir}")
    if summary:
        # ranked index so the prominent regions are easy to pick for the chapter grid
        idxf = os.path.join(outdir, "_ranked_index.txt")
        with open(idxf, "w") as fh:
            fh.write(f"{'region':<26}{'ensemble':<22}{'N':>4}{'meanSLE':>11}{'sigma':>10}\n")
            for nm, ens, n, mu, sd in sorted(summary, key=lambda z: -abs(z[3])):
                fh.write(f"{nm:<26}{ens:<22}{n:>4}{mu:>11.3f}{sd:>10.4f}\n")
        print(f"ranked index -> {idxf}")
        print(f"\ntop 10 by |final SLE|:")
        for nm, ens, n, mu, sd in sorted(summary, key=lambda z: -abs(z[3]))[:10]:
            print(f"   {nm:<24} {ens:<20} n={n:<3} {mu:9.3f} mm  sigma={sd:.4f}")


if __name__ == "__main__":
    main()
