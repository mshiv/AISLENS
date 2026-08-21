#!/usr/bin/env python3
"""
fig_all_regions_one_ensemble.py -- every region of ONE ensemble, on one figure.

Companion to fig_regional_all_ensembles.py. That one compares all five ensembles but
is limited to the three regions equivalent between the 133- and 16-region masks. This
one drops the cross-ensemble comparison and so is free to plot EVERY region the
ensemble has -- 16 for SSP126/SSP585/varScaled10x, 133 for CTRL/SSP585-3X.

Each panel: ensemble mean with mean +/- 1 sigma shaded. Optionally thin per-member
lines on top (--members).

TWO BLOCKS, STACKED
    upper  VAF -> sea-level equivalent (mm). Only ice above flotation, so this is the
           sea-level relevant quantity.
    lower  total ice volume change (10^12 m3). Includes FLOATING ice, so it moves for
           reasons that never reach sea level -- shelf thinning above all. The pair is
           informative precisely because they diverge: a region losing total volume
           while VAF holds steady is thinning its shelf, not its grounded ice.

Regions are ordered by |mean change| at the final year, so the panels that matter come
first, and are capped by --max-regions (133 regions do not fit on a legible page).

Usage
    python3 fig_all_regions_one_ensemble.py --ensemble SSP585
    python3 fig_all_regions_one_ensemble.py --ensemble CTRL --max-regions 24 --members
"""
from __future__ import annotations
import os, sys, glob, csv, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO_I, RHO_O, A_O = 910.0, 1028.0, 3.625e14
MEMBER_PAT = {"SSP585_varScaled10x": "SSP585_[0-9][0-9]"}


def region_names(n):
    """Names for an n-region mask; falls back to indices."""
    if n == 16:
        f = "data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"
        if os.path.exists(f):
            d = netCDF4.Dataset(f)
            if "regionNames" in d.variables:
                nm = [b"".join(r).decode(errors="ignore").strip().replace("ISMIP6 Basin ", "")
                      for r in np.asarray(d["regionNames"][:]).astype("S1")]
                d.close(); return nm
            d.close()
    if n == 133:
        f = "docs/region_mapping_133_to_ismip6.csv"
        if os.path.exists(f):
            nm = [""] * 133
            for r in csv.DictReader(open(f)):
                i = int(r["idx_133"])
                if i < 133:
                    nm[i] = r["name_133"]
            return nm
    return [f"region {i}" for i in range(n)]


def load(root, ens, keys):
    """Per-member (year, {key: (time, nRegions)}). Restart fragments dropped."""
    pat = MEMBER_PAT.get(ens, f"{ens}_[0-9][0-9]")
    out = []
    for d in sorted(glob.glob(os.path.join(root, ens, pat))):
        f = os.path.join(d, "regionalStats.nc")
        if not os.path.exists(f):
            continue
        try:
            ds = netCDF4.Dataset(f)
            yr = np.asarray(ds["daysSinceStart"][:], float) / 365.0
            V = {k: np.asarray(ds[k][:], float) for k in keys if k in ds.variables}
            ds.close()
        except Exception:
            continue
        if not V or len(yr) < 50 or yr[0] > 5.0:
            continue
        out.append((yr, V))
    return out


def series(mem, key, r, grid, to_sle):
    """(member, year) for region r on the common grid, referenced to each member's t0."""
    M = np.full((len(mem), grid.size), np.nan)
    for i, (yr, V) in enumerate(mem):
        if key not in V:
            continue
        v = V[key][:, r]
        ok = np.isfinite(yr) & np.isfinite(v)
        if ok.sum() < 10:
            continue
        d = v[ok] - v[ok][0]
        d = -d * (RHO_I / RHO_O) / A_O * 1e3 if to_sle else d / 1e12
        M[i] = np.interp(grid, yr[ok], d, left=np.nan, right=np.nan)
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/MALI/diagnostics/ENSEMBLES")
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--out-dir", default="reports/dissertation/figures/tierB")
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--max-regions", type=int, default=16)
    ap.add_argument("--min-members", type=int, default=5,
                    help="panels are blanked where fewer members remain (3X thins late)")
    ap.add_argument("--members", action="store_true", help="overlay thin per-member lines")
    ap.add_argument("--ncols", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    KEYS = ["regionalVolumeAboveFloatation", "regionalIceVolume"]
    mem = load(a.root, a.ensemble, KEYS)
    if len(mem) < 3:
        sys.exit(f"{a.ensemble}: only {len(mem)} usable members")
    nreg = mem[0][1][KEYS[0]].shape[1]
    names = region_names(nreg)
    grid = np.arange(1.0, a.horizon + 1e-9, 1.0)
    print(f"{a.ensemble}: N={len(mem)}  nRegions={nreg}")

    def stat(M):
        n = np.sum(np.isfinite(M), axis=0)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0, ddof=1)
        mu[n < a.min_members] = np.nan; sd[n < a.min_members] = np.nan
        return mu, sd, n

    # rank regions by |final mean VAF change| so the important panels come first
    rank = []
    for r in range(nreg):
        mu, _, _ = stat(series(mem, KEYS[0], r, grid, True))
        f = np.isfinite(mu)
        rank.append((r, abs(mu[np.where(f)[0][-1]]) if f.any() else 0.0))
    order = [r for r, _ in sorted(rank, key=lambda z: -z[1])][:a.max_regions]

    ncol = a.ncols
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(3.6 * ncol, 2.5 * nrow * 2),
                             sharex=True, squeeze=False)
    for k, r in enumerate(order):
        br, bc = divmod(k, ncol)
        for blk, (key, to_sle, lab, col) in enumerate([
                (KEYS[0], True,  "SLE (mm)",           "#C1121F"),
                (KEYS[1], False, "tot vol (10$^{12}$ m$^3$)", "#0072B2")]):
            ax = axes[br * 2 + blk][bc]
            M = series(mem, key, r, grid, to_sle)
            mu, sd, n = stat(M)
            if a.members:
                for row in M:
                    ax.plot(grid, row, color=col, lw=0.4, alpha=0.35)
            ax.plot(grid, mu, color=col, lw=1.6)
            ax.fill_between(grid, mu - sd, mu + sd, color=col, alpha=0.25, lw=0)
            ax.axhline(0, color="k", lw=0.4, ls=":")
            ax.grid(alpha=.25)
            nm = names[r] if r < len(names) and names[r] else f"region {r}"
            if blk == 0:
                ax.set_title(f"{nm}", fontsize=8.5)
            # sigma/|mean| is often ~0.1%, so the shaded band is thinner than the
            # line itself. Annotate the final sigma so the spread is legible even
            # when the band is not -- the smallness IS the result, but it should be
            # readable rather than merely invisible.
            fin = np.isfinite(mu) & np.isfinite(sd)
            if fin.any():
                j = np.where(fin)[0][-1]
                u = "mm" if to_sle else ""
                ax.text(0.03, 0.94, f"$\\sigma$={sd[j]:.3g}{u}  (n={int(n[j])})",
                        transform=ax.transAxes, fontsize=6.5, va="top",
                        color=col, alpha=0.9)
            if bc == 0:
                ax.set_ylabel(lab, fontsize=7.5)
            ax.tick_params(labelsize=7)
            if br * 2 + blk == nrow * 2 - 1:
                ax.set_xlabel("model year", fontsize=8)
    # blank any unused panels
    for k in range(len(order), nrow * ncol):
        br, bc = divmod(k, ncol)
        for blk in (0, 1):
            axes[br * 2 + blk][bc].set_visible(False)

    fig.suptitle(f"{a.ensemble}  —  N={len(mem)}, {nreg}-region mask, "
                 f"top {len(order)} regions by |final VAF change|\n"
                 f"per region: upper = sea-level equivalent, lower = total ice volume; "
                 f"line = ensemble mean, shading = $\\pm1\\sigma$",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = os.path.join(a.out_dir, f"F13_all_regions_{a.ensemble}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
