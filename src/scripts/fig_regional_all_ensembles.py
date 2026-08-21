#!/usr/bin/env python3
"""
fig_regional_all_ensembles.py -- one figure, all five ensembles, global + regional.

THE MASK PROBLEM, AND WHY ONLY THREE REGIONS APPEAR
    CTRL and SSP585-3X write regionalStats on a 133-region draft-dependent mask;
    SSP126, SSP585 and varScaled10x use the 16 ISMIP6 basins. Putting them on one
    axis requires regions that are the SAME SET OF CELLS in both masks.

    The 133 regions are NOT a partition -- they sum to 1.66M cells on a 385k mesh
    (4.3x overcounting) because nested aggregates ("Antarctica", "West Antarctica",
    "Peninsula") sit on top of IMBIE basins and individual shelves. So summing
    133 -> 16 double-counts and is invalid.

    `docs/region_mapping_133_to_ismip6.csv` reports overlap_pct, but that measures
    CONTAINMENT ONE WAY: Pine_Island is 97% inside basin G-H, yet Pine_Island is
    4996 cells and G-H is 14331, so they are not the same region. The correct test
    is the Jaccard index |A and B| / |A or B|, which is symmetric. Measured:

        Filchner-Ronne <-> J-K    J = 0.969   EQUIVALENT
        Ross           <-> E-F    J = 0.905   EQUIVALENT
        Amery          <-> B-C    J = 0.922   EQUIVALENT
        Ronne          <-> J-K    J = 0.606   subset, NOT comparable
        Pine_Island    <-> G-H    J = 0.335   subset, NOT comparable
        Thwaites       <-> G-H    J = 0.216   subset, NOT comparable

    Only the three above may be plotted across all five ensembles. Everything else
    would compare a shelf against a whole drainage basin.

Usage
    python3 fig_regional_all_ensembles.py [--root DIR] [--out-dir DIR]
"""
from __future__ import annotations
import os, sys, glob, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO_I, RHO_O, A_O = 910.0, 1028.0, 3.625e14

# (label, index in the 133-region mask, index in the 16-region mask, Jaccard)
REGIONS = [
    ("Antarctica (global)", None, None, None),
    ("Filchner-Ronne",        0, 14, 0.969),
    ("Ross",                  1,  7, 0.905),
    ("Amery",                34,  2, 0.922),
]

# ensemble -> (member glob, which mask it uses, colour)
ENS = [
    ("CTRL",                "CTRL_[0-9][0-9]",      133, "#888888"),
    ("SSP126",              "SSP126_[0-9][0-9]",     16, "#0072B2"),
    ("SSP585",              "SSP585_[0-9][0-9]",     16, "#D55E00"),
    ("SSP585_varScaled10x", "SSP585_[0-9][0-9]",     16, "#7B3FA0"),
    ("SSP585-3X",           "SSP585-3X_[0-9][0-9]", 133, "#8B0000"),
]


def sle(v):
    """VAF (m3) -> sea-level equivalent (mm), referenced to each member's own t=0."""
    return -(v - v[0]) * (RHO_I / RHO_O) / A_O * 1e3


def load(root, ens, pat, nreg_expected, region_idx):
    """Per-member (year, SLE) for one region; region_idx None means global."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, ens, pat))):
        f = os.path.join(d, "regionalStats.nc" if region_idx is not None else "globalStats.nc")
        if not os.path.exists(f):
            continue
        try:
            ds = netCDF4.Dataset(f)
            yr = np.asarray(ds["daysSinceStart"][:], float) / 365.0
            if region_idx is None:
                v = np.asarray(ds["volumeAboveFloatation"][:], float)
            else:
                nreg = len(ds.dimensions["nRegions"])
                if nreg != nreg_expected:          # mask is not what we assumed -> skip
                    ds.close(); continue
                v = np.asarray(ds["regionalVolumeAboveFloatation"][:], float)[:, region_idx]
            ds.close()
        except Exception:
            continue
        ok = np.isfinite(yr) & np.isfinite(v) & (v > 0)
        if ok.sum() < 50 or yr[ok][0] > 5.0:        # restart fragment, not a member
            continue
        out.append((yr[ok], sle(v[ok])))
    return out


def stats(members, grid, min_members=3):
    M = np.full((len(members), grid.size), np.nan)
    for i, (yr, s) in enumerate(members):
        M[i] = np.interp(grid, yr, s, left=np.nan, right=np.nan)
    n = np.sum(np.isfinite(M), axis=0)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0, ddof=1)
    mu[n < min_members] = np.nan; sd[n < min_members] = np.nan
    return mu, sd, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/MALI/diagnostics/ENSEMBLES")
    ap.add_argument("--out-dir", default="reports/dissertation/figures/tierB")
    ap.add_argument("--horizon", type=float, default=300.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    grid = np.arange(1.0, a.horizon + 1e-9, 1.0)
    fig, axes = plt.subplots(2, len(REGIONS), figsize=(4.6 * len(REGIONS), 7.6),
                             sharex=True, squeeze=False)

    for col, (label, i133, i16, jac) in enumerate(REGIONS):
        axM, axS = axes[0][col], axes[1][col]
        for ens, pat, mask, colr in ENS:
            idx = None if i133 is None else (i133 if mask == 133 else i16)
            mem = load(a.root, ens, pat, mask, idx)
            if len(mem) < 3:
                continue
            mu, sd, n = stats(mem, grid)
            axM.plot(grid, mu, color=colr, lw=1.8, label=f"{ens} (n={len(mem)})")
            axM.fill_between(grid, mu - sd, mu + sd, color=colr, alpha=0.18, lw=0)
            axS.plot(grid, sd, color=colr, lw=1.8)
        ttl = label if jac is None else f"{label}\n(Jaccard {jac:.2f} between masks)"
        axM.set_title(ttl, fontsize=10)
        axM.grid(alpha=.3); axS.grid(alpha=.3)
        axS.set_xlabel("model year")
        if col == 0:
            axM.set_ylabel("ensemble mean SLE (mm)\nshading = $\\pm\\sigma$")
            axS.set_ylabel("ensemble spread $\\sigma$ (mm SLE)")
            axM.legend(fontsize=7, loc="upper left")

    fig.suptitle("AISLENS: global and regional response, all five ensembles\n"
                 "regions shown are the only ones equivalent between the 133- and "
                 "16-region masks", fontsize=11)
    fig.tight_layout()
    out = os.path.join(a.out_dir, "F12_regional_all_ensembles.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")

    # numbers for the text
    print(f"\n{'region':<22} {'ensemble':<22} {'N':>3} {'mean@end':>10} {'sigma@end':>10}")
    for label, i133, i16, _ in REGIONS:
        for ens, pat, mask, _c in ENS:
            idx = None if i133 is None else (i133 if mask == 133 else i16)
            mem = load(a.root, ens, pat, mask, idx)
            if len(mem) < 3:
                continue
            mu, sd, n = stats(mem, grid)
            f = np.isfinite(mu)
            if f.any():
                j = np.where(f)[0][-1]
                print(f"{label:<22} {ens:<22} {int(n[j]):>3} {mu[j]:10.2f} {sd[j]:10.3f}"
                      f"   (yr {grid[j]:.0f})")


if __name__ == "__main__":
    main()
