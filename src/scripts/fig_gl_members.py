#!/usr/bin/env python3
"""
fig_gl_members.py — grounding-line migration drawn from members, not the ensemble mean.

Averaging a geometry across members smears a moving ice edge, and the averaged geometry
satisfies no member's flotation condition. Section at the final year with every member drawn,
plus each member's grounding-line position against time.

Needs the per-member extract from hpc_extract_member_thickness.py in
reports/dissertation/figures/spatial/members/.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds                      # noqa: E402
from fig_gl_transect import (           # noqa: E402
    MESH, SHELF_MASK, BASIN_MASK, RHO_I, RHO_O,
    region_names, rd, build_transect, gl_position,
)

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MEMDIR = f"{ROOT}/reports/dissertation/figures/spatial/members"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shelf", default="Thwaites")
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--members", default=MEMDIR)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    npz = os.path.join(a.members, f"member_thickness_{a.ensemble}.npz")
    if not os.path.exists(npz):
        sys.exit(f"missing {npz}\n"
                 f"  run src/scripts/hpc_extract_member_thickness.py on the HPC first, "
                 f"then copy the .npz into {a.members}/")
    out = a.out or (f"{ROOT}/reports/dissertation/figures/slides/"
                    f"fig_gl_members_{a.shelf}_{a.ensemble}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    z = np.load(npz, allow_pickle=True)
    cells, years, members, H = z["cells"], z["years"], z["members"], z["h"]

    names, masks = region_names(SHELF_MASK)
    x, y = rd(MESH, "xCell"), rd(MESH, "yCell")
    bed, h0 = rd(MESH, "bedTopography"), rd(MESH, "thickness")
    bm = netCDF4.Dataset(BASIN_MASK)
    basins = np.asarray(bm["regionCellMasks"][:]); bm.close()
    tree = cKDTree(np.column_stack([x, y]))

    sel = masks[:, names.index(a.shelf)] > 0
    home = np.bincount(np.argmax(basins[np.where(sel)[0]], axis=1),
                       minlength=basins.shape[1]).argmax()
    s, pts, _, _ = build_transect(x, y, sel, bed, h0, basins[:, home] > 0)
    _, idx = tree.query(pts)
    inb = basins[idx, home] > 0
    if inb.any():
        j0, j1 = np.where(inb)[0][[0, -1]]
        s, pts, idx = s[j0:j1+1], pts[j0:j1+1], idx[j0:j1+1]

    b = bed[idx]
    hflot = (RHO_O / RHO_I) * np.maximum(0.0, -b)
    s_gl0, _ = gl_position(s, h0[idx], hflot)
    sk = (s - (s_gl0 if np.isfinite(s_gl0) else 0.0)) / 1e3

    # map full-mesh cell index -> position in the extracted subset
    lut = -np.ones(x.size, np.int64)
    lut[cells] = np.arange(cells.size)
    col_idx = lut[idx]
    ok = col_idx >= 0
    if ok.sum() < 20:
        sys.exit("the extract does not cover this transect — re-run with a larger --radius-km")

    def member_h(mi, ti):
        v = np.full(sk.size, np.nan)
        v[ok] = H[mi, ti, col_idx[ok]]
        return v

    fig = plt.figure(figsize=(12.6, 6.0))
    axa = fig.add_axes([0.060, 0.545, 0.905, 0.335])
    axb = fig.add_axes([0.060, 0.105, 0.905, 0.300])

    # ---------------- (a) final-year section, every member
    ti = len(years) - 1
    axa.fill_between(sk, b, 0, where=(b < 0), color=ds.ICE_TINT, alpha=.45,
                     linewidth=0, zorder=1)
    axa.fill_between(sk, b, b.min() - 800, color="#DCD3C2", linewidth=0, zorder=2)
    axa.plot(sk, b, color="#7C6F58", lw=1.6, zorder=3)
    axa.axhline(0, color=ds.INK_SOFT, lw=.8, ls=(0, (4, 3)), zorder=3)

    gl_t = np.full((len(members), len(years)), np.nan)
    for mi in range(len(members)):
        h = member_h(mi, ti)
        floating = h < hflot - 1e-6
        base = np.where(floating, -(RHO_I / RHO_O) * h, b)
        surf = base + h
        m = h > 1.0
        axa.plot(sk, np.where(m, surf, np.nan), color=ds.ICE, lw=1.0, alpha=.55, zorder=5)
        axa.plot(sk, np.where(m, base, np.nan), color=ds.ICE, lw=1.0, alpha=.55, zorder=5)
        for tj in range(len(years)):
            g, _ = gl_position(s, member_h(mi, tj), hflot)
            gl_t[mi, tj] = (g - s_gl0) / 1e3 if np.isfinite(g) else np.nan
        if np.isfinite(gl_t[mi, ti]):
            axa.plot([gl_t[mi, ti]], [np.interp(gl_t[mi, ti] * 1e3 + s_gl0, s, b)],
                     "v", ms=7, color=ds.INK, alpha=.8, zorder=7, clip_on=False)

    fin = gl_t[:, ti][np.isfinite(gl_t[:, ti])]
    xhi = min(sk.max(), (np.nanmax(fin) if fin.size else 120) + 70)
    win = (sk >= max(sk.min(), -80)) & (sk <= xhi)
    axa.set_xlim(max(sk.min(), -80), xhi)
    axa.set_ylim(float(np.nanmin(b[win])) - 120, 2200)
    ds.strip(axa)
    axa.set_ylabel("elevation  (m)", labelpad=6)
    axa.tick_params(length=3)
    axa.text(0.0, 1.10, f"{a.shelf.replace('_',' ')} · {a.ensemble} · "
             f"every member at year {years[ti]-2000}",
             transform=axa.transAxes, fontsize=14, color=ds.INK, ha="left", va="bottom")
    axa.text(0.0, 1.02, f"one line per realisation (N = {len(members)}) — no averaging, "
             "so every section shown is one the model actually produced",
             transform=axa.transAxes, fontsize=10, color=ds.INK_SOFT,
             ha="left", va="bottom")

    # ---------------- (b) grounding line vs time, per member
    t = np.concatenate([[0], years - 2000])
    G = np.column_stack([np.zeros(len(members)), gl_t])
    for mi in range(len(members)):
        axb.plot(t, G[mi], color=ds.ICE, lw=1.1, alpha=.6, zorder=3)
    axb.plot(t, np.nanmean(G, axis=0), color=ds.INK, lw=2.6, zorder=4)
    ds.strip(axb)
    axb.set_xlim(0, t[-1])
    axb.set_xticks(t)
    axb.set_xlabel("model year", labelpad=6)
    axb.set_ylabel("grounding line  (km inland)", labelpad=6)
    axb.tick_params(length=3)
    rng = np.nanmax(G[:, -1]) - np.nanmin(G[:, -1])
    axb.text(0.0, 1.04,
             f"member spread at year {t[-1]}: {rng:.1f} km across {np.isfinite(G[:,-1]).sum()} "
             "members — thick line is the ensemble mean of the member positions",
             transform=axb.transAxes, fontsize=10, color=ds.INK_SOFT,
             ha="left", va="bottom")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
    print(f"wrote {out}")
    for tj, yr in enumerate(t):
        v = G[:, tj][np.isfinite(G[:, tj])]
        if v.size:
            print(f"  year {yr:3d}: mean {v.mean():7.1f} km  range {v.min():7.1f} to "
                  f"{v.max():7.1f}  ({v.max()-v.min():5.1f} km spread, N={v.size})")


if __name__ == "__main__":
    main()
