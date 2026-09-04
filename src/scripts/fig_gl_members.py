#!/usr/bin/env python3
"""
fig_gl_members.py — grounding-line migration drawn from members, not the ensemble mean.

Averaging geometry across members smears a moving ice edge, and the averaged geometry satisfies
no member's flotation condition. Everything here is computed per member and only then compared.

Default: grounding-line position against time for every member of every ensemble, which puts
scenario separation and realisation spread in the same units on the same axis.
--section ENS: the year-slice view, every member's ice surface and base along the transect.

Needs the per-member extract from hpc_extract_member_thickness.py in
reports/dissertation/figures/spatial/members/.

Validated for Thwaites. The 1-D crossing rule depends on where the transect is drawn -- see
fig_gl_transect.py -- so use fig_grounded_area.py for multi-shelf comparison.
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
OUTDIR = f"{ROOT}/reports/dissertation/figures/slides"

# order is the story order: no trend, weak trend, strong trend, louder, stronger
ENSEMBLES = [
    ("CTRL",                "control",          ds.INK_SOFT),
    ("SSP126",              "SSP1-2.6",         ds.ICE),
    ("SSP585",              "SSP5-8.5",         ds.MARSH),
    ("SSP585_varScaled10x", "SSP5-8.5, 10x",    ds.MARSH_DEEP),
    ("SSP585-3X",           "SSP5-8.5, 3x trend", ds.INK),
]
BACKSTEP_KM = 50.0      # a seaward jump this large means the crossing rule has failed


def geometry(shelf):
    """Transect, bed and flotation thickness along it, plus the map into the extract."""
    names, masks = region_names(SHELF_MASK)
    x, y = rd(MESH, "xCell"), rd(MESH, "yCell")
    bed, h0 = rd(MESH, "bedTopography"), rd(MESH, "thickness")
    bm = netCDF4.Dataset(BASIN_MASK)
    basins = np.asarray(bm["regionCellMasks"][:]); bm.close()
    tree = cKDTree(np.column_stack([x, y]))

    sel = masks[:, names.index(shelf)] > 0
    home = np.bincount(np.argmax(basins[np.where(sel)[0]], axis=1),
                       minlength=basins.shape[1]).argmax()
    s, pts, _, _ = build_transect(x, y, sel, bed, h0, basins[:, home] > 0)
    _, idx = tree.query(pts)
    inb = basins[idx, home] > 0
    if inb.any():
        j0, j1 = np.where(inb)[0][[0, -1]]
        s, idx = s[j0:j1 + 1], idx[j0:j1 + 1]

    b = bed[idx]
    hflot = (RHO_O / RHO_I) * np.maximum(0.0, -b)
    s_gl0, _ = gl_position(s, h0[idx], hflot)
    s_gl0 = s_gl0 if np.isfinite(s_gl0) else 0.0
    return dict(s=s, sk=(s - s_gl0) / 1e3, b=b, hflot=hflot, s_gl0=s_gl0,
                idx=idx, ncell=x.size)


def series(ens, g, memdir):
    """(years, gl_t (M,T) km inland, H, got, members) for one ensemble, or None."""
    npz = os.path.join(memdir, f"member_thickness_{ens}.npz")
    if not os.path.exists(npz):
        return None
    z = np.load(npz, allow_pickle=True)
    cells, years, members, H = z["cells"], z["years"], z["members"], z["h"]
    got = z["year_got"] >= 0

    lut = -np.ones(g["ncell"], np.int64)
    lut[cells] = np.arange(cells.size)
    col = lut[g["idx"]]
    ok = col >= 0
    if ok.sum() < 20:
        return None

    def h_of(mi, ti):
        v = np.full(g["sk"].size, np.nan)
        v[ok] = H[mi, ti, col[ok]]
        return v

    gl = np.full((len(members), len(years)), np.nan)
    for mi in range(len(members)):
        for ti in range(len(years)):
            if got[mi, ti]:
                p, _ = gl_position(g["s"], h_of(mi, ti), g["hflot"])
                gl[mi, ti] = (p - g["s_gl0"]) / 1e3 if np.isfinite(p) else np.nan
        # once the line jumps back seaward the crossing rule has picked a different
        # feature; everything after that is not a grounding line for this glacier
        run = -np.inf
        for ti in range(len(years)):
            if np.isfinite(gl[mi, ti]):
                if gl[mi, ti] < run - BACKSTEP_KM:
                    gl[mi, ti:] = np.nan
                    break
                run = max(run, gl[mi, ti])
    return dict(years=years, gl=gl, H=H, got=got, members=members, h_of=h_of)


def fig_scenarios(g, memdir, shelf, out):
    fig = plt.figure(figsize=(15.0, 6.6))
    ax = fig.add_axes([0.058, 0.115, 0.735, 0.795])
    rows = []
    for ens, label, colr in ENSEMBLES:
        d = series(ens, g, memdir)
        if d is None:
            print(f"  ! no extract for {ens}"); continue
        t = np.concatenate([[0], d["years"] - 2000])
        G = np.column_stack([np.zeros(len(d["members"])), d["gl"]])
        for mi in range(G.shape[0]):
            ax.plot(t, G[mi], color=colr, lw=0.9, alpha=.45, zorder=3)
        ax.plot(t, np.nanmean(G, axis=0), color=colr, lw=2.6, zorder=4)

        # no inline labels: the three SSP5-8.5 variants converge on the same
        # end point and their labels land on top of each other. The key names them.
        fin = np.flatnonzero(np.isfinite(G).sum(axis=0) >= 2)[-1]
        live = np.isfinite(G).sum(axis=0) >= 2
        sp = np.where(live, np.nanmax(np.where(live, G, np.nan), axis=0)
                      - np.nanmin(np.where(live, G, np.nan), axis=0), np.nan)
        j = int(np.nanargmax(sp))
        rows.append((label, colr, t[fin], np.nanmean(G[:, fin]), sp[j], t[j],
                     np.nanmedian(sp[np.isfinite(sp)])))

    ds.strip(ax)
    ax.set_xlim(0, 300)
    ax.set_xlabel("model year", labelpad=7)
    ax.set_ylabel("grounding-line retreat  (km inland of its year-0 position)", labelpad=6)
    ax.tick_params(length=3)
    ax.text(0.0, 1.055, f"{shelf.replace('_', ' ')} · one line per realisation, "
            f"bold line is that ensemble's mean",
            transform=ax.transAxes, fontsize=11, color=ds.INK_SOFT, ha="left", va="bottom")

    # the numbers the figure exists to make: separation between ensembles against
    # spread inside them
    y0 = 0.93
    ax.text(1.030, y0 + .055, "peak spread across realisations",
            transform=ax.transAxes, fontsize=10.5, color=ds.INK, ha="left", va="top")
    for k, (label, colr, tf, gf, spmax, tsp, spmed) in enumerate(rows):
        ax.text(1.030, y0 - .085 * k, f"{label}", transform=ax.transAxes,
                fontsize=10.5, color=colr, ha="left", va="top")
        ax.text(1.030, y0 - .085 * k - .040,
                f"{spmax:.0f} km at year {tsp:.0f} · median {spmed:.1f} km",
                transform=ax.transAxes, fontsize=9.5, color=ds.INK_SOFT,
                ha="left", va="top")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote {out}")
    for label, _, tf, gf, spmax, tsp, spmed in rows:
        print(f"  {label:20s} last year {tf:3.0f}  retreat {gf:6.1f} km   "
              f"peak spread {spmax:5.1f} km at year {tsp:3.0f}   median {spmed:4.1f} km")


def fig_section(ens, g, memdir, shelf, out):
    d = series(ens, g, memdir)
    if d is None:
        sys.exit(f"no usable extract for {ens}")
    years, gl, got, members, sk, b = (d["years"], d["gl"], d["got"], d["members"],
                                      g["sk"], g["b"])
    need = max(2, int(0.8 * len(members)))
    ti = int(np.max(np.where(np.isfinite(gl).sum(axis=0) >= need)[0]))
    ti_data = int(np.max(np.where(got.sum(axis=0) >= need)[0]))

    fig = plt.figure(figsize=(15.0, 6.0))
    ax = fig.add_axes([0.060, 0.135, 0.905, 0.700])
    ax.fill_between(sk, b, 0, where=(b < 0), color=ds.ICE_TINT, alpha=.45,
                    linewidth=0, zorder=1)
    ax.fill_between(sk, b, b.min() - 800, color="#DCD3C2", linewidth=0, zorder=2)
    ax.plot(sk, b, color="#7C6F58", lw=1.6, zorder=3)
    ax.axhline(0, color=ds.INK_SOFT, lw=.8, ls=(0, (4, 3)), zorder=3)

    for mi in range(len(members)):
        if not got[mi, ti]:
            continue
        h = d["h_of"](mi, ti)
        floating = h < g["hflot"] - 1e-6
        base = np.where(floating, -(RHO_I / RHO_O) * h, b)
        m = h > 1.0
        ax.plot(sk, np.where(m, base + h, np.nan), color=ds.ICE, lw=1.0, alpha=.55, zorder=5)
        ax.plot(sk, np.where(m, base, np.nan), color=ds.ICE, lw=1.0, alpha=.55, zorder=5)
        if np.isfinite(gl[mi, ti]):
            ax.plot([gl[mi, ti]], [np.interp(gl[mi, ti], sk, b)], "v", ms=7,
                    color=ds.INK, alpha=.8, zorder=7, clip_on=False)

    fin = gl[:, ti][np.isfinite(gl[:, ti])]
    xhi = min(sk.max(), (fin.max() if fin.size else 120) + 70)
    win = (sk >= max(sk.min(), -80)) & (sk <= xhi)
    ax.set_xlim(max(sk.min(), -80), xhi)
    ax.set_ylim(float(np.nanmin(b[win])) - 120, 420)
    ds.strip(ax)
    ax.set_xlabel("distance from the year-0 grounding line, seaward → inland  (km)", labelpad=7)
    ax.set_ylabel("elevation  (m)", labelpad=6)
    ax.tick_params(length=3)
    ax.text(0.0, 1.10, f"{shelf.replace('_', ' ')} · {ens} · every member at "
            f"year {years[ti] - 2000}", transform=ax.transAxes, fontsize=14,
            color=ds.INK, ha="left", va="bottom")
    tail = "" if ti == ti_data else " · later the grounding line leaves this section"
    ax.text(0.0, 1.02, f"N = {len(fin)} realisations, no averaging — every section drawn "
            f"is one the model produced · spread {fin.max() - fin.min():.1f} km{tail}",
            transform=ax.transAxes, fontsize=10, color=ds.INK_SOFT, ha="left", va="bottom")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  year {years[ti]-2000}: mean {fin.mean():.1f} km, "
          f"range {fin.min():.1f}-{fin.max():.1f} ({fin.max()-fin.min():.1f} km), N={fin.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shelf", default="Thwaites")
    ap.add_argument("--section", default=None, metavar="ENSEMBLE",
                    help="draw the per-member year slice for one ensemble instead")
    ap.add_argument("--members", default=MEMDIR)
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    if not os.path.isdir(a.members):
        sys.exit(f"missing {a.members} — run hpc_extract_member_thickness.py first")
    os.makedirs(a.outdir, exist_ok=True)
    ds.apply()

    g = geometry(a.shelf)
    if a.section:
        fig_section(a.section, g, a.members, a.shelf,
                    f"{a.outdir}/fig_gl_members_{a.shelf}_{a.section}.png")
    else:
        fig_scenarios(g, a.members, a.shelf, f"{a.outdir}/fig_gl_scenarios_{a.shelf}.png")


if __name__ == "__main__":
    main()
