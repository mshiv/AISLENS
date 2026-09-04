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
    region_names, rd, build_transect, build_flowline, gl_position, gl_position_main,
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


def geometry(shelf, flowline=True):
    """Transect, bed and flotation thickness along it, plus the map into the extract.

    The section follows observed ice flow by default. The principal-axis alternative
    runs across flow on coast-parallel shelves -- Getz and Ross never left grounded
    ice on it, so no grounding line existed to find.
    """
    names, masks = region_names(SHELF_MASK)
    x, y = rd(MESH, "xCell"), rd(MESH, "yCell")
    bed, h0 = rd(MESH, "bedTopography"), rd(MESH, "thickness")
    tree = cKDTree(np.column_stack([x, y]))
    sel = masks[:, names.index(shelf)] > 0

    if flowline:
        vx, vy = rd(MESH, "observedSurfaceVelocityX"), rd(MESH, "observedSurfaceVelocityY")
        s, pts = build_flowline(x, y, sel, vx, vy, tree, bed, h0)
        idx = tree.query(pts)[1]
    else:
        bm = netCDF4.Dataset(BASIN_MASK)
        basins = np.asarray(bm["regionCellMasks"][:]); bm.close()
        home = np.bincount(np.argmax(basins[np.where(sel)[0]], axis=1),
                           minlength=basins.shape[1]).argmax()
        s, pts, _, _ = build_transect(x, y, sel, bed, h0, basins[:, home] > 0)
        idx = tree.query(pts)[1]
        inb = basins[idx, home] > 0
        if inb.any():
            j0, j1 = np.where(inb)[0][[0, -1]]
            s, idx, pts = s[j0:j1 + 1], idx[j0:j1 + 1], pts[j0:j1 + 1]

    b = bed[idx]
    hflot = (RHO_O / RHO_I) * np.maximum(0.0, -b)
    s_gl0, _ = gl_position_main(s, h0[idx], hflot)
    s_gl0 = s_gl0 if np.isfinite(s_gl0) else 0.0
    return dict(s=s, sk=(s - s_gl0) / 1e3, b=b, hflot=hflot, s_gl0=s_gl0,
                idx=idx, ncell=x.size, x=x, y=y, ice=h0 > 1.0, shelf=sel,
                tx=pts[:, 0], ty=pts[:, 1])


def locator(fig, rect, g, shelf):
    """Small plan view: the sheet, this shelf, and where the section is cut."""
    ax = fig.add_axes(rect)
    x, y, ice = g["x"] / 1e3, g["y"] / 1e3, g["ice"]
    ax.scatter(x[ice][::12], y[ice][::12], s=.12, color=ds.RULE, linewidths=0,
               rasterized=True, zorder=1)
    ax.scatter(x[g["shelf"]], y[g["shelf"]], s=2.2, color=ds.ICE, linewidths=0,
               rasterized=True, zorder=2)
    ax.plot(g["tx"] / 1e3, g["ty"] / 1e3, color=ds.MARSH_DEEP, lw=2.4, zorder=3)
    ax.plot([g["tx"][0] / 1e3], [g["ty"][0] / 1e3], "o", ms=3.0,
            color=ds.MARSH_DEEP, zorder=4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, -0.02, f"section through {shelf.replace('_', ' ')}",
            transform=ax.transAxes, fontsize=9, color=ds.INK_SOFT,
            ha="center", va="top")
    return ax


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
                p, _ = gl_position_main(g["s"], h_of(mi, ti), g["hflot"])
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
    # Spread is only meaningful across members that ran the whole record. SSP585_10
    # and _11 hold output from 2200 only and sit a full output step behind the ten
    # production members; counting them turns a 0.7 km spread into 73 km.
    complete = got.all(axis=1)
    return dict(years=years, gl=gl, H=H, got=got, members=members, h_of=h_of,
                complete=complete)


def fig_scenarios(g, memdir, shelf, out, xlim=None, ylim=None):
    """Grounding line against time, every member, every ensemble, plus how far apart
    the realisations get. Spread uses complete-record members only."""
    fig = plt.figure(figsize=(15.4, 7.8))
    ax = fig.add_axes([0.055, 0.400, 0.700, 0.520])
    axs = fig.add_axes([0.055, 0.078, 0.700, 0.225], sharex=ax)
    rows = []
    for ens, label, colr in ENSEMBLES:
        d = series(ens, g, memdir)
        if d is None:
            print(f"  ! no extract for {ens}"); continue
        keep = d["complete"]
        if keep.sum() < 3:
            keep = np.ones(len(d["members"]), bool)      # 3X: almost none run to 2300
        t = np.concatenate([[0], d["years"] - 2000])
        G = np.column_stack([np.zeros(keep.sum()), d["gl"][keep]])

        for mi in range(G.shape[0]):
            ax.plot(t, G[mi], color=colr, lw=0.9, alpha=.45, zorder=3)
        ax.plot(t, np.nanmean(G, axis=0), color=colr, lw=2.6, zorder=4)

        # interquartile spread: robust to the single member that jumps a step early
        live = np.isfinite(G).sum(axis=0) >= min(4, G.shape[0])
        q = np.where(live,
                     np.nanpercentile(np.where(live, G, np.nan), 75, axis=0)
                     - np.nanpercentile(np.where(live, G, np.nan), 25, axis=0), np.nan)
        axs.plot(t, q, color=colr, lw=2.0, zorder=3)

        fin = np.flatnonzero(np.isfinite(G).sum(axis=0) >= 2)[-1]
        j = int(np.nanargmax(q)) if np.isfinite(q).any() else 0
        rows.append((label, colr, t[fin], np.nanmean(G[:, fin]), q[j], t[j],
                     np.nanmedian(q[np.isfinite(q)]), keep.sum(), len(d["members"])))

    for a in (ax, axs):
        ds.strip(a)
        a.set_xlim(*(xlim or (0, 300)))
        a.tick_params(length=3)
    ax.tick_params(labelbottom=False)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel("grounding-line retreat\n(km inland of year 0)", labelpad=6)
    axs.set_ylabel("spread across\nrealisations (km, IQR)", labelpad=6)
    axs.set_xlabel("model year", labelpad=7)
    ax.text(0.0, 1.045, f"{shelf.replace('_', ' ')} · one line per realisation, "
            f"bold line is that ensemble's mean",
            transform=ax.transAxes, fontsize=11, color=ds.INK_SOFT, ha="left", va="bottom")

    y0 = 0.97
    ax.text(1.035, y0 + .05, "retreat by its last year · peak IQR",
            transform=ax.transAxes, fontsize=10.5, color=ds.INK, ha="left", va="top")
    for k, (label, colr, tf, gf, qmax, tq, qmed, nk, nall) in enumerate(rows):
        ax.text(1.035, y0 - .132 * k, label, transform=ax.transAxes,
                fontsize=10.5, color=colr, ha="left", va="top")
        ax.text(1.035, y0 - .132 * k - .045,
                f"{gf:.0f} km by year {tf:.0f}  ·  N = {nk}"
                + ("" if nk == nall else f" of {nall}"),
                transform=ax.transAxes, fontsize=9.5, color=ds.INK_SOFT,
                ha="left", va="top")
        ax.text(1.035, y0 - .132 * k - .085,
                f"IQR median {qmed:.2f} km, peak {qmax:.0f} km at year {tq:.0f}",
                transform=ax.transAxes, fontsize=9.5, color=ds.INK_SOFT,
                ha="left", va="top")

    locator(fig, [0.815, 0.075, 0.165, 0.235], g, shelf)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote {out}")
    for label, _, tf, gf, qmax, tq, qmed, nk, nall in rows:
        print(f"  {label:20s} N={nk:2d}/{nall:2d}  last yr {tf:3.0f}  retreat {gf:6.1f} km   "
              f"IQR median {qmed:5.2f} km, peak {qmax:5.1f} km at yr {tq:3.0f}")

def fig_section(ens, g, memdir, shelf, out, xlim=None, ylim=None, year=None):
    d = series(ens, g, memdir)
    if d is None:
        sys.exit(f"no usable extract for {ens}")
    years, gl, got, members, sk, b = (d["years"], d["gl"], d["got"], d["members"],
                                      g["sk"], g["b"])
    need = max(2, int(0.8 * len(members)))
    if year is not None:
        ti = int(np.argmin(np.abs(years - (year + 2000))))
    else:
        ti = int(np.max(np.where(np.isfinite(gl).sum(axis=0) >= need)[0]))
    ti_data = int(np.max(np.where(got.sum(axis=0) >= need)[0]))

    # member fields first: the ocean fill has to stop where the ice is grounded,
    # and members disagree about where that is
    HH, BB = [], []
    for mi in range(len(members)):
        if not got[mi, ti]:
            continue
        hm = d["h_of"](mi, ti)
        fl = hm < g["hflot"] - 1e-6
        HH.append(hm)
        BB.append(np.where(fl, -(RHO_I / RHO_O) * hm, b))
    HH, BB = np.array(HH), np.array(BB)
    gr_frac = np.nanmean((HH > 1.0) & (HH > g["hflot"]), axis=0)
    wet = (b < 0) & (gr_frac < 0.5)

    fig = plt.figure(figsize=(15.0, 6.0))
    ax = fig.add_axes([0.060, 0.135, 0.905, 0.700])
    # water tops out at the shallowest member's ice base where a shelf is present
    otop = np.where(np.nanmin(np.where(HH > 1.0, BB, 0.0), axis=0) < 0,
                    np.nanmin(np.where(HH > 1.0, BB, 0.0), axis=0), 0.0)
    ax.fill_between(sk, b, otop, where=wet & (b < otop), color=ds.ICE,
                    alpha=.32, linewidth=0, zorder=1)
    ax.fill_between(sk, np.nanmedian(BB, axis=0),
                    np.nanmedian(BB + np.where(HH > 1.0, HH, np.nan), axis=0),
                    color=ds.FIELD, linewidth=0, zorder=2)
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
    ax.set_xlim(*(xlim or (max(sk.min(), -80), xhi)))
    ax.set_ylim(*(ylim or (float(np.nanmin(b[win])) - 120, 420)))
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
    ap.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="zoom: model years for the default figure, km along the section for --section")
    ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="zoom: km of retreat for the default figure, elevation for --section")
    ap.add_argument("--principal-axis", action="store_true",
                    help="use the old principal-axis transect instead of the flowline")
    ap.add_argument("--suffix", default="", help="appended to the output filename")
    ap.add_argument("--year", type=int, default=None,
                    help="--section only: model year to draw (default: last year most members have a GL)")
    a = ap.parse_args()
    if not os.path.isdir(a.members):
        sys.exit(f"missing {a.members} — run hpc_extract_member_thickness.py first")
    os.makedirs(a.outdir, exist_ok=True)
    ds.apply()

    g = geometry(a.shelf, flowline=not a.principal_axis)
    if a.section:
        fig_section(a.section, g, a.members, a.shelf,
                    f"{a.outdir}/fig_gl_members_{a.shelf}_{a.section}{a.suffix}.png",
                    a.xlim, a.ylim, a.year)
    else:
        fig_scenarios(g, a.members, a.shelf,
                      f"{a.outdir}/fig_gl_scenarios_{a.shelf}{a.suffix}.png",
                      a.xlim, a.ylim)


if __name__ == "__main__":
    main()
