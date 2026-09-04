#!/usr/bin/env python3
"""
fig_gl_transect.py — ice geometry and grounding line along a shelf transect.

Bed, ocean, ice surface and base along the shelf's principal axis, confined to its own ISMIP6
basin, with the grounding line at several years and sigma_h/|dg/ds| as the implied spread in
its position.

Validated for Thwaites only. A 1-D crossing rule depends on where the transect is drawn:
Pine Island differs by hundreds of km between two defensible directions, and Ronne loses the
crossing once it ungrounds. Use fig_grounded_area.py for any multi-shelf comparison, and
fig_gl_members.py once per-member fields are available -- the ensemble mean geometry is not
a member, so it is drawn dashed where thickness_min <= 1 m < thickness_max.
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
import slidestyle as ds  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SPAT = f"{ROOT}/reports/dissertation/figures/spatial/stats_sample"
try:
    from aislens.config import config
    _MALI = str(config.DIR_MALI)
except Exception:
    _MALI = os.path.join(os.environ.get("AISLENS_DATA_DIR", ROOT), "data", "MALI")

MESH = os.path.join(_MALI, "AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m"
                    "_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu"
                    "_meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
SHELF_MASK = os.path.join(_MALI, "aislens_draftDepen_regionMasks.nc")
BASIN_MASK = os.path.join(_MALI, "AIS_4to20km_r01_20220907.regionMask_ismip6.nc")
ENS_ROOT = os.path.join(_MALI, "diagnostics", "ENSEMBLES")

RHO_I, RHO_O = 910.0, 1028.0
STEP_M = 2000.0          # sampling interval along the transect
EXTEND_SEA_KM = 60.0     # how far past the shelf's seaward end to start
EXTEND_LAND_KM = 190.0   # how far inland to run

YEARS = [2100, 2200, 2300]
# Validated for Thwaites only.  A 1-D crossing rule is unstable where the transect
# is not along flow: Pine Island differs by hundreds of km between two defensible
# transect choices, and Ronne loses the crossing once it ungrounds.  Use
# fig_grounded_area.py for any multi-shelf comparison.
SHELVES = ["Thwaites"]


# ---------------------------------------------------------------- loading
def region_names(path):
    d = netCDF4.Dataset(path)
    raw = d["regionNames"][:]
    names = ["".join(x.decode() if isinstance(x, bytes) else str(x) for x in row)
             .replace("\x00", "").strip("-").strip() for row in raw]
    masks = np.asarray(d["regionCellMasks"][:])
    d.close()
    return names, masks


def rd(path, var):
    d = netCDF4.Dataset(path)
    a = np.ma.filled(np.asarray(d[var][:], dtype=float), np.nan)
    d.close()
    return np.ravel(a) if a.ndim > 1 else a


# ---------------------------------------------------------------- transect
def build_transect(x, y, sel, bed, h0, basin_sel=None):
    """Transect from the shelf toward the grounded interior of its own basin.

    Direction is the vector from the shelf's centroid to the centroid of the
    grounded ice in the same drainage basin.  That is always along-flow and always
    points inland.  A principal-axis fit is not used: for a roughly square shelf
    the long axis is arbitrary and can run across flow rather than along it.
    """
    P = np.column_stack([x[sel], y[sel]])
    c = P.mean(axis=0)
    u = np.linalg.svd(P - c, full_matrices=False)[2][0]      # shelf long axis

    # orient +u inland: the inland end has more grounded ice around it
    hf_all = (RHO_O / RHO_I) * np.maximum(0.0, -bed)
    def groundedness(sign):
        probe = c + sign * u * 60e3
        near = np.hypot(x - probe[0], y - probe[1]) < 40e3
        if near.sum() == 0:
            return -np.inf
        return float(np.nanmean((h0[near] > hf_all[near] + 1.0)))
    if groundedness(-1) > groundedness(+1):
        u = -u

    proj = (P - c) @ u
    s0, s1 = proj.min() - EXTEND_SEA_KM * 1e3, proj.max() + EXTEND_LAND_KM * 1e3
    s = np.arange(s0, s1 + STEP_M, STEP_M)
    pts = c[None, :] + s[:, None] * u[None, :]
    return s, pts, c, u


def build_flowline(x, y, sel, vx, vy, tree, bed, h0, step_m=2000.0,
                   n_up=500, n_down=120, smooth=9, tail_up_km=380.0, tail_down_km=40.0):
    """Transect traced along observed ice flow through the shelf centroid.

    build_transect uses the shelf's principal axis, which is the long axis of the
    polygon. For a coast-parallel shelf like Getz or Ross that axis runs across flow,
    so the section never leaves grounded ice and no grounding line exists on it. Here
    the path is integrated up- and downstream through the observed velocity field, so
    it is along flow by construction.

    Returns (s, pts) with s increasing inland, matching build_transect.
    """
    P = np.column_stack([x[sel], y[sel]])
    c = P.mean(axis=0)
    spd = np.hypot(vx, vy)

    # Filchner-Ronne and Ross are 500+ km long, so a fixed step count never reaches
    # grounded ice; march until the condition is met instead, then run on a little.
    hflot_all = (RHO_O / RHO_I) * np.maximum(0.0, -bed)
    grounded_all = (h0 > 1.0) & (h0 > hflot_all)

    def march(p0, sign, n, want, tail_km):
        tail = int(tail_km * 1e3 / step_m)
        pts, p, hit = [], p0.copy(), -1
        for k in range(n):
            j = tree.query(p)[1]
            v = np.array([vx[j], vy[j]])
            m = np.hypot(*v)
            if not np.isfinite(m) or m <= 0:
                break
            p = p + sign * step_m * v / m
            pts.append(p.copy())
            if hit < 0 and want(tree.query(p)[1]):
                hit = k
            if hit >= 0 and k - hit >= tail:
                break
        return np.array(pts) if pts else np.empty((0, 2))

    # start from the fastest cell in the shelf: the centroid can sit on a slow margin
    idx_shelf = np.flatnonzero(sel)
    start = np.column_stack([x, y])[idx_shelf[np.argmax(spd[idx_shelf])]]
    # the inland tail must outrun the retreat itself: Thwaites moves 300 km, and a
    # line that stops just past the present grounding line goes fully afloat by year 30
    up = march(start, -1.0, n_up, lambda j: grounded_all[j], tail_up_km)
    down = march(start, +1.0, n_down, lambda j: h0[j] <= 1.0, tail_down_km)
    pts = np.vstack([down[::-1], start[None, :], up])
    if smooth > 1 and pts.shape[0] > smooth:
        k = np.ones(smooth) / smooth
        pts = np.column_stack([np.convolve(pts[:, 0], k, "same"),
                               np.convolve(pts[:, 1], k, "same")])
        pts[:smooth], pts[-smooth:] = pts[smooth], pts[-smooth - 1]
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    return d - d[down.shape[0]], pts


def gl_position_main(s, h, hflot, min_run_km=15.0):
    """Seaward edge of the grounded body connected to the interior. (s_gl, slope).

    gl_position below takes the seaward-most floating->grounded crossing, so any
    pinning point, ice rise or bedrock bump seaward of the real grounding line
    captures it -- that is what returns -203 km for Filchner. The transect always
    ends in the grounded interior, so the run containing the inland end is the ice
    sheet proper; its seaward edge is the grounding line. Runs shorter than
    min_run_km are ignored as pinning points.
    """
    grounded = (h > 1.0) & (h > hflot)
    if not grounded.any():
        return np.nan, np.nan
    d = np.diff(grounded.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if grounded[0]:
        starts = np.r_[0, starts]
    if grounded[-1]:
        ends = np.r_[ends, grounded.size]
    runs = [(a, b) for a, b in zip(starts, ends)
            if s[b - 1] - s[a] >= min_run_km * 1e3]
    if not runs:
        return np.nan, np.nan
    a, _ = runs[-1]              # most inland surviving run = the ice sheet
    if a == 0:
        return s[0], np.nan      # grounded from the seaward end of the section
    g = h - hflot
    g = np.where(h > 1.0, g, -np.abs(g) - 1.0)
    g0, g1 = g[a - 1], g[a]
    if g1 == g0:
        return s[a], np.nan
    t = -g0 / (g1 - g0)
    return s[a - 1] + t * (s[a] - s[a - 1]), (g1 - g0) / (s[a] - s[a - 1])


def gl_position(s, h, hflot):
    """Seaward-most floating->grounded crossing, refined linearly. (s_gl, slope).

    NOTE: this rule is only trustworthy where the transect runs along flow and the
    shelf has no large seaward pinning points.  See the module docstring -- it is
    validated for Thwaites and NOT used for other shelves.
    """
    g = h - hflot
    g = np.where(h > 1.0, g, -np.abs(g) - 1.0)      # ice-free counts as not grounded
    sgn = np.sign(g)
    idx = np.where((sgn[:-1] < 0) & (sgn[1:] >= 0))[0]
    if idx.size == 0:
        return np.nan, np.nan
    i = idx[0]
    g0, g1 = g[i], g[i + 1]
    if g1 == g0:
        return s[i], np.nan
    t = -g0 / (g1 - g0)
    slope = (g1 - g0) / (s[i + 1] - s[i])
    return s[i] + t * (s[i + 1] - s[i]), slope


# ---------------------------------------------------------------- figure
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shelf", default="Thwaites")
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--all", action="store_true",
                    help="every shelf in SHELVES (validated for Thwaites only)")
    ap.add_argument("--outdir", default=f"{ROOT}/reports/dissertation/figures/slides")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ds.apply()

    names, masks = region_names(SHELF_MASK)
    bm = netCDF4.Dataset(BASIN_MASK)
    basins = np.asarray(bm["regionCellMasks"][:])
    bm.close()
    x, y = rd(MESH, "xCell"), rd(MESH, "yCell")
    bed, h0 = rd(MESH, "bedTopography"), rd(MESH, "thickness")
    tree = cKDTree(np.column_stack([x, y]))

    def sample(field, pts):
        """Inverse-distance blend of the 3 nearest cells -- the MALI mesh is 4-20 km,
        so plain nearest-neighbour turns a smooth bed into a staircase."""
        d, idx = tree.query(pts, k=3)
        w = 1.0 / np.maximum(d, 1.0)
        w /= w.sum(axis=1, keepdims=True)
        return np.nansum(field[idx] * w, axis=1)

    shelves = SHELVES if a.all else [a.shelf]
    for shelf in shelves:
        if shelf not in names:
            print(f"  ! {shelf} not in the region mask"); continue
        sel = masks[:, names.index(shelf)] > 0
        home0 = np.bincount(np.argmax(basins[np.where(sel)[0]], axis=1),
                            minlength=basins.shape[1]).argmax()
        s, pts, c, u = build_transect(x, y, sel, bed, h0, basins[:, home0] > 0)

        # confine the section to the shelf's own ISMIP6 drainage basin: a long
        # principal axis can run over a divide into a neighbouring catchment, and a
        # "grounding line" found there would belong to a different glacier
        _, idx_all = tree.query(pts)
        in_basin = basins[idx_all, home0] > 0
        if in_basin.any():
            j0, j1 = np.where(in_basin)[0][[0, -1]]
            keep = slice(j0, j1 + 1)
            s, pts = s[keep], pts[keep]

        b = sample(bed, pts)
        prof = {0: sample(h0, pts)}
        spread, disagree = {}, {}
        for yr in YEARS:
            f = f"{SPAT}/{a.ensemble}_{yr}.nc"
            if not os.path.exists(f):
                continue
            prof[yr - 2000] = sample(rd(f, "thickness_mean"), pts)
            spread[yr - 2000] = sample(rd(f, "thickness_std"), pts)
            # where the thinnest member has no ice but the thickest does, the
            # ensemble MEAN thickness describes no member at all
            _, i1 = tree.query(pts)
            disagree[yr - 2000] = ((rd(f, "thickness_max")[i1] > 1.0) &
                                   (rd(f, "thickness_min")[i1] <= 1.0))

        hflot = (RHO_O / RHO_I) * np.maximum(0.0, -b)

        # re-origin: s = 0 is the INITIAL grounding line, so the axis reads as
        # "how far has the grounding line moved", which is the question
        s_gl0, _ = gl_position(s, prof[0], hflot)
        if np.isfinite(s_gl0):
            s = s - s_gl0
        sk = s / 1e3

        fig = plt.figure(figsize=(12.4, 5.6))
        ax = fig.add_axes([0.062, 0.135, 0.905, 0.70])

        # ocean and bed
        ax.fill_between(sk, b, 0, where=(b < 0), color=ds.ICE_TINT, alpha=.55,
                        linewidth=0, zorder=1)
        ax.fill_between(sk, b, b.min() - 800, color="#DCD3C2", linewidth=0, zorder=2)
        ax.plot(sk, b, color="#7C6F58", lw=1.8, zorder=3)
        ax.axhline(0, color=ds.INK_SOFT, lw=.8, ls=(0, (4, 3)), zorder=3)

        shades = [ds.LINEWORK, "#7FA9C4", ds.ICE, ds.INK]
        gl_rows = []
        for k, (yr, h) in enumerate(sorted(prof.items())):
            floating = h < hflot - 1e-6
            base = np.where(floating, -(RHO_I / RHO_O) * h, b)
            surf = base + h
            base = np.where(h > 1.0, base, np.nan)
            surf = np.where(h > 1.0, surf, np.nan)
            col = shades[min(k, len(shades) - 1)]
            lw = 2.4 if yr in (0, max(prof)) else 1.5
            dis = disagree.get(yr, np.zeros_like(sk, dtype=bool))
            solid = np.where(dis, np.nan, 1.0)
            ax.plot(sk, surf * solid, color=col, lw=lw, zorder=5)
            ax.plot(sk, base * solid, color=col, lw=lw, zorder=5)
            if dis.any():          # ensemble mean is not a member here -- dash it
                ax.plot(sk, np.where(dis, surf, np.nan), color=col, lw=lw,
                        ls=(0, (2, 2.2)), alpha=.75, zorder=5)
                ax.plot(sk, np.where(dis, base, np.nan), color=col, lw=lw,
                        ls=(0, (2, 2.2)), alpha=.75, zorder=5)
            if yr == 0:
                ax.fill_between(sk, base, surf, color=col, alpha=.16,
                                linewidth=0, zorder=4)

            s_gl, slope = gl_position(s, h, hflot)
            if not np.isfinite(s_gl):
                continue
            sig_s = np.nan
            if yr in spread and np.isfinite(slope) and abs(slope) > 1e-9:
                j = int(np.argmin(np.abs(s - s_gl)))
                sig_s = spread[yr][j] / abs(slope)
            gl_rows.append((yr, s_gl / 1e3, sig_s / 1e3 if np.isfinite(sig_s) else np.nan, col))

            ax.plot([s_gl / 1e3], [np.interp(s_gl, s, b)], "v", ms=9, color=col,
                    zorder=7, clip_on=False)
            ax.axvline(s_gl / 1e3, color=col, lw=.9, alpha=.45, zorder=3)
            if np.isfinite(sig_s):
                ax.plot([(s_gl - sig_s) / 1e3, (s_gl + sig_s) / 1e3],
                        [np.interp(s_gl, s, b)] * 2, color=col, lw=3.2, alpha=.5,
                        solid_capstyle="butt", zorder=6)

        # where members disagree on whether ice is present at all
        last = max(prof)
        dlast = disagree.get(last)

        # crop to where the action is: seaward of the initial GL to just past the last one
        gl_all = [g[1] for g in gl_rows if np.isfinite(g[1])]
        xlo = max(sk.min(), -80.0)
        xhi = min(sk.max(), (max(gl_all) if gl_all else 120.0) + 70.0)
        inwin = (sk >= xlo) & (sk <= xhi)

        ds.strip(ax)
        ax.set_xlim(xlo, xhi)
        lo = float(np.nanmin(b[inwin]))
        tops = [np.nanmax(np.where(h[inwin] > 1.0,
                                   np.where(h[inwin] < hflot[inwin],
                                            (1 - RHO_I / RHO_O) * h[inwin],
                                            b[inwin] + h[inwin]), np.nan))
                for h in prof.values()]
        hi = float(np.nanmax(tops))
        ax.set_ylim(lo - 210, hi + 160)
        if dlast is not None and dlast.any():
            yb = lo - 40
            ax.fill_between(sk, yb - 55, yb, where=dlast, color=ds.MARSH, alpha=.55,
                            linewidth=0, zorder=6)
            frac = 100 * float(np.mean(dlast[(sk >= 0)]))
            ax.text(0.995, 0.052,
                    f"dashed: the ensemble mean is not a member — some realisations keep ice "
                    f"here, others are open water  ({frac:.0f}% of the section)",
                    transform=ax.transAxes, fontsize=10, color=ds.MARSH,
                    ha="right", va="bottom")

        ax.set_xlabel("distance from the initial grounding line, seaward → inland  (km)",
                      labelpad=7)
        ax.set_ylabel("elevation  (m)", labelpad=6)
        ax.tick_params(length=3)

        ax.text(0.006, 1.135, f"{shelf.replace('_',' ')} · {a.ensemble}",
                transform=ax.transAxes, fontsize=15, color=ds.INK, ha="left", va="bottom")
        ax.text(0.006, 1.055,
                "ensemble-mean ice surface and base · triangles mark the grounding line",
                transform=ax.transAxes, fontsize=10, color=ds.INK_SOFT,
                ha="left", va="bottom")

        # headline: the migration, and how little of it depends on the realisation
        if len(gl_rows) >= 2:
            last_in = [g for g in gl_rows if g[1] <= xhi]
            ref = last_in[-1] if last_in else gl_rows[-1]
            sig_max = np.nanmax([g[2] for g in gl_rows if np.isfinite(g[2])] or [np.nan])
            ax.text(0.995, 1.135, f"{ref[1] - gl_rows[0][1]:+.0f} km",
                    transform=ax.transAxes, fontsize=21, color=ds.MARSH,
                    ha="right", va="bottom")
            ax.text(0.995, 1.06, f"grounding-line migration by year {ref[0]}",
                    transform=ax.transAxes, fontsize=10, color=ds.INK_SOFT,
                    ha="right", va="bottom")
            if np.isfinite(sig_max):
                ax.text(0.995, 0.955,
                        f"± {sig_max:.1f} km across realisations",
                        transform=ax.transAxes, fontsize=12.5, color=ds.ICE,
                        ha="right", va="bottom")

        # per-year key, with any grounding line that has left the window called out
        for k, (yr, sg, sig, col) in enumerate(gl_rows):
            lab = f"year {yr}"
            if sg > xhi:
                lab += f"   {sg:+.0f} km — ungrounded through the trough"
            ax.text(0.006, 0.44 - 0.072 * k, lab, transform=ax.transAxes,
                    fontsize=10.5, color=col, ha="left", va="center")

        out = f"{a.outdir}/fig_gl_transect_{shelf}.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
        plt.close(fig)
        print(f"wrote {out}")
        for yr, sg, sig, _ in gl_rows:
            sg_txt = f"{sg:8.1f} km"
            sig_txt = f"±{sig:4.1f} km" if np.isfinite(sig) else "     n/a"
            print(f"    year {yr:3d}   GL at {sg_txt}   {sig_txt}")


if __name__ == "__main__":
    main()
