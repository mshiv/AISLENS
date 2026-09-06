#!/usr/bin/env python3
"""
fig_shelf_geometry.py -- what happens to the shelves themselves, from draft and damage.

Two fields that the deck never shows. The melt the shelves receive is set from their draft,
so the draft is the input the scheme reads, and it moves as the shelves thin. Damage is the
model's own measure of weakened ice, which is the nearest thing in the run to a map of where
buttressing is being lost.

Four panels: the draft at the start, how far it shoals by the end, where damage grows, and
the shelf draft distribution at both ends. The last one is the point -- the melt scheme is
draft dependent, so a shift in that distribution is a shift in the forcing the shelves get,
produced by the ice itself rather than by the ocean.

No melt is computed here. The coefficients are on the mesh and the scheme is linear in
draft, but the mean melt field is set from satellite observations rather than from those
coefficients, so evaluating the parameterisation and calling it the applied melt would be
wrong. This stays with what was measured.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm, Normalize, ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import oceancolors as oc         # noqa: E402
import fig_gl_transect as glt    # noqa: E402

FIELDS = f"{glt.ROOT}/reports/dissertation/figures/spatial/members_state/fields"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=os.path.join(
        glt.ROOT, "reports/dissertation/figures/slides"))
    a = ap.parse_args()
    out = a.out or f"{a.outdir}/fig_shelf_geometry.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    import netCDF4
    mesh = os.path.join(glt.ROOT, "data/MALI", os.path.basename(glt.MESH))
    d = netCDF4.Dataset(mesh); cov = np.asarray(d["cellsOnVertex"][:]) - 1; d.close()
    x, y = glt.rd(mesh, "xCell"), glt.rd(mesh, "yCell")
    bed = glt.rd(mesh, "bedTopography")

    z = np.load(f"{FIELDS}/member_thickness_{a.ensemble}.npz", allow_pickle=True)
    keep = (z["year_got"] >= 0).all(axis=1)
    cells, yrs = z["cells"], np.asarray(z["years"])
    thk = z["thickness"][keep]
    draft = z["lowerSurface"][keep]
    dmg = z["damage"][keep]
    print(f"  {a.ensemble}: {int(keep.sum())} complete members, "
          f"years {yrs[0]-2000:.0f}..{yrs[-1]-2000:.0f}")

    tri = cov[(cov >= 0).all(axis=1)]
    T = mtri.Triangulation(x, y, tri)

    ice = thk > 1.0
    grounded = ice & (thk > (glt.RHO_O / glt.RHO_I) * np.maximum(0.0, -bed[cells])[None, None, :])
    shelf = ice & ~grounded                       # (member, year, cell)

    # ensemble mean over members, restricted to cells that are shelf at that time
    def shelf_mean(arr, t):
        v = np.where(shelf[:, t], arr[:, t], np.nan)
        return np.nanmean(v, axis=0)

    d0, d1 = shelf_mean(draft, 0), shelf_mean(draft, -1)
    g0, g1 = shelf_mean(dmg, 0), shelf_mean(dmg, -1)
    shoal = d1 - d0                               # positive = base has risen
    dgrow = g1 - g0

    everywhere = shelf.any(axis=(0, 1))
    full = np.full(x.size, np.nan)

    def onto(vals, sel=None):
        f = full.copy()
        f[cells] = vals
        if sel is not None:
            f[cells[~sel]] = np.nan
        return f

    lim = lambda v: float(np.nanpercentile(np.abs(v[np.isfinite(v)]), 98)) or 1.0
    # ice anywhere at any time, on the full mesh -- the earlier version indexed a
    # subset array with whole-mesh triangle indices, so the context never drew
    ever_ice = np.zeros(x.size, bool)
    ever_ice[cells] = ice.any(axis=(0, 1))

    panels = [
        ("draft change to year 300  (m, positive is shoaling)",
         onto(np.where(everywhere, shoal, np.nan)), "tendency",
         TwoSlopeNorm(vmin=-lim(shoal), vcenter=0.0, vmax=lim(shoal))),
        ("damage change to year 300", onto(np.where(everywhere, dgrow, np.nan)),
         "tendency", TwoSlopeNorm(vmin=-lim(dgrow), vcenter=0.0, vmax=lim(dgrow))),
    ]

    fig = plt.figure(figsize=(17.4, 6.6))
    for i, (lab, field, role, norm) in enumerate(panels):
        ax = fig.add_axes([0.010 + 0.300 * i, 0.070, 0.270, 0.840])
        T.set_mask(~ever_ice[tri].all(axis=1))
        ax.tripcolor(T, np.ones(x.size), cmap=ListedColormap(["#D6D8DA"]),
                     shading="gouraud", rasterized=True, zorder=1)
        T.set_mask(~np.isfinite(field)[tri].all(axis=1))
        tp = ax.tripcolor(T, field, cmap=oc.cmap(role, "cmocean"), norm=norm,
                          shading="gouraud", rasterized=True, zorder=2)
        cb = fig.colorbar(tp, ax=ax, fraction=0.031, pad=0.02)
        cb.ax.tick_params(labelsize=10.5); cb.outline.set_visible(False)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.0, 1.005, lab, transform=ax.transAxes, fontsize=12.5,
                color=ds.INK, ha="left", va="bottom")

    # the distribution the melt scheme reads, at both ends
    axh = fig.add_axes([0.665, 0.130, 0.315, 0.760])
    b = np.linspace(-1600, 0, 60)
    for v, lab, col in ((d0, "year 0", ds.ICE), (d1, "year 300", ds.MARSH)):
        vv = v[np.isfinite(v)]
        axh.hist(vv, bins=b, histtype="step", lw=2.4, color=col, density=True,
                 label=f"{lab}   median {np.median(vv):.0f} m")
    ds.strip(axh)
    axh.legend(loc="upper left", frameon=False, fontsize=11.5)
    axh.set_xlabel("ice draft  (m)", labelpad=7)
    axh.set_ylabel("share of shelf cells", labelpad=7)
    axh.set_yticks([])
    axh.text(0.0, 1.005, "the draft distribution the melt scheme reads",
             transform=axh.transAxes, fontsize=12.5, color=ds.INK, ha="left", va="bottom")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.12, dpi=175)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  shelf draft median {np.nanmedian(d0):.0f} m at year 0, "
          f"{np.nanmedian(d1):.0f} m at year 300")
    # fractions over cells that are actually shelf, not over the NaN padding
    fs, fd = shoal[np.isfinite(shoal)], dgrow[np.isfinite(dgrow)]
    print(f"  shoaling: median {np.median(fs):+.1f} m over {fs.size} shelf cells, "
          f"{100*np.mean(fs > 0):.0f}% rose")
    print(f"  damage:   median change {np.median(fd):+.3f}, "
          f"{100*np.mean(fd > 0):.0f}% grew more damaged")


if __name__ == "__main__":
    main()
