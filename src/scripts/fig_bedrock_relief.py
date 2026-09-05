#!/usr/bin/env python3
"""
fig_bedrock_relief.py -- the bed map with shaded relief under the hypsometric tint.

The technique behind most modern terrain cartography: a hillshade multiplied under a
colour ramp, so elevation carries the hue and slope carries the shading. Troughs read as
troughs rather than as blue patches.

Two honesty controls, because the mesh is 4-20 km and relief invents detail if pushed:

  --smooth-km    blurs the grid before shading, so the hillshade cannot resolve finer
                 than the mesh does
  --fine-only    fades the relief out where the mesh is coarse, leaving the interior
                 flat-tinted and the well-resolved margins shaded

Lighting is multidirectional rather than a single azimuth, which is softer and does not
manufacture a false ridge line along one bearing.

--compare writes the tint-only, relief, and difference versions side by side.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm, LightSource

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import fig_gl_transect as glt    # noqa: E402
import oceancolors as oc         # noqa: E402

AZIMUTHS = (315.0, 45.0, 135.0, 225.0)
WEIGHTS = (0.45, 0.25, 0.15, 0.15)      # north-west dominant, the cartographic convention


def gaussian(a, sigma_px):
    """Separable Gaussian blur that tolerates NaN."""
    if sigma_px <= 0:
        return a
    r = int(max(1, round(3 * sigma_px)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_px) ** 2)
    k /= k.sum()
    m = np.isfinite(a).astype(float)
    b = np.where(np.isfinite(a), a, 0.0)
    for ax in (0, 1):
        b = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), ax, b)
        m = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), ax, m)
    out = np.where(m > 1e-6, b / np.maximum(m, 1e-6), np.nan)
    return out


def multishade(ls_z, dx, dy, vert_exag):
    """Weighted mean of hillshades from several bearings."""
    acc = np.zeros_like(ls_z, dtype=float)
    for az, w in zip(AZIMUTHS, WEIGHTS):
        ls = LightSource(azdeg=az, altdeg=45.0)
        acc += w * ls.hillshade(ls_z, vert_exag=vert_exag, dx=dx, dy=dy)
    return np.clip(acc / sum(WEIGHTS), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-km", type=float, default=4.0)
    ap.add_argument("--smooth-km", type=float, default=8.0,
                    help="blur before shading; keeps relief no finer than the mesh")
    ap.add_argument("--vert-exag", type=float, default=45.0)
    ap.add_argument("--strength", type=float, default=0.72,
                    help="0 is tint only, 1 is full hillshade")
    ap.add_argument("--fine-only", action="store_true",
                    help="fade relief out where the mesh is coarse")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(
        glt.ROOT, "reports/dissertation/figures/relief"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ds.apply()

    import netCDF4
    d = netCDF4.Dataset(glt.MESH)
    cov = np.asarray(d["cellsOnVertex"][:]) - 1
    d.close()
    x, y = glt.rd(glt.MESH, "xCell"), glt.rd(glt.MESH, "yCell")
    bed = glt.rd(glt.MESH, "bedTopography")
    thk = glt.rd(glt.MESH, "thickness")
    spacing = glt.rd(glt.MESH, "gridSpacing")
    tri = cov[(cov >= 0).all(axis=1)]
    T = mtri.Triangulation(x, y, tri)

    # regular grid, because a hillshade needs gradients
    step = a.grid_km * 1000.0
    gx = np.arange(x.min(), x.max() + step, step)
    gy = np.arange(y.min(), y.max() + step, step)
    GX, GY = np.meshgrid(gx, gy)
    Z = mtri.LinearTriInterpolator(T, bed)(GX, GY).filled(np.nan)
    S = mtri.LinearTriInterpolator(T, spacing)(GX, GY).filled(np.nan)
    print(f"  grid {Z.shape[1]}x{Z.shape[0]} at {a.grid_km:g} km, "
          f"{100*np.isfinite(Z).mean():.0f}% inside the domain")

    Zs = gaussian(Z, a.smooth_km / a.grid_km)
    shade = multishade(np.nan_to_num(Zs, nan=0.0), step, step, a.vert_exag)

    strength = np.full_like(shade, a.strength)
    if a.fine_only:
        # full relief at 4 km, none by 15 km
        f = np.clip((15000.0 - S) / (15000.0 - 4000.0), 0.0, 1.0)
        strength = strength * np.nan_to_num(f, nan=0.0)
        print(f"  relief faded by mesh size: full on {100*np.nanmean(f>0.9):.0f}% of the grid")

    norm = TwoSlopeNorm(vmin=-2500.0, vcenter=0.0, vmax=2500.0)
    cmap = oc.cmap("topography", "cmocean")
    rgb = cmap(norm(Z))[..., :3]
    # soft-light style: shade darkens and lightens about 0.5 without crushing hue
    lit = np.clip(rgb * (0.55 + 0.9 * shade[..., None]), 0, 1)
    blended = rgb * (1 - strength[..., None]) + lit * strength[..., None]
    blended[~np.isfinite(Z)] = 1.0

    ice = thk > 1.0
    hflot = (glt.RHO_O / glt.RHO_I) * np.maximum(0.0, -bed)
    grounded = ice & (thk > hflot)
    ext = (gx[0], gx[-1], gy[0], gy[-1])

    panels = [("relief", blended)]
    if a.compare:
        flat = rgb.copy(); flat[~np.isfinite(Z)] = 1.0
        panels = [("tint only", flat), ("with shaded relief", blended)]

    fig = plt.figure(figsize=(9.0 * len(panels), 9.4))
    for i, (lab, img) in enumerate(panels):
        ax = fig.add_axes([0.02 + i / len(panels), 0.045, 0.96 / len(panels), 0.885]
                          if len(panels) > 1 else [0.02, 0.045, 0.96, 0.885])
        ax.imshow(img, origin="lower", extent=ext, interpolation="bilinear", zorder=1)
        ax.tricontour(T, grounded.astype(float), levels=[0.5], colors=[ds.INK],
                      linewidths=1.2, zorder=4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.02, 0.985, lab, transform=ax.transAxes, fontsize=15,
                color=ds.INK, ha="left", va="top")

    tag = "compare" if a.compare else ("relief_fine" if a.fine_only else "relief")
    out = f"{a.outdir}/fig_bedrock_{tag}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.10, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
