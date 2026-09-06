#!/usr/bin/env python3
"""
fig_mesh_fields.py -- survey maps of the MALI mesh input fields.

One map per field, drawn on the native triangulation from cellsOnVertex. Fields are
converted to the units they would be quoted in, and each is masked to the domain where
it means anything: melt to the shelves, friction to grounded ice, and so on.

Tier 1 fields carry the argument, tier 2 answer likely questions, tier 3 are inversion
products and boundary conditions kept for the appendix. --contact writes a single sheet.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import (LinearSegmentedColormap, TwoSlopeNorm, LogNorm,
                               Normalize, BoundaryNorm, ListedColormap)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds          # noqa: E402
import fig_gl_transect as glt    # noqa: E402
import oceancolors as oc         # noqa: E402

SEC_PER_YEAR = 3.15576e7
RHO_ICE = 910.0

BEDCOL = LinearSegmentedColormap.from_list("bed", [
    (0.00, "#0B2545"), (0.16, "#17456E"), (0.32, "#2E76A8"),
    (0.44, "#7FB2D4"), (0.50, "#EDE7D8"), (0.62, "#CBB68B"),
    (0.80, "#9C8A63"), (1.00, "#6B5F49")])

# name, tier, title, unit label, domain, how to build the field, colour spec
FIELDS = [
    # ---------------------------------------------------------------- tier 1
    ("surface_speed", 1, "Observed surface speed", "m yr$^{-1}$", "ice",
     lambda R: np.hypot(R("observedSurfaceVelocityX"),
                        R("observedSurfaceVelocityY")) * SEC_PER_YEAR,
     dict(role="speed", norm=LogNorm(1.0, 4000.0))),

    ("basal_melt", 1, "Mean basal melt applied under the shelves", "m yr$^{-1}$ ice", "shelf",
     lambda R: -R("floatingBasalMassBal") * SEC_PER_YEAR / RHO_ICE,
     dict(role="melt", pct=(1, 99), diverging=True)),

    ("thickness_tendency", 1, "Observed thickness change", "m yr$^{-1}$", "ice",
     lambda R: R("observedThicknessTendency") * SEC_PER_YEAR,
     dict(role="tendency", norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0))),

    # ---------------------------------------------------------------- tier 2
    ("bed_edit", 2, "Bed as used, minus bed as delivered", "m", "all",
     lambda R: R("bedTopography") - R("bedTopographyOriginal"),
     dict(role="difference", pct=(1, 99), diverging=True)),

    ("draft_paramtype", 2, "Where melt is draft-dependent rather than constant",
     "0 draft-dependent / 1 constant", "shelf",
     lambda R: R("draftDepenBasalMelt_paramType"),
     dict(cmap=ListedColormap(["#D8D2C4", ds.ICE]), norm=BoundaryNorm([-.5, .5, 1.5], 2))),

    ("draft_alpha0", 2, "Draft-dependence intercept", "m yr$^{-1}$ ice", "shelf",
     lambda R: -R("draftDepenBasalMeltAlpha0") * SEC_PER_YEAR / RHO_ICE,
     dict(role="melt", pct=(2, 98), diverging=True)),

    ("draft_alpha1", 2, "Draft-dependence slope", "m yr$^{-1}$ ice per m draft", "shelf",
     lambda R: -R("draftDepenBasalMeltAlpha1") * SEC_PER_YEAR / RHO_ICE,
     dict(role="melt", pct=(2, 98), diverging=True)),

    ("draft_mindraft", 2, "Minimum draft in the melt parameterisation", "m", "shelf",
     lambda R: R("draftDepenBasalMelt_minDraft"),
     dict(role="depth", norm=Normalize(-900.0, 0.0))),

    ("smb", 2, "Surface mass balance", "m yr$^{-1}$ ice", "ice",
     lambda R: R("sfcMassBal") * SEC_PER_YEAR / RHO_ICE,
     dict(role="precip", norm=Normalize(0.0, 2.0))),

    # ---------------------------------------------------------------- tier 3
    ("mu_friction", 3, "Basal friction coefficient from the inversion", "Pa yr m$^{-1}$",
     "grounded", lambda R: R("muFriction"),
     dict(role="friction", norm=LogNorm(0.1, 2000.0))),

    ("stiffness", 3, "Ice stiffness factor from the inversion", "dimensionless", "ice",
     lambda R: R("stiffnessFactor"),
     dict(role="ratio", pct=(2, 98), diverging=True, center=1.0)),

    ("thickness", 3, "Ice thickness", "m", "ice",
     lambda R: R("thickness"), dict(role="thickness", norm=Normalize(0.0, 4000.0))),

    ("thickness_uncertainty", 3, "Thickness uncertainty", "m", "ice",
     lambda R: R("thicknessUncertainty"), dict(role="magnitude", pct=(2, 98))),

    ("velocity_uncertainty", 3, "Surface velocity uncertainty", "m yr$^{-1}$", "ice",
     lambda R: np.where(R("observedSurfaceVelocityUncertainty") >= 0.99,
                        np.nan, R("observedSurfaceVelocityUncertainty") * SEC_PER_YEAR),
     dict(role="magnitude", pct=(2, 98))),

    ("basal_heat_flux", 3, "Geothermal heat flux", "W m$^{-2}$", "all",
     lambda R: R("basalHeatFlux"), dict(role="heat", norm=Normalize(0.04, 0.15))),

    ("surface_air_temperature", 3, "Surface air temperature", "$^\\circ$C", "all",
     lambda R: R("surfaceAirTemperature") - 273.15,
     dict(role="temperature", norm=Normalize(-55.0, 0.0))),

    ("grid_spacing", 3, "Mesh resolution", "km", "ice",
     lambda R: R("gridSpacing") / 1000.0, dict(role="resolution", norm=Normalize(3.0, 23.0))),

    ("bed", 3, "Bed elevation", "m", "all",
     lambda R: R("bedTopography"),
     dict(role="topography", norm=TwoSlopeNorm(vmin=-2500.0, vcenter=0.0, vmax=2500.0))),
]


def load():
    import netCDF4
    d = netCDF4.Dataset(glt.MESH)
    cov = np.asarray(d["cellsOnVertex"][:]) - 1
    d.close()
    x, y = glt.rd(glt.MESH, "xCell"), glt.rd(glt.MESH, "yCell")
    tri = cov[(cov >= 0).all(axis=1)]
    T = mtri.Triangulation(x, y, tri)

    R = lambda v: glt.rd(glt.MESH, v).astype(float)
    thk, bed = R("thickness"), R("bedTopography")
    ice = thk > 1.0
    grounded = ice & (thk > (glt.RHO_O / glt.RHO_I) * np.maximum(0.0, -bed))
    dom = {"all": np.ones_like(ice), "ice": ice,
           "grounded": grounded, "shelf": ice & ~grounded}
    return T, tri, R, dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tier", type=int, nargs="*", default=None)
    ap.add_argument("--contact", action="store_true")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--palette", default=None, choices=["legacy", "cmocean"])
    ap.add_argument("--outdir", default=os.path.join(
        glt.ROOT, "reports/dissertation/figures/mesh"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ds.apply()

    T, tri, R, dom = load()
    made = []

    for name, tier, title, unit, domain, build, col in FIELDS:
        if a.only and name not in a.only:
            continue
        if a.tier and tier not in a.tier:
            continue
        f = build(R)
        m = dom[domain].astype(bool)
        col = dict(col)
        if "role" in col:
            col["cmap"] = oc.cmap(col.pop("role"), a.palette)
        pct, diverging, center = col.pop("pct", None), col.pop("diverging", False), col.pop("center", 0.0)
        if pct is not None:
            v = f[m & np.isfinite(f)]
            lo, hi = np.percentile(v, pct[0]), np.percentile(v, pct[1])
            if diverging:
                r = max(abs(lo - center), abs(hi - center))
                col["norm"] = TwoSlopeNorm(vmin=center - r, vcenter=center, vmax=center + r)
            else:
                col["norm"] = Normalize(lo, hi)
        fig = plt.figure(figsize=(8.6, 8.0))
        ax = fig.add_axes([0.02, 0.055, 0.96, 0.865])
        # the shelves are a thin rim at this scale, so the rest of the ice goes
        # underneath in grey -- otherwise a shelf-only field reads as a blank page
        if domain in ("shelf", "grounded"):
            T.set_mask(~dom["ice"][tri].all(axis=1))
            ax.tripcolor(T, np.ones_like(f), cmap=ListedColormap(["#ECECE8"]),
                         shading="gouraud", rasterized=True, zorder=1)
        T.set_mask(~m[tri].all(axis=1))
        tp = ax.tripcolor(T, f, shading="gouraud", rasterized=True, zorder=2, **col)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        cax = fig.add_axes([0.10, 0.085, 0.26, 0.019])
        cb = fig.colorbar(tp, cax=cax, orientation="horizontal")
        cb.set_label(unit, fontsize=10, labelpad=5)
        cb.ax.tick_params(length=3, labelsize=9)
        cb.outline.set_visible(False)

        v = f[m & np.isfinite(f)]
        fig.text(0.02, 0.985, title, fontsize=14, color=ds.INK, ha="left", va="top")
        fig.text(0.02, 0.950,
                 f"tier {tier} · {domain} · median {np.median(v):.4g}, "
                 f"range {v.min():.4g} to {v.max():.4g}",
                 fontsize=10.5, color=ds.INK_SOFT, ha="left", va="top")

        dst = f"{a.outdir}/mesh_{name}.png"
        fig.savefig(dst, bbox_inches="tight", pad_inches=0.09, dpi=a.dpi)
        plt.close(fig)
        made.append((name, tier, dst))
        print(f"  tier{tier}  {name:26s} median {np.median(v):11.4g}  "
              f"[{v.min():.4g}, {v.max():.4g}]")

    if a.contact and made:
        from PIL import Image
        thumbs = []
        for name, tier, dst in made:
            im = Image.open(dst); im.thumbnail((560, 560))
            thumbs.append((f"T{tier}  {name}", im))
        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        cw = max(t.size[0] for _, t in thumbs) + 20
        ch = max(t.size[1] for _, t in thumbs) + 40
        sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
        from PIL import ImageDraw
        d = ImageDraw.Draw(sheet)
        for i, (lab, t) in enumerate(thumbs):
            x0, y0 = (i % cols) * cw + 10, (i // cols) * ch + 30
            sheet.paste(t, (x0, y0))
            d.text((x0, y0 - 20), lab, fill="#0B2545")
        out = f"{a.outdir}/mesh_contact_sheet.png"
        sheet.save(out)
        print(f"\nwrote {out}")

    print(f"\n{len(made)} maps -> {a.outdir}")


if __name__ == "__main__":
    main()
