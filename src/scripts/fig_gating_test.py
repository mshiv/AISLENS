#!/usr/bin/env python3
"""
fig_gating_test.py -- does the spread need a warm cavity and a moving grounding line?

The slide claims variability reaches grounded ice only where the cavity is already warm
and the grounding line is already retreating. F8 never tested that: it regressed per-basin
spread on the forcing low-frequency fraction, found nothing, and left the mechanism asserted.

This tests the sentence directly. For each ISMIP6 basin:

  warmth   mean basal melt applied under that basin's shelves, from the mesh
  motion   the basin's own mean dVAF over the ensemble -- how far it actually moves
  spread   sigma of dVAF across members, the quantity to be explained

If the conditional holds, spread should be large only where warmth and motion are both
large, and small wherever either is small. The lower panels show each axis separately;
the main panel shows them together.

Correlations are on ranks as well as values, because sixteen basins with a couple of
outliers will not support a Pearson coefficient on its own.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds              # noqa: E402
import oceancolors as oc             # noqa: E402
import ensemble_io as eio            # noqa: E402
import fig_gl_transect as glt        # noqa: E402
from connect_forcing_response import load_forcing_lowfreq  # noqa: E402

SEC_PER_YEAR = 3.15576e7
RHO_ICE = 910.0


def basin_series(root, ensemble, include):
    """(years, member x year x basin) of dVAF in mm SLE, loaded once."""
    ens_dir = os.path.join(root, ensemble)
    stacks, nmin = [], None
    for name, path in eio.discover_members(ens_dir, stats_filename="regionalStats.nc",
                                           include=include):
        try:
            d = eio.to_year_dim(eio.load_member_regionalstats(path))
        except Exception:
            continue
        if "regionalVolumeAboveFloatation" not in d:
            continue
        vaf = d["regionalVolumeAboveFloatation"]
        yr = d["year"].values
        if yr[0] > 5.0 or len(yr) < 10:
            continue
        sle = np.column_stack([eio.vaf_to_sle_mm(vaf.isel(nRegions=r).values, reference="first")
                               for r in range(vaf.sizes["nRegions"])])
        stacks.append((yr, sle))
        nmin = len(yr) if nmin is None else min(nmin, len(yr))
    if len(stacks) < 3:
        sys.exit("fewer than three usable members")
    years = stacks[0][0][:nmin]
    arr = np.stack([s[:nmin] for _, s in stacks], axis=0)
    return years, arr, len(stacks)


def at(years, arr, horizon):
    i = int(np.argmin(np.abs(years - horizon)))
    return (np.nanmean(arr[:, i, :], axis=0), np.nanstd(arr[:, i, :], axis=0, ddof=1),
            float(years[i]))


def basin_melt(mesh, mask):
    """Mean applied basal melt (m/yr ice) over each basin's floating cells."""
    dm = xr.open_dataset(mesh, decode_times=False)
    get = lambda v: np.asarray(dm[v].values).squeeze()
    bmb = -get("floatingBasalMassBal") * SEC_PER_YEAR / RHO_ICE
    thk, bed = get("thickness"), get("bedTopography")
    ice = thk > 1.0
    floating = ice & ~(thk > (glt.RHO_O / glt.RHO_I) * np.maximum(0.0, -bed))
    masks = np.asarray(xr.open_dataset(mask, decode_times=False)["regionCellMasks"].values)
    if masks.shape[0] != thk.size and masks.shape[1] == thk.size:
        masks = masks.T
    out = []
    for r in range(masks.shape[1]):
        sel = (masks[:, r] > 0) & floating
        out.append(float(np.nanmean(bmb[sel])) if sel.sum() > 20 else np.nan)
    return np.array(out)


def mesh_context(mesh, mask):
    """Triangulation, basin index per cell, and an in-any-basin flag."""
    import netCDF4
    d = netCDF4.Dataset(mesh)
    cov = np.asarray(d["cellsOnVertex"][:]) - 1
    x = np.asarray(d["xCell"][:]).ravel()
    y = np.asarray(d["yCell"][:]).ravel()
    d.close()
    tri = cov[(cov >= 0).all(axis=1)]
    masks = np.asarray(xr.open_dataset(mask, decode_times=False)["regionCellMasks"].values)
    if masks.shape[0] != x.size and masks.shape[1] == x.size:
        masks = masks.T
    return mtri.Triangulation(x, y, tri), tri, np.argmax(masks, axis=1), masks.sum(axis=1) > 0


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan, int(ok.sum())
    pr, pp = pearsonr(x[ok], y[ok])
    sr, sp = spearmanr(x[ok], y[ok])
    return pr, pp, sr, sp, int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--members", default=r"^SSP585_\d+$")
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--forcing-csv", default="reports/spectrum_percell_generated0.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=os.path.join(
        glt.ROOT, "reports/dissertation/figures/slides"))
    a = ap.parse_args()
    out = a.out or f"{a.outdir}/fig_gating_test.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    names, _, lf = load_forcing_lowfreq(a.forcing_csv)
    years, arr, n = basin_series(a.root, a.ensemble, a.members)
    mean, sig, hy = at(years, arr, a.horizon)
    melt = basin_melt(os.path.join(glt.ROOT, "data/MALI", os.path.basename(glt.MESH)),
                      glt.BASIN_MASK)
    motion = np.abs(mean)

    k = min(len(names), len(sig), len(melt))
    names, sig, melt, motion, lf = names[:k], sig[:k], melt[:k], motion[:k], lf[:k]

    print(f"{a.ensemble}: {n} members, year {hy:.0f}, {k} basins")
    print(f"  {'basin':8s} {'melt m/yr':>10s} {'|mean| mm':>10s} {'sigma mm':>9s}")
    for i in np.argsort(-sig):
        print(f"  {names[i]:8s} {melt[i]:10.2f} {motion[i]:10.1f} {sig[i]:9.3f}")

    tests = [("spread vs cavity warmth", melt, sig),
             ("spread vs basin motion", motion, sig),
             ("spread vs forcing low-freq (the old F8 test)", lf * 100, sig)]
    print()
    stats = {}
    for lab, xv, yv in tests:
        pr, pp, sr, sp, nn = corr(xv, yv)
        stats[lab] = (pr, pp, sr, sp, nn)
        print(f"  {lab:46s} Pearson {pr:+.2f} (p={pp:.3f})   Spearman {sr:+.2f} (p={sp:.3f})   n={nn}")

    # sweep the horizons: which predictor explains the spread, and when
    hs = np.arange(20.0, float(years[-1]) + 1e-6, 10.0)
    rows = []
    for h in hs:
        m_, s_, y_ = at(years, arr, h)
        w = corr(melt, s_[:k])[2:4]
        mo = corr(np.abs(m_[:k]), s_[:k])[2:4]
        fr = corr(lf * 100, s_[:k])[2:4]
        rows.append((y_, w[0], w[1], mo[0], mo[1], fr[0], fr[1]))
    R = np.array(rows)
    print("\n  horizon sweep, Spearman rho (p):")
    for r in R[::4]:
        print(f"   yr {r[0]:5.0f}   warmth {r[1]:+.2f} (p={r[2]:.3f})   "
              f"motion {r[3]:+.2f} (p={r[4]:.3f})   forcing {r[5]:+.2f} (p={r[6]:.3f})")

    # ---------------------------------------------------------------- figure
    T, tri, cell_basin, in_any = mesh_context(
        os.path.join(glt.ROOT, "data/MALI", os.path.basename(glt.MESH)), glt.BASIN_MASK)

    fig = plt.figure(figsize=(19.4, 6.4))
    axm = fig.add_axes([0.012, 0.075, 0.290, 0.830])
    ax = fig.add_axes([0.372, 0.135, 0.260, 0.755])
    axr = fig.add_axes([0.735, 0.135, 0.250, 0.755])

    # panel A -- where the spread actually is, the half of F8 worth keeping
    cell_sig = np.where(in_any, sig[np.clip(cell_basin, 0, len(sig) - 1)], np.nan)
    T.set_mask(~in_any[tri].all(axis=1))
    tp = axm.tripcolor(T, cell_sig, shading="gouraud", rasterized=True,
                       cmap=oc.cmap("magnitude", "cmocean"),
                       vmin=0, vmax=float(np.nanpercentile(sig, 97)))
    cbm = fig.colorbar(tp, ax=axm, fraction=0.036, pad=0.01)
    cbm.set_label(f"σ ΔVAF at year {hy:.0f}  (mm)", fontsize=12)
    cbm.outline.set_visible(False)
    for tag in ("G-H", "J-K", "A-Ap"):
        if tag in names:
            sel = (cell_basin == names.index(tag)) & in_any
            if sel.any():
                axm.text(T.x[sel].mean(), T.y[sel].mean(), tag, fontsize=12.5, color="white",
                         ha="center", va="center", zorder=6,
                         bbox=dict(boxstyle="round,pad=0.18", fc=ds.INK, ec="none", alpha=.72))
    axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(False)
    axm.text(0.0, 1.005, "where the spread is", transform=axm.transAxes,
             fontsize=12.5, color=ds.INK, ha="left", va="bottom")

    ok = np.isfinite(melt) & np.isfinite(sig) & np.isfinite(motion)
    sc = ax.scatter(motion[ok], sig[ok], s=190, c=melt[ok],
                    cmap=oc.cmap("temperature", "cmocean"), edgecolor=ds.INK,
                    linewidth=0.9, zorder=3)
    for i2 in np.flatnonzero(ok):
        ax.annotate(names[i2], (motion[i2], sig[i2]), fontsize=11.5, color=ds.INK,
                    xytext=(9, 3), textcoords="offset points", zorder=4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.040, pad=0.015)
    cb.set_label("mean basal melt under the basin  (m yr$^{-1}$)", fontsize=12)
    cb.outline.set_visible(False)
    ds.strip(ax)
    ax.set_xscale("symlog", linthresh=1.0); ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel(f"how far the basin itself moves, |mean ΔVAF| at year {hy:.0f}  (mm)", labelpad=8)
    ax.set_ylabel(f"spread across realizations, σ ΔVAF  (mm)", labelpad=8)
    _, _, sr, sp, _ = stats["spread vs basin motion"]
    ax.text(0.0, 1.035, f"what explains it — Spearman {sr:+.2f}, p = {sp:.4f}",
            transform=ax.transAxes, fontsize=12.5, color=ds.INK, ha="left", va="bottom")

    axr.axhspan(-1, 1, color=ds.FIELD, alpha=0.0)
    axr.axhline(0, color=ds.RULE, lw=1.0, zorder=2)
    for col, lab, c, style in ((3, "how far the basin is already moving", ds.MARSH, "-"),
                               (1, "how warm its cavity is", ds.ICE, "-"),
                               (5, "forcing low-frequency fraction (the old test)",
                                ds.INK_SOFT, "--")):
        axr.plot(R[:, 0], R[:, col], style, color=c, lw=2.6, zorder=4)
        sig_pts = R[R[:, col + 1] < 0.05]
        if len(sig_pts):
            axr.plot(sig_pts[:, 0], sig_pts[:, col], "o", ms=5.5, color=c, zorder=5)
        never = "" if len(sig_pts) else "   (never significant)"
        axr.text(0.015, 0.965 - 0.072 * [3, 1, 5].index(col), lab + never,
                 transform=axr.transAxes, fontsize=12, color=c, ha="left", va="top")
    ds.strip(axr)
    axr.set_xlim(R[0, 0], R[-1, 0]); axr.set_ylim(-0.55, 1.0)
    axr.set_xlabel("horizon  (model year)", labelpad=8)
    axr.set_ylabel("rank correlation with the spread  (Spearman ρ)", labelpad=8)
    axr.text(0.0, 1.035, "and when · filled markers are p < 0.05",
             transform=axr.transAxes, fontsize=12, color=ds.INK_SOFT, ha="left", va="bottom")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
