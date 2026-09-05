#!/usr/bin/env python3
"""
fig_drift_robustness.py -- is the per-basin drift sign a real pattern or ensemble noise?

fig_drift_basin_map states, in its docstring and on the slide, that the sign pattern
transfers even though the magnitude is a 10x-amplitude result. That was asserted, never
shown. Two things can be checked without another simulation:

  persistence   the same basin sign at every horizon, or a pattern that wanders
  separation    drift against the ensemble spread it was drawn from

Separation uses per-member basin totals, not per-cell statistics. Summing per-cell errors in
quadrature would treat neighbouring cells as independent and inflate the significance by
roughly the square root of the cell count; the members are the independent samples here, so
the spread across members is what the drift has to beat.

What this cannot do is test transfer to 1x-versus-deterministic; that needs the twin run.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds              # noqa: E402
import oceancolors as oc             # noqa: E402
import fig_drift_basin_map as D      # noqa: E402
import ensemble_io as eio            # noqa: E402
from fig_gating_test import basin_series, at   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=os.path.join(
        D.ROOT, "reports/dissertation/figures/slides"))
    a = ap.parse_args()
    out = a.out or f"{a.outdir}/fig_drift_robustness.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    root = eio.default_ensembles_root()
    y1, a1, n1 = basin_series(root, "SSP585", r"^SSP585_\d+$")
    y10, a10, n10 = basin_series(root, "SSP585_varScaled10x", r".*")
    print(f"  SSP585 n={n1}, SSP585_varScaled10x n={n10}")

    # evenly spaced horizons that both ensembles actually cover
    hi = min(float(y1[-1]), float(y10[-1]))
    used = [float(v) for v in np.linspace(40.0, hi, 9)]
    L = D.LETTERS[:a1.shape[2]]
    drift, se, sdv = [], [], []
    for yv in used:
        m1, s1, _ = at(y1, a1, yv)
        m10, s10, _ = at(y10, a10, yv)
        drift.append(m10 - m1)
        # members are the independent samples; this is the honest denominator
        se.append(np.sqrt(s10 ** 2 / n10 + s1 ** 2 / n1))
        sdv.append(np.sqrt(0.5 * (s10 ** 2 + s1 ** 2)))
    drift = np.array(drift).T[:len(L)]
    se = np.array(se).T[:len(L)]
    sdv = np.array(sdv).T[:len(L)]

    sign = np.sign(drift)
    persistent = (np.abs(sign.sum(axis=1)) == len(used))
    zse = np.where(se > 0, drift / se, np.nan)
    zsd = np.where(sdv > 0, drift / sdv, np.nan)

    order = np.argsort(-np.abs(drift[:, -1]))
    print(f"  horizons {used}, {len(L)} basins")
    print(f"  {'basin':7s} {'drift@end':>10s} {'|z| SE':>8s} {'|z| sd':>8s}  same sign at every horizon")
    for i in order:
        print(f"  {L[i]:7s} {drift[i,-1]:+10.2f} {abs(zse[i,-1]):8.1f} {abs(zsd[i,-1]):8.2f}  "
              f"{'yes' if persistent[i] else 'NO'}")
    n_p = int(persistent.sum())
    n_se = int((np.abs(zse[:, -1]) > 2).sum())
    n_sd = int((np.abs(zsd[:, -1]) > 2).sum())
    print(f"\n  sign persists at all {len(used)} horizons: {n_p}/{len(L)} basins")
    print(f"  |drift| > 2 x standard error at the end: {n_se}/{len(L)}")
    print(f"  |drift| > 2 x ensemble sd at the end:    {n_sd}/{len(L)}")

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(15.2, 7.2))
    ax = fig.add_axes([0.075, 0.130, 0.470, 0.760])
    axr = fig.add_axes([0.630, 0.130, 0.350, 0.760])

    yrs = np.array(used, dtype=float)
    big = np.argsort(-np.abs(drift[:, -1]))[:6]
    for i in range(len(L)):
        lead = i in big
        ax.plot(yrs, drift[i], "-", lw=2.6 if lead else 1.0,
                color=(ds.MARSH if drift[i, -1] > 0 else ds.ICE) if lead else ds.RULE,
                zorder=4 if lead else 2)
        if lead:
            ax.annotate(f"{L[i]}  {D.NAMES[L[i]]}", (yrs[-1], drift[i, -1]), fontsize=11.5,
                        color=ds.MARSH if drift[i, -1] > 0 else ds.ICE,
                        xytext=(8, 0), textcoords="offset points", va="center", zorder=5)
    ax.axhline(0, color=ds.INK, lw=1.1, zorder=3)
    ds.strip(ax)
    ax.set_xlim(yrs[0], yrs[-1] + 42)
    ax.set_xlabel("model year", labelpad=8)
    ax.set_ylabel("basin drift, VAF(10×) − VAF(1×)  (mm SLE)", labelpad=8)
    ax.text(0.0, 1.035, f"sign holds at every horizon for {n_p} of {len(L)} basins  ·  "
                        f"warm is more ice lost, cool is less",
            transform=ax.transAxes, fontsize=12.5, color=ds.INK_SOFT, ha="left", va="bottom")

    o = np.argsort(drift[:, -1])
    yy = np.arange(len(L))
    axr.barh(yy, drift[o, -1], height=.68, zorder=3, linewidth=0,
             color=[ds.MARSH if v > 0 else ds.ICE for v in drift[o, -1]])
    for k, i in enumerate(o):
        mark = "✓" if persistent[i] else "·"
        axr.text(0.02, k, mark, transform=axr.get_yaxis_transform(), fontsize=12,
                 color=ds.INK_SOFT, va="center", ha="left", zorder=5)
    axr.axvline(0, color=ds.INK, lw=1.1, zorder=4)
    axr.set_yticks(yy); axr.set_yticklabels([L[i] for i in o], fontsize=11.5)
    axr.tick_params(axis="y", length=0)
    ds.strip(axr)
    axr.set_xlabel(f"drift at year {yrs[-1]:.0f}  (mm SLE)", labelpad=8)
    axr.text(0.0, 1.035, f"✓ marks a sign that never flips  ·  "
                         f"{n_sd} of {len(L)} exceed twice the ensemble spread",
             transform=axr.transAxes, fontsize=12, color=ds.INK_SOFT, ha="left", va="bottom")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
