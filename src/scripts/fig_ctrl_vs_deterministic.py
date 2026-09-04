#!/usr/bin/env python3
"""
fig_ctrl_vs_deterministic.py -- does time variation in the applied melt displace the mean?

DET-CTRL is integrated with the background melt parameterisation alone: no seasonal cycle
and no generated variability, so its forcing is B while every control member receives
B + S + V_i. The difference

    D_SV(t) = mean(CTRL)(t) - DET-CTRL(t)

is therefore the COMBINED effect of seasonality and variability at realistic amplitude, not
the effect of the stochastic component alone. Isolating the stochastic term needs the
amplitude experiment, in which the seasonal cycle is identical between the two ensembles
being compared.

The band is two standard errors of the ensemble mean, sigma/sqrt(n). The difference is
positive at every year but passes inside the band for part of the integration, which the
figure shows rather than hides.

Usage
    python3 fig_ctrl_vs_deterministic.py --no-member-counts
"""
from __future__ import annotations
import os, glob, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.0,
    "savefig.dpi": 300,
})

RHO_I, RHO_O, A_O = 910.0, 1028.0, 3.625e14
ROOT = "data/MALI/diagnostics/ENSEMBLES"


def sle(path):
    d = netCDF4.Dataset(path)
    yr = np.asarray(d["daysSinceStart"][:], float) / 365.0
    v = np.asarray(d["volumeAboveFloatation"][:], float)
    d.close()
    k = np.isfinite(yr) & np.isfinite(v) & (v > 0)
    return yr[k], -(v[k] - v[k][0]) * (RHO_I / RHO_O) / A_O * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", default="CTRL")
    # DETERMINISTIC/DET-CTRL/output is the 308-year run behind the reported numbers.
    # DET-CTRL/DET-CTRL is a longer 522-year extension; the two agree over the overlap.
    ap.add_argument("--deterministic", default="DETERMINISTIC/DET-CTRL/output")
    ap.add_argument("--no-member-counts", action="store_true",
                    help="omit '(n=N)' from legend labels (publication style)")
    ap.add_argument("--out-dir", default="reports/dissertation/figures/tierA")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    g = np.arange(1.0, 301.0)

    M = []
    for f in sorted(glob.glob(f"{ROOT}/{a.ensemble}/{a.ensemble}_[0-9][0-9]/globalStats.nc")):
        yr, s = sle(f)
        if yr.size > 50 and yr[0] < 5:
            M.append(np.interp(g, yr, s, left=np.nan, right=np.nan))
    M = np.array(M)
    n = int(np.median(np.sum(np.isfinite(M), axis=0)))
    mu = np.nanmean(M, 0)
    sd = np.nanstd(M, 0, ddof=1)
    se = sd / np.sqrt(np.maximum(np.sum(np.isfinite(M), axis=0), 1))

    dpath = None
    for cand in (f"{ROOT}/{a.deterministic}/globalStats.nc",
                 f"{ROOT}/{a.deterministic}/{os.path.basename(a.deterministic)}/globalStats.nc"):
        if os.path.exists(cand):
            dpath = cand; break
    if dpath is None:
        raise SystemExit(f"deterministic run not found under {ROOT}/{a.deterministic}")
    yd, sd_ = sle(dpath)
    det = np.interp(g, yd, sd_, left=np.nan, right=np.nan)
    diff = mu - det

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    lbl = "CTRL ensemble mean" if a.no_member_counts else f"CTRL ensemble mean (n={n})"
    for m in M:
        axA.plot(g, m, color="#BBBBBB", lw=.6, zorder=1)
    axA.fill_between(g, mu - sd, mu + sd, color="#0072B2", alpha=.20, lw=0, zorder=2,
                     label="$\\pm 1\\sigma$")
    axA.plot(g, mu, color="#0072B2", lw=2.2, zorder=3, label=lbl)
    axA.plot(g, det, color="#C1121F", lw=2.0, ls="--", zorder=4,
             label="DET-CTRL ($B$ only)")
    axA.set_ylabel("sea-level contribution (mm SLE)\n[negative = ice gain]")
    axA.text(0.01, 0.98, "(a)", transform=axA.transAxes, ha="left", va="top",
             fontsize=15, fontweight="bold")
    axA.legend()

    axB.fill_between(g, diff - 2 * se, diff + 2 * se, color="#999999", alpha=.35, lw=0,
                     label="$\\pm 2$ standard errors of the mean")
    axB.plot(g, diff, color="#222222", lw=2.0)
    axB.axhline(0, color="k", lw=.7, ls=":")
    # contiguous intervals where the band reaches zero, not merely the first and last
    # such year: an isolated crossing is not the same as a sustained unresolved window
    unres = (diff - 2 * se <= 0) & np.isfinite(diff)
    runs, i = [], 0
    while i < unres.size:
        if unres[i]:
            j = i
            while j + 1 < unres.size and unres[j + 1]:
                j += 1
            if g[j] - g[i] >= 5:
                runs.append((g[i], g[j]))
            i = j + 1
        else:
            i += 1
    for lo, hi in runs:
        axB.axvspan(lo, hi, color="#C1121F", alpha=.07, lw=0, zorder=0)
        print(f"  within 2 SE of zero over yr {lo:.0f}-{hi:.0f}")
    axB.set_ylabel("$D_{SV}$ = mean(CTRL) $-$ DET  (mm)")
    axB.text(0.98, 0.98, "(b)", transform=axB.transAxes, ha="right", va="top",
             fontsize=15, fontweight="bold")
    axB.legend(loc="upper left")

    for ax in (axA, axB):
        ax.set_xlabel("model year"); ax.grid(alpha=.3); ax.set_xlim(0, 300)
    for y in (25, 100, 300):
        i = int(y - 1)
        print(f"  yr {y:3d}: diff {diff[i]:+.2f} mm  = {diff[i]/se[i]:.1f} SE")

    fig.tight_layout()
    o = f"{a.out_dir}/F22_ctrl_vs_deterministic.png"
    fig.savefig(o, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(o)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", o)


if __name__ == "__main__":
    main()
