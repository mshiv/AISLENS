#!/usr/bin/env python3
"""
fig_ocean_variability.py -- what the generator resamples, and at which periods.

Left: the domain-mean melt anomaly of the SORRM variability component, F_v. Right: the share
of its variance in each frequency band, for the domain and for four sectors.

Reads the products of hpc_sorrm_variability.py under reports/. The raw component pair is used;
the extrapolated pair fills the interior and dilutes the domain aggregate.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BANDS = [("seasonal", "< 1.5 yr"), ("interannual", "1.5-10 yr"),
         ("decadal", "10-30 yr"), ("multidecadal", "> 30 yr")]
SHADE = [ds.MARSH_TINT, ds.MARSH, ds.ICE, ds.INK]
ORDER = ["Amundsen", "Weddell", "Ross", "East", "domain"]


def read_bands(path):
    rows, ratio = {}, None
    for line in open(path).read().splitlines():
        if line.startswith("# seasonal_fraction_var_ratio"):
            ratio = float(line.split(",")[1]); continue
        c = line.split(",")
        if c[0] in ("sector", ""):
            continue
        rows[c[0]] = (int(c[1]), np.array([float(x) for x in c[2:6]]))
    return rows, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="raw", choices=["raw", "extrap"])
    ap.add_argument("--window-years", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"{ROOT}/reports/dissertation/figures/slides/fig_ocean_variability.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.apply()

    rows, ratio = read_bands(f"{ROOT}/reports/sorrm_{a.tag}_bands.csv")
    z = np.load(f"{ROOT}/reports/sorrm_{a.tag}_series.npz")
    dom, dt = z["domain"], float(z["dt_years"])
    t = np.arange(dom.size) * dt

    fig = plt.figure(figsize=(15.0, 5.8))
    ax = fig.add_axes([0.052, 0.150, 0.445, 0.740])
    axb = fig.add_axes([0.590, 0.150, 0.310, 0.740])

    n = int(a.window_years / dt)
    s = dom[:n] / np.std(dom)
    ax.axhline(0, color=ds.RULE, lw=.9, zorder=2)
    ax.plot(t[:n], s, color=ds.ICE, lw=0.9, zorder=3)
    k = max(1, int(10.0 / dt))                       # 10-year running mean
    sm = np.convolve(s, np.ones(k) / k, "same")
    ax.plot(t[:n], sm, color=ds.INK, lw=2.4, zorder=4)
    ds.strip(ax)
    ax.set_xlim(0, a.window_years)
    ax.set_xlabel("years of the SORRM record", labelpad=7)
    ax.set_ylabel("melt anomaly  (standard deviations)", labelpad=7)
    ax.tick_params(length=3)
    ax.text(0.0, 1.04, "domain-mean variability component · thick line is a 10-year mean",
            transform=ax.transAxes, fontsize=11, color=ds.INK_SOFT, ha="left", va="bottom")

    names = [k for k in ORDER if k in rows]
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for bi, ((bnm, per), col) in enumerate(zip(BANDS, SHADE)):
        w = np.array([rows[k][1][bi] for k in names]) * 100
        axb.barh(y, w, left=left, height=.62, color=col, linewidth=0, zorder=3)
        for yi, (l, ww) in enumerate(zip(left, w)):
            if ww > 7:
                axb.text(l + ww / 2, yi, f"{ww:.0f}", ha="center", va="center",
                         fontsize=10, color="white" if bi == 3 else ds.INK, zorder=4)
        left += w
        axb.text(0.0, 1.10 - 0.055 * bi, f"{bnm}  {per}", transform=axb.transAxes,
                 fontsize=10, color=col, ha="left", va="bottom")

    axb.set_yticks(y)
    axb.set_yticklabels([f"{k}" if k != "domain" else "all shelves" for k in names],
                        fontsize=11.5)
    axb.tick_params(axis="y", length=0)
    axb.tick_params(axis="x", length=3)
    ds.strip(axb, keep=("bottom",))
    axb.set_xlim(0, 100)
    axb.set_xlabel("share of F$_v$ variance  (%)", labelpad=7)
    axb.set_ylim(-0.6, len(names) - 0.4)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  record {dom.size * dt:.0f} years at {dt*12:.2f} months")
    for k in names:
        nc, v = rows[k]
        print(f"  {k:9s} n={nc:7,}  " +
              "  ".join(f"{b[0][:4]} {100*x:5.1f}%" for b, x in zip(BANDS, v)))
    if ratio:
        print(f"  var(F_s)/(var(F_s)+var(F_v)) = {100*ratio:.1f}%")


if __name__ == "__main__":
    main()
