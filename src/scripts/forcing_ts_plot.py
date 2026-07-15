#!/usr/bin/env python3
"""
forcing_ts_plot.py — overlay the nCells-reduced forcing time series made by forcing_ts_extract.sh.

Reads <root>/<label>/*.nc, plots VAR(Time) against a real year axis (from xtime if present, else index/12),
one colour per ensemble folder, one faint line per sampled member, plus the per-folder member-mean in bold.
Prints a quick magnitude table (mean / std / min / max) per folder.

Usage:
  python forcing_ts_plot.py --root <out-root>/ts [--var floatingBasalMassBalAdjustment] [--annual] [--out fig.png]
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset, chartostring


def load(path, var):
    d = Dataset(path)
    y = np.asarray(d.variables[var][:], dtype=float).squeeze()
    if "xtime" in d.variables:
        xt = [str(s).strip() for s in chartostring(d.variables["xtime"][:])]
        yr = np.array([int(s[:4]) + (int(s[5:7]) - 1) / 12.0 for s in xt])
    else:
        yr = np.arange(y.size) / 12.0
    d.close()
    return yr, y


def to_annual(yr, y):
    k = (y.size // 12) * 12
    if k < 12:
        return yr, y
    return yr[:k].reshape(-1, 12).mean(1), y[:k].reshape(-1, 12).mean(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--var", default="floatingBasalMassBalAdjustment")
    ap.add_argument("--annual", action="store_true", help="12-month-mean each series (drops the seasonal cycle)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    labels = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))
    if not labels:
        raise SystemExit(f"no <label>/ subdirs under {a.root}")
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(11, 5))

    print(f"{'folder':45s} {'n':>2s} {'mean':>11s} {'std':>11s} {'min':>11s} {'max':>11s}")
    for i, lab in enumerate(labels):
        c = cmap(i % 10)
        files = sorted(glob.glob(os.path.join(a.root, lab, "*.nc")))
        if not files:
            continue
        series, allvals, ref_yr = [], [], None
        for j, f in enumerate(files):
            yr, y = load(f, a.var)
            if a.annual:
                yr, y = to_annual(yr, y)
            ax.plot(yr, y, color=c, alpha=0.35, lw=0.8, label=lab if j == 0 else None)
            series.append(y); allvals.append(y); ref_yr = yr
        # bold member-mean where lengths match
        try:
            M = np.vstack([s for s in series if s.size == series[0].size])
            ax.plot(ref_yr[:M.shape[1]], M.mean(0), color=c, lw=2.2)
        except Exception:
            pass
        v = np.concatenate(allvals)
        print(f"{lab:45s} {len(files):2d} {np.nanmean(v):11.3e} {np.nanstd(v):11.3e} "
              f"{np.nanmin(v):11.3e} {np.nanmax(v):11.3e}")

    ax.set_xlabel("year"); ax.set_ylabel(f"{a.var}  (nCells-reduced{', annual' if a.annual else ''})")
    ax.set_title("Forcing time series — ensemble comparison")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    out = a.out or os.path.join(a.root, "forcing_ts_compare.png")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
