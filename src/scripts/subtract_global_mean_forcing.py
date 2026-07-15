#!/usr/bin/env python3
"""
subtract_global_mean_forcing.py — center a MALI-grid forcing realization by subtracting ONE global scalar
(the mean over Time x nCells) from the field, in place or to a new file.

WHY: the 1000-yr realizations came out with a constant positive DC offset (~1.2e-5 in the nCells-reduced
series) that the 300-yr ensemble doesn't have -- the meanAdjust (notebook 06's single-scalar subtraction)
either didn't take or the generator used the un-adjusted variability. The variability std is identical,
so it's purely a missing constant. This reproduces notebook 06 exactly: it subtracted the global scalar
from filled(0) cells too, so subtracting it from the whole realization (shelf + non-shelf) matches the
300-yr construction. Do this to the base realizations BEFORE adding xtime / scenario trend.

By default computes each file's OWN global scalar and subtracts it (each file ends up centered at 0). Pass
--scalar VALUE to subtract a fixed shared scalar instead (e.g. to reuse one member's value for all).

Memory-safe: two chunked passes; writes HIGH-INDEX-FIRST so an unlimited Time dim is never truncated.

Usage:
  # in place on a whole dir (array-friendly with --file):
  python subtract_global_mean_forcing.py --dir <dir> --in-place
  # one file to a new location:
  python subtract_global_mean_forcing.py --file in.nc --out out.nc
"""
from __future__ import annotations
import argparse, glob, os, shutil, sys
import numpy as np
from netCDF4 import Dataset


def global_scalar(v, chunk):
    """nan-aware mean over the whole (Time, ...) variable, chunked over Time."""
    Nf = v.shape[0]
    tot = 0.0; cnt = 0
    for t0 in range(0, Nf, chunk):
        t1 = min(t0 + chunk, Nf)
        b = np.asarray(v[t0:t1], dtype=float)
        m = np.isfinite(b)
        tot += float(b[m].sum()); cnt += int(m.sum())
    return tot / cnt if cnt else 0.0


def process(path, var, out, scalar, chunk):
    if out and out != path:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        shutil.copy2(path, out)
        target = out
    else:
        target = path
    with Dataset(target, "r+") as ds:
        v = ds.variables[var]
        if v.dimensions[0].lower() not in ("time",):
            sys.exit(f"{var} first dim is {v.dimensions[0]}, expected Time")
        Nf = v.shape[0]
        s = scalar if scalar is not None else global_scalar(v, chunk)
        # HIGH-INDEX-FIRST: pins an unlimited Time at Nf so low-slice writes don't truncate the file.
        for t0 in reversed(range(0, Nf, chunk)):
            t1 = min(t0 + chunk, Nf)
            b = np.asarray(v[t0:t1], dtype=float)
            v[t0:t1] = b - s
        if len(ds.dimensions[v.dimensions[0]]) != Nf:
            sys.exit(f"FATAL: Time truncated {Nf} -> {len(ds.dimensions[v.dimensions[0]])}")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--dir")
    ap.add_argument("--pattern", default="AIS_*Forcing_*.nc")
    ap.add_argument("--var", default="floatingBasalMassBalAdjustment")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--out", help="output path (single --file) ")
    ap.add_argument("--out-dir", help="output dir (with --dir)")
    ap.add_argument("--scalar", type=float, default=None, help="subtract this fixed scalar instead of each file's own mean")
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args()

    files = [a.file] if a.file else sorted(glob.glob(os.path.join(a.dir, a.pattern)))
    if not files:
        sys.exit("no files matched")
    for f in files:
        if a.in_place:
            out = f
        elif a.file:
            out = a.out or sys.exit("need --out or --in-place")
        else:
            out = os.path.join(a.out_dir or sys.exit("need --out-dir or --in-place"), os.path.basename(f))
        s = process(f, a.var, out, a.scalar, a.chunk)
        print(f"{os.path.basename(f)}: subtracted scalar={s:.6e} -> {out}")


if __name__ == "__main__":
    main()
