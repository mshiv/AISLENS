#!/usr/bin/env python3
"""
find_resume_timestep.py — for partially-interpolated MALI forcing files (interpolate_to_mpasli_grid_mod.py
died mid-run), report how many timesteps were written and the --timestart to resume from.

How it works: that interpolator writes `var[timelev,:]` and `sync()`s after each step into an UNLIMITED
Time dimension, so the DEST file's current Time SIZE == number of steps already written. Resume point =
last_written - overlap, where last_written = Time_size - 1. A fallback scan (last time slice that has
spatial variation, i.e. is not a constant fill) covers the rare case where Time is fixed-full.

Usage:
  # survey a whole dir (table):
  python find_resume_timestep.py --dir <DEST_DIR> --timeend 11999
  # single file, print ONLY the resume --timestart (for use inside an sbatch):
  python find_resume_timestep.py --file <DEST.nc> --timeend 11999 --overlap 1 --print-timestart
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
from netCDF4 import Dataset


def slice_is_written(v, t):
    """A written melt slice varies spatially; an unwritten fill slice is constant/NaN."""
    a = np.asarray(v[t, ...], dtype=float)
    finite = a[np.isfinite(a)]
    return finite.size > 0 and float(finite.min()) != float(finite.max())


def last_written_index(path, var, timeend):
    ds = Dataset(path, "r")
    if var not in ds.variables:
        ds.close(); raise KeyError(f"{var} not in {path}")
    v = ds.variables[var]
    tdim = v.dimensions[0]
    nt = len(ds.dimensions[tdim])
    full = timeend + 1
    if nt < full:
        # partial unlimited dim -> size is exactly the written count
        k = nt - 1
        mode = f"Time size {nt} < {full} (partial)"
    else:
        # looks complete: verify by scanning back for the last non-constant slice
        k = nt - 1
        while k >= 0 and not slice_is_written(v, k):
            k -= 1
        mode = "Time is full; scanned for last non-fill slice" if k < nt - 1 else "Time full & last slice written (COMPLETE?)"
    ds.close()
    return nt, k, mode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--dir")
    ap.add_argument("--pattern", default="AIS_*Forcing_*.nc")
    ap.add_argument("--var", default="floatingBasalMassBalAdjustment")
    ap.add_argument("--timeend", type=int, default=11999)
    ap.add_argument("--overlap", type=int, default=1, help="resume this many steps before the last written")
    ap.add_argument("--print-timestart", action="store_true",
                    help="print ONLY the integer --timestart (single --file only)")
    a = ap.parse_args()

    files = [a.file] if a.file else sorted(glob.glob(os.path.join(a.dir, a.pattern)))
    if not files:
        sys.exit("no files matched")

    if a.print_timestart:
        nt, k, _ = last_written_index(files[0], a.var, a.timeend)
        print(max(0, k - a.overlap))
        return

    print(f"{'file':60s} {'Time':>7s} {'last_wr':>8s} {'resume@':>8s}  note")
    for f in files:
        try:
            nt, k, mode = last_written_index(f, a.var, a.timeend)
        except Exception as e:
            print(f"{os.path.basename(f):60s}  ERROR: {e}"); continue
        ts = max(0, k - a.overlap)
        done = " COMPLETE (skip)" if k >= a.timeend else ""
        print(f"{os.path.basename(f):60s} {nt:7d} {k:8d} {ts:8d}  {mode}{done}")


if __name__ == "__main__":
    main()
