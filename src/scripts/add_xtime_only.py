#!/usr/bin/env python3
"""
add_xtime_only.py — write a monthly `xtime` char variable onto a MALI forcing file, WITHOUT touching the
data field or adding any trend. Length is taken from the file (n_steps = len(Time)), so a 12000-step file
gets xtime 2300-01 .. 3299-12 with --start-year 2300.

Use this for a no-trend / CTRL-style extension, or when you only want the time axis. For SSP585/SSP126
(which need the scenario 2300-mean added), use add_trend_and_xtime_monthly.py instead — it writes xtime
in the same pass, so there's no reason to run this separately there.

xtime is IDENTICAL across ensemble members (pure time axis), so the fast path is:
    python add_xtime_only.py --file member0.nc --start-year 2300      # build once
    for f in member{1..9}.nc; do ncks -A -v xtime member0.nc "$f"; done   # copy to the rest (no field rewrite)
"""
from __future__ import annotations
import argparse
import numpy as np
from netCDF4 import Dataset, stringtochar


def build_monthly_xtime(y, m, d, n):
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}-{d:02d}_00:00:00")
        m += 1
        if m > 12:
            m = 1; y += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="forcing file to modify IN PLACE")
    ap.add_argument("--start-year", type=int, default=2300)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--start-day", type=int, default=1)
    ap.add_argument("--strlen", type=int, default=64)
    a = ap.parse_args()

    with Dataset(a.file, "r+") as ds:
        if "Time" not in ds.dimensions:
            raise ValueError("file has no Time dimension")
        n = len(ds.dimensions["Time"])
        stamps = build_monthly_xtime(a.start_year, a.start_month, a.start_day, n)
        if "StrLen" not in ds.dimensions:
            ds.createDimension("StrLen", a.strlen)
        if "xtime" in ds.variables:
            xv = ds.variables["xtime"]; sl = xv.shape[1]
        else:
            sl = len(ds.dimensions["StrLen"])
            xv = ds.createVariable("xtime", "S1", ("Time", "StrLen"))
        # single bulk write (all rows at once) instead of one write per timestep
        xv[:] = stringtochar(np.array([s.ljust(sl)[:sl] for s in stamps], dtype=f"S{sl}"))
    print(f"wrote xtime[{n}] = {stamps[0]} .. {stamps[-1]}  into {a.file}")


if __name__ == "__main__":
    main()
