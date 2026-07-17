#!/usr/bin/env python3
"""
add_scenario_trend_2000_3000.py — build a full 2000->3000 scenario forcing from a shared, scenario-agnostic
1000-yr realization (on the MALI grid) by adding a scenario trend that:
    * leaves 2000 -> --trend-start-year (default 2015) as VARIABILITY-ONLY (no trend), matching the original
      production -- the ISMIP6 scenario trend file only spans 2015-2300, so the first 15 years carry only
      the seasonality+variability anomalies,
    * adds the resampled ISMIP6 trend element-wise over its span (2015-2300), then
    * HOLDS the last (2300) slice constant for every year after (2300-3000).
It also writes a monthly xtime (2000-01 .. 2999-12 by default).

This keeps CTRL and SSP585/126 PAIRED: all three use the same realization indexed by the same calendar
year (xtime 2000-2999), so member i differs across scenarios only by this trend offset. CTRL needs no
trend at all -- use add_xtime_only.py for it.

Alignment is by index from --start-year (both realization[0] and trend[0] are assumed = start-year-01).
The script prints the alignment (trend length, hold-from year) so you can sanity-check before trusting it.

Memory-safe: chunked over Time; never loads the whole (12000 x nCells) field.

Example:
  python add_scenario_trend_2000_3000.py \
     --realization .../vargen_realizations-ssn-1000y-mali/AIS_..._Forcing_0.nc \
     --trend .../RESAMPLED_forcing_trend_expAE05_SSP585_2015-2300_negAdj.nc \
     --out   .../vargen_realizations-ssn-2000-3000-ssp585/AIS_..._Forcing_0.nc \
     --start-year 2000
"""
from __future__ import annotations
import argparse, os, shutil, sys
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


def write_xtime(ds, start_year, start_month, start_day, strlen):
    n = len(ds.dimensions["Time"])
    stamps = build_monthly_xtime(start_year, start_month, start_day, n)
    if "StrLen" not in ds.dimensions:
        ds.createDimension("StrLen", strlen)
    if "xtime" in ds.variables:
        xv = ds.variables["xtime"]; sl = xv.shape[1]
    else:
        sl = len(ds.dimensions["StrLen"])
        xv = ds.createVariable("xtime", "S1", ("Time", "StrLen"))
    xv[:] = stringtochar(np.array([s.ljust(sl)[:sl] for s in stamps], dtype=f"S{sl}"))
    return stamps[0], stamps[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--realization", required=True, help="1000-yr MALI-grid realization (scenario-agnostic)")
    ap.add_argument("--trend", required=True, help="resampled scenario trend (covers 2000-2300)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--varname", default="floatingBasalMassBalAdjustment")
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--start-day", type=int, default=1)
    ap.add_argument("--trend-start-year", type=int, default=2015,
                    help="calendar year of the trend file's first step (ISMIP6 scenarios start 2015); "
                         "years before this get NO trend (variability-only)")
    ap.add_argument("--strlen", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=200, help="Time-steps per read/write block")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    for p in (a.realization, a.trend):
        if not os.path.isfile(p):
            sys.exit(f"not found: {p}")
    if os.path.exists(a.out):
        if not a.overwrite:
            sys.exit(f"output exists (use --overwrite): {a.out}")
        os.remove(a.out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # open the trend but DO NOT load it whole -- read only its last slice now (for the held-constant tail);
    # the element-wise span is read chunk-by-chunk in the loop below, so peak RAM stays ~one chunk instead
    # of the whole (~10 GB monthly) trend. Trend must have Time as dim 0 (matches the realization cadence).
    td = Dataset(a.trend, "r")
    tv = td.variables[a.varname] if a.varname in td.variables else td.variables[a.varname + "_var"]
    if "Time" in tv.dimensions and tv.dimensions.index("Time") != 0:
        sys.exit(f"trend var {tv.name} must have Time as dim 0, got {tv.dimensions}")
    Tt = tv.shape[0] if "Time" in tv.dimensions else 1
    last = np.squeeze(np.asarray(tv[-1] if Tt > 1 else tv[:]))   # (nCells,)

    shutil.copy2(a.realization, a.out)
    with Dataset(a.out, "r+") as ds:
        v = ds.variables[a.varname]
        tax = v.dimensions.index("Time")
        if tax != 0:
            sys.exit(f"expected Time as dim 0 of {a.varname}, got {v.dimensions}")
        Nf = v.shape[0]
        off = (a.trend_start_year - a.start_year) * 12          # steps of variability-only lead-in (2000-2015)
        if off < 0:
            sys.exit(f"trend-start-year {a.trend_start_year} < start-year {a.start_year}")
        trend_end = off + Tt                                    # first step of the held-constant tail
        hold_year = a.start_year + trend_end // 12
        print(f"realization Time={Nf}, trend Time={Tt}, trend begins at step {off} (year {a.trend_start_year})")
        print(f"  steps 0..{off-1}        : variability-only, NO trend ({a.start_year}..{a.trend_start_year})")
        print(f"  steps {off}..{trend_end-1}  : trend element-wise ({a.trend_start_year}.. )")
        print(f"  steps {trend_end}..{Nf-1}   : last trend slice held constant (~{hold_year}.. )")
        if trend_end > Nf:
            sys.exit(f"trend end {trend_end} (off {off} + Tt {Tt}) exceeds realization Time {Nf} -- check inputs")

        # IMPORTANT: write chunks HIGH-INDEX-FIRST. On a netCDF UNLIMITED Time dim, assigning a low
        # slice (e.g. v[0:200]) shrinks the record count to that slice; writing the top chunk first
        # pins Time at Nf so subsequent lower writes don't truncate the file. (Safe for fixed Time too.)
        for t0 in reversed(range(0, Nf, a.chunk)):
            t1 = min(t0 + a.chunk, Nf)
            block = np.asarray(v[t0:t1], dtype=float)          # (nt, nCells)
            # region [0,off): variability-only -> add nothing.
            # region [off,trend_end): trend element-wise.
            lo, hi = max(t0, off), min(t1, trend_end)
            if lo < hi:                                          # read only this trend slice (bounded RAM)
                block[lo - t0: hi - t0] += np.squeeze(np.asarray(tv[lo - off: hi - off]))
            # region [trend_end,Nf): held last slice (broadcast).
            lo, hi = max(t0, trend_end), min(t1, Nf)
            if lo < hi:
                block[lo - t0: hi - t0] += last
            v[t0:t1] = block

        if len(ds.dimensions["Time"]) != Nf:
            sys.exit(f"FATAL: Time truncated {Nf} -> {len(ds.dimensions['Time'])} during write")

        first, lastst = write_xtime(ds, a.start_year, a.start_month, a.start_day, a.strlen)
    td.close()
    print(f"wrote {a.out}  (xtime {first} .. {lastst})")


if __name__ == "__main__":
    main()
