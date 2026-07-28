#!/usr/bin/env python
'''
Concatenate base-period ensemble members with their restart extensions.

For each scenario SCEN (e.g. SSP585), the base run lives in
    <root>/SCEN/SCEN_NN/{global,regional}Stats.nc            (2000 -> 2299-12)
and its restart extension in
    <root>/SCEN-EXT/SCEN-EXT_NN/{global,regional}Stats.nc     (2300 -> ...)

The two are stitched along the Time dimension into a new "-FULL" ensemble
    <root>/SCEN-FULL/SCEN-FULL_NN/{global,regional}Stats.nc
so the existing plot_ensemble*Stats.py tools can plot the complete series
(-r <root> -b SCEN-FULL -e 'SCEN-FULL_*').

The extension's daysSinceStart is already on the base's continuous clock
(it starts just past the base's final value), and the calendars do not
overlap, so this is a straight concat along Time -- no offset, no dedup.
The join is nonetheless checked per member and anomalies are reported.

Only the clean 'SCEN_NN' base members (matching the EXT members) are paired;
the legacy 'SCENNN' / 'SCEN-EMn' naming variants are ignored.

Shiva Muruganandham, 2026
'''

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import xarray as xr
from netCDF4 import Dataset

STATS_FILES = ("globalStats.nc", "regionalStats.nc")
# variables that legitimately have no Time dimension: copy from base, don't concat
NONTIME_KEEP = ("simulationStartTime",)


def source_format(path):
    """Return the xarray-compatible format string matching the source file."""
    with Dataset(path) as nc:
        model = nc.data_model  # e.g. NETCDF3_64BIT_OFFSET, NETCDF4, ...
    if model in ("NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA", "NETCDF3_64BIT"):
        return "NETCDF3_64BIT"
    if model == "NETCDF3_CLASSIC":
        return "NETCDF3_CLASSIC"
    if model == "NETCDF4_CLASSIC":
        return "NETCDF4_CLASSIC"
    return "NETCDF4"


def decode_xtime(arr):
    """Turn an (Time, StrLen) char array into a list of stripped strings."""
    if arr.ndim == 2:
        return ["".join(c.decode() if isinstance(c, bytes) else c
                        for c in row).strip() for row in arr]
    return [str(x) for x in arr]


def trim_trailing_garbage(ds):
    """Number of rows to keep (from the start) after trimming trailing incomplete-write rows.

    An HPC run killed mid-write (e.g. storage full) can emit a final timestep with
    daysSinceStart reset to 0 and zero-filled fields (observed in SSP585-EXT_04). Such
    rows sit at the END and break strict time monotonicity. We trim ONLY from the tail,
    so legitimate *interior* restart overlaps (small backward day-jumps where a checkpoint
    re-simulated a few steps, with physical field values) are preserved.
    """
    d = np.asarray(ds["daysSinceStart"].values, dtype=float)
    end = len(d)
    while end >= 2 and not (d[end - 1] > d[end - 2]):
        end -= 1
    return end


def concat_pair(base_path, ext_path, out_path):
    """Concatenate one base+ext stats file pair along Time; write out_path."""
    base = xr.open_dataset(base_path, decode_times=False)
    ext = xr.open_dataset(ext_path, decode_times=False)

    # sanity: variable sets and non-Time dims must match
    if set(base.variables) != set(ext.variables):
        raise ValueError(f"variable mismatch:\n  {base_path}\n  {ext_path}")

    # trim trailing incomplete-write rows (unfinished runs); interior overlaps kept
    for tag, src in (("base", base_path), ("ext", ext_path)):
        ds = base if tag == "base" else ext
        n = ds.sizes["Time"]
        end = trim_trailing_garbage(ds)
        if end < n:
            print(f"  [trim {tag}] removed {n - end} trailing incomplete-write row(s) "
                  f"from {os.path.basename(os.path.dirname(src))}")
            if tag == "base":
                base = base.isel(Time=slice(0, end))
            else:
                ext = ext.isel(Time=slice(0, end))

    # report the seam
    bx = decode_xtime(base["xtime"].values)
    ex = decode_xtime(ext["xtime"].values)
    b_days = base["daysSinceStart"].values
    e_days = ext["daysSinceStart"].values
    gap = e_days[0] - b_days[-1]
    note = ""
    if e_days[0] <= b_days[-1]:
        note = "  [WARN: daysSinceStart not increasing across seam]"
    if bx[-1] == ex[0]:
        note += "  [WARN: duplicate xtime at seam]"

    time_vars = [v for v in base.data_vars if "Time" in base[v].dims]
    comb = xr.concat([base[time_vars], ext[time_vars]], dim="Time",
                     data_vars="all", coords="all")

    # carry over the scalar/static variables from the base
    for v in base.variables:
        if v not in comb.variables:
            comb[v] = base[v]
    comb.attrs = dict(base.attrs)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fmt = source_format(base_path)
    # preserve char (|S1) variables as-is; strip stale encoding that can clash
    for v in comb.variables:
        comb[v].encoding.pop("contiguous", None)
    comb.to_netcdf(out_path, format=fmt)

    base.close()
    ext.close()
    comb.close()
    return (len(bx), len(ex), bx[-1], ex[0], gap, note)


def member_id(name, scen):
    """Extract NN from a clean base member 'SCEN_NN'; else None."""
    m = re.fullmatch(re.escape(scen) + r"_(\d+)", name)
    return m.group(1) if m else None


def process_scenario(root, scen, out_suffix, dry_run, only_members=None):
    base_dir = os.path.join(root, scen)
    ext_dir = os.path.join(root, f"{scen}-EXT")
    out_scen = f"{scen}-{out_suffix}"
    out_dir = os.path.join(root, out_scen)

    if not os.path.isdir(base_dir):
        print(f"  [skip] no base dir: {base_dir}")
        return 0
    if not os.path.isdir(ext_dir):
        print(f"  [skip] no ext dir:  {ext_dir}")
        return 0

    # extension members drive the set (base has many more members than EXT)
    ext_members = {}
    for name in os.listdir(ext_dir):
        m = re.fullmatch(re.escape(scen) + r"-EXT_(\d+)", name)
        if m:
            ext_members[m.group(1)] = name

    made = 0
    for nn in sorted(ext_members):
        if only_members is not None and nn not in only_members:
            continue
        base_member = f"{scen}_{nn}"
        ext_member = ext_members[nn]
        out_member = f"{out_scen}_{nn}"
        bmdir = os.path.join(base_dir, base_member)
        emdir = os.path.join(ext_dir, ext_member)
        if not os.path.isdir(bmdir):
            print(f"  [skip {nn}] no base member {base_member}")
            continue
        wrote_any = False
        for fname in STATS_FILES:
            bpath = os.path.join(bmdir, fname)
            epath = os.path.join(emdir, fname)
            opath = os.path.join(out_dir, out_member, fname)
            if not (os.path.exists(bpath) and os.path.exists(epath)):
                print(f"  [skip {nn}/{fname}] missing base or ext file")
                continue
            if dry_run:
                print(f"  [dry] {out_member}/{fname}")
                continue
            nb, ne, bl, ef, gap, note = concat_pair(bpath, epath, opath)
            print(f"  {out_member}/{fname:16s} {nb}+{ne}={nb+ne}  "
                  f"seam {bl} -> {ef}  (+{gap:.1f} d){note}")
            wrote_any = True
        made += 1 if wrote_any else 0
    print(f"  -> {made} member(s) written to {out_dir}")
    return made


def main():
    default_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "data", "MALI", "diagnostics", "ENSEMBLES")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-r", "--root", default=os.path.normpath(default_root),
                   help="ENSEMBLES root directory (default: repo data path)")
    p.add_argument("-s", "--scenarios", default="SSP585,SSP126",
                   help="comma-separated scenario names (default: SSP585,SSP126)")
    p.add_argument("--suffix", default="FULL",
                   help="output ensemble suffix (default: FULL -> SCEN-FULL)")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="list what would be written without writing")
    p.add_argument("-m", "--members", default=None,
                   help="comma-separated member ids to process (e.g. '00,01'); "
                        "default: all members with extension files")
    args = p.parse_args()
    only = None
    if args.members:
        only = set(x.strip() for x in args.members.split(",") if x.strip())

    print(f"Root: {args.root}")
    total = 0
    for scen in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        print(f"[{scen}] -> {scen}-{args.suffix}")
        total += process_scenario(args.root, scen, args.suffix, args.dry_run, only)
    print(f"Done. {total} member(s) processed.")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
