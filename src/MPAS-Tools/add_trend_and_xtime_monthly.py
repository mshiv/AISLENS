#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
from netCDF4 import Dataset, stringtochar


def build_monthly_xtime(start_year: int, start_month: int, start_day: int, n_steps: int) -> list[str]:
    timestamps = []
    year = start_year
    month = start_month
    day = start_day

    for _ in range(n_steps):
        timestamps.append(f"{year:04d}-{month:02d}-{day:02d}_00:00:00")
        month += 1
        if month > 12:
            month = 1
            year += 1

    return timestamps

def load_last_trend_slice(trend_ds: Dataset, varname: str) -> np.ndarray:
    if varname not in trend_ds.variables:
        raise KeyError(f"Variable '{varname}' not found in trend file")

    trend_var = trend_ds.variables[varname]
    trend_data = np.asarray(trend_var[:])

    if trend_data.ndim == 0:
        return np.asarray(trend_data)

    if "Time" in trend_var.dimensions:
        time_axis = trend_var.dimensions.index("Time")
    else:
        time_axis = 0

    last_slice = np.take(trend_data, -1, axis=time_axis)
    return np.asarray(np.squeeze(last_slice))


def add_constant_trend(out_ds: Dataset, trend_slice: np.ndarray, varname: str) -> None:
    if varname not in out_ds.variables:
        raise KeyError(f"Variable '{varname}' not found in forcing file")

    out_var = out_ds.variables[varname]
    out_shape = out_var.shape

    if "Time" in out_var.dimensions:
        time_axis = out_var.dimensions.index("Time")
        expected_shape = out_shape[:time_axis] + out_shape[time_axis + 1 :]
        if trend_slice.ndim == 0:
            pass
        elif trend_slice.shape != expected_shape:
            if trend_slice.size == int(np.prod(expected_shape)):
                trend_slice = trend_slice.reshape(expected_shape)
            else:
                raise ValueError(
                    f"Trend slice shape {trend_slice.shape} does not match forcing spatial shape {expected_shape}"
                )

        # Write in chunks, high-index-first: single-index writes on an UNLIMITED
        # Time dim can corrupt the record count; slice access is well-behaved.
        # Writing the top chunk first pins the dimension so later lower writes are safe.
        nt = out_shape[time_axis]
        chunk = 200
        for t0 in reversed(range(0, nt, chunk)):
            t1 = min(t0 + chunk, nt)
            slc = [slice(None)] * out_var.ndim
            slc[time_axis] = slice(t0, t1)
            block = np.asarray(out_var[tuple(slc)])
            out_var[tuple(slc)] = block + trend_slice
    else:
        if trend_slice.ndim > 0 and trend_slice.shape != out_shape:
            if trend_slice.size == int(np.prod(out_shape)):
                trend_slice = trend_slice.reshape(out_shape)
            else:
                raise ValueError(
                    f"Trend slice shape {trend_slice.shape} does not match forcing shape {out_shape}"
                )
        out_var[:] = np.asarray(out_var[:]) + trend_slice


def write_xtime(out_ds: Dataset, start_year: int, start_month: int, start_day: int, xtime_strlen: int) -> None:
    if "Time" not in out_ds.dimensions:
        raise ValueError("forcing file does not have a Time dimension")

    n_steps = len(out_ds.dimensions["Time"])
    timestamps = build_monthly_xtime(start_year, start_month, start_day, n_steps)

    if "StrLen" not in out_ds.dimensions:
        out_ds.createDimension("StrLen", xtime_strlen)

    if "xtime" in out_ds.variables:
        xtime_var = out_ds.variables["xtime"]
        if xtime_var.ndim != 2:
            raise ValueError("Existing xtime variable is not 2D char data")
        str_len = xtime_var.shape[1]
    else:
        str_len = len(out_ds.dimensions["StrLen"])
        xtime_var = out_ds.createVariable("xtime", "S1", ("Time", "StrLen"))

    for i, stamp in enumerate(timestamps):
        padded = stamp.ljust(str_len)[:str_len]
        xtime_var[i, :] = stringtochar(np.array([padded], dtype=f"S{str_len}"))[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add the last trend slice as a constant offset to forcing files and rewrite xtime monthly."
    )
    parser.add_argument("--forcing", required=True, help="Input forcing file to modify")
    parser.add_argument("--trend", required=True, help="Trend file providing the constant offset")
    parser.add_argument("--out", required=True, help="Output file path")
    parser.add_argument("--varname", default="floatingBasalMassBalAdjustment", help="Variable to modify")
    parser.add_argument("--start-year", type=int, default=2300, help="First year for xtime")
    parser.add_argument("--start-month", type=int, default=1, help="First month for xtime")
    parser.add_argument("--start-day", type=int, default=1, help="First day for xtime")
    parser.add_argument("--xtime-strlen", type=int, default=64, help="Character length for xtime entries")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it already exists")

    args = parser.parse_args()

    if not os.path.isfile(args.forcing):
        raise FileNotFoundError(f"Forcing file not found: {args.forcing}")
    if not os.path.isfile(args.trend):
        raise FileNotFoundError(f"Trend file not found: {args.trend}")

    if os.path.exists(args.out):
        if not args.overwrite:
            raise FileExistsError(f"Output file already exists: {args.out}")
        os.remove(args.out)

    shutil.copy2(args.forcing, args.out)

    with Dataset(args.trend, "r") as trend_ds, Dataset(args.out, "r+") as out_ds:
        trend_slice = load_last_trend_slice(trend_ds, args.varname)
        add_constant_trend(out_ds, trend_slice, args.varname)
        write_xtime(out_ds, args.start_year, args.start_month, args.start_day, args.xtime_strlen)

    print(f"Wrote combined forcing file: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())