#!/usr/bin/env python3
"""
subtract_trend_var.py

Copy a TREND+VAR NetCDF to an output path and subtract the VAR-only
component from the copied file in-place to produce the trend-only
`floatingBasalMassBalAdjustment` (or another chosen variable).

This does minimal I/O: it copies the TREND file (fast filesystem copy)
then opens the VAR and OUT files with netCDF4 and updates the variable
in chunks along the leading (time) dimension.

Usage:
  python src/scripts/subtract_trend_var.py \
    --trend trend_plus_var.nc \
    --var var_only.nc \
    --out trend_only_out.nc \
    [--varname floatingBasalMassBalAdjustment] [--chunk 16]

The script preserves all coordinates (including `xtime`) from the TREND
file because it copies the TREND file to the output first.
"""
from pathlib import Path
import argparse
import shutil
import sys
import math
from netCDF4 import Dataset
import numpy as np


def chunked_subtract(trend_path, var_path, out_path, varname, chunk_size=16):
    # Copy TREND -> OUT first (preserves xtime and other coords)
    shutil.copy2(trend_path, out_path)

    # Open datasets
    var_ds = Dataset(var_path, 'r')
    out_ds = Dataset(out_path, 'r+')

    if varname not in var_ds.variables:
        raise KeyError(f"Variable '{varname}' not found in VAR file: {var_path}")
    if varname not in out_ds.variables:
        raise KeyError(f"Variable '{varname}' not found in OUT file (copied from TREND): {out_path}")

    v_var = var_ds.variables[varname]
    v_out = out_ds.variables[varname]

    # Determine axis to iterate over (assume leading/time axis is first)
    # Use shape and treat first axis as time axis
    if len(v_out.shape) == 0:
        # scalar: just subtract directly
        val_out = v_out[()]
        val_var = v_var[()]
        v_out[()] = val_out - val_var
        out_ds.sync()
        var_ds.close()
        out_ds.close()
        return

    time_len = v_out.shape[0]
    n_chunks = math.ceil(time_len / float(chunk_size))

    print(f"Performing in-place subtraction for '{varname}': length={time_len}, chunk={chunk_size}, chunks={n_chunks}")

    for i in range(0, time_len, chunk_size):
        j = min(time_len, i + chunk_size)
        # read slices (netCDF4 supports numpy-style slicing)
        out_slice = v_out[i:j, ...]
        var_slice = v_var[i:j, ...]
        # promote to float for safe arithmetic (preserve dtype later)
        # handle missing-value types by using numpy arrays directly
        try:
            # compute difference
            result = out_slice - var_slice
        except Exception:
            # fallback to elementwise np subtraction
            result = np.array(out_slice) - np.array(var_slice)

        # write back
        v_out[i:j, ...] = result
        print(f"  wrote chunk {i}:{j}")

    out_ds.sync()
    var_ds.close()
    out_ds.close()


def main():
    parser = argparse.ArgumentParser(description='Create trend-only file by subtracting var-only component from trend+var file')
    parser.add_argument('--trend', required=True, help='TREND+VAR input file (will be copied to --out)')
    parser.add_argument('--var', required=True, help='VAR-only file containing component to subtract')
    parser.add_argument('--out', required=True, help='Output path (copy of TREND file, then in-place subtraction)')
    parser.add_argument('--varname', default='floatingBasalMassBalAdjustment', help='Variable name to subtract (default: floatingBasalMassBalAdjustment)')
    parser.add_argument('--chunk', type=int, default=16, help='Time-chunk size to operate on (default 16). Tune to memory/perf.')
    args = parser.parse_args()

    trend_path = Path(args.trend)
    var_path = Path(args.var)
    out_path = Path(args.out)

    if not trend_path.exists():
        print(f"TREND file not found: {trend_path}", file=sys.stderr)
        sys.exit(2)
    if not var_path.exists():
        print(f"VAR file not found: {var_path}", file=sys.stderr)
        sys.exit(3)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        chunked_subtract(str(trend_path), str(var_path), str(out_path), args.varname, args.chunk)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(10)

    print("Completed subtraction successfully.")


if __name__ == '__main__':
    main()
