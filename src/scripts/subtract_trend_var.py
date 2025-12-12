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
import logging


def chunked_subtract(trend_path, var_path, out_path, varname, chunk_size=16):
    # Copy TREND -> OUT first (preserves xtime and other coords)
    logger = logging.getLogger(__name__)
    logger.info(f"Copying TREND -> OUT: {trend_path} -> {out_path}")
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

    # Determine time axis for both variables. Prefer dimension named 'Time',
    # otherwise pick the axis whose size matches the other's time length.
    def detect_time_axis(nc_var, other_time_len=None):
        # prefer a dimension literally named 'Time' (case-sensitive)
        dims = getattr(nc_var, 'dimensions', None)
        if dims:
            for i, d in enumerate(dims):
                if d == 'Time' or d == 'time':
                    return i
        # fallback: choose first axis if nothing else
        if other_time_len is None:
            return 0
        # try to find a unique axis matching other_time_len
        for i, s in enumerate(nc_var.shape):
            if s == other_time_len:
                return i
        # default to first axis
        return 0

    # Detect time axes
    if len(v_out.shape) == 0:
        # scalar: just subtract directly
        val_out = v_out[()]
        val_var = v_var[()]
        v_out[()] = val_out - val_var
        out_ds.sync()
        var_ds.close()
        out_ds.close()
        return

    # time lengths (attempt to use the first axis of var_ds as source time length)
    var_time_len = v_var.shape[0] if len(v_var.shape) > 0 else 1
    out_time_axis = detect_time_axis(v_out, other_time_len=var_time_len)
    var_time_axis = detect_time_axis(v_var, other_time_len=var_time_len)
    # determine full time length from out variable using detected axis
    time_len = v_out.shape[out_time_axis]

    logger.info(f"Variable '{varname}' shapes: OUT={v_out.shape}, VAR={v_var.shape}")
    logger.info(f"Detected time axes: OUT axis={out_time_axis}, VAR axis={var_time_axis}, time_len={time_len}")
    n_chunks = math.ceil(time_len / float(chunk_size))

    logger.info(f"Performing in-place subtraction for '{varname}': length={time_len}, chunk={chunk_size}, chunks={n_chunks}")

    for i in range(0, time_len, chunk_size):
        j = min(time_len, i + chunk_size)
        # build slice tuples for out and var according to detected time axes
        out_slice_idx = [slice(None)] * len(v_out.shape)
        out_slice_idx[out_time_axis] = slice(i, j)
        var_slice_idx = [slice(None)] * len(v_var.shape)
        var_slice_idx[var_time_axis] = slice(i, j)

        out_slice = v_out[tuple(out_slice_idx)]
        var_slice = v_var[tuple(var_slice_idx)]
        # log basic stats for this slice when verbose
        try:
            o_mean = float(np.nanmean(np.array(out_slice)))
            v_mean = float(np.nanmean(np.array(var_slice)))
            logger.debug(f"  chunk {i}:{j} shapes out={out_slice.shape} var={var_slice.shape} means out={o_mean:.6g} var={v_mean:.6g}")
        except Exception:
            logger.debug(f"  chunk {i}:{j} shapes out={getattr(out_slice, 'shape', None)} var={getattr(var_slice, 'shape', None)}")
        # If time axes positions differ, move var_slice's time axis to match out_slice
        if var_slice.ndim == out_slice.ndim and var_time_axis != out_time_axis:
            try:
                var_slice = np.moveaxis(var_slice, var_time_axis, out_time_axis)
            except Exception:
                # if moveaxis fails, fall back to elementwise subtraction attempt
                pass
        # promote to float for safe arithmetic (preserve dtype later)
        # handle missing-value types by using numpy arrays directly
        try:
            # compute difference
            result = out_slice - var_slice
        except Exception:
            # fallback to elementwise np subtraction
            result = np.array(out_slice) - np.array(var_slice)

        # write back (use same indexing tuple)
        v_out[tuple(out_slice_idx)] = result
        # verify write by re-reading the slice and checking residual
        try:
            written = np.array(v_out[tuple(out_slice_idx)])
            residual = np.nanmean(np.abs(written - result))
            logger.debug(f"  wrote chunk {i}:{j} residual_mean={residual:.6g}")
        except Exception as e:
            logger.warning(f"  wrote chunk {i}:{j} but verification failed: {e}")

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
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug logging')
    args = parser.parse_args()

    # configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s')

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
        logging.exception(f"ERROR during subtraction: {e}")
        sys.exit(10)

    logging.info("Completed subtraction successfully.")


if __name__ == '__main__':
    main()
