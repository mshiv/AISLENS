#!/usr/bin/env python3
"""
subtract_trend_var_xarray.py

Xarray-based variant of subtract_trend_var that mirrors the structure
used by the addition script (opens with xarray, performs arithmetic,
and writes output). This is often easier to debug and behaves like the
existing addition workflow.

Usage:
  python src/scripts/subtract_trend_var_xarray.py \
    --trend TREND_PLUS_VAR.nc --var VAR_ONLY.nc --out OUT_TREND_ONLY.nc \
    [--varname floatingBasalMassBalAdjustment] [--chunk 12] [--verbose]

Behavior:
  - Loads TREND and VAR with optional chunking.
  - Aligns on Time dimension by trimming to the shorter length if needed.
  - Computes diff = trend[varname] - var[varname] and assigns into trend
  - Writes the modified trend dataset to `--out` (preserving coords like xtime).
"""
from pathlib import Path
import argparse
import logging
import xarray as xr
import numpy as np


def main():
    p = argparse.ArgumentParser(description='Subtract VAR from TREND using xarray')
    p.add_argument('--trend', required=True, help='TREND+VAR input file (will be used as base)')
    p.add_argument('--var', required=True, help='VAR-only file to subtract')
    p.add_argument('--out', required=True, help='Output file path (will be overwritten)')
    p.add_argument('--varname', default='floatingBasalMassBalAdjustment', help='Variable name to subtract')
    p.add_argument('--chunk', type=int, default=12, help='Time chunk size for dask (default 12)')
    p.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = p.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    trend_path = Path(args.trend)
    var_path = Path(args.var)
    out_path = Path(args.out)

    if not trend_path.exists():
        logger.error(f"TREND file not found: {trend_path}")
        raise SystemExit(2)
    if not var_path.exists():
        logger.error(f"VAR file not found: {var_path}")
        raise SystemExit(3)

    logger.info(f"Opening TREND: {trend_path}")
    trend_ds = xr.open_dataset(trend_path, chunks={"Time": args.chunk})
    logger.info(f"Opening VAR: {var_path}")
    var_ds = xr.open_dataset(var_path, chunks={"Time": args.chunk})

    if args.varname not in trend_ds.data_vars:
        logger.error(f"{args.varname} not found in TREND dataset")
        raise SystemExit(4)
    if args.varname not in var_ds.data_vars:
        logger.error(f"{args.varname} not found in VAR dataset")
        raise SystemExit(5)

    tda = trend_ds[args.varname]
    vda = var_ds[args.varname]

    # Align on Time dimension: if lengths differ, trim to min length
    tlen = tda.sizes.get('Time')
    vlen = vda.sizes.get('Time')
    logger.info(f"TREND {args.varname} Time length={tlen}; VAR length={vlen}")
    if tlen != vlen:
        minlen = min(tlen, vlen)
        logger.warning(f"Time lengths differ; trimming to min length {minlen}")
        t_sel = slice(0, minlen)
        v_sel = slice(0, minlen)
        tda = tda.isel(Time=t_sel)
        vda = vda.isel(Time=v_sel)
        # also trim any time coordinate in trend_ds for writing continuity
        trend_ds = trend_ds.isel(Time=t_sel)

    # Compute difference (lazy with dask if chunks provided)
    logger.info(f"Computing difference: trend - var for variable '{args.varname}'")
    diff = tda - vda
    # assign into trend_ds (will align coords)
    trend_ds[args.varname] = diff

    # Write output with compression
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {args.varname: {'zlib': True, 'complevel': 4}}
    logger.info(f"Writing output to {out_path}")
    trend_ds.to_netcdf(out_path, format='NETCDF4', encoding=encoding)
    logger.info("Write complete")


if __name__ == '__main__':
    main()
