#!/usr/bin/env python3
"""
mean_adjust_variability.py — produce the `..._meanAdjusted.nc` the generator expects.

Reproduces what notebooks/06-AIS-variability-extrapolation.ipynb does:
    filled   = extrapolated.fillna(0)              # zero any residual NaN
    scalar   = filled[FLUX_VAR].mean()             # ONE global scalar (mean over time+y+x), ~2.4e-6
    adjusted = filled - scalar                     # subtract that single scalar everywhere

i.e. it removes a tiny domain-wide bias the extrapolation introduces; it does NOT touch per-cell
means or spatial/temporal structure. (This corrects an earlier per-cell-demean version.)

Memory-safe (dask-chunked) and writes UNLIMITED Time + (1, ny, nx) chunks to match fill_extrapolate.py.

Example:
  python mean_adjust_variability.py \
    --input  .../model-fast-1000y/sorrm_variability_extrapolated_fillNA.nc \
    --output .../model-fast-1000y/sorrm_variability_extrapolated_fillNA_meanAdjusted.nc
"""
from __future__ import annotations
import argparse
import xarray as xr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--var", default="timeMonthly_avg_landIceFreshwaterFlux")
    ap.add_argument("--time-dim", default="Time")
    ap.add_argument("--time-chunk", type=int, default=200)
    args = ap.parse_args()

    ds = xr.open_dataset(args.input, chunks={args.time_dim: args.time_chunk})
    da = ds[args.var].fillna(0.0)                       # zero residual NaNs (as in the notebook)

    scalar_mean = float(da.mean().compute())           # ONE scalar: mean over ALL dims
    print(f"scalar mean subtracted (mean-adjust): {scalar_mean:.6e}")

    adj = da - scalar_mean
    ds_out = ds.copy()
    ds_out[args.var] = adj

    spatial = [d for d in ds[args.var].dims if d != args.time_dim]
    ny, nx = ds[args.var].sizes[spatial[-2]], ds[args.var].sizes[spatial[-1]]
    enc = {args.var: {"chunksizes": (1, ny, nx), "zlib": False}}
    ds_out.to_netcdf(args.output, encoding=enc, unlimited_dims=[args.time_dim])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
