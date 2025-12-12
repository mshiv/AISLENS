#!/usr/bin/env python3
"""
resample_to_monthly_300yrs.py

Resample forcing/output files to a long monthly-timeseries of fixed length
(default: 300 years x 12 months = 3600 time slices).

- Compute a numeric "years since start" coordinate from `daysSinceStart` or
  `deltat` if available (robust to datetime64 / timedelta64 / numeric types).
- Create target monthly time points: 0.0, 1/12, 2/12, ..., up to (years-1/12).
- Linearly interpolate each time-dependent data variable onto the monthly
  target years using xarray `interp`.
- Optionally fill values outside the original time range using the nearest
  boundary value (`--fill nearest`) or leave them as NaN (default).
- Replace the time coordinate with integer indices 0..(n_time-1) as requested.

Notes:
- This script uses xarray for interpolation. NCO can do some
  resampling but is less flexible for the mixed time encodings and large
  multidimensional fields.
- The script preserves mesh coordinates and copies
  non-time-varying variables to the output.

Usage example:
  python src/scripts/resample_to_monthly_300yrs.py \
    --infile /path/to/input.nc \
    --out /path/to/out.nc \
    --years 300 --months-per-year 12 --fill nearest

"""
from pathlib import Path
import argparse
import logging
import numpy as np
import xarray as xr

from aislens.config import config

logger = logging.getLogger(__name__)


def compute_years_from_ds(ds, time_dim=config.TIME_DIM):
    """Return a 1D numpy array of years relative to start using the dataset.
    Handles `daysSinceStart` (datetime64/timedelta64/numeric) and `deltat`.
    """
    if 'daysSinceStart' in ds:
        days = ds['daysSinceStart'].values
        if np.issubdtype(getattr(days, 'dtype', type(days)), np.datetime64):
            delta_days = (days - days[0]) / np.timedelta64(1, 'D')
            years = delta_days.astype(float) / 365.0
        elif np.issubdtype(getattr(days, 'dtype', type(days)), np.timedelta64):
            delta_days = (days - days[0]) / np.timedelta64(1, 'D')
            years = delta_days.astype(float) / 365.0
        else:
            years = np.asarray(days, dtype=float) / 365.0
            years = years - years[0]
        return years

    # fallback to deltat
    if 'deltat' in ds:
        dt = ds['deltat'].values
        if np.issubdtype(getattr(dt, 'dtype', type(dt)), np.timedelta64):
            secs = (dt / np.timedelta64(1, 's')).astype(float)
            # cumulative seconds, start at zero
            cumsum_secs = np.cumsum(np.r_[0.0, secs[:-1]])
            return cumsum_secs / 3.15e7
        else:
            # numeric seconds (scalar or array)
            if np.isscalar(dt):
                step_year = float(dt) / 3.15e7
                return np.arange(ds.sizes.get(time_dim)) * step_year
            else:
                secs = np.asarray(dt, dtype=float)
                cumsum_secs = np.cumsum(np.r_[0.0, secs[:-1]])
                return cumsum_secs / 3.15e7

    raise RuntimeError('No daysSinceStart or deltat found to compute years')


def resample_to_monthly(infile, outpath, years_total=300, months_per_year=12,
                        time_dim=config.TIME_DIM, fill='nan'):
    ds = xr.open_dataset(infile, chunks={time_dim: 12})
    logger.info(f"Opened input: {infile}")

    years = compute_years_from_ds(ds, time_dim=time_dim)
    logger.info(f"Computed years: start={years[0]:.3f}, end={years[-1]:.3f}, n_time={len(years)}")

    # target monthly years: 0, 1/12, 2/12, ..., up to years_total (exclusive)
    target = np.arange(0, years_total, 1.0 / months_per_year)
    n_target = target.size
    logger.info(f"Target monthly points: {n_target} slices (years={years_total}, per-year={months_per_year})")

    # attach numeric 'years' coord to time dimension for interpolation
    # find name of time dim in the dataset variable dims (use config.TIME_DIM)
    if time_dim not in ds.dims and time_dim not in ds.coords:
        # pick first dim that looks like time
        time_dim = next((d for d in ds.dims if 'time' in d.lower()), ds.dims[0])
        logger.info(f"Using inferred time dim: {time_dim}")

    # create a new coordinate 'years_coord' (temporary) aligned to time dim
    ds = ds.assign_coords({'years_coord': (time_dim, years)})

    # Select variables that contain the time dimension
    time_vars = [v for v in ds.data_vars if time_dim in ds[v].dims]
    logger.info(f"Variables to resample (time-dependent): {time_vars}")

    out_vars = {}
    # Interpolate each time-dependent variable onto target years
    for v in time_vars:
        logger.info(f"Interpolating variable: {v}")
        # use linear interpolation in the 'years_coord' dimension
        try:
            da = ds[v].interp(years_coord=target, method='linear')
        except Exception as e:
            logger.warning(f"xarray interp failed for {v}: {e}; attempting fallback by stacking spatial dims")
            # Fallback: stack non-time dims and apply numpy interp per 1D timeseries
            non_time_dims = [d for d in ds[v].dims if d != time_dim]
            stacked = ds[v].stack(allpoints=non_time_dims)
            data = stacked.values  # shape (time, allpoints)
            # use np.apply_along_axis with interpolation for each column
            interp_cols = np.empty((n_target, data.shape[1]), dtype=float)
            for col in range(data.shape[1]):
                colvals = data[:, col]
                try:
                    interp_cols[:, col] = np.interp(target, years, colvals, left=np.nan, right=np.nan)
                except Exception:
                    interp_cols[:, col] = np.nan
            da = xr.DataArray(interp_cols, dims=('years_coord', 'allpoints'))
            da = da.unstack('allpoints')

        # after interp da has coord 'years_coord'
        out_vars[v] = da

    # Build output dataset: include mesh/static variables, and interpolated vars
    out_ds = xr.Dataset()
    # copy non-time-varying data_vars and coords (mesh variables)
    for v in ds.data_vars:
        if v not in out_vars:
            if time_dim not in ds[v].dims:
                out_ds[v] = ds[v]

    # add interpolated vars, rename years_coord -> time_dim and set integer time index
    for v, da in out_vars.items():
        # ensure dims are (years_coord, ...). Rename coord to temporary 'years_coord' if necessary
        # assign integer time index
        da = da.assign_coords({'years_coord': target})
        # rename coord to time_dim for output and set integer index
        da = da.rename({'years_coord': time_dim})
        # create integer time indices 0..n_target-1
        int_time = np.arange(n_target, dtype=int)
        da = da.assign_coords({time_dim: int_time})
        out_ds[v] = da

    # copy supportive coords from input (xCell, yCell, etc.) if present
    for coord in ['xCell', 'yCell', 'dcEdge', 'nCells']:
        if coord in ds.coords or coord in ds:
            out_ds.coords[coord] = ds.coords.get(coord, ds.get(coord))

    # If fill requested as 'nearest', fill NaNs outside original range with nearest boundary
    if fill == 'nearest':
        # determine original time range in years
        y0, y1 = years[0], years[-1]
        # for each var, where target < y0 set values to var.sel(time_dim=0), similarly for > y1
        for v in out_vars.keys():
            da = out_ds[v]
            left_idx = np.where(target < y0)[0]
            right_idx = np.where(target > y1)[0]
            if left_idx.size > 0:
                left_val = ds[v].isel({time_dim: 0})
                out_ds[v][{time_dim: left_idx}] = left_val
            if right_idx.size > 0:
                right_val = ds[v].isel({time_dim: -1})
                out_ds[v][{time_dim: right_idx}] = right_val

    # write output
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {'zlib': True, 'complevel': 4} for v in out_ds.data_vars}
    logger.info(f"Writing resampled file: {outpath}")
    out_ds.to_netcdf(outpath, format='NETCDF4', engine='netcdf4', encoding=encoding)
    logger.info("Write complete")


def main():
    parser = argparse.ArgumentParser(description='Resample forcing/output to monthly fixed-length timeseries')
    parser.add_argument('--infile', required=True, help='Input NetCDF file to resample')
    parser.add_argument('--out', required=True, help='Output NetCDF file path')
    parser.add_argument('--years', type=int, default=300, help='Total years for output (default 300)')
    parser.add_argument('--months-per-year', type=int, default=12, help='Months per year (default 12)')
    parser.add_argument('--time-dim', default=config.TIME_DIM, help='Name of time dimension in file (default from config)')
    parser.add_argument('--fill', choices=['nan', 'nearest'], default='nan', help="How to fill outside original time range: 'nan' or 'nearest' (default 'nan')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    resample_to_monthly(args.infile, args.out, years_total=args.years,
                        months_per_year=args.months_per_year,
                        time_dim=args.time_dim, fill=args.fill)


if __name__ == '__main__':
    main()
