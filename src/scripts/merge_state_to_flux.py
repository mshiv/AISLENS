#!/usr/bin/env python3
"""
merge_state_to_flux.py

Utility: interpolate `lowerSurface` from a MALI `state` file onto the
`daysSinceStart` time levels of a MALI `flux` file and append the result
into a new copy of the flux dataset.

This produces a new NetCDF file containing all variables from the original
flux file plus an appended `lowerSurface` variable taken from the state file
and interpolated to the flux time axis.

Usage example:
    python src/scripts/merge_state_to_flux.py --flux path/to/flux.nc --state path/to/state.nc

Options:
  --method: interpolation method for numeric days (default: 'linear', fallback 'nearest')
  --outdir: directory to write the merged file (default: flux file directory)
  --overwrite: if set and `lowerSurface` exists in flux file, overwrite it
"""
from pathlib import Path
import argparse
import logging
import numpy as np
import xarray as xr
from aislens.config import config

logger = logging.getLogger(__name__)


def _to_days(vals):
    vals = np.asarray(vals)
    # Datetime64 -> seconds -> days
    if np.issubdtype(vals.dtype, np.datetime64):
        secs = vals.astype('datetime64[s]').astype('int64')
        return secs.astype('float64') / 86400.0
    # Timedelta64 -> seconds -> days
    if np.issubdtype(vals.dtype, np.timedelta64):
        secs = vals.astype('timedelta64[s]').astype('int64')
        return secs.astype('float64') / 86400.0
    # Large integers: likely nanoseconds since epoch
    if np.issubdtype(vals.dtype, np.integer):
        try:
            mag = np.nanmax(np.abs(vals))
        except Exception:
            mag = None
        if mag is not None and mag > 1e12:
            return vals.astype('float64') / 1e9 / 86400.0
        else:
            return vals.astype('float64')
    if np.issubdtype(vals.dtype, np.floating):
        return vals.astype('float64')
    # fallback: elementwise
    try:
        out = np.array([np.datetime64(v).astype('datetime64[s]').astype('int64')/86400.0 for v in vals])
        return out.astype('float64')
    except Exception:
        return vals.astype('float64')


def merge_state_lower_surface(flux_path, state_path, outdir=None, method='linear', overwrite=False):
    flux_path = Path(flux_path)
    state_path = Path(state_path)

    logger.info(f'Loading flux: {flux_path}')
    ds_flux = xr.open_dataset(flux_path)
    logger.info(f'Loading state: {state_path}')
    ds_state = xr.open_dataset(state_path)

    # Create a sign-reversed copy of the flux variable named
    # 'floatingBasalMassBalAdjustment' and leave the original variable unchanged.
    try:
        var_flux = config.MALI_FLOATINGBMB_VAR
        adj_name = 'floatingBasalMassBalAdjustment'
        if var_flux in ds_flux:
            # create adjustment variable as negative of the original
            ds_flux[adj_name] = -1 * ds_flux[var_flux]
            logger.info(f"Created sign-reversed variable '{adj_name}' from '{var_flux}' in ds_flux")
        else:
            logger.warning(f"Flux variable '{var_flux}' not found in ds_flux; cannot create '{adj_name}'")
    except Exception:
        logger.exception('Failed to create sign-reversed adjustment variable; continuing')

    var_draft = 'lowerSurface'

    if var_draft not in ds_state:
        raise RuntimeError(f"'{var_draft}' not found in state file: {state_path}")

    # Prefer daysSinceStart coord for robust alignment
    if 'daysSinceStart' not in ds_flux or 'daysSinceStart' not in ds_state:
        logger.warning('One of the files is missing daysSinceStart coordinate. Falling back to Time coordinate if present.')

    # Build numeric days coordinates
    try:
        if 'daysSinceStart' in ds_flux:
            days_flux = _to_days(ds_flux['daysSinceStart'].values)
        else:
            days_flux = _to_days(ds_flux['Time'].values)
    except Exception:
        logger.exception('Failed to extract numeric days for flux; aborting')
        raise

    try:
        if 'daysSinceStart' in ds_state:
            days_state = _to_days(ds_state['daysSinceStart'].values)
        else:
            days_state = _to_days(ds_state['Time'].values)
    except Exception:
        logger.exception('Failed to extract numeric days for state; aborting')
        raise

    # We'll interpolate draft values from state -> flux along numeric days
    days_target = days_flux

    # select lowerSurface and interpolate along numeric days
    da_draft = ds_state[var_draft]

    # If draft has a Time dim name other than 'Time', try to interpolate along the correct dim
    time_dim = None
    for d in da_draft.dims:
        if d.lower().startswith('time'):
            time_dim = d
            break
    if time_dim is None and da_draft.dims:
        # fallback to first dim that matches length of days_state
        for d in da_draft.dims:
            if da_draft.sizes[d] == days_state.size:
                time_dim = d
                break

    if time_dim is None:
        raise RuntimeError('Could not find time dimension in lowerSurface DataArray')

    # assign numeric coord under a consistent name and make it a dimension
    da_draft = da_draft.assign_coords({'daysSinceStart_num': (time_dim, days_state)})

    logger.info(f'Interpolating {var_draft} from state times ({days_state.size}) onto flux times ({days_target.size}) using method={method}')
    try:
        # swap the time_dim for a numeric dimension so interp accepts the indexer
        try:
            da_for_interp = da_draft.swap_dims({time_dim: 'daysSinceStart_num'})
        except Exception:
            # If swap_dims fails, fall back to using the original DataArray (interp may still work)
            da_for_interp = da_draft

        # prefer linear on numeric days; if it fails, fallback to nearest
        try:
            merged = da_for_interp.interp({'daysSinceStart_num': days_target}, method=method)
        except Exception:
            logger.warning('Primary interpolation failed; falling back to nearest')
            merged = da_for_interp.interp({'daysSinceStart_num': days_target}, method='nearest')

        # If we swapped dims, merged will have 'daysSinceStart_num' as a dim; map back to flux Time
        if 'daysSinceStart_num' in merged.dims and 'Time' in ds_flux.coords:
            try:
                merged = merged.assign_coords({'Time': ds_flux['Time'].values})
                # if desired, collapse daysSinceStart_num dim name to Time by swapping dims
                try:
                    merged = merged.swap_dims({'daysSinceStart_num': 'Time'})
                except Exception:
                    pass
            except Exception:
                logger.debug('Could not assign flux Time coords to merged result')

    except Exception:
        logger.exception('Unexpected error during interpolation of lowerSurface')
        raise

    # Ensure the merged DataArray uses the same time DIM as the flux
    # If interpolation produced a 'daysSinceStart_num' dimension, rename it
    # to the project's `Time` dim when lengths match and assign the flux Time coords.
    try:
        if 'daysSinceStart_num' in merged.dims:
            time_dim_name = config.TIME_DIM if hasattr(config, 'TIME_DIM') else 'Time'
            if time_dim_name in ds_flux.coords and merged.sizes['daysSinceStart_num'] == ds_flux.sizes[time_dim_name]:
                merged = merged.rename({'daysSinceStart_num': time_dim_name})
                merged = merged.assign_coords({time_dim_name: ds_flux[time_dim_name].values})
                logger.info(f"Renamed 'daysSinceStart_num' dim -> '{time_dim_name}' and assigned flux Time coords")
            else:
                # If lengths match the numeric target length, still rename to 'Time' for downstream consistency
                if merged.sizes['daysSinceStart_num'] == len(days_target):
                    merged = merged.rename({'daysSinceStart_num': time_dim_name})
                    try:
                        if time_dim_name in ds_flux.coords:
                            merged = merged.assign_coords({time_dim_name: ds_flux[time_dim_name].values})
                    except Exception:
                        logger.debug('Could not assign flux Time coords after renaming; leaving numeric days as coord values')
    except Exception:
        logger.debug('Failed to normalize merged time dimension; merged may still be along daysSinceStart_num')

    # Decide variable name to put into ds_flux
    out_varname = var_draft
    if out_varname in ds_flux and not overwrite:
        out_varname = var_draft + '_from_state'
        logger.info(f"Flux already contains '{var_draft}'; storing interpolated draft as '{out_varname}'")
    else:
        logger.info(f"Appending interpolated draft as '{out_varname}' (overwrite={overwrite})")

    # Insert into a copy of ds_flux
    out_ds = ds_flux.copy()
    # If merged still has the temporary numeric coord, drop it from coords (avoid duplicates)
    try:
        if 'daysSinceStart_num' in merged.coords:
            try:
                merged = merged.reset_coords('daysSinceStart_num', drop=True)
            except Exception:
                # fallback: attempt to delete directly
                try:
                    del merged.coords['daysSinceStart_num']
                except Exception:
                    pass
    except Exception:
        logger.debug('No daysSinceStart_num coord to remove from merged')

    # Align dims: if merged has dims (Time/nCells) and flux has (Time, nCells), transpose if necessary
    try:
        if set(merged.dims) <= set(out_ds.dims) and merged.dims != tuple(out_ds.dims):
            target_order = tuple(d for d in out_ds.dims if d in merged.dims)
            merged = merged.transpose(*target_order)
    except Exception:
        logger.debug('Could not transpose merged to match flux dims; proceeding')

    out_ds[out_varname] = merged

    # Ensure the dataset does not contain the temporary numeric coord as a top-level coord
    try:
        if 'daysSinceStart_num' in out_ds.coords:
            del out_ds.coords['daysSinceStart_num']
    except Exception:
        logger.debug('No top-level daysSinceStart_num coord to remove from output dataset')

    # Write output
    if outdir is None:
        outdir = flux_path.parent
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / f"{flux_path.stem}_with_state.nc"
    logger.info(f'Writing merged file to {out_path}')
    encoding = {out_varname: {'zlib': True, 'complevel': 4}}
    out_ds.to_netcdf(out_path, format='NETCDF4', engine='netcdf4', encoding=encoding)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description='Interpolate lowerSurface from state onto flux daysSinceStart and append into flux file')
    parser.add_argument('--flux', required=True, help='Path to flux NetCDF file')
    parser.add_argument('--state', required=True, help='Path to state NetCDF file')
    parser.add_argument('--outdir', default=None, help='Output directory (default: flux file dir)')
    parser.add_argument('--method', default='linear', choices=['linear', 'nearest'], help='Interpolation method for numeric days')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing lowerSurface in flux if present')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    merged_path = merge_state_lower_surface(args.flux, args.state, outdir=args.outdir, method=args.method, overwrite=args.overwrite)
    logger.info(f'Merged file saved: {merged_path}')


if __name__ == '__main__':
    main()
