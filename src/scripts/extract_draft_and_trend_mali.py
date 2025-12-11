#!/usr/bin/env python3
"""
Script: extract_draft_and_trend_mali.py

Purpose:
- For each MALI forcing scenario (default: config.FILE_ISMIP6_SSP126_FORCING and
  config.FILE_ISMIP6_SSP585_FORCING), clip by ice-shelf region masks and compute a
  simple linear draft-dependence (slope, intercept) per region.
- Remove the draft-dependent component from the full `floatingBasalMassBalApplied`
  time series (per region), merge region outputs into a full-nCells-by-Time field,
  and save a dedrafted NetCDF file (with `daysSinceStart` and `deltat` passed through).
- Extract the time-trend using the existing breakpoint detrending helper
  (`detrend_with_breakpoints_vectorized`) and save a trend NetCDF file per scenario.

Notes:
- This script uses the helper `dedraft_unstructured_region` added to `aislens.dataprep`.
- Output directory defaults to `config.DIR_MALI_FORCING_TRENDS`.

Usage examples:
    python src/scripts/extract_draft_and_trend_mali.py --scenarios 126 585

"""
from pathlib import Path
import argparse
import logging
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression

from aislens.config import config
from aislens.dataprep import dedraft_unstructured_region, detrend_with_breakpoints_vectorized

logger = logging.getLogger(__name__)


def process_scenario(flux_path, state_path, masks_path, out_dir, regions=None, stream=False):
    logger.info(f"Processing flux={flux_path} state={state_path}")

    ds_flux = xr.open_dataset(flux_path, chunks={config.TIME_DIM: 12})
    ds_state = xr.open_dataset(state_path, chunks={config.TIME_DIM: 12})
    ds_masks = xr.open_dataset(masks_path)

    # Reverse sign of the floatingBasalMassBalApplied variable in the flux
    # dataset before performing dedrafting (user-requested behavior).
    try:
        var_name = config.MALI_FLOATINGBMB_VAR
        if var_name in ds_flux:
            try:
                ds_flux[var_name] = -1 * ds_flux[var_name]
                logger.info(f"Reversed sign of {var_name} in ds_flux")
            except Exception:
                logger.exception(f"Failed to reverse sign of {var_name}; proceeding with original values")
        else:
            logger.warning(f"Variable {var_name} not found in ds_flux; cannot reverse sign")
    except Exception:
        logger.exception('Unexpected error while attempting to reverse flux sign')

    # Helper to coerce daysSinceStart-like arrays to numeric days (float)
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
            # use safe reduction (avoid nanmax on incompatible dtypes)
            try:
                mag = np.nanmax(np.abs(vals))
            except Exception:
                mag = None
            if mag is not None and mag > 1e12:
                # treat as nanoseconds
                return vals.astype('float64') / 1e9 / 86400.0
            else:
                return vals.astype('float64')
        if np.issubdtype(vals.dtype, np.floating):
            return vals.astype('float64')
        # object or unknown dtype: try elementwise conversion
        try:
            out = np.array([np.datetime64(v).astype('datetime64[s]').astype('int64')/86400.0 for v in vals])
            return out.astype('float64')
        except Exception:
            return vals.astype('float64')

    # Prefer aligning state -> flux using daysSinceStart if available
    try:
        if 'daysSinceStart' in ds_flux and 'daysSinceStart' in ds_state:
            # convert to numeric days for robust interpolation
            days_flux_num = _to_days(ds_flux['daysSinceStart'].values)
            days_state_num = _to_days(ds_state['daysSinceStart'].values)
            # assign numeric daysSinceStart as a coordinate along Time for interp
            ds_flux = ds_flux.assign_coords({'daysSinceStart_num': (config.TIME_DIM, days_flux_num)})
            ds_state = ds_state.assign_coords({'daysSinceStart_num': (config.TIME_DIM, days_state_num)})

            if days_flux_num.size != days_state_num.size or not np.allclose(days_flux_num, days_state_num):
                logger.info('Aligning state to flux using nearest interpolation on daysSinceStart (numeric)')
                try:
                    ds_state = ds_state.interp({'daysSinceStart_num': days_flux_num}, method='nearest')
                    # after interp, rename coord so later code can still look for 'daysSinceStart' if needed
                    ds_state = ds_state.assign_coords({'daysSinceStart': (config.TIME_DIM, days_flux_num)})
                except Exception:
                    logger.warning('Interpolation on numeric daysSinceStart failed; falling back to Time-based alignment')
                    raise RuntimeError('daysSinceStart interp failed')
    except RuntimeError:
        # Fallback to aligning on Time index if daysSinceStart-based alignment not possible
        if config.TIME_DIM in ds_flux.coords and config.TIME_DIM in ds_state.coords:
            t_flux = ds_flux.coords[config.TIME_DIM]
            t_state = ds_state.coords[config.TIME_DIM]
            if t_flux.size != t_state.size or not np.all(t_flux.values == t_state.values):
                logger.info("Aligning state Time to flux Time using nearest interpolation/reindex")
                try:
                    # Try interpolation (works for numeric/time coords)
                    ds_state = ds_state.interp({config.TIME_DIM: t_flux}, method='nearest')
                except Exception:
                    try:
                        # Fallback to reindex with nearest -- may be less flexible but works for many cases
                        ds_state = ds_state.reindex({config.TIME_DIM: t_flux}, method='nearest')
                    except Exception:
                        logger.warning("Could not align Time coords exactly; operations may fail if shapes differ")

    var_flux = config.MALI_FLOATINGBMB_VAR
    var_draft = 'lowerSurface'

    # Prepare output DataArray (same coords/dims as flux variable)
    flux_da = ds_flux[var_flux]
    # initialize with NaNs
    dedrafted = xr.full_like(flux_da, np.nan)

    # iterate requested regions
    if regions is None:
        # default to config ICE_SHELF_REGIONS
        regions = list(config.ICE_SHELF_REGIONS)

    region_coefs = []

    for i in regions:
        logger.info(f"  Region {i}: applying mask and fitting linear draft dependence")
        mask_da = ds_masks.regionCellMasks.isel(nRegions=i)
        # Ensure int/bool
        mask_bool = (mask_da == 1)

        # Apply mask to variables
        region_flux = flux_da.where(mask_bool)
        region_draft = ds_state[var_draft].where(mask_bool)

        # Fit slope/intercept and get predicted component for full time
        slope, intercept, predicted_full = dedraft_unstructured_region(region_flux, region_draft, mask_da, time_dim=config.TIME_DIM)

        logger.info(f"    slope={slope:.4g}, intercept={intercept:.4g}")
        region_coefs.append({'region': int(i), 'slope': float(slope), 'intercept': float(intercept)})

        # Align predicted_full to flux time axis if necessary (use daysSinceStart first)
        try:
            # prefer numeric days coord if present (we created 'daysSinceStart_num' above)
            if 'daysSinceStart_num' in ds_flux.coords:
                days_target = ds_flux['daysSinceStart_num'].values
                if 'daysSinceStart' in predicted_full.coords:
                    # convert predicted_full days to numeric
                    try:
                        pred_days = _to_days(predicted_full['daysSinceStart'].values)
                        predicted_full = predicted_full.assign_coords({'daysSinceStart_num': (config.TIME_DIM, pred_days)})
                    except Exception:
                        pass
                if 'daysSinceStart_num' in predicted_full.coords:
                    logger.info('    Aligning predicted component to flux via numeric daysSinceStart')
                    predicted_full = predicted_full.interp({'daysSinceStart_num': days_target}, method='nearest')
                elif config.TIME_DIM in ds_flux.coords and config.TIME_DIM in predicted_full.coords:
                    logger.info('    Aligning predicted component to flux via Time')
                    predicted_full = predicted_full.interp({config.TIME_DIM: ds_flux[config.TIME_DIM].values}, method='nearest')
            elif config.TIME_DIM in ds_flux.coords and config.TIME_DIM in predicted_full.coords:
                logger.info('    Aligning predicted component to flux via Time')
                predicted_full = predicted_full.interp({config.TIME_DIM: ds_flux[config.TIME_DIM].values}, method='nearest')
        except Exception:
            logger.warning('    Interpolation alignment of predicted component failed; attempting reindex')
            try:
                if 'daysSinceStart' in ds_flux and 'daysSinceStart' in predicted_full.coords:
                    predicted_full = predicted_full.reindex({'daysSinceStart': ds_flux['daysSinceStart'].values}, method='nearest')
                elif config.TIME_DIM in ds_flux.coords and config.TIME_DIM in predicted_full.coords:
                    predicted_full = predicted_full.reindex({config.TIME_DIM: ds_flux[config.TIME_DIM].values}, method='nearest')
            except Exception:
                logger.warning('    Reindex alignment also failed; proceeding and subtraction may error')

        # Ensure dims order matches region_flux (transpose if same dim names differ in order)
        try:
            if set(region_flux.dims) == set(predicted_full.dims) and region_flux.dims != predicted_full.dims:
                predicted_full = predicted_full.transpose(*region_flux.dims)
        except Exception:
            logger.debug('    Could not transpose predicted_full to match region_flux dims')

        # dedrafted region field = region_flux - predicted_full
        dedrafted_region = region_flux - predicted_full

        # merge into full-field (where mask==1, use dedrafted_region)
        dedrafted = xr.where(mask_bool, dedrafted_region, dedrafted)

    # For any cells not covered by any region (still NaN), fall back to original flux
    dedrafted = dedrafted.where(np.isfinite(dedrafted), other=flux_da)

    # Build Dataset to save: include daysSinceStart and deltat if available
    out_ds = xr.Dataset()
    out_ds[var_flux] = dedrafted

    # copy time coords and any existing variables
    for coord in ['Time']:
        if coord in ds_flux.coords:
            out_ds.coords[coord] = ds_flux.coords[coord]

    for attr_var in ['daysSinceStart', 'deltat']:
        if attr_var in ds_flux:
            out_ds[attr_var] = ds_flux[attr_var]

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    dedrafted_path = out_dir_path / f"dedrafted_{Path(flux_path).stem}.nc"
    logger.info(f"Saving dedrafted -> {dedrafted_path}")

    # Rechunk to sensible sizes for writing (reduce number of small writes)
    da = out_ds[var_flux]
    # Ensure dims order is (Time, nCells) for chunk calculation and optional streaming
    if config.TIME_DIM in da.dims and 'nCells' in da.dims and da.dims != (config.TIME_DIM, 'nCells'):
        da = da.transpose(config.TIME_DIM, 'nCells')

    time_len = da.sizes.get(config.TIME_DIM, None)
    nCells_len = da.sizes.get('nCells', None)

    # Choose chunk sizes (tunable)
    time_chunk = min(120, time_len) if time_len is not None else None
    nCells_chunk = min(4096, nCells_len) if nCells_len is not None else None

    chunk_kwargs = {}
    if time_chunk is not None:
        chunk_kwargs[config.TIME_DIM] = time_chunk
    if nCells_chunk is not None:
        chunk_kwargs['nCells'] = nCells_chunk

    # Rechunk if dask-backed or large
    try:
        if chunk_kwargs:
            da = da.chunk(chunk_kwargs)
    except Exception:
        logger.debug('Could not rechunk dedrafted DataArray; proceeding without rechunk')

    # Build encoding for efficient netCDF writing (keep same precision)
    chunksizes = None
    if time_chunk is not None and nCells_chunk is not None:
        chunksizes = (time_chunk, nCells_chunk)

    encoding = {
        var_flux: {
            'zlib': True,
            'complevel': 4,
            **({'chunksizes': chunksizes} if chunksizes is not None else {})
        }
    }

    # If streaming mode requested, write by time-slice using netCDF4 low-level API
    if stream:
        logger.info('Streaming write enabled: writing dedrafted file by time-slice')
        try:
            import netCDF4 as nc

            # Ensure da is in (Time, nCells) order for streaming
            if config.TIME_DIM in da.dims and 'nCells' in da.dims and da.dims != (config.TIME_DIM, 'nCells'):
                da = da.transpose(config.TIME_DIM, 'nCells')

            time_dim_len = da.sizes[config.TIME_DIM]
            nCells_dim_len = da.sizes['nCells']

            with nc.Dataset(str(dedrafted_path), 'w', format='NETCDF4') as dsout:
                dsout.createDimension(config.TIME_DIM, time_dim_len)
                dsout.createDimension('nCells', nCells_dim_len)

                # create coordinate variable for Time if coordinate exists as numeric days or similar
                if 'daysSinceStart' in out_ds:
                    # create daysSinceStart variable
                    dsout.createVariable('daysSinceStart', out_ds['daysSinceStart'].dtype.str, (config.TIME_DIM,))
                    dsout.variables['daysSinceStart'][:] = out_ds['daysSinceStart'].values

                var_out = dsout.createVariable(var_flux, 'f8', (config.TIME_DIM, 'nCells'), zlib=True, complevel=4)

                # Write by time-slice
                for ti in range(time_dim_len):
                    arr = da.isel({config.TIME_DIM: ti}).values
                    # ensure numpy array
                    if hasattr(arr, 'compute'):
                        arr = arr.compute()
                    var_out[ti, :] = arr
                    if (ti + 1) % 10 == 0:
                        dsout.sync()

        except Exception:
            logger.exception('Streaming write failed; falling back to xarray to_netcdf')
            out_ds[var_flux] = da
            out_ds.to_netcdf(dedrafted_path, format='NETCDF4', engine='netcdf4', encoding=encoding)
    else:
        # Non-stream path: attach rechunked DataArray back and write with encoding
        out_ds[var_flux] = da
        out_ds.to_netcdf(dedrafted_path, format='NETCDF4', engine='netcdf4', encoding=encoding)

    # Trend extraction using existing helper
    logger.info("Computing trend via breakpoint detrend (vectorized)")
    # detrend_with_breakpoints_vectorized expects a DataArray
    # Ensure the core time dimension is a single dask chunk as required by
    # xarray.apply_ufunc when used with dask='parallelized'. If rechunking
    # fails, attempt to compute into memory as a fallback (may be large).
    try:
        if config.TIME_DIM in dedrafted.dims:
            time_len = dedrafted.sizes.get(config.TIME_DIM, None)
            nCells_len = dedrafted.sizes.get('nCells', None)
            chunk_dict = {config.TIME_DIM: -1}
            # keep a reasonable chunk for nCells if present
            if nCells_len is not None:
                chunk_dict['nCells'] = min(4096, nCells_len)
            logger.info(f'Rechunking dedrafted for detrending with chunks={chunk_dict}')
            try:
                dedrafted = dedrafted.chunk(chunk_dict)
            except Exception:
                logger.warning('Rechunk failed; attempting to compute dedrafted into memory for detrending')
                dedrafted = dedrafted.compute()
                logger.info('Computed dedrafted into memory')
    except Exception:
        logger.exception('Unexpected error while preparing dedrafted for detrending; proceeding to detrend may fail')

    detrended = detrend_with_breakpoints_vectorized(dedrafted, dim=config.TIME_DIM, deg=1, model='rbf', penalty=10)

    trend_da = dedrafted - detrended
    trend_ds = xr.Dataset({var_flux: trend_da})
    for attr_var in ['daysSinceStart', 'deltat']:
        if attr_var in ds_flux:
            trend_ds[attr_var] = ds_flux[attr_var]

    trend_path = out_dir_path / f"trend_{Path(flux_path).stem}.nc"
    logger.info(f"Saving trend -> {trend_path}")
    trend_ds.to_netcdf(trend_path)

    return {'dedrafted': str(dedrafted_path), 'trend': str(trend_path), 'coefs': region_coefs}


def main():
    parser = argparse.ArgumentParser(description='Extract region-wise draft dependence and trend for MALI forcing scenarios')
    parser.add_argument('--scenarios', nargs='+', choices=['126', '585'], default=['126','585'], help='Which scenarios to process')
    parser.add_argument('--masks', type=str, default=str(config.DATA_ROOT / 'data' / 'external' / 'aislens_draftDepen_regionMasks.nc'), help='Path to regionCellMasks NetCDF')
    parser.add_argument('--outdir', type=str, default=str(config.DIR_MALI_FORCING_TRENDS), help='Output directory')
    parser.add_argument('--stream', action='store_true', help='Stream write dedrafted output by time-slice (lower memory, slower CPU)')
    parser.add_argument('--regions', nargs='*', type=int, default=None, help='List of region indices (e.g. 33 34 35)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    mapping = {
        '126': (str(config.FILE_ISMIP6_SSP126_FORCING), None),
        '585': (str(config.FILE_ISMIP6_SSP585_FORCING), None),
    }

    results = {}
    for sc in args.scenarios:
        flux_path = mapping[sc][0]
        # derive state path from flux filename by replacing floatingBMB with state equivalent if present in same folder
        # User-provided layout: the state files are in the same folder with name extracted_state_* matching the forcing
        # Try a simple guess: replace 'floatingBasalMassBalApplied' in path with 'state' patterns used elsewhere.
        # Fallback: require user to place matching state files next to the flux files with standard naming.
        # For now, attempt to infer the state file by replacing 'floatingBasalMassBalApplied' with 'state' substring patterns.
        state_guess = flux_path.replace('_flux_', '_state_')
        # if guessed file does not exist, look for nearby file pattern by swapping 'flux' <-> 'state' in common naming
        # Fall back to using the same directory with 'extracted_state' naming
        try:
            state_path = state_guess
        except Exception:
            state_path = flux_path

        # Best effort: check existence and try alternative name
        # A conservative guess used in notebooks: files named 'extracted_state_expAE05_2015-2300.nc'
        flux_p = Path(flux_path)
        alt_state = flux_p.parent.parent / 'draft-depen' / ("extracted_state_" + flux_p.name.split('extracted_flux_')[-1])
        if not Path(state_path).exists() and alt_state.exists():
            state_path = str(alt_state)

        if not Path(state_path).exists():
            logger.warning(f"State file for scenario {sc} not found at {state_path}; trying to proceed but results may be incorrect")

        out = process_scenario(flux_path, state_path, args.masks, args.outdir, regions=args.regions, stream=args.stream)
        results[sc] = out

    logger.info("All scenarios processed")
    for sc, info in results.items():
        logger.info(f"Scenario {sc}: dedrafted={info['dedrafted']} trend={info['trend']}")


if __name__ == '__main__':
    main()
