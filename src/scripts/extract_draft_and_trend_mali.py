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
import os

from aislens.config import config
from aislens.dataprep import dedraft_unstructured_region, detrend_with_breakpoints_vectorized

logger = logging.getLogger(__name__)


def process_scenario(flux_path, state_path, masks_path, out_dir, regions=None, stream=False):
    logger.info(f"Processing flux={flux_path} state={state_path}")

    # Debug: log existence and basic info for input/output paths before opening files
    try:
        flux_p = Path(flux_path)
        state_p = Path(state_path)
        masks_p = Path(masks_path)
        out_p = Path(out_dir)

        def _exists_info(p):
            if p.exists():
                try:
                    return f"exists=True size={p.stat().st_size} bytes"
                except Exception:
                    return "exists=True size=unknown"
            return "exists=False"

        logger.info(f"INPUT CHECK: flux: {flux_path} -> {_exists_info(flux_p)}")
        logger.info(f"INPUT CHECK: state: {state_path} -> {_exists_info(state_p)}")
        logger.info(f"INPUT CHECK: masks: {masks_path} -> {_exists_info(masks_p)}")
        writability = None
        if out_p.exists():
            writability = os.access(str(out_p), os.W_OK)
        else:
            # parent directory writable check
            parent = out_p.parent if out_p.parent.exists() else Path('.')
            writability = os.access(str(parent), os.W_OK)
        logger.info(f"OUTPUT CHECK: outdir: {out_dir} -> exists={out_p.exists()} writable_parent={writability}")
    except Exception:
        logger.exception('Failed to stat input/output paths for debug logging')

    ds_flux = xr.open_dataset(flux_path, chunks={config.TIME_DIM: 12})
    ds_state = xr.open_dataset(state_path, chunks={config.TIME_DIM: 12})
    ds_masks = xr.open_dataset(masks_path)

    # Helper to derive a numeric daysSinceStart coordinate for a dataset
    def _ensure_days_num_global(ds):
        # If already present, return
        if 'daysSinceStart_num' in ds.coords:
            return ds
        # Prefer an explicit daysSinceStart coord
        try:
            if 'daysSinceStart' in ds.coords:
                days_vals = _to_days(ds['daysSinceStart'].values)
                return ds.assign_coords({'daysSinceStart_num': (config.TIME_DIM, days_vals)})
        except Exception:
            logger.debug('Could not convert existing daysSinceStart to numeric')
        # Else try to derive from deltat
        try:
            if 'deltat' in ds:
                delta_days = _to_days(ds['deltat'].values)
                cumsum = np.cumsum(delta_days)
                cumsum = cumsum - cumsum[0]
                return ds.assign_coords({'daysSinceStart_num': (config.TIME_DIM, cumsum)})
        except Exception:
            logger.debug('Could not derive daysSinceStart_num from deltat')
        # Finally, try to derive from Time coordinate (datetime64 or numeric)
        try:
            if config.TIME_DIM in ds.coords:
                tvals = ds[config.TIME_DIM].values
                tdays = _to_days(tvals)
                tdays = tdays - tdays[0]
                return ds.assign_coords({'daysSinceStart_num': (config.TIME_DIM, tdays)})
        except Exception:
            logger.debug('Could not derive daysSinceStart_num from Time coord')
        return ds

    # Create numeric daysSinceStart coordinate for both flux and state (global)
    try:
        ds_flux = _ensure_days_num_global(ds_flux)
    except Exception:
        logger.debug('Failed to ensure numeric days on ds_flux')
    try:
        ds_state = _ensure_days_num_global(ds_state)
    except Exception:
        logger.debug('Failed to ensure numeric days on ds_state')

    # (Sign reversal will be applied after `var_flux` is defined below.)

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

    # Reverse sign of the flux variable (straightforward negation).
    ds_flux[var_flux] = -ds_flux[var_flux]
    logger.info(f'Reversed sign of flux variable: {var_flux}')

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
        # Pass the canonical ds_flux Time coord as `target_time` so the helper
        # can produce a predicted_full that is already aligned to the flux Time
        # axis (this avoids later xarray index-identity alignment errors).
        slope, intercept, predicted_full = dedraft_unstructured_region(
            region_flux, region_draft, mask_da, time_dim=config.TIME_DIM,
            target_time=ds_flux.coords.get(config.TIME_DIM, None)
        )

        logger.info(f"    slope={slope:.4g}, intercept={intercept:.4g}")
        region_coefs.append({'region': int(i), 'slope': float(slope), 'intercept': float(intercept)})

        # Align predicted_full to flux time axis using the simpler, robust strategy
        # (match behavior from the older working script):
        # - If the main dataset (`ds_flux`) has a numeric `daysSinceStart_num` coord,
        #   prefer interpolating `predicted_full` on that coord (if `predicted_full` has a matching coord).
        # - Otherwise, fall back to interpolating/reindexing on `Time` if both have it.
        try:
            if 'daysSinceStart_num' in ds_flux.coords:
                # target numeric days (from ds_flux)
                days_target = ds_flux['daysSinceStart_num'].values

                # Try several ways to derive numeric days for predicted_full.
                pred_days_assigned = False
                # 1) If predicted_full has an explicit daysSinceStart coord
                try:
                    if 'daysSinceStart' in predicted_full.coords:
                        pred_days = _to_days(predicted_full['daysSinceStart'].values)
                        predicted_full = predicted_full.assign_coords({'daysSinceStart_num': (config.TIME_DIM, pred_days)})
                        pred_days_assigned = True
                except Exception:
                    logger.debug('Could not convert predicted_full daysSinceStart to numeric')

                # 2) If predicted_full has a Time coord (maybe integer/datetime)
                try:
                    if (not pred_days_assigned) and config.TIME_DIM in predicted_full.coords:
                        tvals = predicted_full[config.TIME_DIM].values
                        # If Time coord already numeric/datetime-like, convert
                        try:
                            pred_days = _to_days(tvals)
                            predicted_full = predicted_full.assign_coords({'daysSinceStart_num': (config.TIME_DIM, pred_days)})
                            pred_days_assigned = True
                        except Exception:
                            # If Time coord is simple integer index (0..N-1), map it across the flux days range
                            try:
                                if np.issubdtype(np.asarray(tvals).dtype, np.integer) and 'daysSinceStart_num' in ds_flux.coords:
                                    ft = ds_flux['daysSinceStart_num'].values
                                    nl = predicted_full.sizes.get(config.TIME_DIM)
                                    if nl and ft.size >= 2:
                                        pred_days = np.linspace(float(ft[0]), float(ft[-1]), num=nl)
                                        predicted_full = predicted_full.assign_coords({'daysSinceStart_num': (config.TIME_DIM, pred_days)})
                                        pred_days_assigned = True
                                        logger.debug('Mapped integer predicted_full Time index linearly over flux days range')
                            except Exception:
                                logger.debug('Could not map integer Time coord for predicted_full')
                except Exception:
                    logger.debug('Could not convert predicted_full Time coord to numeric days')

                # 3) If predicted_full has deltat, derive cumulative days
                try:
                    if (not pred_days_assigned) and 'deltat' in predicted_full.coords:
                        pd = _to_days(predicted_full['deltat'].values)
                        pcumsum = np.cumsum(pd)
                        pcumsum = pcumsum - pcumsum[0]
                        predicted_full = predicted_full.assign_coords({'daysSinceStart_num': (config.TIME_DIM, pcumsum)})
                        pred_days_assigned = True
                except Exception:
                    logger.debug('Could not derive predicted_full daysSinceStart_num from deltat')

                # If we now have numeric days on predicted_full, interpolate to the flux numeric days
                if 'daysSinceStart_num' in predicted_full.coords:
                    try:
                        logger.info('    Aligning predicted component to flux via numeric daysSinceStart')
                        predicted_full = predicted_full.interp({'daysSinceStart_num': days_target}, method='nearest')
                    except Exception as e:
                        logger.debug(f'    numeric days interp failed for predicted_full: {e!r}')
                else:
                    # Fall back to Time-based interpolation if both have Time coords
                    if config.TIME_DIM in ds_flux.coords and config.TIME_DIM in predicted_full.coords:
                        try:
                            logger.info('    Aligning predicted component to flux via Time (fallback)')
                            predicted_full = predicted_full.interp({config.TIME_DIM: ds_flux[config.TIME_DIM].values}, method='nearest')
                        except Exception:
                            logger.debug('    Time-based interp of predicted_full failed')
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

        # After attempts, verify Time lengths match the flux; if not, try a final reindex to target Time
        try:
            if config.TIME_DIM in ds_flux.coords and config.TIME_DIM in predicted_full.coords:
                t_flux = ds_flux[config.TIME_DIM].values
                t_pred = predicted_full[config.TIME_DIM].values
                if t_flux.size != t_pred.size or not np.array_equal(t_flux, t_pred):
                    logger.warning('    Predicted component Time index still differs from flux; forcing reindex with nearest/fill')
                    predicted_full = predicted_full.reindex({config.TIME_DIM: t_flux}, method='nearest', fill_value=np.nan)
        except Exception:
            logger.exception('    Final reindex of predicted component failed; subtraction may raise an error')

        # Ensure predicted_full explicitly uses the same Time coordinate object as ds_flux
        try:
            if config.TIME_DIM in predicted_full.dims and config.TIME_DIM in ds_flux.coords:
                # If lengths match, assign the flux Time values (this makes indexes identical)
                if predicted_full.sizes.get(config.TIME_DIM, 0) == ds_flux.sizes.get(config.TIME_DIM, 0):
                    predicted_full = predicted_full.assign_coords({config.TIME_DIM: ds_flux[config.TIME_DIM].values})
                else:
                    # Force reindex as a last resort (should have been handled above)
                    predicted_full = predicted_full.reindex({config.TIME_DIM: ds_flux[config.TIME_DIM].values}, method='nearest', fill_value=np.nan)
        except Exception:
            logger.exception('    Could not force predicted_full Time coords to match ds_flux')

        # Ensure predicted_full has a Time-like dimension with the same name/length as ds_flux
        try:
            if config.TIME_DIM in ds_flux.coords:
                t_flux = ds_flux[config.TIME_DIM].values
                # If predicted has a different dim whose length equals the flux Time length,
                # rename it to the canonical Time dim and assign the flux Time values so
                # the index objects are identical for downstream alignment.
                renamed = False
                for d in predicted_full.dims:
                    if predicted_full.sizes.get(d, None) == ds_flux.sizes.get(config.TIME_DIM, None):
                        if d != config.TIME_DIM:
                            try:
                                predicted_full = predicted_full.rename({d: config.TIME_DIM})
                                logger.info(f"    Renamed predicted dim '{d}' -> '{config.TIME_DIM}' to match flux Time length")
                                renamed = True
                            except Exception:
                                logger.debug(f"    Could not rename dim {d} to {config.TIME_DIM}")
                        # assign the exact flux Time coordinate values (makes index equality)
                        try:
                            predicted_full = predicted_full.assign_coords({config.TIME_DIM: t_flux})
                            logger.info('    Assigned flux Time coordinate values to predicted component')
                        except Exception:
                            logger.debug('    Could not assign flux Time values to predicted component')
                        break
        except Exception:
            logger.exception('    Error while coercing predicted_full to flux Time dim')

        # Ensure dims order matches region_flux (transpose if same dim names differ in order)
        try:
            if set(region_flux.dims) == set(predicted_full.dims) and region_flux.dims != predicted_full.dims:
                predicted_full = predicted_full.transpose(*region_flux.dims)
        except Exception:
            logger.debug('    Could not transpose predicted_full to match region_flux dims')

        # dedrafted region field = region_flux - predicted_full
        dedrafted_region = region_flux - predicted_full

        # Diagnostic: log Time-index info if alignment problems persist
        try:
            if logger.isEnabledFor(logging.DEBUG):
                def _sample_info(da, name):
                    info = {'dims': da.dims}
                    if config.TIME_DIM in da.coords:
                        info['time_len'] = da.sizes.get(config.TIME_DIM)
                        tvals = da.coords[config.TIME_DIM].values
                        info['time_dtype'] = getattr(tvals, 'dtype', type(tvals))
                        try:
                            info['time_head'] = tvals[:3].tolist()
                        except Exception:
                            info['time_head'] = str(tvals[:3])
                    logger.debug(f"    TIME DIAG ({name}): {info}")
                _sample_info(region_flux, 'region_flux')
                _sample_info(predicted_full, 'predicted_full')
                _sample_info(dedrafted, 'dedrafted')
        except Exception:
            logger.debug('    Could not produce time-diags for region arrays')

        # Ensure dedrafted_region uses the exact same Time index object as ds_flux
        # This prevents xarray from trying to align differing index objects when
        # performing xr.where later.
        try:
            if config.TIME_DIM in dedrafted_region.dims and config.TIME_DIM in ds_flux.coords:
                if dedrafted_region.sizes.get(config.TIME_DIM) == ds_flux.sizes.get(config.TIME_DIM):
                    # assign the exact Index object (not just values) from ds_flux
                    dedrafted_region = dedrafted_region.assign_coords({config.TIME_DIM: ds_flux.coords[config.TIME_DIM]})
                    logger.debug('    Assigned ds_flux Time index object to dedrafted_region')
        except Exception:
            logger.debug('    Could not assign ds_flux Time index to dedrafted_region')

        # Before merging, log and (if necessary) force exact index equality
        try:
            if logger.isEnabledFor(logging.DEBUG):
                tf = ds_flux.coords.get(config.TIME_DIM, None)
                tr = dedrafted_region.coords.get(config.TIME_DIM, None)
                td = dedrafted.coords.get(config.TIME_DIM, None)
                logger.debug(f"    Pre-merge Time index objects: ds_flux Time repr={repr(tf)}")
                logger.debug(f"    Pre-merge Time index objects: dedrafted_region Time repr={repr(tr)}")
                logger.debug(f"    Pre-merge Time index objects: dedrafted Time repr={repr(td)}")
                try:
                    identical_rr = (tr is not None and tf is not None and tr.identical(tf))
                except Exception:
                    identical_rr = False
                try:
                    identical_dd = (td is not None and tf is not None and td.identical(tf))
                except Exception:
                    identical_dd = False
                logger.debug(f"    Time index identical check: dedrafted_region==ds_flux: {identical_rr}, dedrafted==ds_flux: {identical_dd}")
        except Exception:
            logger.debug('    Could not produce pre-merge index introspection')

        # If any index isn't identical to ds_flux's Time index, force-assign the ds_flux
        try:
            if config.TIME_DIM in ds_flux.coords:
                # Prefer assigning the actual Index object (ensures .identical() succeeds)
                tf_index = None
                try:
                    tf_index = ds_flux.indexes.get(config.TIME_DIM, None)
                except Exception:
                    tf_index = None

                # Fall back to the DataArray coord values if Index not available
                tf_coord = ds_flux.coords.get(config.TIME_DIM, None)

                # Assign index object when possible (this creates true identity)
                if tf_index is not None:
                    try:
                        if config.TIME_DIM in dedrafted_region.coords and not dedrafted_region.coords[config.TIME_DIM].identical(tf_index):
                            dedrafted_region = dedrafted_region.assign_coords({config.TIME_DIM: tf_index})
                            logger.debug('    Assigned ds_flux Index object to dedrafted_region Time coord')
                    except Exception:
                        logger.debug('    Could not assign ds_flux Index object to dedrafted_region')
                    try:
                        if config.TIME_DIM in dedrafted.coords and not dedrafted.coords[config.TIME_DIM].identical(tf_index):
                            dedrafted = dedrafted.assign_coords({config.TIME_DIM: tf_index})
                            logger.debug('    Assigned ds_flux Index object to dedrafted Time coord')
                    except Exception:
                        logger.debug('    Could not assign ds_flux Index object to dedrafted')
                elif tf_coord is not None:
                    # Fallback: assign the coordinate values (may not create identical Index objects,
                    # but often sufficient when values and dtypes match)
                    try:
                        if config.TIME_DIM in dedrafted_region.coords and not dedrafted_region.coords[config.TIME_DIM].identical(tf_coord):
                            dedrafted_region = dedrafted_region.assign_coords({config.TIME_DIM: tf_coord.values})
                            logger.debug('    Assigned ds_flux Time values to dedrafted_region Time coord (fallback)')
                    except Exception:
                        logger.debug('    Could not assign ds_flux Time values to dedrafted_region')
                    try:
                        if config.TIME_DIM in dedrafted.coords and not dedrafted.coords[config.TIME_DIM].identical(tf_coord):
                            dedrafted = dedrafted.assign_coords({config.TIME_DIM: tf_coord.values})
                            logger.debug('    Assigned ds_flux Time values to dedrafted Time coord (fallback)')
                    except Exception:
                        logger.debug('    Could not assign ds_flux Time values to dedrafted')
        except Exception:
            logger.debug('    Could not force assign ds_flux Time index to arrays; proceeding to xr.where')

        # Ensure dedrafted_region is indexed like the target `dedrafted` before merging.
        # `reindex_like` will align coordinates (and create missing values) so that
        # xarray won't attempt deep alignment during `xr.where` and raise index errors.
        try:
            # reindex time and other coordinates to match dedrafted
            dedrafted_region = dedrafted_region.reindex_like(dedrafted)
            logger.debug('    Reindexed dedrafted_region to match dedrafted (pre-merge)')
        except Exception:
            logger.debug('    reindex_like failed for dedrafted_region; proceeding to xr.where (may error)')

        # Also ensure the boolean mask uses the same Time index (if it has one).
        try:
            if config.TIME_DIM in mask_bool.coords:
                try:
                    mask_bool = mask_bool.reindex_like(dedrafted, method='nearest', fill_value=False)
                    logger.debug('    Reindexed mask_bool to match dedrafted (pre-merge)')
                except Exception:
                    # Fallback: assign the ds_flux Time index object or values
                    try:
                        tf_idx = ds_flux.indexes.get(config.TIME_DIM, None)
                        if tf_idx is not None:
                            mask_bool = mask_bool.assign_coords({config.TIME_DIM: tf_idx})
                            logger.debug('    Assigned ds_flux Time index object to mask_bool (fallback)')
                        else:
                            mask_bool = mask_bool.assign_coords({config.TIME_DIM: ds_flux.coords[config.TIME_DIM].values})
                            logger.debug('    Assigned ds_flux Time values to mask_bool (fallback)')
                    except Exception:
                        logger.debug('    Could not align mask_bool Time coord; proceeding (may error)')
        except Exception:
            logger.debug('    Could not check/reindex mask_bool; proceeding to xr.where')

        # Merge into full-field (where mask==1, use dedrafted_region).
        # Build an explicit boolean mask with identical coords to `dedrafted` to avoid
        # xarray deep-align/index-equality issues. We use numpy broadcasting to
        # create a (Time, nCells) mask that shares the exact Time index from ds_flux.
        try:
            # If mask_bool lacks the Time dim, expand it explicitly across Time.
            if config.TIME_DIM not in mask_bool.dims:
                time_vals = ds_flux[config.TIME_DIM].values
                ntime = int(ds_flux.sizes.get(config.TIME_DIM, 0))
                # ensure mask_bool has nCells dim
                if 'nCells' in mask_bool.dims:
                    mask_vals = mask_bool.values
                    # broadcast to (Time, nCells)
                    mask_2d = np.broadcast_to(mask_vals, (ntime, mask_vals.shape[-1]))
                    mask_expanded = xr.DataArray(mask_2d, coords={config.TIME_DIM: time_vals, 'nCells': ds_flux['nCells'].values}, dims=(config.TIME_DIM, 'nCells'))
                else:
                    # unexpected shape; try to broadcast using dedrafted's nCells
                    mask_vals = mask_bool.values
                    mask_2d = np.broadcast_to(mask_vals, (ntime, dedrafted.sizes.get('nCells')))
                    mask_expanded = xr.DataArray(mask_2d, coords={config.TIME_DIM: time_vals, 'nCells': ds_flux['nCells'].values}, dims=(config.TIME_DIM, 'nCells'))
            else:
                # If mask already has Time dim, reindex/assign exact Time index from ds_flux
                try:
                    mask_expanded = mask_bool.reindex({config.TIME_DIM: ds_flux[config.TIME_DIM].values}, method='nearest', fill_value=False)
                except Exception:
                    try:
                        mask_expanded = mask_bool.assign_coords({config.TIME_DIM: ds_flux.indexes.get(config.TIME_DIM, ds_flux[config.TIME_DIM].values)})
                    except Exception:
                        mask_expanded = mask_bool

            # Ensure dims order matches (Time, nCells)
            if set(mask_expanded.dims) == set(dedrafted.dims) and mask_expanded.dims != dedrafted.dims:
                mask_expanded = mask_expanded.transpose(*dedrafted.dims)

            dedrafted = xr.where(mask_expanded, dedrafted_region, dedrafted)
            logger.debug('    Merged dedrafted_region into dedrafted using explicit expanded mask')
        except Exception:
            logger.exception('    Failed to merge dedrafted_region via explicit mask; attempting xr.where fallback')
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

                logger.info(f'Finished streaming write of dedrafted file: {dedrafted_path}')

        except Exception:
            logger.exception('Streaming write failed; falling back to xarray to_netcdf')
            out_ds[var_flux] = da
            out_ds.to_netcdf(dedrafted_path, format='NETCDF4', engine='netcdf4', encoding=encoding)
    else:
        # Non-stream path: attach rechunked DataArray back and write with encoding
        out_ds[var_flux] = da
        out_ds.to_netcdf(dedrafted_path, format='NETCDF4', engine='netcdf4', encoding=encoding)
    logger.info(f'Wrote dedrafted file: {dedrafted_path}')

    # Trend extraction using existing helper
    logger.info("Computing trend via breakpoint detrend (vectorized)")
    # detrend_with_breakpoints_vectorized expects a DataArray
    # Ensure the Time dimension is a single chunk before detrending (dask-aware).
    if config.TIME_DIM in dedrafted.dims:
        logger.info(f'Chunking dedrafted for detrending: {config.TIME_DIM} -> single chunk')
        dedrafted = dedrafted.chunk({config.TIME_DIM: -1})

    detrended = detrend_with_breakpoints_vectorized(dedrafted, dim=config.TIME_DIM, deg=1, model='rbf', penalty=10)

    trend_da = dedrafted - detrended
    trend_ds = xr.Dataset({var_flux: trend_da})
    for attr_var in ['daysSinceStart', 'deltat']:
        if attr_var in ds_flux:
            trend_ds[attr_var] = ds_flux[attr_var]

    trend_path = out_dir_path / f"trend_{Path(flux_path).stem}.nc"
    logger.info(f"Saving trend -> {trend_path}")
    # Write trend using dask-friendly chunking and compact encoding (float32).
    time_len = int(trend_ds.sizes.get(config.TIME_DIM, 1))
    nCells_len = int(trend_ds.sizes.get('nCells', 1)) if 'nCells' in trend_ds.dims else None
    trend_ds = trend_ds.chunk({config.TIME_DIM: -1})
    chunksizes = (time_len, nCells_len) if nCells_len is not None else (time_len,)
    trend_encoding = {
        var_flux: {
            'zlib': True,
            'complevel': 4,
            'dtype': 'float32',
            'chunksizes': chunksizes
        }
    }
    trend_ds.to_netcdf(trend_path, format='NETCDF4', engine='netcdf4', encoding=trend_encoding)
    logger.info(f'Wrote trend file: {trend_path}')

    return {'dedrafted': str(dedrafted_path), 'trend': str(trend_path), 'coefs': region_coefs}


def main():
    parser = argparse.ArgumentParser(description='Extract region-wise draft dependence and trend for MALI forcing scenarios')
    parser.add_argument('--scenarios', nargs='+', choices=['126', '585'], default=['126','585'], help='Which scenarios to process')
    parser.add_argument('--masks', type=str, default=str(config.DATA_ROOT / 'data' / 'external' / 'aislens_draftDepen_regionMasks.nc'), help='Path to regionCellMasks NetCDF')
    parser.add_argument('--outdir', type=str, default=str(config.DIR_MALI_FORCING_TRENDS), help='Output directory')
    parser.add_argument('--stream', action='store_true', help='Stream write dedrafted output by time-slice (lower memory, slower CPU)')
    parser.add_argument('--regions', nargs='*', type=int, default=None, help='List of region indices (e.g. 33 34 35)')
    parser.add_argument('--region-start', type=int, default=None, help='Start index for a contiguous region range (inclusive)')
    parser.add_argument('--region-end', type=int, default=None, help='End index for a contiguous region range (inclusive)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')
    args = parser.parse_args()

    # configure logging from CLI flag
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level)

    mapping = {
        # mapping: scenario -> (flux_path, state_path)
        '126': (str(config.FILE_ISMIP6_SSP126_FORCING), str(config.FILE_ISMIP6_SSP126_zDRAFT)),
        '585': (str(config.FILE_ISMIP6_SSP585_FORCING), str(config.FILE_ISMIP6_SSP585_zDRAFT)),
    }

    results = {}
    for sc in args.scenarios:
        flux_path, state_path = mapping[sc]
        # Prefer the configured state_path; warn if it's missing so the user can fix it.
        if not Path(state_path).exists():
            logger.warning(f"Configured state file for scenario {sc} not found at {state_path}; results may be incorrect. Please verify the path.")

        # Interpret region selection: explicit list takes precedence, otherwise
        # use a start/end range if provided; if neither is given, process the
        # default regions from config inside the function.
        regions_arg = None
        if args.regions is not None and len(args.regions) > 0:
            regions_arg = args.regions
        elif args.region_start is not None or args.region_end is not None:
            if args.region_start is None or args.region_end is None:
                raise SystemExit('Both --region-start and --region-end must be provided to use a range')
            if args.region_end < args.region_start:
                raise SystemExit('--region-end must be >= --region-start')
            regions_arg = list(range(args.region_start, args.region_end + 1))

        logger.info(f'Processing regions: {regions_arg if regions_arg is not None else "(default from config)"}')
        out = process_scenario(flux_path, state_path, args.masks, args.outdir, regions=regions_arg, stream=args.stream)
        results[sc] = out

    logger.info("All scenarios processed")
    for sc, info in results.items():
        logger.info(f"Scenario {sc}: dedrafted={info['dedrafted']} trend={info['trend']}")


if __name__ == '__main__':
    main()
