#!/usr/bin/env python3
"""
Debug workflow: stepwise prepare + diagnostics for model simulation data.

This script performs a clear, debuggable sequence of steps:
  - load model and subset by years or indices
  - compute polynomial trend fit and detrended dataset (save)
  - deseasonalize detrended data and save seasonality (save)
  - compute time-mean of deseasonalized data and run per-catchment dedraft
  - merge per-catchment predictions (save)
  - compute variability (deseasonalized - draft_pred) and also residual = original - (trend + seasonality + draft_pred)
  - save variability & residual
  - compute spatial-mean time-series for all components and plot them together
  - compute simple summary stats and write JSON

Usage:
    python debug_prepare_workflow.py [--start-year YYYY] [--end-year YYYY]
                                     [--start-index SI] [--end-index EI]
                                     [--coarsen N]
                                     [--output-dir OUTPUT_DIR]
                                     [--draft-dir DRAFT_DIR]
                                     [--precomputed-mean PRECOMPUTED_MEAN.nc]
                                     [--skip-dedraft]
"""

import argparse
import logging
import json
from pathlib import Path
from time import time

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt

from aislens.dataprep import detrend_dim, deseasonalize, dedraft_catchment
from aislens.utils import (
    merge_catchment_files,
    subset_dataset_by_time,
    write_crs,
    setup_logging,
    ensure_dataset_for_var,
)
from aislens.config import config

logger = logging.getLogger(__name__)


def compute_polyfit_and_detrend(da, time_dim):
    """Compute polynomial fit (deg=1) and detrended array while returning the fit.

    Returns: fit (DataArray, same dims as da), detrended (DataArray)
    """
    # fit coefficients
    p = da.polyfit(dim=time_dim, deg=1)
    fit = xr.polyval(da[time_dim], p.polyfit_coefficients)
    # preserve original mean per-pixel
    original_mean = da.mean(dim=time_dim)
    detrended = da - fit + original_mean
    return fit, detrended


def spatial_mean_timeseries(da, dims=('x', 'y')):
    # returns DataArray with Time dim if present
    if all(d in da.dims for d in dims):
        return da.mean(dim=dims)
    else:
        # no spatial dims -> return mean scalar broadcast to Time if time exists
        if 'Time' in da.dims:
            return da.mean()
        else:
            return da


def save_stats(outpath, stats):
    with open(outpath, 'w') as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Debug data-prep workflow with diagnostics')
    parser.add_argument('--start-year', type=int, default=None)
    parser.add_argument('--end-year', type=int, default=None)
    parser.add_argument('--start-index', type=int, default=None,
                        help='Optional explicit start index (takes precedence over start-year)')
    parser.add_argument('--end-index', type=int, default=None,
                        help='Optional explicit end index (inclusive)')
    parser.add_argument('--coarsen', type=int, default=1,
                        help='Optional spatial coarsen factor')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to write debug outputs')
    parser.add_argument('--draft-dir', type=str, default=None,
                        help='Directory to write per-catchment draft predictions')
    parser.add_argument('--precomputed-mean', type=str, default=None,
                        help='Path to precomputed time-mean to use for dedraft')
    parser.add_argument('--skip-dedraft', action='store_true')
    args = parser.parse_args()

    outdir = Path(args.output_dir) if args.output_dir else Path(config.DIR_PROCESSED) / 'debug_workflow'
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir, 'debug_prepare_workflow')

    draft_dir = Path(args.draft_dir) if args.draft_dir else Path(config.DIR_ICESHELF_DEDRAFT_MODEL) / 'debug'
    draft_dir.mkdir(parents=True, exist_ok=True)

    logger.info('DEBUG WORKFLOW START')
    t0 = time()

    # Load model
    logger.info('Loading model: %s', config.FILE_MPASO_MODEL)
    ds = xr.open_dataset(config.FILE_MPASO_MODEL, chunks={config.TIME_DIM: 36})
    ds = write_crs(ds, config.CRS_TARGET)

    # Subset by years or indices
    if args.start_index is not None or args.end_index is not None:
        si = 0 if args.start_index is None else args.start_index
        ei = len(ds[config.TIME_DIM]) - 1 if args.end_index is None else args.end_index
        logger.info('Subsetting by index %s:%s', si, ei)
        ds_sub = ds.isel({config.TIME_DIM: slice(si, ei + 1)})
    else:
        sy = args.start_year or config.SORRM_START_YEAR
        ey = args.end_year or config.SORRM_END_YEAR
        logger.info('Subsetting by years %s-%s', sy, ey)
        ds_sub = subset_dataset_by_time(ds, time_dim=config.TIME_DIM, start_year=sy, end_year=ey)

    if args.coarsen and args.coarsen > 1:
        logger.info('Coarsening spatially by factor %d', args.coarsen)
        ds_sub = ds_sub.coarsen(x=args.coarsen, y=args.coarsen, boundary='trim').mean()

    flux_var = config.SORRM_FLUX_VAR
    draft_var = config.SORRM_DRAFT_VAR

    if flux_var not in ds_sub.data_vars:
        raise KeyError(f'Flux variable {flux_var} not found in dataset')

    original = ds_sub[flux_var]

    # Trend fit + detrend (we compute fit so we can examine it)
    logger.info('Computing polynomial trend fit and detrended dataset')
    fit, detrended = compute_polyfit_and_detrend(original, time_dim=config.TIME_DIM)
    detrended_path = outdir / 'detrended.nc'
    logger.info('Saving detrended -> %s', detrended_path)
    detrended.to_netcdf(detrended_path)

    # Deseasonalize
    logger.info('Deseasonalizing detrended dataset')
    deseas = deseasonalize(detrended)
    seasonality = detrended - deseas
    seasonality_path = outdir / 'seasonality.nc'
    logger.info('Saving seasonality -> %s', seasonality_path)
    # save only the seasonality variable
    try:
        seasonality[[flux_var]].to_netcdf(seasonality_path)
    except Exception:
        seasonality.to_netcdf(seasonality_path)

    # Dedraft: compute time-mean and per-catchment predictions
    if args.skip_dedraft:
        logger.info('Skipping dedraft as requested')
        merged_pred = None
    else:
        if args.precomputed_mean:
            logger.info('Loading precomputed mean: %s', args.precomputed_mean)
            ds_mean = xr.open_dataset(args.precomputed_mean)
        else:
            logger.info('Computing time-mean for dedraft (flux from deseasonalized, draft from model)')
            # The dedraft routine expects a Dataset containing both the flux and the draft
            # variables (so it can access ds[flux_var] and ds[draft_var]). `deseas` here is a
            # DataArray (only the flux variable), so build a time-mean Dataset that contains
            # the deseasonalized flux mean and the model draft mean from `ds_sub`.
            flux_mean = deseas.mean(dim=config.TIME_DIM).compute()
            # get draft mean from the original subset (ds_sub) so dedraft has draft field
            if draft_var in ds_sub:
                draft_mean = ds_sub[draft_var].mean(dim=config.TIME_DIM).compute()
            else:
                # fallback: try config name explicitly
                draft_mean = ds_sub[config.SORRM_DRAFT_VAR].mean(dim=config.TIME_DIM).compute()

            # Ensure both are Datasets and merge into a single Dataset for dedraft_catchment
            flux_ds = flux_mean.to_dataset(name=flux_var) if isinstance(flux_mean, xr.DataArray) else flux_mean
            draft_ds = draft_mean.to_dataset(name=draft_var) if isinstance(draft_mean, xr.DataArray) else draft_mean
            ds_mean = xr.merge([flux_ds, draft_ds])

            mean_file = draft_dir / '_temp_time_mean_debug.nc'
            ds_mean.to_netcdf(mean_file)
            logger.info('Saved time-mean to %s', mean_file)

        # run dedraft per shelf
        icems = gpd.read_file(config.FILE_ICESHELFMASKS).to_crs({'init': config.CRS_TARGET})
        shelves = [(i, icems.name.values[i]) for i in config.ICE_SHELF_REGIONS]
        for idx, (i, name) in enumerate(shelves, 1):
            logger.info('Dedrafting [%d/%d] %s', idx, len(shelves), name)
            try:
                dedraft_catchment(i, icems, ds_mean, config, save_dir=draft_dir, save_pred=True, save_coefs=False)
            except Exception as e:
                logger.exception('dedraft_catchment failed for %s: %s', name, e)

        pred_files = [draft_dir / f'draftDepenModelPred_{icems.name.values[i]}.nc' for i in config.ICE_SHELF_REGIONS]
        logger.info('Merging %d per-catchment predictions', len(pred_files))
        merged_pred = merge_catchment_files(pred_files)

        # align and ensure variable name
        try:
            if 'x' in merged_pred.coords and 'y' in merged_pred.coords:
                merged_pred = merged_pred.interp(x=deseas['x'], y=deseas['y'], method='nearest')
            else:
                merged_pred = merged_pred.reindex_like(deseas)
        except Exception:
            try:
                merged_pred = merged_pred.reindex_like(deseas)
            except Exception:
                logger.exception('Failed to align merged_pred; proceeding as-is')

        if flux_var not in merged_pred.data_vars:
            try:
                pv = next(iter(merged_pred.data_vars))
                merged_pred = merged_pred.rename({pv: flux_var})
                logger.info('Renamed merged pred var %s -> %s', pv, flux_var)
            except StopIteration:
                # create NaN placeholder
                logger.warning('Merged pred has no data variables; creating NaN placeholder')
                zero_da = xr.DataArray(np.full(deseas[flux_var].shape, np.nan, dtype=float), coords=deseas[flux_var].coords, dims=deseas[flux_var].dims)
                merged_pred = xr.Dataset({flux_var: zero_da})

        pred_out = outdir / 'draft_dependence_merged.nc'
        logger.info('Saving merged draft predictions -> %s', pred_out)
        merged_pred.to_netcdf(pred_out)

    # Compute variability and residual
    logger.info('Computing variability (deseasonalized - draft_pred) and residual')
    if merged_pred is None:
        variability = deseas
        logger.warning('No merged_pred available; variability set to deseasonalized')
    else:
        # merged_pred may be 2D (no Time); if so broadcast to Time
        if config.TIME_DIM in deseas.dims and config.TIME_DIM not in merged_pred.dims:
            mp = merged_pred[flux_var]
            mp_time = mp.expand_dims({config.TIME_DIM: deseas[config.TIME_DIM].values}).transpose(config.TIME_DIM, 'y', 'x')
            merged_pred_da = mp_time
        else:
            merged_pred_da = merged_pred[flux_var]

        variability = deseas - merged_pred_da

    variability_path = outdir / 'variability_pipeline.nc'
    logger.info('Saving variability -> %s', variability_path)
    variability.to_netcdf(variability_path)

    # residual = original - (fit + seasonality + draft_pred)
    if merged_pred is None:
        residual = original - (fit + seasonality)
    else:
        # ensure draft_pred broadcast
        dp = merged_pred[flux_var]
        if config.TIME_DIM in original.dims and config.TIME_DIM not in dp.dims:
            dp_time = dp.expand_dims({config.TIME_DIM: original[config.TIME_DIM].values}).transpose(config.TIME_DIM, 'y', 'x')
            dp_da = dp_time
        else:
            dp_da = dp
        residual = original - (fit + seasonality + dp_da)

    residual_path = outdir / 'residual_original_minus_components.nc'
    logger.info('Saving residual -> %s', residual_path)
    residual.to_netcdf(residual_path)

    # Time-series: spatial means
    logger.info('Computing spatial-mean time-series for components')
    ts = {}
    ts['original'] = spatial_mean_timeseries(original)
    ts['detrended'] = spatial_mean_timeseries(detrended)
    ts['deseasonalized'] = spatial_mean_timeseries(deseas)
    ts['seasonality'] = spatial_mean_timeseries(seasonality)
    if merged_pred is not None:
        # merged pred spatial mean (2D) -> scalar, repeat to Time length
        pred_mean = merged_pred[flux_var].mean(dim=('x', 'y'))
        if np.isscalar(pred_mean.values):
            # scalar
            if config.TIME_DIM in original.dims:
                ts['draft_pred'] = xr.DataArray(np.repeat(float(pred_mean.values), len(original[config.TIME_DIM])), dims=[config.TIME_DIM], coords={config.TIME_DIM: original[config.TIME_DIM].values})
            else:
                ts['draft_pred'] = xr.DataArray(pred_mean.values)
        else:
            ts['draft_pred'] = pred_mean
    else:
        ts['draft_pred'] = None

    ts['variability_pipeline'] = spatial_mean_timeseries(variability)
    ts['residual'] = spatial_mean_timeseries(residual)

    # Build and save a time-series Dataset for exact reproducibility of the plot
    logger.info('Assembling time-series Dataset and saving to disk')
    ts_vars = {}
    for name, arr in ts.items():
        if arr is None:
            continue
        # If it's already an xarray DataArray with time dim, preserve coords
        if isinstance(arr, xr.DataArray) and config.TIME_DIM in arr.dims:
            da = xr.DataArray(arr.values, dims=arr.dims, coords=arr.coords, name=name)
        else:
            # For scalars or 1D arrays without TIME_DIM, broadcast to the original time
            if config.TIME_DIM in original.dims:
                if hasattr(arr, 'values') and getattr(arr, 'values').ndim == 1:
                    # 1D without TIME_DIM (unlikely) - assume length matches time
                    vals = arr.values
                    da = xr.DataArray(vals, dims=[config.TIME_DIM], coords={config.TIME_DIM: original[config.TIME_DIM].values}, name=name)
                else:
                    # scalar -> repeat across time
                    vals = np.repeat(float(arr.values) if hasattr(arr, 'values') else float(arr), len(original[config.TIME_DIM]))
                    da = xr.DataArray(vals, dims=[config.TIME_DIM], coords={config.TIME_DIM: original[config.TIME_DIM].values}, name=name)
            else:
                # No Time dim anywhere; store as 0-d or 1-d as-is
                if hasattr(arr, 'values'):
                    da = xr.DataArray(arr.values, name=name)
                else:
                    da = xr.DataArray(float(arr), name=name)

        ts_vars[name] = da

    if ts_vars:
        ts_ds = xr.merge({k: v for k, v in ts_vars.items()})
        ts_out = outdir / 'time_series_components.nc'
        logger.info('Saving time-series Dataset -> %s', ts_out)
        ts_ds.to_netcdf(ts_out)

    # convert to numpy arrays for plotting
    logger.info('Plotting time-series to %s', outdir / 'time_series_components.png')
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    time_coord = None
    for name, arr in ts.items():
        if arr is None:
            continue
        if hasattr(arr, 'coords') and config.TIME_DIM in arr.dims:
            x = arr[config.TIME_DIM].values
            y = arr.values
            time_coord = x
        else:
            # scalar or 1D without Time dim
            if hasattr(arr, 'values') and arr.values.ndim == 1:
                y = arr.values
                x = np.arange(len(y))
            else:
                # scalar
                y = np.repeat(float(arr.values), len(original[config.TIME_DIM])) if config.TIME_DIM in original.dims else np.array([float(arr.values)])
                x = original[config.TIME_DIM].values if config.TIME_DIM in original.dims else np.arange(len(y))

        ax.plot(x, y, label=name)

    ax.set_xlabel('Time')
    ax.set_ylabel(f'Spatial mean of {flux_var}')
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(outdir / 'time_series_components.png', dpi=150)
    plt.close(fig)

    # Summary stats
    logger.info('Computing summary stats')
    stats = {}
    def comp_stats(da):
        arr = da.values
        return {
            'nan_frac': float(np.isnan(arr).sum()) / float(arr.size),
            'mean': float(np.nanmean(arr)),
            'median': float(np.nanmedian(arr)),
            'std': float(np.nanstd(arr)),
        }

    stats['original'] = comp_stats(original.mean(dim=config.TIME_DIM) if config.TIME_DIM in original.dims else original)
    stats['detrended'] = comp_stats(detrended.mean(dim=config.TIME_DIM) if config.TIME_DIM in detrended.dims else detrended)
    stats['deseasonalized'] = comp_stats(deseas.mean(dim=config.TIME_DIM) if config.TIME_DIM in deseas.dims else deseas)
    stats['seasonality'] = comp_stats(seasonality.mean(dim=config.TIME_DIM) if config.TIME_DIM in seasonality.dims else seasonality)
    if merged_pred is not None:
        stats['draft_pred'] = comp_stats(merged_pred[flux_var])
    stats['variability_pipeline'] = comp_stats(variability.mean(dim=config.TIME_DIM) if config.TIME_DIM in variability.dims else variability)
    stats['residual'] = comp_stats(residual.mean(dim=config.TIME_DIM) if config.TIME_DIM in residual.dims else residual)

    stats_path = outdir / 'component_stats.json'
    logger.info('Saving stats -> %s', stats_path)
    save_stats(stats_path, stats)

    elapsed = time() - t0
    logger.info('DEBUG WORKFLOW COMPLETE (%.1fs). Outputs in %s', elapsed, outdir)


if __name__ == '__main__':
    main()
