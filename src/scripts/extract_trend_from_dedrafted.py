#!/usr/bin/env python3
"""
extract_trend_from_dedrafted.py

Create a trend file from a dedrafted MALI merged file.

- If a dedrafted input file is supplied it will be used. If not, and a
  merged file is supplied, a dedrafted file will be created by computing
  per-region linear draft dependence (using `dedraft_unstructured_region`)
  and subtracting the predicted component.
- The dedrafted timeseries is centered by subtracting the spatial mean for
  each time slice.
- The time series is split into 25-year segments (years calculated using
  `daysSinceStart`) and each segment is detrended with `detrend_dim`.
  The trend contribution for the segment is computed as `segment - detrended`.
- All trend segments are concatenated in original temporal order and
  written to `trend_{input_stem}.nc`. `daysSinceStart` and `deltat` are
  copied to the output file.

  """
from pathlib import Path
import argparse
import logging

import numpy as np
import xarray as xr

from aislens.config import config
from aislens.dataprep import (
    dedraft_unstructured_region,
    detrend_dim,
)

logger = logging.getLogger(__name__)


def create_dedrafted_from_merged(merged_path, masks_path, out_dir, regions=None):
    """Create a dedrafted file from a merged MALI output by removing
    per-region linear draft dependence. Returns path to the dedrafted file."""
    merged_path = Path(merged_path)
    ds = xr.open_dataset(merged_path, chunks={config.TIME_DIM: 12})
    logger.info(f"Opened merged file: {merged_path}")
    logger.info(f"  variables: {list(ds.data_vars)}; coords: {list(ds.coords)}")

    flux_var = 'floatingBasalMassBalAdjustment'
    draft_var = 'lowerSurface'

    if flux_var not in ds:
        raise RuntimeError(f"Expected '{flux_var}' in merged file {merged_path}")
    if draft_var not in ds:
        raise RuntimeError(f"Expected '{draft_var}' in merged file {merged_path}")

    ds_masks = xr.open_dataset(masks_path)

    flux_da = ds[flux_var]
    logger.info(f"Found flux variable '{flux_var}' with dims={flux_da.dims} and shape={tuple(flux_da.shape)}")

    # Prepare output dedrafted array (same shape/coords)
    dedrafted = xr.full_like(flux_da, np.nan)

    if regions is None:
        regions = list(config.ICE_SHELF_REGIONS)

    for i in regions:
        logger.info(f'Processing region {i}')
        mask_da = ds_masks.regionCellMasks.isel(nRegions=i)
        mask_bool = (mask_da == 1)
        n_cells_total = mask_da.sizes.get('nCells', 'unknown')
        # try to get number of selected cells (best-effort, may trigger compute)
        try:
            n_selected = int(mask_bool.sum().values)
        except Exception:
            n_selected = 'unknown'
        logger.info(f"  mask nCells={n_cells_total}, selected={n_selected}")

        region_flux = flux_da.where(mask_bool)
        region_draft = ds[draft_var].where(mask_bool)

        slope, intercept, predicted_full = dedraft_unstructured_region(
            region_flux, region_draft, mask_da, time_dim=config.TIME_DIM
        )

        dedrafted_region = region_flux - predicted_full
        dedrafted = xr.where(mask_bool, dedrafted_region, dedrafted)

    dedrafted = dedrafted.where(np.isfinite(dedrafted), other=flux_da)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dedrafted_path = out_dir / f"dedrafted_{merged_path.stem}.nc"
    logger.info(f'Writing dedrafted file: {dedrafted_path}')

    da = dedrafted
    # simple chunk heuristics
    time_len = da.sizes.get(config.TIME_DIM, None)
    nCells_len = da.sizes.get('nCells', None)
    chunk_kwargs = {}
    if time_len:
        chunk_kwargs[config.TIME_DIM] = min(120, time_len)
    if nCells_len:
        chunk_kwargs['nCells'] = min(4096, nCells_len)

    if chunk_kwargs:
        da = da.chunk(chunk_kwargs)

    encoding = {da.name: {'zlib': True, 'complevel': 4}}

    ds_out = xr.Dataset({da.name: da})
    # copy supportive coords/vars
    for coord in [config.TIME_DIM, 'daysSinceStart', 'deltat']:
        if coord in ds.coords or coord in ds:
            ds_out.coords[coord] = ds.coords.get(coord, ds.get(coord))

    ds_out.to_netcdf(dedrafted_path, format='NETCDF4', engine='netcdf4', encoding=encoding)
    logger.info(f'Wrote dedrafted file: {dedrafted_path}')

    return str(dedrafted_path)


def compute_trend_by_segments(dedrafted_path, out_dir, segment_years=25):
    dedrafted_path = Path(dedrafted_path)
    ds = xr.open_dataset(dedrafted_path, chunks={config.TIME_DIM: 12})
    
    logger.info(f"Opened dedrafted file: {dedrafted_path}")

    flux_var = 'floatingBasalMassBalAdjustment'
    if flux_var not in ds:
        raise RuntimeError(f"Expected '{flux_var}' in dedrafted file {dedrafted_path}")

    da = ds[flux_var]
    logger.info(f"Working on variable '{flux_var}' with dims={da.dims} shape={tuple(da.shape)}")

    # center each time slice by subtracting the spatial mean
    # compute mean over spatial dims (all dims except time)
    spatial_dims = [d for d in da.dims if d != config.TIME_DIM]
    # keepdims so broadcasting works
    time_means = da.mean(dim=spatial_dims)
    da_centered = da - time_means
    logger.info(f"Centered data by spatial mean along dims={spatial_dims}")

    # compute years from daysSinceStart (like plotting scripts)
    if 'daysSinceStart' in ds:
        days = ds['daysSinceStart'].values
        # days may be datetime64, timedelta64 or numeric days; handle all cases
        if np.issubdtype(getattr(days, 'dtype', type(days)), np.datetime64):
            # compute days since first entry in calendar days
            delta_days = (days - days[0]) / np.timedelta64(1, 'D')
            years = delta_days.astype(float) / 365.0
        elif np.issubdtype(getattr(days, 'dtype', type(days)), np.timedelta64):
            delta_days = (days - days[0]) / np.timedelta64(1, 'D')
            years = delta_days.astype(float) / 365.0
        else:
            # numeric days
            years = np.asarray(days, dtype=float) / 365.0
            years = years - years[0]
    else:
        # fallback: construct from index using deltat if present
        if 'deltat' in ds:
            dt = ds['deltat'].values
            # deltat may be timedelta64 or numeric seconds
            if np.issubdtype(getattr(dt, 'dtype', type(dt)), np.timedelta64):
                # convert each deltat to days, cumulative from zero
                secs = (dt / np.timedelta64(1, 's')).astype(float)
                cumsum_secs = np.cumsum(np.r_[0.0, secs[:-1]])
                years = cumsum_secs / 3.15e7
            else:
                # numeric seconds (scalar or array)
                if np.isscalar(dt):
                    step_year = float(dt) / 3.15e7
                    years = np.arange(da.sizes[config.TIME_DIM]) * step_year
                else:
                    secs = np.asarray(dt, dtype=float)
                    cumsum_secs = np.cumsum(np.r_[0.0, secs[:-1]])
                    years = cumsum_secs / 3.15e7
        else:
            raise RuntimeError('No daysSinceStart or deltat found to compute years')

    total_years = float(years[-1])
    seg_len = segment_years

    # build segment boundaries (start years)
    starts = np.arange(0, total_years + 1e-6, seg_len)
    
    logger.info(f"Computed years vector: start={years[0]:.3f}, end={years[-1]:.3f}, n_time={len(years)}")
    logger.info(f"Segmenting into {len(starts)} start-points with segment_years={segment_years}")

    segments = []
    for seg_i, s in enumerate(starts):
        e = s + seg_len
        # mask indices where years >= s and years < e, include last point on final segment
        if e >= years[-1] - 1e-12:
            sel_idx = np.where(years >= s)[0]
        else:
            sel_idx = np.where((years >= s) & (years < e))[0]

        if sel_idx.size == 0:
            continue

        # report segment selection
        logger.info(f"Segment {seg_i}: years {s:.2f}..{min(e, years[-1]):.2f} -> indices {sel_idx[0]}..{sel_idx[-1]} (n={sel_idx.size})")

        seg = da_centered.isel({config.TIME_DIM: sel_idx})
        # ensure chunking: we want whole-time chunk for polyfit
        seg = seg.chunk({config.TIME_DIM: -1})
        logger.info(f"  Detrending segment {seg_i} with {seg.sizes.get(config.TIME_DIM)} timesteps")
        detrended = detrend_dim(seg, dim=config.TIME_DIM, deg=1)
        trend_seg = seg - detrended
        logger.info(f"  Computed trend segment {seg_i} shape={tuple(trend_seg.shape)}")
        segments.append(trend_seg)

    if len(segments) == 0:
        raise RuntimeError('No segments produced for trend computation')

    trend_full = xr.concat(segments, dim=config.TIME_DIM)
    logger.info(f"Concatenated {len(segments)} trend segments -> full trend dims={trend_full.dims} shape={tuple(trend_full.shape)}")

    # restore original coords for time (use from original dataset)
    trend_full = trend_full.assign_coords({config.TIME_DIM: ds[config.TIME_DIM]})

    out_ds = xr.Dataset({flux_var: trend_full})
    # add daysSinceStart and deltat if present in source
    for v in ['daysSinceStart', 'deltat']:
        if v in ds:
            out_ds[v] = ds[v]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trend_path = out_dir / f"trend_{dedrafted_path.stem}.nc"
    logger.info(f'Writing trend file: {trend_path}')

    # simple encoding/chunking
    da_out = out_ds[flux_var]
    time_len = da_out.sizes.get(config.TIME_DIM, None)
    nCells_len = da_out.sizes.get('nCells', None)
    chunk_kwargs = {}
    if time_len:
        chunk_kwargs[config.TIME_DIM] = min(120, time_len)
    if nCells_len:
        chunk_kwargs['nCells'] = min(4096, nCells_len)

    if chunk_kwargs:
        da_out = da_out.chunk(chunk_kwargs)

    encoding = {flux_var: {'zlib': True, 'complevel': 4}}
    out_ds[flux_var] = da_out
    out_ds.to_netcdf(trend_path, format='NETCDF4', engine='netcdf4', encoding=encoding)
    logger.info(f'Wrote trend file: {trend_path}')

    return str(trend_path)


def main():
    parser = argparse.ArgumentParser(description='Compute trend file from dedrafted MALI outputs')
    parser.add_argument('--dedrafted', type=str, help='Path to dedrafted input file (preferred)')
    parser.add_argument('--merged', type=str, help='Path to merged input file (used to create dedrafted if --dedrafted missing)')
    parser.add_argument('--masks', type=str, default=str(config.DATA_ROOT / 'data' / 'external' / 'aislens_draftDepen_regionMasks.nc'), help='Path to region masks NetCDF')
    parser.add_argument('--outdir', type=str, default=str(config.DIR_MALI_FORCING_TRENDS), help='Output directory')
    parser.add_argument('--regions', nargs='*', type=int, default=None, help='List of region indices to process when creating dedrafted file')
    parser.add_argument('--segment-years', type=int, default=25, help='Segment length in years (default: 25)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    dedrafted_path = args.dedrafted
    if dedrafted_path is None:
        if args.merged is None:
            raise RuntimeError('Either --dedrafted or --merged must be supplied')
        logger.info(f"No dedrafted file supplied; creating from merged: {args.merged}")
        dedrafted_path = create_dedrafted_from_merged(args.merged, args.masks, args.outdir, regions=args.regions)
    else:
        logger.info(f"Using dedrafted file: {dedrafted_path}")

    trend_path = compute_trend_by_segments(dedrafted_path, args.outdir, segment_years=args.segment_years)
    logger.info(f'Trend file created: {trend_path}')


if __name__ == '__main__':
    main()
