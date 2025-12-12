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

    flux_var = 'floatingBasalMassBalAdjustment'
    draft_var = 'lowerSurface'

    if flux_var not in ds:
        raise RuntimeError(f"Expected '{flux_var}' in merged file {merged_path}")
    if draft_var not in ds:
        raise RuntimeError(f"Expected '{draft_var}' in merged file {merged_path}")

    ds_masks = xr.open_dataset(masks_path)

    flux_da = ds[flux_var]

    # Prepare output dedrafted array (same shape/coords)
    dedrafted = xr.full_like(flux_da, np.nan)

    if regions is None:
        regions = list(config.ICE_SHELF_REGIONS)

    for i in regions:
        logger.info(f'Processing region {i}')
        mask_da = ds_masks.regionCellMasks.isel(nRegions=i)
        mask_bool = (mask_da == 1)

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

    return str(dedrafted_path)


def compute_trend_by_segments(dedrafted_path, out_dir, segment_years=25):
    dedrafted_path = Path(dedrafted_path)
    ds = xr.open_dataset(dedrafted_path, chunks={config.TIME_DIM: 12})

    flux_var = 'floatingBasalMassBalAdjustment'
    if flux_var not in ds:
        raise RuntimeError(f"Expected '{flux_var}' in dedrafted file {dedrafted_path}")

    da = ds[flux_var]

    # center each time slice by subtracting the spatial mean
    # compute mean over spatial dims (all dims except time)
    spatial_dims = [d for d in da.dims if d != config.TIME_DIM]
    # keepdims so broadcasting works
    time_means = da.mean(dim=spatial_dims)
    da_centered = da - time_means

    # compute years from daysSinceStart (like plotting scripts)
    if 'daysSinceStart' in ds:
        days = ds['daysSinceStart'].values
        years = days / 365.0
        years = years - years[0]
    else:
        # fallback: construct from index using deltat if present
        if 'deltat' in ds:
            dt = ds['deltat'].values
            # deltat typically scalar or array of same len as Time
            # convert seconds to years
            if np.isscalar(dt):
                step_year = dt / 3.15e7
                years = np.arange(da.sizes[config.TIME_DIM]) * step_year
            else:
                years = np.cumsum(np.r_[0.0, dt[:-1]]) / 3.15e7
                years = years - years[0]
        else:
            raise RuntimeError('No daysSinceStart or deltat found to compute years')

    total_years = years[-1]
    seg_len = segment_years

    # build segment boundaries (start years)
    starts = np.arange(0, total_years + 1e-6, seg_len)

    segments = []
    for s in starts:
        e = s + seg_len
        # mask indices where years >= s and years < e, include last point on final segment
        if e >= years[-1] - 1e-12:
            sel_idx = np.where(years >= s)[0]
        else:
            sel_idx = np.where((years >= s) & (years < e))[0]

        if sel_idx.size == 0:
            continue

        seg = da_centered.isel({config.TIME_DIM: sel_idx})
        # ensure chunking: we want whole-time chunk for polyfit
        seg = seg.chunk({config.TIME_DIM: -1})

        detrended = detrend_dim(seg, dim=config.TIME_DIM, deg=1)
        trend_seg = seg - detrended
        segments.append(trend_seg)

    if len(segments) == 0:
        raise RuntimeError('No segments produced for trend computation')

    trend_full = xr.concat(segments, dim=config.TIME_DIM)

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
        dedrafted_path = create_dedrafted_from_merged(args.merged, args.masks, args.outdir, regions=args.regions)

    trend_path = compute_trend_by_segments(dedrafted_path, args.outdir, segment_years=args.segment_years)
    logger.info(f'Trend file created: {trend_path}')


if __name__ == '__main__':
    main()
