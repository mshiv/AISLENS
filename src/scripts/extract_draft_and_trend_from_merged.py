#!/usr/bin/env python3
"""
extract_draft_and_trend_from_merged.py

Read merged MALI output files (those produced by `merge_state_to_flux.py`),
compute per-region linear draft dependence using `lowerSurface` and
`floatingBasalMassBalAdjustment`, remove the draft-dependent component,
and extract the trend using the existing breakpoint detrending helper.

Differences from `extract_draft_and_trend_mali.py`:
- Does NOT reverse sign of the flux variable (the merged file already
  contains `floatingBasalMassBalAdjustment`).
- Does NOT perform any interpolation along time; the merged input is
  expected to already have `lowerSurface` aligned to the `Time` axis.

Outputs (per input merged file):
- dedrafted_{input_stem}.nc  : dedrafted `floatingBasalMassBalAdjustment`
- trend_{input_stem}.nc      : trend component (same variable name)

The final trend variable will include attributes describing the
transformations applied: conversion, draft-dependence removal, trend.
"""
from pathlib import Path
import argparse
import logging
import csv

import numpy as np
import xarray as xr

from aislens.config import config
from aislens.dataprep import dedraft_unstructured_region, detrend_with_breakpoints_vectorized

logger = logging.getLogger(__name__)


def process_merged(merged_path, masks_path, out_dir, regions=None, stream=False, save_coefs=True):
    merged_path = Path(merged_path)
    logger.info(f'Opening merged file: {merged_path}')
    ds = xr.open_dataset(merged_path, chunks={config.TIME_DIM: 12})

    # Expect variables already present in merged file
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

    region_coefs = []

    for i in regions:
        logger.info(f'Processing region {i}')
        mask_da = ds_masks.regionCellMasks.isel(nRegions=i)
        mask_bool = (mask_da == 1)

        region_flux = flux_da.where(mask_bool)
        region_draft = ds[draft_var].where(mask_bool)

        slope, intercept, predicted_full = dedraft_unstructured_region(region_flux, region_draft, mask_da, time_dim=config.TIME_DIM)
        logger.info(f'  region {i}: slope={slope:.6g}, intercept={intercept:.6g}')
        region_coefs.append({'region': int(i), 'slope': float(slope), 'intercept': float(intercept)})

        # predicted_full should already be aligned to the same Time axis as inputs
        # Subtract predicted component only where mask==1
        dedrafted_region = region_flux - predicted_full

        dedrafted = xr.where(mask_bool, dedrafted_region, dedrafted)

    # For any cells not covered by any region (still NaN), fall back to original
    dedrafted = dedrafted.where(np.isfinite(dedrafted), other=flux_da)

    out_ds = xr.Dataset({flux_var: dedrafted})

    # copy Time and supporting coords/vars if present
    for coord in [config.TIME_DIM, 'daysSinceStart', 'deltat']:
        if coord in ds.coords or coord in ds:
            try:
                out_ds.coords[coord] = ds.coords.get(coord, ds.get(coord))
            except Exception:
                # best-effort copy
                pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dedrafted_path = out_dir / f"dedrafted_{merged_path.stem}.nc"
    logger.info(f'Writing dedrafted file: {dedrafted_path}')

    # Rechunk heuristics (keep small and simple)
    da = out_ds[flux_var]
    if config.TIME_DIM in da.dims and 'nCells' in da.dims and da.dims != (config.TIME_DIM, 'nCells'):
        da = da.transpose(config.TIME_DIM, 'nCells')

    time_len = da.sizes.get(config.TIME_DIM, None)
    nCells_len = da.sizes.get('nCells', None)
    time_chunk = min(120, time_len) if time_len is not None else None
    nCells_chunk = min(4096, nCells_len) if nCells_len is not None else None

    chunk_kwargs = {}
    if time_chunk:
        chunk_kwargs[config.TIME_DIM] = time_chunk
    if nCells_chunk:
        chunk_kwargs['nCells'] = nCells_chunk

    if chunk_kwargs:
        try:
            da = da.chunk(chunk_kwargs)
        except Exception:
            logger.debug('Could not rechunk dedrafted DataArray; continuing without rechunk')

    encoding = {flux_var: {'zlib': True, 'complevel': 4}}

    # Write dedrafted file (non-stream path is fine here)
    out_ds[flux_var] = da
    out_ds.to_netcdf(dedrafted_path, format='NETCDF4', engine='netcdf4', encoding=encoding)

    # Optionally save region coefficients CSV
    if save_coefs:
        coefs_path = out_dir / f"region_coefs_{merged_path.stem}.csv"
        logger.info(f'Saving region coefficients to: {coefs_path}')
        with open(coefs_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['region', 'slope', 'intercept'])
            writer.writeheader()
            for r in region_coefs:
                writer.writerow(r)

    # Prepare for detrending: ensure single dask chunk along Time
    try:
        dedrafted_for_detrend = da
        if config.TIME_DIM in dedrafted_for_detrend.dims:
            chunk_dict = {config.TIME_DIM: -1}
            if nCells_len is not None:
                chunk_dict['nCells'] = min(4096, nCells_len)
            dedrafted_for_detrend = dedrafted_for_detrend.chunk(chunk_dict)
    except Exception:
        logger.warning('Failed to rechunk dedrafted for detrending; computing into memory')
        dedrafted_for_detrend = dedrafted_for_detrend.compute()

    logger.info('Running breakpoint detrend (vectorized)')
    detrended = detrend_with_breakpoints_vectorized(dedrafted_for_detrend, dim=config.TIME_DIM, deg=1, model='rbf', penalty=10)

    trend_da = dedrafted_for_detrend - detrended
    trend_ds = xr.Dataset({flux_var: trend_da})

    # copy supportive variables
    for attr_var in ['daysSinceStart', 'deltat']:
        if attr_var in ds:
            trend_ds[attr_var] = ds[attr_var]

    # Add processing attributes describing the transformations
    proc_text = (
        "Converted original floatingBasalMassBalApplied to floatingBasalMassBalAdjustment; "
        "removed draft dependence per-region (slope/intercept fit and subtraction); "
        "extracted trend component by breakpoint detrending."
    )
    trend_ds[flux_var].attrs['processing_steps'] = proc_text
    trend_ds[flux_var].attrs['source_merged_file'] = str(merged_path.name)

    trend_path = out_dir / f"trend_{merged_path.stem}.nc"
    logger.info(f'Writing trend file: {trend_path}')
    trend_ds.to_netcdf(trend_path)

    return {'dedrafted': str(dedrafted_path), 'trend': str(trend_path), 'coefs': str(coefs_path) if save_coefs else None}


def main():
    parser = argparse.ArgumentParser(description='Compute draft-dependence removal and trend from merged MALI files')
    parser.add_argument('--merged', nargs='+', required=True, help='One or more merged input files (output of merge_state_to_flux.py)')
    parser.add_argument('--masks', type=str, default=str(config.DATA_ROOT / 'data' / 'external' / 'aislens_draftDepen_regionMasks.nc'), help='Path to region masks NetCDF')
    parser.add_argument('--outdir', type=str, default=str(config.DIR_MALI_FORCING_TRENDS), help='Output directory')
    parser.add_argument('--regions', nargs='*', type=int, default=None, help='List of region indices to process')
    parser.add_argument('--no-coefs', action='store_true', help="Don't save per-region coefficients CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results = {}
    for infile in args.merged:
        res = process_merged(infile, args.masks, args.outdir, regions=args.regions, save_coefs=not args.no_coefs)
        results[infile] = res
        logger.info(f"Processed {infile}: dedrafted={res['dedrafted']} trend={res['trend']} coefs={res['coefs']}")


if __name__ == '__main__':
    main()
