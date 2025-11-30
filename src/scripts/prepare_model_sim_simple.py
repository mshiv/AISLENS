#!/usr/bin/env python3
"""
Simplified model-prep script: same logical steps as the optimized pipeline but
with a minimal, easy-to-read flow. This aims to be a readable, reproducible
single-file implementation that is safe to run interactively or on HPC.

Features:
- load & subset model
- optional spatial coarsening
- detrend -> deseasonalize -> compute time-mean for dedraft
- per-catchment dedraft (skips already-existing files)
- merge preds, compute variability & seasonality, save
- extrapolate to full grid and save (optional)

This intentionally keeps behavior simple and predictable; it avoids
aggressive rechunking and extra heuristics so outputs are easy to compare
with the legacy `prepare_data.py` flow.
"""

import argparse
import logging
from pathlib import Path
from time import time

import xarray as xr
import geopandas as gpd
import numpy as np

from aislens.dataprep import (
    detrend_dim,
    deseasonalize,
    dedraft_catchment,
    extrapolate_catchment_over_time,
)
from aislens.utils import (
    merge_catchment_files,
    subset_dataset_by_time,
    initialize_directories,
    collect_directories,
    write_crs,
    setup_logging,
)
from aislens.config import config

logger = logging.getLogger(__name__)


def compute_time_mean(ds, time_dim='Time'):
    """Simple time-mean helper (computes and returns a fully-computed Dataset).

    This intentionally does a plain mean + compute() so behavior is
    deterministic and easy to inspect.
    """
    logger.info("Computing time-mean (simple, deterministic)...")
    ds_mean = ds.mean(dim=time_dim)
    return ds_mean.compute()


def coarsen_dataset(ds, factor):
    if factor is None or factor <= 1:
        return ds
    logger.info(f"Coarsening spatially by factor {factor} (simple average)")
    return ds.coarsen(x=factor, y=factor, boundary='trim').mean()


def main():
    parser = argparse.ArgumentParser(description='Simplified model preparation pipeline')
    parser.add_argument('--start-year', type=int, default=None)
    parser.add_argument('--end-year', type=int, default=None)
    parser.add_argument('--coarsen', type=int, default=1,
                        help='Spatial coarsen factor (1 = no coarsening)')
    parser.add_argument('--skip-extrapolation', action='store_true')
    parser.add_argument('--init-dirs', action='store_true')
    parser.add_argument('--precomputed-mean', type=str, default=None,
                        help='Optional precomputed time-mean to use for dedraft')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Optional override for output directory')
    args = parser.parse_args()

    start_year = args.start_year or config.SORRM_START_YEAR
    end_year = args.end_year or config.SORRM_END_YEAR

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.DIR_PROCESSED)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, 'prepare_model_sim_simple')

    if args.init_dirs:
        initialize_directories(collect_directories(config))

    logger.info('SIMPLE MPAS-OCEAN MODEL PREPROCESSOR')
    logger.info(f'Time range: {start_year}-{end_year} | coarsen={args.coarsen} | skip_extrap={args.skip_extrapolation}')

    t0 = time()

    # Load model (moderate chunking)
    logger.info(f'Loading model: {config.FILE_MPASO_MODEL}')
    model = xr.open_dataset(config.FILE_MPASO_MODEL, chunks={config.TIME_DIM: 36})
    model = write_crs(model, config.CRS_TARGET)

    logger.info('Subsetting to requested years...')
    model_subset = subset_dataset_by_time(model, time_dim=config.TIME_DIM, start_year=start_year, end_year=end_year)

    if args.coarsen and args.coarsen > 1:
        model_subset = coarsen_dataset(model_subset, args.coarsen)

    # Step: detrend
    logger.info('Detrending...')
    model_detrended = model_subset.copy()
    model_detrended[config.SORRM_FLUX_VAR] = detrend_dim(model_subset[config.SORRM_FLUX_VAR], dim=config.TIME_DIM, deg=1)

    # Step: deseasonalize
    logger.info('Deseasonalizing...')
    model_deseasonalized = deseasonalize(model_detrended)

    # Prepare names
    file_seasonality = output_dir / 'sorrm_seasonality.nc'
    file_variability = output_dir / 'sorrm_variability.nc'
    file_seasonality_extrapl = output_dir / 'sorrm_seasonality_extrapolated.nc'
    file_variability_extrapl = output_dir / 'sorrm_variability_extrapolated.nc'

    # Load ice shelf masks
    logger.info('Loading ice-shelf masks...')
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs({'init': config.CRS_TARGET})

    # Handle dedraft per-catchment (use precomputed mean if provided)
    pred_dir = Path(config.DIR_ICESHELF_DEDRAFT_MODEL)
    pred_dir.mkdir(parents=True, exist_ok=True)

    pred_files = [pred_dir / f'draftDepenModelPred_{icems.name.values[i]}.nc' for i in config.ICE_SHELF_REGIONS]
    missing = [f for f in pred_files if not f.exists()]

    if missing:
        logger.info(f'{len(missing)} draft-pred files missing; computing per-catchment predictions')
        if args.precomputed_mean:
            logger.info(f'Using provided time-mean: {args.precomputed_mean}')
            model_mean = xr.open_dataset(args.precomputed_mean)
        else:
            model_mean = compute_time_mean(model_deseasonalized, time_dim=config.TIME_DIM)
            tmp_mean = pred_dir / '_temp_time_mean.nc'
            model_mean.to_netcdf(tmp_mean)
            logger.info(f'Wrote temp time-mean: {tmp_mean}')

        shelves_to_process = [(i, icems.name.values[i]) for i in config.ICE_SHELF_REGIONS if not (pred_dir / f'draftDepenModelPred_{icems.name.values[i]}.nc').exists()]
        for idx, (i, name) in enumerate(shelves_to_process, 1):
            logger.info(f'[{idx}/{len(shelves_to_process)}] Processing {name}')
            dedraft_catchment(i, icems, model_mean, config, save_dir=pred_dir, save_pred=True, save_coefs=False)
    else:
        logger.info('All per-catchment draft predictions exist; skipping dedraft step')

    # Merge draft predictions and align to grid
    logger.info('Merging draft-dependence predictions...')
    draft_dependence_pred = merge_catchment_files(pred_files)
    try:
        draft_dependence_pred = draft_dependence_pred.reindex_like(model_deseasonalized)
    except Exception:
        logger.warning('Reindex_like on draft_dependence_pred failed; attempting interp')
        try:
            draft_dependence_pred = draft_dependence_pred.interp(x=model_deseasonalized['x'], y=model_deseasonalized['y'], method='nearest')
        except Exception:
            logger.warning('Interpolation failed; proceeding with merged prediction as-is')

    # Compute components
    logger.info('Computing variability and seasonality components...')
    model_variability = model_deseasonalized - draft_dependence_pred
    model_seasonality = model_detrended - model_deseasonalized

    logger.info(f'Saving seasonality -> {file_seasonality} and variability -> {file_variability}')
    model_seasonality.to_netcdf(file_seasonality)
    model_variability.to_netcdf(file_variability)

    # Extrapolate if requested
    if not args.skip_extrapolation:
        logger.info('Extrapolating variability to full grid...')
        var_ex = extrapolate_catchment_over_time(model_variability, icems, config, config.SORRM_FLUX_VAR)
        var_ex = var_ex.fillna(0)
        logger.info('Extrapolating seasonality to full grid...')
        sea_ex = extrapolate_catchment_over_time(model_seasonality, icems, config, config.SORRM_FLUX_VAR)
        sea_ex = sea_ex.fillna(0)
        logger.info(f'Saving extrapolated outputs: {file_variability_extrapl}, {file_seasonality_extrapl}')
        var_ex.to_netcdf(file_variability_extrapl)
        sea_ex.to_netcdf(file_seasonality_extrapl)
    else:
        logger.info('Skipping extrapolation (--skip-extrapolation)')

    elapsed = time() - t0
    logger.info(f'Complete (elapsed {elapsed:.1f}s). Outputs in {output_dir}')


if __name__ == '__main__':
    main()
