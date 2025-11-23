#!/usr/bin/env python3
"""
Fast version of model simulation data preparation for MALI forcing generation.

This optimized script processes ocean model output with the following speedup strategies:
1. Aggressive chunking to optimize dask operations
2. Pre-computed time-mean for draft dependence (avoids repeated computation)
3. Lazy evaluation with strategic compute() calls
4. Memory-efficient operations with explicit rechunking
5. Optional spatial coarsening for faster processing

Usage:
    python prepare_model_sim_fast.py [--start-year YYYY] [--end-year YYYY] [--coarsen N] [--skip-extrapolation]
"""

import argparse
import logging
import sys
from pathlib import Path
import xarray as xr
import geopandas as gpd
import numpy as np
from time import time

from aislens.dataprep import detrend_dim, deseasonalize, dedraft_catchment, extrapolate_catchment_over_time
from aislens.utils import merge_catchment_files, subset_dataset_by_time, initialize_directories, collect_directories, write_crs, setup_logging
from aislens.config import config

logger = logging.getLogger(__name__)


def coarsen_dataset(ds, factor, time_dim='Time'):
    """
    Coarsen a dataset spatially by averaging.
    
    Args:
        ds: xarray Dataset
        factor: Coarsening factor (e.g., 2 = half resolution)
        time_dim: Name of time dimension to preserve
        
    Returns:
        Coarsened dataset
    """
    if factor == 1:
        return ds
    
    logger.info(f"Coarsening by factor {factor}...")
    
    coarsen_dict = {'x': factor, 'y': factor}
    
    ds_coarse = ds.coarsen(dim=coarsen_dict, boundary='trim').mean()
    
    logger.info(f"  Spatial dimensions after coarsening: x={len(ds_coarse.x)}, y={len(ds_coarse.y)}")
    
    return ds_coarse


def compute_time_mean_efficient(ds, time_dim, chunk_size=36):
    """
    Compute time-mean efficiently using optimized chunking.
    
    Args:
        ds: xarray Dataset
        time_dim: Name of time dimension
        chunk_size: Chunk size for temporal dimension
        
    Returns:
        Time-mean dataset
    """
    logger.info("Computing time-mean with optimized chunking...")
    
    ds_rechunked = ds.chunk({time_dim: chunk_size, 'x': -1, 'y': -1})
    
    ds_mean = ds_rechunked.mean(dim=time_dim)
    
    logger.info("  Computing (this may take a moment)...")
    ds_mean = ds_mean.compute()
    
    return ds_mean


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Fast preparation of MPAS-Ocean model simulation data')
    
    parser.add_argument('--start-year', type=int, default=None,
                        help=f'Start year for time subsetting (default: {config.SORRM_START_YEAR})')
    parser.add_argument('--end-year', type=int, default=None,
                        help=f'End year for time subsetting (default: {config.SORRM_END_YEAR})')
    parser.add_argument('--coarsen', type=int, default=1,
                        help='Coarsen factor (e.g., 2 = half resolution, 4 = quarter resolution)')
    parser.add_argument('--skip-extrapolation', action='store_true',
                        help='Skip extrapolation step (for testing)')
    parser.add_argument('--init-dirs', action='store_true',
                        help='Initialize required directories before processing')
    parser.add_argument('--precomputed-mean', type=str, default=None,
                        help='Path to pre-computed time-mean file (skips time-mean computation)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for final products (overrides config.DIR_PROCESSED)')
    parser.add_argument('--draft-dir', type=str, default=None,
                        help='Custom directory for draft dependence files (overrides config.DIR_ICESHELF_DEDRAFT_MODEL)')
    
    args = parser.parse_args()
    
    start_year = args.start_year if args.start_year is not None else config.SORRM_START_YEAR
    end_year = args.end_year if args.end_year is not None else config.SORRM_END_YEAR
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.DIR_PROCESSED)
    draft_dir = Path(args.draft_dir) if args.draft_dir else Path(config.DIR_ICESHELF_DEDRAFT_MODEL)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    draft_dir.mkdir(parents=True, exist_ok=True)
    
    file_seasonality = output_dir / "sorrm_seasonality.nc"
    file_variability = output_dir / "sorrm_variability.nc"
    file_seasonality_extrapl = output_dir / "sorrm_seasonality_extrapolated_fillNA.nc"
    file_variability_extrapl = output_dir / "sorrm_variability_extrapolated_fillNA_meanAdjusted.nc"
    
    if args.init_dirs:
        initialize_directories(collect_directories(config))
    
    setup_logging(output_dir, "prepare_model_sim_fast")
    
    logger.info("FAST MPAS-OCEAN MODEL SIMULATION PREPROCESSOR")
    logger.info(f"Time range: {start_year}-{end_year}")
    logger.info(f"Coarsening factor: {args.coarsen}")
    logger.info(f"Skip extrapolation: {args.skip_extrapolation}")
    if args.output_dir:
        logger.info(f"Custom output directory: {output_dir}")
    if args.draft_dir:
        logger.info(f"Custom draft directory: {draft_dir}")
    
    start_time = time()
    
    # Step 1: Load and subset model data with optimized chunking
    logger.info(f"\nStep 1: Loading model data")
    logger.info(f"  File: {config.FILE_MPASO_MODEL}")
    
    model = xr.open_dataset(config.FILE_MPASO_MODEL, 
                           chunks={config.TIME_DIM: 24, 'x': 200, 'y': 200})
    model = write_crs(model, config.CRS_TARGET)
    
    logger.info(f"  Subsetting to {start_year}-{end_year}...")
    model_subset = subset_dataset_by_time(model, time_dim=config.TIME_DIM,
                                          start_year=start_year, end_year=end_year)
    
    if args.coarsen > 1:
        model_subset = coarsen_dataset(model_subset, args.coarsen, time_dim=config.TIME_DIM)
        model_subset = model_subset.chunk({config.TIME_DIM: 36, 'x': 200, 'y': 200})
    
    step1_time = time()
    logger.info(f"  Step 1 complete ({step1_time - start_time:.1f}s)")
    
    # Step 2: Detrend
    logger.info(f"\nStep 2: Detrending")
    
    model_detrended = model_subset.copy()
    model_detrended[config.SORRM_FLUX_VAR] = detrend_dim(
        model_subset[config.SORRM_FLUX_VAR], dim=config.TIME_DIM, deg=1
    )
    
    model_detrended = model_detrended.chunk({config.TIME_DIM: 36, 'x': 200, 'y': 200})
    
    step2_time = time()
    logger.info(f"  Step 2 complete ({step2_time - step1_time:.1f}s)")
    
    # Step 3: Deseasonalize
    logger.info(f"\nStep 3: Deseasonalizing")
    
    model_deseasonalized = deseasonalize(model_detrended)
    
    model_deseasonalized = model_deseasonalized.chunk({config.TIME_DIM: 36, 'x': 200, 'y': 200})
    
    step3_time = time()
    logger.info(f"  Step 3 complete ({step3_time - step2_time:.1f}s)")
    
    # Step 4: Draft dependence calculation
    logger.info(f"\nStep 4: Draft dependence calculation")
    
    # Load ice shelf masks
    logger.info("  Loading ice shelf masks...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs({'init': config.CRS_TARGET})
    
    # Check which ice shelves need processing
    pred_files = [
        draft_dir / f'draftDepenModelPred_{icems.name.values[i]}.nc'
        for i in config.ICE_SHELF_REGIONS
    ]
    missing = [f for f in pred_files if not f.exists()]
    
    if missing:
        logger.info(f"  Processing {len(missing)} ice shelves...")
        
        # Option 1: Use pre-computed time-mean if provided
        if args.precomputed_mean:
            logger.info(f"  Loading pre-computed time-mean from: {args.precomputed_mean}")
            model_deseasonalized_mean = xr.open_dataset(args.precomputed_mean)
        else:
            # Option 2: Compute time-mean efficiently
            logger.info("  Computing time-mean (this is the slow part)...")
            model_deseasonalized_mean = compute_time_mean_efficient(
                model_deseasonalized, config.TIME_DIM, chunk_size=36
            )
            
            # Optionally save for reuse
            temp_mean_file = draft_dir / '_temp_time_mean.nc'
            logger.info(f"  Saving time-mean to: {temp_mean_file}")
            model_deseasonalized_mean.to_netcdf(temp_mean_file)
        
        # Process ice shelves sequentially
        ice_shelves_to_process = [
            (i, icems.name.values[i]) for i in config.ICE_SHELF_REGIONS
            if not (draft_dir / f'draftDepenModelPred_{icems.name.values[i]}.nc').exists()
        ]
        
        for idx, (i, catchment_name) in enumerate(ice_shelves_to_process, 1):
            logger.info(f"  [{idx}/{len(ice_shelves_to_process)}] Processing {catchment_name}")
            dedraft_catchment(
                i, icems, model_deseasonalized_mean, config,
                save_dir=draft_dir,
                save_pred=True,
                save_coefs=False
            )
    else:
        logger.info("  All ice shelves already processed")
    
    step4_time = time()
    logger.info(f"  Step 4 complete ({step4_time - step3_time:.1f}s)")
    
    # Step 5: Merge draft dependence predictions
    logger.info(f"\nStep 5: Merging draft dependence predictions")
    
    draft_dependence_pred = merge_catchment_files(pred_files)
    draft_dependence_pred = draft_dependence_pred.reindex_like(model_deseasonalized)
    
    step5_time = time()
    logger.info(f"  Step 5 complete ({step5_time - step4_time:.1f}s)")
    
    # Step 6: Calculate components
    logger.info(f"\nStep 6: Calculating variability and seasonality components")
    
    model_variability = model_deseasonalized - draft_dependence_pred
    model_seasonality = model_detrended - model_deseasonalized
    
    step6_time = time()
    logger.info(f"  Step 6 complete ({step6_time - step5_time:.1f}s)")
    
    # Step 7: Save intermediate components
    logger.info(f"\nStep 7: Saving intermediate components")
    logger.info(f"  Seasonality: {file_seasonality}")
    logger.info(f"  Variability: {file_variability}")
    
    model_seasonality.to_netcdf(file_seasonality)
    model_variability.to_netcdf(file_variability)
    
    step7_time = time()
    logger.info(f"  Step 7 complete ({step7_time - step6_time:.1f}s)")
    
    # Step 8: Extrapolate and save final components
    if args.skip_extrapolation:
        logger.info(f"\nStep 8: Skipping extrapolation (--skip-extrapolation)")
    else:
        logger.info(f"\nStep 8: Extrapolating components")
        
        logger.info("  Extrapolating variability...")
        model_variability_extrapl = extrapolate_catchment_over_time(
            model_variability, icems, config, config.SORRM_FLUX_VAR
        )
        model_variability_extrapl = model_variability_extrapl.fillna(0)
        
        logger.info("  Extrapolating seasonality...")
        model_seasonality_extrapl = extrapolate_catchment_over_time(
            model_seasonality, icems, config, config.SORRM_FLUX_VAR
        )
        model_seasonality_extrapl = model_seasonality_extrapl.fillna(0)
        
        logger.info(f"  Saving extrapolated components...")
        logger.info(f"    Variability: {file_variability_extrapl}")
        logger.info(f"    Seasonality: {file_seasonality_extrapl}")
        
        model_variability_extrapl.to_netcdf(file_variability_extrapl)
        model_seasonality_extrapl.to_netcdf(file_seasonality_extrapl)
        
        step8_time = time()
        logger.info(f"  Step 8 complete ({step8_time - step7_time:.1f}s)")
    
    # Summary
    total_time = time() - start_time
    logger.info(f"PROCESSING COMPLETE!")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Draft dependence directory: {draft_dir}")


if __name__ == "__main__":
    main()
