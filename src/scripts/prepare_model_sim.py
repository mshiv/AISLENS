#!/usr/bin/env python3
"""
Prepare MPAS-Ocean model simulation data for MALI forcing generation.

This script processes ocean model output to generate seasonality and variability
components for EOF decomposition and ensemble generation.

Usage:
    python prepare_model_sim.py [--start-year YYYY] [--end-year YYYY] [--skip-extrapolation]
"""

import argparse
import logging
import sys
from pathlib import Path
import xarray as xr
import geopandas as gpd

from aislens.dataprep import detrend_dim, deseasonalize, dedraft_catchment, extrapolate_catchment_over_time
from aislens.utils import merge_catchment_files, subset_dataset_by_time, initialize_directories, collect_directories, write_crs, setup_logging
from aislens.config import config

logger = logging.getLogger(__name__)


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Prepare MPAS-Ocean model simulation data for MALI forcing generation')
    
    parser.add_argument('--start-year', type=int, default=None,
                        help=f'Start year for time subsetting (default: {config.SORRM_START_YEAR})')
    parser.add_argument('--end-year', type=int, default=None,
                        help=f'End year for time subsetting (default: {config.SORRM_END_YEAR})')
    parser.add_argument('--skip-extrapolation', action='store_true',
                        help='Skip extrapolation step (for testing)')
    parser.add_argument('--init-dirs', action='store_true',
                        help='Initialize required directories before processing')
    
    args = parser.parse_args()
    
    # Use config defaults if not specified
    start_year = args.start_year if args.start_year is not None else config.SORRM_START_YEAR
    end_year = args.end_year if args.end_year is not None else config.SORRM_END_YEAR
    
    # Initialize directories if requested
    if args.init_dirs:
        initialize_directories(collect_directories(config))
    
    # Setup logging
    output_dir = Path(config.DIR_PROCESSED)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "prepare_model_sim")
    
    logger.info("MPAS-OCEAN MODEL SIMULATION PREPROCESSOR")
    logger.info(f"Time range: {start_year}-{end_year}")
    
    # Step 1: Load and subset model data
    logger.info(f"Loading model: {config.FILE_MPASO_MODEL}")
    model = xr.open_dataset(config.FILE_MPASO_MODEL, chunks={config.TIME_DIM: 36})
    model = write_crs(model, config.CRS_TARGET)
    
    logger.info(f"Subsetting to {start_year}-{end_year}...")
    model_subset = subset_dataset_by_time(model, time_dim=config.TIME_DIM,
                                          start_year=start_year, end_year=end_year)
    
    # Step 2: Detrend and deseasonalize
    logger.info("Detrending...")
    model_detrended = model_subset.copy()
    model_detrended[config.SORRM_FLUX_VAR] = detrend_dim(
        model_subset[config.SORRM_FLUX_VAR], dim=config.TIME_DIM, deg=1
    )
    
    logger.info("Deseasonalizing...")
    model_deseasonalized = deseasonalize(model_detrended)
    
    # Step 3: Load ice shelf masks
    logger.info("Loading ice shelf masks...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs({'init': config.CRS_TARGET})
    
    # Step 4: Calculate or load draft dependence predictions
    pred_files = [
        config.DIR_ICESHELF_DEDRAFT_MODEL / f'draftDepenModelPred_{icems.name.values[i]}.nc'
        for i in config.ICE_SHELF_REGIONS
    ]
    
    missing = [f for f in pred_files if not f.exists()]
    
    if missing:
        logger.info(f"Calculating draft dependence ({len(missing)} missing)...")
        config.DIR_ICESHELF_DEDRAFT_MODEL.mkdir(parents=True, exist_ok=True)
        
        # Save time-mean to temporary file to avoid recomputing
        temp_mean_file = config.DIR_ICESHELF_DEDRAFT_MODEL / '_temp_time_mean.nc'
        
        if not temp_mean_file.exists():
            logger.info("Computing time-mean (this will take a few minutes)...")
            # Compute in chunks and save to disk
            model_deseasonalized.mean(dim=config.TIME_DIM).to_netcdf(temp_mean_file)
            logger.info(f"Time-mean saved to: {temp_mean_file}")
        else:
            logger.info(f"Loading existing time-mean from: {temp_mean_file}")
        
        # Load time-mean from disk with spatial chunking
        model_deseasonalized_mean = xr.open_dataset(temp_mean_file, chunks={'x': 200, 'y': 200})
        
        # Process ice shelves sequentially
        ice_shelves_to_process = [
            (i, icems.name.values[i]) for i in config.ICE_SHELF_REGIONS
            if not (config.DIR_ICESHELF_DEDRAFT_MODEL / f'draftDepenModelPred_{icems.name.values[i]}.nc').exists()
        ]
        
        for idx, (i, catchment_name) in enumerate(ice_shelves_to_process, 1):
            logger.info(f"[{idx}/{len(ice_shelves_to_process)}] {catchment_name}")
            dedraft_catchment(
                i, icems, model_deseasonalized_mean, config,
                save_dir=config.DIR_ICESHELF_DEDRAFT_MODEL,
                save_pred=True,
                save_coefs=False
            )
    
    # Step 5: Merge draft dependence predictions
    logger.info("Merging draft dependence predictions...")
    draft_dependence_pred = merge_catchment_files(pred_files)
    draft_dependence_pred = draft_dependence_pred.reindex_like(model_deseasonalized)
    
    # Step 6: Calculate components
    logger.info("Calculating variability and seasonality...")
    model_variability = model_deseasonalized - draft_dependence_pred
    model_seasonality = model_detrended - model_deseasonalized
    
    # Step 7: Save intermediate components
    logger.info(f"Saving to: {config.FILE_SEASONALITY.name}, {config.FILE_VARIABILITY.name}")
    model_seasonality.to_netcdf(config.FILE_SEASONALITY)
    model_variability.to_netcdf(config.FILE_VARIABILITY)
    
    # Step 8: Extrapolate and save final components
    if args.skip_extrapolation:
        logger.info("Skipping extrapolation")
    else:
        logger.info("Extrapolating variability...")
        model_variability_extrapl = extrapolate_catchment_over_time(
            model_variability, icems, config, config.SORRM_FLUX_VAR
        )
        model_variability_extrapl = model_variability_extrapl.fillna(0)
        
        logger.info("Extrapolating seasonality...")
        model_seasonality_extrapl = extrapolate_catchment_over_time(
            model_seasonality, icems, config, config.SORRM_FLUX_VAR
        )
        model_seasonality_extrapl = model_seasonality_extrapl.fillna(0)
        
        logger.info(f"Saving to: {config.FILE_VARIABILITY_EXTRAPL.name}, {config.FILE_SEASONALITY_EXTRAPL.name}")
        model_variability_extrapl.to_netcdf(config.FILE_VARIABILITY_EXTRAPL)
        model_seasonality_extrapl.to_netcdf(config.FILE_SEASONALITY_EXTRAPL)
    
    logger.info(f"Complete! Outputs in: {config.DIR_PROCESSED}")


if __name__ == "__main__":
    main()
 