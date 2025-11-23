#!/usr/bin/env python3
"""
Prepare satellite observations and model simulation data for MALI forcing generation.

This is the FIRST script to be run in the workflow.

Satellite observations:
  1. Load the satellite observation dataset.
  2. Detrend the data along the time dimension (retain the mean value).
  3. Deseasonalize the data.
  4. Take the time-mean of the dataset.
  5. Ensure that melt is converted to flux, if not, and is in SI units.
  6. Save the prepared data fields of meltflux and draft.

Model simulation data:
  1. Load the model simulation dataset.
  2. Subset the dataset to the desired time range.
  3. Detrend the data along the time dimension.
  4. Deseasonalize the data.
  5. Dedraft the data.
  6. Save the seasonality and variability components thus obtained.

Usage:
    python prepare_data.py [--satobs] [--model]
"""

import argparse
import logging
from pathlib import Path
from time import time
import numpy as np
import xarray as xr
import geopandas as gpd

from aislens.dataprep import detrend_dim, deseasonalize, dedraft_catchment, extrapolate_catchment_over_time
from aislens.utils import merge_catchment_files, subset_dataset_by_time, collect_directories, initialize_directories, write_crs, setup_logging
from aislens.config import config

logger = logging.getLogger(__name__)



def prepare_satellite_observations():
    """
    Prepare satellite observation data for forcing generation.
    
    Loads Paolo et al. 2023 satellite observations, detrends, deseasonalizes,
    and saves the prepared dataset.
    """
    logger.info("Preparing satellite observations...")
    logger.debug(f"  Loading from: {config.FILE_PAOLO23_SATOBS}")
    
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS)
    logger.info("  Satellite observations loaded")
    
    # Detrend the data along the time dimension
    logger.info("  Detrending...")
    satobs_deseasonalized = satobs.copy()
    satobs_detrended = detrend_dim(satobs_deseasonalized[config.SATOBS_FLUX_VAR], dim=config.TIME_DIM, deg=1)
    logger.debug("  Detrending complete")
    
    # Deseasonalize the data
    logger.info("  Deseasonalizing...")
    satobs_deseasonalized[config.SATOBS_FLUX_VAR] = deseasonalize(satobs_detrended)
    logger.debug("  Deseasonalization complete")
    
    # TODO: Take the time-mean of the dataset
    # TODO: Ensure that melt is converted to flux, if not, and is in SI units

    # Save the prepared data
    logger.info(f"  Saving to: {config.FILE_PAOLO23_SATOBS_PREPARED}")
    satobs_deseasonalized.to_netcdf(config.FILE_PAOLO23_SATOBS_PREPARED)
    logger.info("Satellite observations prepared successfully")


def prepare_model_simulation():
    """
    Prepare MPAS-Ocean model simulation data for forcing generation.
    
    Loads model data, subsets to time range, detrends, deseasonalizes,
    calculates draft dependence, and extrapolates to entire ice sheet grid.
    """
    start_time = time()
    
    logger.info("Preparing model simulation data...")
    
    # Load model simulation dataset
    logger.info(f"  Loading model: {config.FILE_MPASO_MODEL}")
    model = xr.open_dataset(config.FILE_MPASO_MODEL, chunks={config.TIME_DIM: 36})
    model = write_crs(model, config.CRS_TARGET)
    logger.debug("  Model loaded")
    
    # Subset by time
    logger.info(f"  Subsetting to years {config.SORRM_START_YEAR}-{config.SORRM_END_YEAR}...")
    model_subset = subset_dataset_by_time(model,
                                          time_dim=config.TIME_DIM,
                                          start_year=config.SORRM_START_YEAR,
                                          end_year=config.SORRM_END_YEAR)
    logger.debug(f"  Subsetted to {len(model_subset[config.TIME_DIM])} timesteps")
    
    # Detrend
    logger.info("  Detrending...")
    model_detrended = model_subset.copy()
    model_detrended[config.SORRM_FLUX_VAR] = detrend_dim(model_subset[config.SORRM_FLUX_VAR], dim=config.TIME_DIM, deg=1)
    logger.debug("  Detrending complete")
    
    # Deseasonalize
    logger.info("  Deseasonalizing...")
    model_deseasonalized = deseasonalize(model_detrended)
    logger.debug("  Deseasonalization complete")
    
    # Draft dependence calculation
    logger.info("  Calculating draft dependence...")
    logger.debug("  Loading ice shelf masks...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs({'init': config.CRS_TARGET})
    
    logger.debug(f"  Processing {len(config.ICE_SHELF_REGIONS)} ice shelves...")
    for idx, i in enumerate(config.ICE_SHELF_REGIONS, 1):
        shelf_name = icems.name.values[i]
        logger.debug(f"    [{idx}/{len(config.ICE_SHELF_REGIONS)}] {shelf_name}")
        dedraft_catchment(i, icems, model_deseasonalized, config, 
                          save_dir=config.DIR_ICESHELF_DEDRAFT_MODEL,
                          save_pred=True)
    
    logger.info("  Merging draft dependence predictions...")
    draft_dependence_pred = merge_catchment_files([config.DIR_ICESHELF_DEDRAFT_MODEL / f'draftDepenModelPred_{icems.name.values[i]}.nc'
                                                   for i in config.ICE_SHELF_REGIONS])
    logger.debug("  Merge complete")
    
    # Calculate components
    logger.info("  Calculating variability and seasonality components...")
    model_variability = model_deseasonalized - draft_dependence_pred
    model_seasonality = model_detrended - model_deseasonalized
    
    # Save intermediate components
    logger.info(f"  Saving components:")
    logger.info(f"    Seasonality: {config.FILE_SEASONALITY.name}")
    logger.info(f"    Variability: {config.FILE_VARIABILITY.name}")
    model_seasonality.to_netcdf(config.FILE_SEASONALITY)
    model_variability.to_netcdf(config.FILE_VARIABILITY)
    
    # Extrapolate to entire ice sheet grid
    logger.info("  Extrapolating variability to entire grid...")
    model_variability_extrapl = extrapolate_catchment_over_time(model_variability, 
                                                                icems, config, 
                                                                config.SORRM_FLUX_VAR)
    
    logger.info("  Extrapolating seasonality to entire grid...")
    model_seasonality_extrapl = extrapolate_catchment_over_time(model_seasonality, 
                                                                icems, config, 
                                                                config.SORRM_FLUX_VAR)
    logger.debug("  Extrapolation complete")
    
    # Save final components
    logger.info(f"  Saving extrapolated components:")
    logger.info(f"    Seasonality: {config.FILE_SEASONALITY_EXTRAPL.name}")
    logger.info(f"    Variability: {config.FILE_VARIABILITY_EXTRAPL.name}")
    model_seasonality_extrapl.to_netcdf(config.FILE_SEASONALITY_EXTRAPL)
    model_variability_extrapl.to_netcdf(config.FILE_VARIABILITY_EXTRAPL)
    
    elapsed = time() - start_time
    logger.info(f"Model simulation data prepared successfully ({elapsed:.1f}s)")


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Prepare satellite and model data for MALI forcing generation')
    
    parser.add_argument('--satobs', action='store_true',
                        help='Prepare satellite observations')
    parser.add_argument('--model', action='store_true',
                        help='Prepare model simulation data (default if no flags given)')
    parser.add_argument('--init-dirs', action='store_true',
                        help='Initialize required directories before processing')
    
    args = parser.parse_args()
    
    # If no flags specified, default to model processing
    if not args.satobs and not args.model:
        args.model = True
    
    # Initialize directories if requested
    if args.init_dirs:
        dirs_to_create = collect_directories(config)
        initialize_directories(dirs_to_create)
    
    # Setup logging
    output_dir = Path(config.DIR_PROCESSED)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "prepare_data")
    
    logger.info("DATA PREPARATION FOR MALI FORCING GENERATION")
    
    # Process based on flags
    if args.satobs:
        prepare_satellite_observations()
    
    if args.model:
        prepare_model_simulation()
    
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Outputs in: {config.DIR_PROCESSED}")



if __name__ == "__main__":
    main()
 