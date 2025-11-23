#!/usr/bin/env python3
"""
Prepare satellite observation data for MALI forcing generation.

This script processes Paolo et al. 2023 satellite observations by:
1. Loading the satellite observation dataset
2. Detrending along the time dimension (retaining mean value)
3. Deseasonalizing the data
4. Saving the prepared dataset

TODO:
- Take the time-mean of the dataset
- Ensure that melt is converted to flux and is in SI units

Usage:
    python prepare_satobs.py [--init-dirs]
"""

import argparse
import logging
from pathlib import Path
from time import time
import numpy as np
import xarray as xr
import geopandas as gpd

from aislens.dataprep import detrend_dim, deseasonalize
from aislens.utils import collect_directories, initialize_directories, setup_logging
from aislens.config import config

logger = logging.getLogger(__name__)


def prepare_satellite_observations():
    """Prepare satellite observations by detrending and deseasonalizing."""
    start_time = time()
    logger.info("Preparing satellite observations...")
    logger.debug(f"  Loading from: {config.FILE_PAOLO23_SATOBS}")
    
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS)
    logger.info("  Satellite observations loaded")
    
    # Detrend along time dimension
    logger.info("  Detrending...")
    satobs_deseasonalized = satobs.copy()
    satobs_detrended = detrend_dim(satobs_deseasonalized[config.SATOBS_FLUX_VAR], 
                                   dim=config.TIME_DIM, deg=1)
    logger.debug("  Detrending complete")
    
    # Deseasonalize
    logger.info("  Deseasonalizing...")
    satobs_deseasonalized[config.SATOBS_FLUX_VAR] = deseasonalize(satobs_detrended)
    logger.debug("  Deseasonalization complete")
    
    # Save prepared data
    logger.info(f"  Saving to: {config.FILE_PAOLO23_SATOBS_PREPARED}")
    satobs_deseasonalized.to_netcdf(config.FILE_PAOLO23_SATOBS_PREPARED)
    
    elapsed = time() - start_time
    logger.info(f"Satellite observations prepared successfully ({elapsed:.1f}s)")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Prepare satellite observations for MALI forcing generation'
    )
    parser.add_argument('--init-dirs', action='store_true',
                       help='Initialize required directories before processing')
    args = parser.parse_args()
    
    # Initialize directories if requested
    if args.init_dirs:
        dirs_to_create = collect_directories(config)
        initialize_directories(dirs_to_create)
    
    # Setup logging
    output_dir = Path(config.DIR_PROCESSED)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "prepare_satobs")
    
    logger.info("=" * 60)
    logger.info("SATELLITE OBSERVATION DATA PREPARATION")
    logger.info("=" * 60)
    
    prepare_satellite_observations()
    
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Output: {config.FILE_PAOLO23_SATOBS_PREPARED}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
