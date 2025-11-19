#!/usr/bin/env python3
"""
Compute time-mean of deseasonalized model data for draft dependence calculations.

This script pre-computes the time-mean in a memory-efficient way by:
1. Processing one variable at a time
2. Using very small spatial chunks
3. Writing directly to disk with delayed computation

Usage:
    python prepare_time_mean.py [--start-year YYYY] [--end-year YYYY] [--output OUTPUT.nc]
"""

import argparse
import logging
import sys
from pathlib import Path
import xarray as xr
import numpy as np

from aislens.dataprep import detrend_dim, deseasonalize
from aislens.utils import subset_dataset_by_time, write_crs, setup_logging
from aislens.config import config

logger = logging.getLogger(__name__)


def compute_time_mean_variable(data_array, var_name, time_dim, output_file):
    """
    Compute time-mean for a single variable using ultra-small chunks.
    Writes directly to NetCDF to avoid memory buildup.
    """
    logger.info(f"Computing time-mean for variable: {var_name}")
    
    # Get dimensions
    ny, nx = len(data_array.y), len(data_array.x)
    
    # Initialize output array
    result = np.full((ny, nx), np.nan, dtype=np.float32)
    
    # Process in very small spatial chunks
    chunk_size = 50  # Process 50x50 at a time
    n_chunks_x = int(np.ceil(nx / chunk_size))
    n_chunks_y = int(np.ceil(ny / chunk_size))
    total_chunks = n_chunks_x * n_chunks_y
    
    logger.info(f"  Processing {total_chunks} chunks ({chunk_size}x{chunk_size} each)...")
    
    for i in range(n_chunks_x):
        for j in range(n_chunks_y):
            x_start = i * chunk_size
            x_end = min((i + 1) * chunk_size, nx)
            y_start = j * chunk_size
            y_end = min((j + 1) * chunk_size, ny)
            
            chunk_num = i * n_chunks_y + j + 1
            if chunk_num % 50 == 0 or chunk_num == 1:
                logger.info(f"    Chunk {chunk_num}/{total_chunks}")
            
            # Extract tiny chunk and compute mean
            chunk = data_array.isel(x=slice(x_start, x_end), y=slice(y_start, y_end))
            # Rechunk time to very small pieces
            chunk = chunk.chunk({time_dim: 12, 'x': -1, 'y': -1})
            chunk_mean = chunk.mean(dim=time_dim).compute()
            
            # Store result
            result[y_start:y_end, x_start:x_end] = chunk_mean.values
    
    logger.info(f"  Time-mean computed for {var_name}")
    return result


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Compute time-mean for draft dependence calculations')
    
    parser.add_argument('--start-year', type=int, default=None,
                        help=f'Start year for time subsetting (default: {config.SORRM_START_YEAR})')
    parser.add_argument('--end-year', type=int, default=None,
                        help=f'End year for time subsetting (default: {config.SORRM_END_YEAR})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: data/interim/draft_dependence/_temp_time_mean.nc)')
    
    args = parser.parse_args()
    
    # Use config defaults if not specified
    start_year = args.start_year if args.start_year is not None else config.SORRM_START_YEAR
    end_year = args.end_year if args.end_year is not None else config.SORRM_END_YEAR
    
    # Setup output
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = config.DIR_ICESHELF_DEDRAFT_MODEL / '_temp_time_mean.nc'
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    output_dir = Path(config.DIR_PROCESSED)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "prepare_time_mean")
    
    logger.info("TIME-MEAN PREPROCESSOR FOR DRAFT DEPENDENCE")
    logger.info(f"Time range: {start_year}-{end_year}")
    logger.info(f"Output: {output_file}")
    
    # Check if output already exists
    if output_file.exists():
        logger.warning(f"Output file already exists: {output_file}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted")
            return
        output_file.unlink()
    
    # Step 1: Load and subset model data
    logger.info(f"Loading model: {config.FILE_MPASO_MODEL}")
    model = xr.open_dataset(config.FILE_MPASO_MODEL, chunks={config.TIME_DIM: 36, 'x': 250, 'y': 250})
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
    
    # Rechunk after deseasonalize to prevent large chunks
    logger.info("Rechunking after deseasonalize...")
    model_deseasonalized = model_deseasonalized.chunk({config.TIME_DIM: 36, 'x': 250, 'y': 250})
    
    # Step 3: Compute time-mean for each variable
    logger.info("Computing time-mean (this will take 1-3 hours)...")
    
    results = {}
    for var in model_deseasonalized.data_vars:
        result = compute_time_mean_variable(
            model_deseasonalized[var], 
            var, 
            config.TIME_DIM,
            output_file
        )
        results[var] = result
    
    # Step 4: Create dataset and save
    logger.info("Assembling dataset...")
    data_vars = {}
    for var in model_deseasonalized.data_vars:
        data_vars[var] = (('y', 'x'), results[var])
    
    time_mean_ds = xr.Dataset(
        data_vars,
        coords={
            'x': model_deseasonalized.x,
            'y': model_deseasonalized.y
        }
    )
    
    # Copy attributes
    for var in model_deseasonalized.data_vars:
        time_mean_ds[var].attrs = model_deseasonalized[var].attrs
    time_mean_ds.attrs = model_deseasonalized.attrs
    
    # Add metadata
    time_mean_ds.attrs['time_range'] = f'{start_year}-{end_year}'
    time_mean_ds.attrs['description'] = 'Time-mean of detrended and deseasonalized model data'
    
    # Save to file
    logger.info(f"Saving to: {output_file}")
    time_mean_ds.to_netcdf(output_file)
    
    # Report file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Time-mean saved successfully ({file_size_mb:.1f} MB)")
    logger.info(f"Dataset dimensions: {dict(time_mean_ds.dims)}")
    logger.info(f"Variables: {list(time_mean_ds.data_vars)}")
    logger.info("Complete! Use this file with prepare_model_sim.py")


if __name__ == "__main__":
    main()
