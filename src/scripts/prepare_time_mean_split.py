#!/usr/bin/env python3
"""
Compute time-mean from split dataset files.

This processes each file separately, then combines the results.
This is a memory-efficient version of prepare_time_mean.py for higher resolution datasets.

Usage:
    python prepare_time_mean_split.py --input-pattern "path/to/data_*.nc" --output output.nc
"""

import argparse
import logging
import sys
from pathlib import Path
import xarray as xr
import numpy as np
import glob

from aislens.dataprep import detrend_dim, deseasonalize
from aislens.utils import write_crs, setup_logging
from aislens.config import config

logger = logging.getLogger(__name__)


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Compute time-mean from split dataset files')
    
    parser.add_argument('--input-pattern', type=str, required=True,
                        help='Glob pattern for input files (e.g., "data/file_*.nc")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: data/interim/draft_dependence/_temp_time_mean.nc)')
    parser.add_argument('--skip-detrend', action='store_true',
                        help='Skip detrending (use raw data)')
    parser.add_argument('--skip-deseasonalize', action='store_true',
                        help='Skip deseasonalizing (use detrended data only)')
    parser.add_argument('--coarsen', type=int, default=1,
                        help='Coarsen factor (e.g., 2 = half resolution, 4 = quarter resolution)')
    
    args = parser.parse_args()
    
    # Setup output
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = config.DIR_ICESHELF_DEDRAFT_MODEL / '_temp_time_mean.nc'
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    output_dir = Path(config.DIR_PROCESSED)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "prepare_time_mean_split")
    
    logger.info("TIME-MEAN FROM SPLIT FILES")
    logger.info(f"Input pattern: {args.input_pattern}")
    logger.info(f"Output: {output_file}")
    
    # Find all input files
    input_files = sorted(glob.glob(args.input_pattern))
    
    if not input_files:
        logger.error(f"No files found matching pattern: {args.input_pattern}")
        sys.exit(1)
    
    logger.info(f"Found {len(input_files)} files:")
    for f in input_files:
        logger.info(f"  - {f}")
    
    # Process each file and accumulate
    logger.info("Processing files...")
    
    partial_means = []
    timestep_counts = []
    
    for idx, file_path in enumerate(input_files, 1):
        logger.info(f"[{idx}/{len(input_files)}] Processing: {Path(file_path).name}")
        
        # Load file
        ds = xr.open_dataset(file_path, chunks={config.TIME_DIM: 36, 'x': 250, 'y': 250})
        ds = write_crs(ds, config.CRS_TARGET)
        
        n_timesteps = len(ds[config.TIME_DIM])
        logger.info(f"  Timesteps: {n_timesteps}")
        
        # Optional: Coarsen FIRST (before detrend/deseasonalize for efficiency)
        if args.coarsen > 1:
            logger.info(f"  Coarsening by factor {args.coarsen}...")
            # Rechunk to smaller pieces before coarsening
            ds = ds.chunk({config.TIME_DIM: 12, 'x': 100, 'y': 100})
            ds = ds.coarsen(x=args.coarsen, y=args.coarsen, boundary='trim').mean()
            logger.info(f"  New spatial dimensions: {len(ds.x)}×{len(ds.y)}")
        
        # Rechunk after coarsening
        ds = ds.chunk({config.TIME_DIM: 36, 'x': 200, 'y': 200})
        
        # Optional: Detrend
        if not args.skip_detrend:
            logger.info(f"  Detrending...")
            ds_detrended = ds.copy()
            ds_detrended[config.SORRM_FLUX_VAR] = detrend_dim(
                ds[config.SORRM_FLUX_VAR], dim=config.TIME_DIM, deg=1
            )
            ds = ds_detrended
        
        # Optional: Deseasonalize
        if not args.skip_deseasonalize:
            logger.info(f"  Deseasonalizing...")
            ds = deseasonalize(ds)
        
        # Rechunk again after detrend/deseasonalize to prevent large chunks
        ds = ds.chunk({config.TIME_DIM: 36, 'x': 200, 'y': 200})
        
        # Compute mean for this file using spatial tiling
        logger.info(f"  Computing time-mean with spatial tiling...")
        
        # Get dimensions
        nx = len(ds.x)
        ny = len(ds.y)
        tile_size = 100
        
        # Initialize result arrays for this file
        file_result = {}
        for var in ds.data_vars:
            file_result[var] = np.full((ny, nx), np.nan, dtype=np.float32)
        
        # Process in tiles
        n_tiles_x = int(np.ceil(nx / tile_size))
        n_tiles_y = int(np.ceil(ny / tile_size))
        total_tiles = n_tiles_x * n_tiles_y
        
        logger.info(f"    Processing {total_tiles} tiles ({tile_size}x{tile_size} each)...")
        
        for i in range(n_tiles_x):
            for j in range(n_tiles_y):
                x_start = i * tile_size
                x_end = min((i + 1) * tile_size, nx)
                y_start = j * tile_size
                y_end = min((j + 1) * tile_size, ny)
                
                tile_num = i * n_tiles_y + j + 1
                if tile_num % 50 == 0 or tile_num == 1:
                    logger.info(f"      Tile {tile_num}/{total_tiles}")
                
                # Extract tile and compute mean
                tile = ds.isel(x=slice(x_start, x_end), y=slice(y_start, y_end))
                tile = tile.chunk({config.TIME_DIM: 12, 'x': -1, 'y': -1})
                tile_mean = tile.mean(dim=config.TIME_DIM).compute()
                
                # Store result
                for var in ds.data_vars:
                    file_result[var][y_start:y_end, x_start:x_end] = tile_mean[var].values
        
        # Create dataset from tiled results
        file_mean_data_vars = {}
        for var in ds.data_vars:
            file_mean_data_vars[var] = (('y', 'x'), file_result[var])
        
        file_mean = xr.Dataset(
            file_mean_data_vars,
            coords={'x': ds.x, 'y': ds.y}
        )
        
        # Copy attributes
        for var in ds.data_vars:
            file_mean[var].attrs = ds[var].attrs
        file_mean.attrs = ds.attrs
        
        partial_means.append(file_mean)
        timestep_counts.append(n_timesteps)
        
        logger.info(f"  File {idx} complete")
    
    # Compute weighted average across all files
    logger.info("Computing weighted average across all files...")
    
    total_timesteps = sum(timestep_counts)
    weights = [count / total_timesteps for count in timestep_counts]
    
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Weights: {weights}")
    
    # Initialize result with first file's structure
    result_ds = partial_means[0].copy()
    
    # Compute weighted average for each variable
    for var in result_ds.data_vars:
        logger.info(f"Averaging variable: {var}")
        weighted_sum = sum(
            partial_means[i][var] * weights[i] 
            for i in range(len(partial_means))
        )
        result_ds[var].values = weighted_sum.values
    
    # Add metadata
    result_ds.attrs['description'] = 'Time-mean computed from split files'
    result_ds.attrs['n_files'] = len(input_files)
    result_ds.attrs['total_timesteps'] = total_timesteps
    result_ds.attrs['detrended'] = not args.skip_detrend
    result_ds.attrs['deseasonalized'] = not args.skip_deseasonalize
    
    # Save to file
    logger.info(f"Saving to: {output_file}")
    result_ds.to_netcdf(output_file)
    
    # Report file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Time-mean saved successfully ({file_size_mb:.1f} MB)")
    logger.info(f"Dataset dimensions: {dict(result_ds.dims)}")
    logger.info(f"Variables: {list(result_ds.data_vars)}")
    logger.info("Complete! Use this file with prepare_model_sim.py")


if __name__ == "__main__":
    main()
