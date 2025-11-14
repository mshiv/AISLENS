#!/usr/bin/env python3
"""
Concatenate regridded MPAS-Ocean data files along time dimension.

Usage:
    python concatenate_mpaso_data.py --input_dir <path> --output_file <path>
"""

import xarray as xr
import sys
import glob
import os
import argparse
import logging
from pathlib import Path
from aislens.utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Concatenate MPAS-Ocean data files along time dimension')
    
    parser.add_argument('--input_dir', required=True, help='Directory containing files to concatenate')
    parser.add_argument('--output_file', required=True, help='Output filename for concatenated dataset')
    parser.add_argument('--file_pattern', default='Regridded_*.nc', help='Glob pattern (default: "Regridded_*.nc")')
    parser.add_argument('--time_dim', default='Time', help='Time dimension name (default: "Time")')
    
    args = parser.parse_args()
    
    # Validate inputs
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(output_file.parent, "concatenate_mpaso")
    
    logger.info("MPAS-OCEAN DATA CONCATENATION")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Pattern: {args.file_pattern}")
    
    # Find files
    file_pattern = str(input_dir / args.file_pattern)
    input_files = sorted(glob.glob(file_pattern))
    
    if not input_files:
        logger.error(f"No files found matching: {file_pattern}")
        sys.exit(1)
    
    logger.info(f"Found {len(input_files)} file(s)")
    
    # Load datasets
    datasets = []
    for i, file in enumerate(input_files, 1):
        filename = os.path.basename(file)
        logger.info(f"[{i}/{len(input_files)}] {filename}")
        try:
            ds = xr.open_dataset(file)
            datasets.append(ds)
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            sys.exit(1)
    
    # Concatenate
    logger.info(f"Concatenating along '{args.time_dim}' dimension...")
    try:
        combined_ds = xr.concat(datasets, dim=args.time_dim)
        logger.info(f"Combined shape: {dict(combined_ds.dims)}")
    except Exception as e:
        logger.error(f"Concatenation failed: {e}")
        sys.exit(1)
    
    # Save
    logger.info(f"Writing to: {output_file.name}")
    try:
        combined_ds.to_netcdf(output_file)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Complete! Size: {file_size_mb:.2f} MB")
    except Exception as e:
        logger.error(f"Write failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

