#!/usr/bin/env python3
"""
Combine two forcing NetCDF files using NCO tools.

Extracts time slices from two files and adds them together using NCO (NetCDF Operators).
Useful for combining ISMIP6 baseline forcings with synthetic variability realizations.

Prerequisites: NCO tools (ncks, ncbo) must be installed

Usage:
    python combine_forcing_files.py
"""

import argparse
import logging
import subprocess
from pathlib import Path
import xarray as xr

from aislens.utils import setup_logging
from aislens.config import config

logger = logging.getLogger(__name__)


def combine_forcing_files_nco(file1_path, file2_path, output_path, 
                              time1_start, time1_end, time2_start, time2_end):
    """
    Combine two forcing files using NCO, adding specific timesteps.
    
    Args:
        file1_path: Path to first NetCDF file
        file2_path: Path to second NetCDF file  
        output_path: Path for output file
        time1_start, time1_end: Time indices for first file (inclusive)
        time2_start, time2_end: Time indices for second file (inclusive)
    """
    logger.info(f"Combining forcing files:")
    logger.info(f"  File 1: {file1_path} [time {time1_start}:{time1_end}]")
    logger.info(f"  File 2: {file2_path} [time {time2_start}:{time2_end}]")
    logger.info(f"  Output: {output_path}")
    
    temp_file1 = Path(output_path).parent / "temp_file1.nc"
    temp_file2 = Path(output_path).parent / "temp_file2.nc"
    
    try:
        logger.debug("Extracting time slices with ncks...")
        subprocess.run(
            ["ncks", "-O", "-d", f"Time,{time1_start},{time1_end}", 
             str(file1_path), str(temp_file1)],
            check=True, capture_output=True, text=True
        )
        subprocess.run(
            ["ncks", "-O", "-d", f"Time,{time2_start},{time2_end}", 
             str(file2_path), str(temp_file2)],
            check=True, capture_output=True, text=True
        )
        
        logger.debug("Adding files with ncbo...")
        subprocess.run(
            ["ncbo", "-O", "-o", str(output_path), "--op_typ=add", 
             str(temp_file1), str(temp_file2)],
            check=True, capture_output=True, text=True
        )
        
        logger.info(f"Successfully combined files → {output_path}")
    finally:
        # Clean up temporary files
        for temp_file in [temp_file1, temp_file2]:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug(f"Cleaned up {temp_file.name}")


def main():
    """Main function with example usage."""
    parser = argparse.ArgumentParser(
        description='Combine two forcing NetCDF files using NCO tools'
    )
    parser.add_argument('--file1', type=Path, 
                       help='First input file (default: ISMIP6 SSP585 UKESM trend)')
    parser.add_argument('--file2', type=Path,
                       help='Second input file (default: forcing_realization_0.nc)')
    parser.add_argument('--output', type=Path,
                       help='Output file path (default: combined_forcing.nc)')
    parser.add_argument('--time1-start', type=int, default=0,
                       help='Start time index for file1 (default: 0)')
    parser.add_argument('--time1-end', type=int, default=50,
                       help='End time index for file1 (default: 50)')
    parser.add_argument('--time2-start', type=int, default=10,
                       help='Start time index for file2 (default: 10)')
    parser.add_argument('--time2-end', type=int, default=60,
                       help='End time index for file2 (default: 60)')
    args = parser.parse_args()
    
    # Default file paths
    file1 = args.file1 or config.DIR_MALI_ISMIP6_FORCINGS / "ISMIP6_SSP585_UKESM_FLOATINGBMB_TREND.nc"
    file2 = args.file2 or config.DIR_FORCINGS / "forcing_realization_0.nc"
    output = args.output or config.DIR_MALI_ISMIP6_FORCINGS / "combined_forcing.nc"
    
    # Setup logging
    output.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(output.parent, "combine_forcing_files")
    
    logger.info("="*60)
    logger.info("COMBINE FORCING FILES")
    logger.info("="*60)
    
    combine_forcing_files_nco(file1, file2, output,
                             args.time1_start, args.time1_end,
                             args.time2_start, args.time2_end)
    
    logger.info("="*60)


if __name__ == "__main__":
    main()

