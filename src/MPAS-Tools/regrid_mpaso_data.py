#!/usr/bin/env python3
"""
Regrid MPAS-Ocean data to a regular polar grid using pyremap.

Usage:
    python regrid_mpaso_data.py --grid_file <path> --input_dir <path> --output_dir <path>
    python regrid_mpaso_data.py --grid_file <path> --input_dir <path> --dx 5 --dy 5
"""

import pyremap
import xarray as xr
import sys
import glob
import os
import gc
import argparse
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(output_dir):
    """Setup logging to file and console."""
    log_file = Path(output_dir) / f'regrid_mpaso_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Log file: {log_file}")
    return log_file


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Regrid MPAS-Ocean data to a regular polar grid')
    
    # Required arguments
    parser.add_argument('--grid_file', required=True, help='Path to MPAS mesh/grid file')
    parser.add_argument('--input_dir', required=True, help='Directory containing input MPAS data files')
    parser.add_argument('--output_dir', required=True, help='Directory to save regridded output files')
    
    # Optional grid parameters
    parser.add_argument('--dx', type=float, default=10.0, help='Grid spacing in x (km), default: 10.0')
    parser.add_argument('--dy', type=float, default=10.0, help='Grid spacing in y (km), default: 10.0')
    parser.add_argument('--lx', type=float, default=6000.0, help='Grid extent in x (km), default: 6000.0')
    parser.add_argument('--ly', type=float, default=6000.0, help='Grid extent in y (km), default: 6000.0')
    
    # Optional processing parameters
    parser.add_argument('--file_pattern', default='*.ISMF.*.nc', help='Glob pattern for input files')
    parser.add_argument('--grid_name', default=None, help='Name for input grid (auto-detected if not provided)')
    parser.add_argument('--mapping_method', default='bilinear', choices=['bilinear', 'conservative'],
                        help='Remapping method (default: bilinear)')
    
    args = parser.parse_args()
    
    # Convert to Path objects and validate
    grid_file = Path(args.grid_file)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not grid_file.exists():
        print(f"Error: Grid file not found: {grid_file}")
        sys.exit(1)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    # Determine grid name
    inGridName = args.grid_name or (grid_file.stem.split('.')[1] if '.' in grid_file.stem else grid_file.stem)
    
    logger.info("="*70)
    logger.info("MPAS-OCEAN DATA REGRIDDING")
    logger.info("="*70)
    logger.info(f"Grid file: {grid_file.name}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Grid: dx={args.dx}km, dy={args.dy}km, Lx={args.lx}km, Ly={args.ly}km")
    logger.info(f"Method: {args.mapping_method}")
    
    # Create mesh descriptors
    logger.info("Creating mesh descriptors...")
    inDescriptor = pyremap.MpasMeshDescriptor(str(grid_file), inGridName)
    outDescriptor = pyremap.get_polar_descriptor(
        Lx=args.lx, Ly=args.ly, dx=args.dx, dy=args.dy, projection='antarctic'
    )
    outGridName = outDescriptor.meshName
    
    # Create or load mapping file
    mappingFileName = f'map_{inGridName}_to_{outGridName}.nc'
    remapper = pyremap.Remapper(inDescriptor, outDescriptor, mappingFileName)
    
    if not Path(mappingFileName).exists():
        logger.info(f"Building mapping file: {mappingFileName}")
        remapper.build_mapping_file(method=args.mapping_method)
        logger.info("Mapping file created")
    else:
        logger.info(f"Using existing mapping file: {mappingFileName}")
    
    # Find and process files
    file_pattern = str(input_dir / args.file_pattern)
    input_files = sorted(glob.glob(file_pattern))
    
    if not input_files:
        logger.error(f"No files found matching: {file_pattern}")
        sys.exit(1)
    
    logger.info(f"\nProcessing {len(input_files)} file(s)...")
    logger.info("="*50)
    
    success_count = 0
    fail_count = 0
    
    for i, file in enumerate(input_files, 1):
        inFileName = os.path.basename(file)
        logger.info(f"[{i}/{len(input_files)}] {inFileName}")
        
        try:
            ds = xr.open_dataset(file)
            remappedDataset = remapper.remap(ds, renormalizationThreshold=0.01)
            remappedDataset.attrs['history'] = f'Regridded using pyremap: {" ".join(sys.argv)}'
            
            outFileName = inFileName.replace('.nc', f'.{outGridName}.nc')
            output_path = output_dir / f'Regridded_{outFileName}'
            remappedDataset.to_netcdf(output_path)
            
            logger.info(f"  > {output_path.name}")
            success_count += 1
            
            ds.close()
            remappedDataset.close()
            gc.collect()
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            fail_count += 1
    
    logger.info("="*50)
    logger.info(f"Complete! Success: {success_count}, Failed: {fail_count}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*50)
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
