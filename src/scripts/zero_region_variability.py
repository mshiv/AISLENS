#!/usr/bin/env python3
"""
Zero out temporal variability in specified ice shelf regions.

This script sets anomaly/variability values to 0 in selected regions.

Supports:
- Regular grids (x, y, Time dimensions)
- MALI unstructured grids (nCells, Time dimensions)
- Multiple shelf selection
- Optional time range (zero only specific years)

Usage:
    # Zero variability in specific shelves using GeoJSON (raster grids):
    python zero_region_variability.py input.nc output.nc --shelves "Ross Ice Shelf" "Filchner Ice Shelf"
    
    # Zero variability using region mask indices (MALI or raster):
    python zero_region_variability.py input.nc output.nc --regions 45 67 89
    
    # Zero only for specific years (keep variability outside this range):
    python zero_region_variability.py input.nc output.nc --shelves "Ross Ice Shelf" --year-start 2015 --year-end 2100
    
    # For MALI grid files with region mask:
    python zero_region_variability.py input.nc output.nc --regions 45 67 --grid-type mali --mask-file mali_regions.nc

"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aislens import config

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def detect_grid_type(ds: xr.Dataset) -> str:
    """
    Detect whether dataset is raster (x/y) or MALI (nCells).
    
    Returns:
        'raster' or 'mali'
    """
    dims = set(ds.dims.keys())
    if 'nCells' in dims:
        return 'mali'
    elif 'x' in dims and 'y' in dims:
        return 'raster'
    else:
        raise ValueError(f"Cannot detect grid type from dimensions: {dims}")


def detect_time_dim(ds: xr.Dataset) -> str:
    """Detect time dimension name."""
    for dim in ['Time', 'time', 't']:
        if dim in ds.dims:
            return dim
    raise ValueError(f"No time dimension found in {ds.dims.keys()}")


def create_shelf_mask_raster(ds: xr.Dataset, shelf_names: list, 
                              geojson_path: Path = None) -> xr.DataArray:
    """
    Create a boolean mask for specified ice shelves on a raster grid.
    
    Args:
        ds: Dataset with x, y coordinates
        shelf_names: List of ice shelf names to mask
        geojson_path: Path to ice shelf GeoJSON file
        
    Returns:
        Boolean DataArray (True where shelf is present)
    """
    if geojson_path is None:
        geojson_path = config.AISLENSConfig().FILE_ICESHELF_GEOJSON
    
    logger.info(f"Loading ice shelf geometries from {geojson_path}")
    icems = gpd.read_file(geojson_path)
    
    # Set CRS for rasterization
    if ds.rio.crs is None:
        ds = ds.rio.write_crs(config.AISLENSConfig().CRS_TARGET)
    
    combined_mask = xr.zeros_like(ds['x'].broadcast_like(ds['y']), dtype=bool)
    combined_mask = combined_mask.rename('shelf_mask')
    
    # Create coordinate arrays for the mask
    x_coords = ds['x'].values
    y_coords = ds['y'].values
    combined_mask = xr.DataArray(
        np.zeros((len(y_coords), len(x_coords)), dtype=bool),
        dims=['y', 'x'],
        coords={'y': y_coords, 'x': x_coords}
    )
    
    for shelf_name in shelf_names:
        matching = icems[icems['name'].str.lower() == shelf_name.lower()]
        if len(matching) == 0:
            logger.warning(f"Ice shelf '{shelf_name}' not found in GeoJSON. Available: {icems['name'].tolist()[:10]}...")
            continue
        
        logger.info(f"Creating mask for: {shelf_name}")
        
        # Rasterize the geometry
        from rasterio import features
        from affine import Affine
        
        # Calculate transform from coordinates
        dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1000
        dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else -1000
        transform = Affine(dx, 0, x_coords[0] - dx/2, 0, dy, y_coords[0] - dy/2)
        
        geom = matching.geometry.values[0]
        shelf_raster = features.rasterize(
            [(geom, 1)],
            out_shape=(len(y_coords), len(x_coords)),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        combined_mask = combined_mask | xr.DataArray(shelf_raster.astype(bool), dims=['y', 'x'], 
                                                       coords={'y': y_coords, 'x': x_coords})
    
    n_cells = int(combined_mask.sum().values)
    logger.info(f"Total masked cells: {n_cells}")
    
    return combined_mask


def create_region_mask_mali(ds: xr.Dataset, region_indices: list,
                            mask_file: Path = None, mask_var: str = 'regionCellMasks') -> xr.DataArray:
    """
    Create a boolean mask for specified region indices on MALI grid.
    
    Args:
        ds: Dataset with nCells dimension
        region_indices: List of region indices to mask
        mask_file: Path to NetCDF file containing region masks
        mask_var: Variable name in mask file
        
    Returns:
        Boolean DataArray (True where region is present)
    """
    if mask_file is None:
        mask_file = config.AISLENSConfig().FILE_MALI_REGION_MASKS
    
    logger.info(f"Loading region masks from {mask_file}")
    mask_ds = xr.open_dataset(mask_file)
    
    if mask_var not in mask_ds:
        # Try alternate variable names
        for alt_var in ['regionMask', 'cellMask', 'mask']:
            if alt_var in mask_ds:
                mask_var = alt_var
                break
        else:
            raise ValueError(f"Mask variable not found. Available: {list(mask_ds.data_vars)}")
    
    region_mask = mask_ds[mask_var]
    n_cells = ds.dims['nCells']
    
    combined_mask = xr.DataArray(np.zeros(n_cells, dtype=bool), dims=['nCells'])
    
    for idx in region_indices:
        logger.info(f"Adding region index: {idx}")
        if 'nRegions' in region_mask.dims:
            # Mask has shape (nRegions, nCells)
            shelf_mask = region_mask.isel(nRegions=idx).values > 0
        else:
            # Mask is a single integer array
            shelf_mask = region_mask.values == idx
        combined_mask = combined_mask | xr.DataArray(shelf_mask, dims=['nCells'])
    
    n_masked = int(combined_mask.sum().values)
    logger.info(f"Total masked cells: {n_masked}")
    
    mask_ds.close()
    return combined_mask


def get_time_mask(da: xr.DataArray, time_dim: str, 
                  year_start: int = None, year_end: int = None) -> xr.DataArray:
    """
    Create a boolean mask for timesteps within the specified year range.
    
    Args:
        da: DataArray with time dimension
        time_dim: Name of time dimension
        year_start: Start year (None = from beginning)
        year_end: End year (None = to end)
        
    Returns:
        Boolean DataArray along time dimension (True = within range)
    """
    if year_start is None and year_end is None:
        # All timesteps
        return xr.ones_like(da[time_dim], dtype=bool)
    
    time_coord = da[time_dim]
    
    # Extract years from time coordinate
    if hasattr(time_coord.values[0], 'year'):
        # cftime or datetime-like
        years = np.array([t.year for t in time_coord.values])
    elif np.issubdtype(time_coord.dtype, np.datetime64):
        years = time_coord.dt.year.values
    else:
        # Assume numeric years
        years = time_coord.values
        if years.max() < 100:
            logger.warning(f"Time values appear to be indices or fractional. Zeroing all timesteps.")
            return xr.ones_like(time_coord, dtype=bool)
    
    # Create mask
    if year_start is not None and year_end is not None:
        time_mask = (years >= year_start) & (years <= year_end)
    elif year_start is not None:
        time_mask = years >= year_start
    else:
        time_mask = years <= year_end
    
    return xr.DataArray(time_mask, dims=[time_dim], coords={time_dim: time_coord})


def zero_variability(da: xr.DataArray, mask: xr.DataArray, time_dim: str,
                     year_start: int = None, year_end: int = None) -> xr.DataArray:
    """Set anomaly/variability values to zero in masked regions and years."""
    # Create time mask if year range specified
    time_mask = get_time_mask(da, time_dim, year_start, year_end)
    
    if year_start is not None or year_end is not None:
        n_timesteps = int(time_mask.sum().values)
        logger.info(f"Zeroing variability for years {year_start or 'start'}-{year_end or 'end'} ({n_timesteps} timesteps)")
    else:
        logger.info(f"Zeroing variability for all {da[time_dim].size} timesteps")
    
    # Broadcast spatial mask to match da dimensions
    # mask is (y, x) or (nCells,)
    # da is (Time, y, x) or (Time, nCells)
    spatial_mask_broadcast = mask.broadcast_like(da)
    
    # Broadcast time mask to match da dimensions
    time_mask_broadcast = time_mask.broadcast_like(da)
    
    # Combined mask: True where BOTH spatial region matches AND time is in range
    combined_mask = spatial_mask_broadcast & time_mask_broadcast
    
    # Apply: where combined mask is True, set to 0; else keep original
    result = xr.where(combined_mask, 0.0, da)
    
    # Preserve attributes
    result.attrs = da.attrs.copy()
    
    return result


def process_file(input_file: Path, output_file: Path, 
                 variable: str,
                 shelf_names: list = None,
                 region_indices: list = None,
                 grid_type: str = None,
                 year_start: int = None,
                 year_end: int = None,
                 geojson_path: Path = None,
                 mask_file: Path = None,
                 mask_var: str = 'regionCellMasks') -> None:
    """
    Main function.
    """
    logger.info(f"Opening input file: {input_file}")
    ds = xr.open_dataset(input_file)
    
    # Detect grid type if not specified
    if grid_type is None:
        grid_type = detect_grid_type(ds)
    logger.info(f"Grid type: {grid_type}")
    
    # Detect time dimension
    time_dim = detect_time_dim(ds)
    logger.info(f"Time dimension: {time_dim}")
    
    # Check variable exists
    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found. Available: {list(ds.data_vars)}")
    
    da = ds[variable]
    logger.info(f"Processing variable '{variable}' with shape {da.shape}")
    
    # Determine which masking approach to use
    if shelf_names and grid_type == 'raster':
        # Use GeoJSON-based masking for raster grids
        logger.info("Using GeoJSON-based masking for raster grid")
        mask = create_shelf_mask_raster(ds, shelf_names, geojson_path)
        mask_description = f"shelves: {', '.join(shelf_names)}"
    elif region_indices:
        # Use region index masking (works for both grid types)
        logger.info("Using region index masking")
        if grid_type == 'mali':
            mask = create_region_mask_mali(ds, region_indices, mask_file, mask_var)
        else:
            # For raster, need a different approach - use region mask file
            logger.info("Loading region mask for raster grid")
            if mask_file is None:
                mask_file = config.AISLENSConfig().FILE_RASTER_REGION_MASKS
            mask_ds = xr.open_dataset(mask_file)
            # Find the right mask variable
            mask_var_found = None
            for v in ['regionMask', 'region', 'mask', 'iceShelfMask']:
                if v in mask_ds:
                    mask_var_found = v
                    break
            if mask_var_found is None:
                raise ValueError(f"No region mask variable found in {mask_file}")
            
            region_mask = mask_ds[mask_var_found]
            mask = xr.zeros_like(region_mask, dtype=bool)
            for idx in region_indices:
                mask = mask | (region_mask == idx)
            mask_ds.close()
        mask_description = f"region indices: {region_indices}"
    elif shelf_names and grid_type == 'mali':
        raise ValueError("For MALI grids, use --regions with region indices instead of --shelves")
    else:
        raise ValueError("Must specify either --shelves (for raster) or --regions")
    
    # Zero variability
    logger.info(f"Zeroing variability in {mask_description}")
    da_processed = zero_variability(da, mask, time_dim, year_start, year_end)
    
    # Update dataset
    ds_out = ds.copy()
    ds_out[variable] = da_processed
    
    # Add processing history
    history_entry = (
        f"{datetime.now().isoformat()}: zero_region_variability.py - "
        f"Set anomaly values to 0 in {mask_description}"
    )
    if year_start or year_end:
        history_entry += f" for years {year_start or 'start'}-{year_end or 'end'}"
    
    if 'history' in ds_out.attrs:
        ds_out.attrs['history'] = history_entry + '\n' + ds_out.attrs['history']
    else:
        ds_out.attrs['history'] = history_entry
    
    # Save output
    logger.info(f"Saving to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Use compression for efficiency
    encoding = {variable: {'zlib': True, 'complevel': 4}}
    ds_out.to_netcdf(output_file, encoding=encoding)
    
    ds.close()
    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', type=Path,
                        help='Input NetCDF file')
    parser.add_argument('output', type=Path,
                        help='Output NetCDF file')
    
    parser.add_argument('--var', '-v', dest='variable',
                        default='floatingBasalMassBalAdjustment',
                        help='Variable to process (default: floatingBasalMassBalAdjustment)')
    
    # Region selection (mutually exclusive groups)
    region_group = parser.add_argument_group('Region Selection')
    region_group.add_argument('--shelves', '-s', nargs='+', dest='shelf_names',
                              help='Ice shelf names (for raster grids). '
                                   'Example: --shelves "Ross Ice Shelf" "Filchner Ice Shelf"')
    region_group.add_argument('--regions', '-r', nargs='+', type=int, dest='region_indices',
                              help='Region indices (for MALI grids or when using mask file). '
                                   'Example: --regions 45 67 89')
    
    # Grid configuration
    grid_group = parser.add_argument_group('Grid Configuration')
    grid_group.add_argument('--grid-type', choices=['raster', 'mali'],
                            help='Grid type (auto-detected if not specified)')
    grid_group.add_argument('--geojson', type=Path, dest='geojson_path',
                            help='Path to ice shelf GeoJSON file (for raster --shelves)')
    grid_group.add_argument('--mask-file', type=Path,
                            help='Path to region mask NetCDF file (for --regions)')
    grid_group.add_argument('--mask-var', default='regionCellMasks',
                            help='Variable name in mask file (default: regionCellMasks)')
    
    # Time range for zeroing
    time_group = parser.add_argument_group('Time Range (optional - zero only specific years)')
    time_group.add_argument('--year-start', type=int,
                            help='Start year to zero variability (default: all years)')
    time_group.add_argument('--year-end', type=int,
                            help='End year to zero variability (default: all years)')
    
    # Other options
    parser.add_argument('--verbose', '-V', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Validate arguments
    if not args.shelf_names and not args.region_indices:
        parser.error("Must specify either --shelves or --regions")
    
    if args.year_start and not args.year_end:
        parser.error("--year-start requires --year-end")
    if args.year_end and not args.year_start:
        parser.error("--year-end requires --year-start")
    
    process_file(
        input_file=args.input,
        output_file=args.output,
        variable=args.variable,
        shelf_names=args.shelf_names,
        region_indices=args.region_indices,
        grid_type=args.grid_type,
        year_start=args.year_start,
        year_end=args.year_end,
        geojson_path=args.geojson_path,
        mask_file=args.mask_file,
        mask_var=args.mask_var
    )


if __name__ == '__main__':
    main()
