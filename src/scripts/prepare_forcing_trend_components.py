#!/usr/bin/env python3
"""
Detrend ISMIP6 forcing trend components using breakpoint detection.

Processes ISMIP6 forcing data (SSP585, SSP126, etc.) by detrending with polynomial
fit and breakpoint detection (ruptures algorithm).

Input: NetCDF file with floatingBasalMassBal variable and Time dimension
Output: NetCDF file with trend component (detrended data)

Usage:
    python prepare_forcing_trend_components.py input.nc output.nc [--method {vectorized,timeseries}]
    python prepare_forcing_trend_components.py  # Uses default SSP585 file
"""

import argparse
import logging
from pathlib import Path
from time import time
import xarray as xr

from aislens.config import config
from aislens.dataprep import detrend_with_breakpoints_vectorized, detrend_with_breakpoints_ts
from aislens.utils import setup_logging

logger = logging.getLogger(__name__)


def detrend_forcing(forcing_file_path, output_path, method='vectorized'):
    """Detrend forcing data using breakpoint detection."""
    logger.info(f"Loading {forcing_file_path}...")
    ds = xr.open_dataset(forcing_file_path)
    ds[config.MALI_FLOATINGBMB_VAR] = (ds[config.MALI_FLOATINGBMB_VAR].isel(Time=0) - 
                                        ds[config.MALI_FLOATINGBMB_VAR])
    
    if method == 'vectorized':
        logger.info("Detrending (vectorized - all spatial points)...")
        detrended_data = detrend_with_breakpoints_vectorized(
            ds[config.MALI_FLOATINGBMB_VAR], dim="Time", deg=1, model="rbf", penalty=10
        )
    else:
        logger.info("Detrending (time series - spatial mean)...")
        spatial_dims = [dim for dim in ds[config.MALI_FLOATINGBMB_VAR].dims if dim != 'Time']
        spatial_mean_ts = ds[config.MALI_FLOATINGBMB_VAR].mean(dim=spatial_dims)
        detrended_data = detrend_with_breakpoints_ts(
            spatial_mean_ts, dim="Time", deg=1, model="rbf", penalty=10
        )
    
    trend = (ds - detrended_data.to_dataset(name=config.MALI_FLOATINGBMB_VAR))
    trend = trend.rename({config.MALI_FLOATINGBMB_VAR: config.AISLENS_FLOATINGBMB_VAR})
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    trend.to_netcdf(output_path)
    logger.info(f"Trend saved as {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detrend ISMIP6 forcing trend components')
    parser.add_argument('input_file', nargs='?', type=Path,
                       default=config.FILE_ISMIP6_SSP585_FORCING,
                       help='Input forcing file (default: SSP585 from config)')
    parser.add_argument('output_file', nargs='?', type=Path,
                       help='Output trend file (default: auto-named in ISMIP6 forcings dir)')
    parser.add_argument('--method', choices=['vectorized', 'timeseries'], default='vectorized',
                       help='Detrending method (default: vectorized)')
    args = parser.parse_args()
    
    # Auto-generate output name if not provided
    if args.output_file is None:
        input_stem = args.input_file.stem
        output_name = f"{input_stem}_TREND.nc"
        args.output_file = Path(config.DIR_MALI_ISMIP6_FORCINGS) / output_name
    
    setup_logging(args.output_file.parent, "prepare_forcing_trend_components")
    
    logger.info("="*60)
    logger.info("FORCING TREND COMPONENT DETRENDING")
    logger.info(f"Input: {args.input_file}")
    logger.info(f"Output: {args.output_file}")
    logger.info(f"Method: {args.method}")
    logger.info("="*60)
    
    start = time()
    detrend_forcing(args.input_file, args.output_file, args.method)
    logger.info("="*60)
    logger.info(f"COMPLETE ({time() - start:.1f}s)")
    logger.info("="*60)

