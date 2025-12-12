#!/usr/bin/env python3
"""
Combine SSP585 trend component with AISLENS forcing files using Python/xarray.

This is the Python/xarray equivalent of the NCO-based shell script.

Usage:
    python combine_ssp585_forcing_python.py --trend-file trend.nc --forcing-file forcing.nc --output-file combined.nc
    python combine_ssp585_forcing_python.py --ensemble-dir /path/to/ensembles --trend-file trend.nc
"""

import logging
import xarray as xr
import argparse
from pathlib import Path
import numpy as np

from aislens.utils import setup_logging

logger = logging.getLogger(__name__)

# --- Configurable indices (change here for different split points) ---
# Inclusive start index for the combined mid-period (e.g. 168)
COMBINED_START_INDEX = 180
# Exclusive end index for slicing the mid-period (e.g. 3600 -> slices stop at 3599)
COMBINED_END_EXCLUSIVE = 3600
# ---------------------------------------------------------------------------

def combine_ssp585_forcing_xarray(trend_file_path, forcing_file_path, output_file_path):
    """
    Python equivalent of the NCO-based SSP585 forcing combination.
    
    Combines SSP585 trend with AISLENS forcing:
    - Timesteps 0-167 (2000-2014): Original forcing only
    - Timesteps 168-3599 (2015-2299): Original forcing + SSP585 trend
    - Timestep 3600 (2300): Original forcing only
    """
    logger.info(f"Loading trend file: {trend_file_path}")
    trend_ds = xr.open_dataset(trend_file_path)
    
    logger.info(f"Loading forcing file: {forcing_file_path}")
    forcing_ds = xr.open_dataset(forcing_file_path)
    
    if "floatingBasalMassBalAdjustment" not in trend_ds.data_vars:
        if "floatingBasalMassBalApplied" in trend_ds.data_vars:
            logger.info("Renaming floatingBasalMassBalApplied → floatingBasalMassBalAdjustment")
            trend_ds = trend_ds.rename({"floatingBasalMassBalApplied": "floatingBasalMassBalAdjustment"})
        else:
            raise ValueError(f"Expected variable not found. Available: {list(trend_ds.data_vars.keys())}")
    
    if "floatingBasalMassBalAdjustment" not in forcing_ds.data_vars:
        raise ValueError(f"floatingBasalMassBalAdjustment not found. Available: {list(forcing_ds.data_vars.keys())}")
    
    logger.info("Extracting overlapping time periods and aligning Time coordinates...")
    # Forcing mid-period is indices COMBINED_START_INDEX .. (COMBINED_END_EXCLUSIVE-1)
    forcing_subset = forcing_ds.isel(Time=slice(COMBINED_START_INDEX, COMBINED_END_EXCLUSIVE))

    # Prefer aligning by exact Time values if possible (handles differing calendars/lengths)
    try:
        trend_times = trend_ds.Time.values
        forcing_times = forcing_subset.Time.values
        common_times = np.intersect1d(trend_times, forcing_times)
    except Exception:
        common_times = np.array([])

    if common_times.size > 0:
        logger.info(f"Found {common_times.size} exact overlapping Time values; selecting intersection")
        trend_subset = trend_ds.sel(Time=common_times)
        forcing_subset = forcing_subset.sel(Time=common_times)
    else:
        # Fall back to length-based trimming: make both equal to the shorter length
        tlen = len(trend_ds.Time)
        flen = len(forcing_subset.Time)
        minlen = min(tlen, flen)
        logger.warning(f"No exact Time intersection found; trimming to min length={minlen} (trend={tlen}, forcing={flen})")
        trend_subset = trend_ds.isel(Time=slice(0, minlen))
        forcing_subset = forcing_subset.isel(Time=slice(0, minlen))

    logger.debug(f"Aligned Trend: {len(trend_subset.Time)} timesteps, Forcing: {len(forcing_subset.Time)} timesteps")

    logger.info("Adding floatingBasalMassBalAdjustment variables...")
    # Ensure trend time coordinate uses the forcing subset times for safe arithmetic
    forcing_time_coord = forcing_subset.Time
    trend_subset_aligned = trend_subset.assign_coords(Time=forcing_time_coord)
    
    combined_subset = forcing_subset.copy()
    combined_subset["floatingBasalMassBalAdjustment"] = (
        forcing_subset["floatingBasalMassBalAdjustment"] + 
        trend_subset_aligned["floatingBasalMassBalAdjustment"]
    )
    
    logger.info("Creating final output with early period...")
    early_period = forcing_ds.isel(Time=slice(0, COMBINED_START_INDEX))
    logger.debug(f"Early period: {len(early_period.Time)} timesteps")
    
    try:
        final_ds = xr.concat([early_period, combined_subset], dim="Time")
        logger.info("Concatenated periods successfully")
    except Exception as e:
        logger.warning(f"Direct concatenation failed ({e}), using fallback assignment")
        final_ds = forcing_ds.copy()
        # Ensure the Time index slice we write to matches the combined_subset length
        start_idx = COMBINED_START_INDEX
        end_idx = start_idx + len(combined_subset.Time) - 1
        logger.debug(f"Fallback writing to Time indices {start_idx}:{end_idx} (inclusive)")
        try:
            for var_name in combined_subset.data_vars:
                if var_name in final_ds.data_vars:
                    # Use positional indexing to avoid issues with mismatched labels
                    final_ds[var_name].values[start_idx:(start_idx + len(combined_subset.Time)), ...] = combined_subset[var_name].values
                else:
                    logger.warning(f"Variable {var_name} not in original forcing")
            logger.info("Fallback assignment completed")
        except Exception as e2:
            logger.error(f"Fallback assignment failed: {e2}")
            raise
    
    logger.info("Verifying final output...")
    final_time_size = len(final_ds.Time)
    if final_time_size == 3600:
        logger.info(f"Correct Time dimension: {final_time_size}")
    else:
        logger.warning(f"Expected 3600 timesteps, got {final_time_size}")
    
    if "floatingBasalMassBalAdjustment" not in final_ds.data_vars:
        raise ValueError("Output verification failed - variable missing")
    
    logger.debug(f"Time range: {final_ds.Time.values[0]} to {final_ds.Time.values[-1]}")
    
    logger.info(f"Saving → {output_file_path}")
    final_ds.to_netcdf(output_file_path)
    
    logger.info("Processing complete!")
    logger.info("Time breakdown:")
    logger.info("  2000-2014 (0-167): Original forcing")
    logger.info("  2015-2299 (168-3599): Original + SSP585 trend")
    logger.info("  2300 (3600): Original forcing")
    
    return final_ds

def process_single_ensemble(ensemble_dir, trend_file_path, ensemble_name, ensemble_num):
    """Process a single ensemble member."""
    logger.info(f"Processing {ensemble_name} (Ensemble Member {ensemble_num})")
    
    ensemble_path = Path(ensemble_dir) / ensemble_name
    forcing_file = ensemble_path / f"AIS_4to20km_r01_20220907_AISLENS-Forcing_{ensemble_num}.nc"
    output_file = ensemble_path / f"AIS_4to20km_r01_20220907_AISLENS-Forcing_{ensemble_num}_combined.nc"
    
    logger.info(f"Ensemble directory: {ensemble_path}")
    logger.info(f"Input: trend={trend_file_path}, forcing={forcing_file}")
    logger.info(f"Output: {output_file}")
    
    if not Path(trend_file_path).exists():
        raise FileNotFoundError(f"Trend file not found: {trend_file_path}")
    if not forcing_file.exists():
        raise FileNotFoundError(f"Forcing file not found: {forcing_file}")
    
    result = combine_ssp585_forcing_xarray(trend_file_path, forcing_file, output_file)
    logger.info(f"{ensemble_name} completed successfully!")
    return result

def process_all_ensembles(ensemble_parent_dir, trend_file_path, ensemble_members=None):
    """Process multiple ensemble members automatically."""
    if ensemble_members is None:
        ensemble_members = ["SSP585-EM1", "SSP585-EM2", "SSP585-EM4", "SSP585-EM6", "SSP585-EM8"]
    
    logger.info("SSP585 TREND + AISLENS FORCING COMBINATION (MULTI-ENSEMBLE)")
    logger.info(f"Parent directory: {ensemble_parent_dir}")
    logger.info(f"Trend file: {trend_file_path}")
    logger.info(f"Processing {len(ensemble_members)} ensemble members:")
    for ensemble in ensemble_members:
        logger.info(f"  - {ensemble}")
    
    successful, failed, failed_ensembles = 0, 0, []
    
    for ensemble_name in ensemble_members:
        try:
            ensemble_num = ensemble_name.replace("SSP585-EM", "")
            process_single_ensemble(ensemble_parent_dir, trend_file_path, ensemble_name, ensemble_num)
            successful += 1
            logger.info(f"✓ {ensemble_name}: SUCCESS")
        except Exception as e:
            logger.error(f"✗ {ensemble_name}: FAILED - {e}")
            failed += 1
            failed_ensembles.append(ensemble_name)
    
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total: {len(ensemble_members)} | Successful: {successful} | Failed: {failed}")
    
    if failed:
        logger.error("Failed ensemble members:")
        for f in failed_ensembles:
            logger.error(f"  - {f}")
        return False
    else:
        logger.info("All ensemble members processed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Combine SSP585 trend with AISLENS forcing files')
    parser.add_argument('--trend-file', required=True,
                       help='Path to SSP585 trend file')
    parser.add_argument('--forcing-file',
                       help='Path to AISLENS forcing file (single file processing)')
    parser.add_argument('--output-file',
                       help='Path for output combined file (single file processing)')
    parser.add_argument('--ensemble-dir',
                       help='Parent directory containing ensemble subdirectories')
    parser.add_argument('--ensemble-members', nargs='+',
                       default=["SSP585-EM1", "SSP585-EM2", "SSP585-EM4", "SSP585-EM6", "SSP585-EM8"],
                       help='List of ensemble member names to process')
    args = parser.parse_args()
    
    output_dir = Path(args.output_file).parent if args.output_file else Path(args.ensemble_dir) if args.ensemble_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "combine_ssp585_forcing")
    
    if not Path(args.trend_file).exists():
        raise FileNotFoundError(f"Trend file not found: {args.trend_file}")
    
    if args.ensemble_dir:
        if not Path(args.ensemble_dir).exists():
            raise FileNotFoundError(f"Ensemble directory not found: {args.ensemble_dir}")
        success = process_all_ensembles(args.ensemble_dir, args.trend_file, args.ensemble_members)
        exit(0 if success else 1)
    elif args.forcing_file and args.output_file:
        if not Path(args.forcing_file).exists():
            raise FileNotFoundError(f"Forcing file not found: {args.forcing_file}")
        combine_ssp585_forcing_xarray(args.trend_file, args.forcing_file, args.output_file)
    else:
        logger.error("Must specify either:")
        logger.error("  1. --forcing-file and --output-file for single file processing, OR")
        logger.error("  2. --ensemble-dir for multi-ensemble processing")
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()
