#!/usr/bin/env python3
"""
Python script to combine SSP585 trend component with AISLENS forcing files.
This is the Python/xarray equivalent of the NCO-based shell script.

Usage:
    python combine_ssp585_forcing_python.py --trend-file trend.nc --forcing-file forcing.nc --output-file combined.nc
    python combine_ssp585_forcing_python.py --ensemble-dir /path/to/ensembles --trend-file trend.nc
"""

import xarray as xr
import numpy as np
import argparse
from pathlib import Path

def combine_ssp585_forcing_xarray(trend_file_path, forcing_file_path, output_file_path):
    """
    Python equivalent of the NCO-based SSP585 forcing combination.
    
    Args:
        trend_file_path: Path to SSP585 trend file (with floatingBasalMassBalAdjustment variable)
        forcing_file_path: Path to AISLENS forcing file  
        output_file_path: Path for combined output file
    """
    print(f"Loading trend file: {trend_file_path}")
    trend_ds = xr.open_dataset(trend_file_path)
    
    print(f"Loading forcing file: {forcing_file_path}")
    forcing_ds = xr.open_dataset(forcing_file_path)
    
    # Check if trend file has the expected variable name
    if "floatingBasalMassBalAdjustment" not in trend_ds.data_vars:
        if "floatingBasalMassBalApplied" in trend_ds.data_vars:
            print("Renaming floatingBasalMassBalApplied to floatingBasalMassBalAdjustment in trend file...")
            trend_ds = trend_ds.rename({"floatingBasalMassBalApplied": "floatingBasalMassBalAdjustment"})
        else:
            raise ValueError(f"Expected variable not found in trend file. Available variables: {list(trend_ds.data_vars.keys())}")
    
    # Check if forcing file has the expected variable name
    if "floatingBasalMassBalAdjustment" not in forcing_ds.data_vars:
        raise ValueError(f"floatingBasalMassBalAdjustment not found in forcing file. Available variables: {list(forcing_ds.data_vars.keys())}")
    
    # Extract time slices (equivalent to ncks -d Time,start,end)
    print("Extracting overlapping time periods...")
    # Get all timesteps from trend (should be 3432 timesteps: 2015-2300)
    trend_subset = trend_ds  # All timesteps from trend file
    
    # Extract timesteps 168-3599 from forcing file (Jan 2015 - Dec 2299)
    # 168 = year 15 * 12 months (2015 start)
    # 3599 = 168 + 3431 (3432 timesteps - 1 for 0-based indexing)
    forcing_subset = forcing_ds.isel(Time=slice(168, 3600))  # 168-3599 inclusive
    
    # Verify dimensions match
    print(f"Trend time dimension: {len(trend_subset.Time)}")
    print(f"Forcing time dimension: {len(forcing_subset.Time)}")
    
    if len(trend_subset.Time) != len(forcing_subset.Time):
        raise ValueError(f"Time dimensions don't match: {len(trend_subset.Time)} vs {len(forcing_subset.Time)}")
    
    print("  ✓ Time dimensions match")
    
    # Add the variables (equivalent to ncbo --op_typ=add)
    print("Adding floatingBasalMassBalAdjustment variables...")
    
    # Create new time coordinate for alignment - use simple integer indices
    new_time_coord = np.arange(len(trend_subset.Time))
    
    # Align both datasets to use the same time coordinate
    trend_subset_aligned = trend_subset.assign_coords(Time=new_time_coord)
    forcing_subset_aligned = forcing_subset.assign_coords(Time=new_time_coord)
    
    # Add the variables
    combined_subset = forcing_subset_aligned.copy()
    combined_subset["floatingBasalMassBalAdjustment"] = (
        forcing_subset_aligned["floatingBasalMassBalAdjustment"] + 
        trend_subset_aligned["floatingBasalMassBalAdjustment"]
    )
    
    print("  Variables added successfully")
    
    # Create final output (equivalent to ncrcat)
    print("Creating final output with early period...")
    
    # Extract early period (timesteps 0-167) from original forcing file
    early_period = forcing_ds.isel(Time=slice(0, 168))  # 0-167 inclusive
    print(f"  Extracted early period: {len(early_period.Time)} timesteps")
    
    # Create consistent time coordinates for concatenation
    # Early period: keep original time coordinates (0-167)
    # Combined period: continue from 168 onwards
    early_time_coord = np.arange(len(early_period.Time))
    combined_time_coord = np.arange(len(early_period.Time), len(early_period.Time) + len(combined_subset.Time))
    
    # Assign consistent time coordinates
    early_period_aligned = early_period.assign_coords(Time=early_time_coord)
    combined_subset_aligned = combined_subset.assign_coords(Time=combined_time_coord)
    
    # Ensure both datasets have consistent coordinate attributes
    # Remove any problematic time coordinate attributes that might cause conflicts
    for ds in [early_period_aligned, combined_subset_aligned]:
        if 'Time' in ds.coords:
            ds.Time.attrs = {}  # Clear all time attributes to avoid conflicts
            ds.Time.encoding = {}  # Clear encoding as well
    
    # Concatenate early period with combined period
    try:
        final_ds = xr.concat([early_period_aligned, combined_subset_aligned], dim="Time")
    except Exception as concat_error:
        print(f"  Warning: Direct concatenation failed ({concat_error}), trying alternative approach...")
        
        # Alternative approach: manually combine the data
        # Create a new dataset structure
        final_ds = early_period_aligned.copy()
        
        # Extend the dataset with combined data
        for var_name in combined_subset_aligned.data_vars:
            if var_name in final_ds.data_vars:
                # Concatenate the variable data manually
                early_data = final_ds[var_name]
                combined_data = combined_subset_aligned[var_name]
                final_data = xr.concat([early_data, combined_data], dim="Time")
                final_ds[var_name] = final_data
            else:
                # Variable only exists in combined data
                # Create full array with early period filled with appropriate values
                early_shape = list(early_period_aligned[list(early_period_aligned.data_vars.keys())[0]].shape)
                combined_shape = list(combined_subset_aligned[var_name].shape)
                
                # Create early period data (filled with zeros or NaN as appropriate)
                early_fill_data = np.zeros(early_shape)
                early_fill = xr.DataArray(
                    early_fill_data,
                    dims=early_period_aligned[list(early_period_aligned.data_vars.keys())[0]].dims,
                    coords={dim: early_period_aligned.coords[dim] for dim in early_period_aligned[list(early_period_aligned.data_vars.keys())[0]].dims}
                )
                
                # Concatenate with combined data
                final_data = xr.concat([early_fill, combined_subset_aligned[var_name]], dim="Time")
                final_ds[var_name] = final_data
        
        # Update time coordinate to be continuous
        final_time_coord = np.arange(len(final_ds.Time))
        final_ds = final_ds.assign_coords(Time=final_time_coord)
    
    print(f"  Final file created with concatenated periods")
    
    # Verify final output
    print("Verifying final output...")
    final_time_size = len(final_ds.Time)
    print(f"  Final Time dimension: {final_time_size}")
    
    if final_time_size == 3600:
        print("  ✓ Final file has correct Time dimension (3600)")
    else:
        print(f"  ⚠ Warning: Expected 3600 timesteps, got {final_time_size}")
    
    # Check that the variable exists
    if "floatingBasalMassBalAdjustment" in final_ds.data_vars:
        print("  ✓ floatingBasalMassBalAdjustment variable present in output")
    else:
        print("  ✗ Error: floatingBasalMassBalAdjustment variable not found in output")
        raise ValueError("Output verification failed - variable missing")
    
    # Save to output file
    print(f"Saving to: {output_file_path}")
    final_ds.to_netcdf(output_file_path)
    
    print("✓ Processing complete!")
    print()
    print("=== COMPLETION SUMMARY ===")
    print("✓ Time alignment: Trend 2015-2300 added to Forcing 2015-2300 period")
    print(f"✓ Output file: {output_file_path}")
    print(f"✓ Final Time dimension: {final_time_size} timesteps")
    print()
    print("Time period breakdown:")
    print("  Years 2000-2014 (months 0-167):   Original forcing values only")
    print("  Years 2015-2299 (months 168-3599): Original forcing + SSP585 trend")
    print("  Year 2300 (month 3600):            Original forcing values only")
    
    return final_ds

def process_single_ensemble(ensemble_dir, trend_file_path, ensemble_name, ensemble_num):
    """Process a single ensemble member."""
    print(f"\n========================================")
    print(f"Processing {ensemble_name} (Ensemble Member {ensemble_num})")
    print(f"========================================")
    
    ensemble_path = Path(ensemble_dir) / ensemble_name
    
    # Construct file paths
    forcing_file = ensemble_path / f"AIS_4to20km_r01_20220907_AISLENS-Forcing_{ensemble_num}.nc"
    output_file = ensemble_path / f"AIS_4to20km_r01_20220907_AISLENS-Forcing_{ensemble_num}_combined.nc"
    
    print(f"Ensemble directory: {ensemble_path}")
    print(f"Input files:")
    print(f"  Trend file: {trend_file_path}")
    print(f"  Forcing file: {forcing_file}")
    print(f"Output file: {output_file}")
    
    # Check if input files exist
    if not Path(trend_file_path).exists():
        raise FileNotFoundError(f"Trend file not found: {trend_file_path}")
    
    if not forcing_file.exists():
        raise FileNotFoundError(f"Forcing file not found: {forcing_file}")
    
    # Process this ensemble
    result = combine_ssp585_forcing_xarray(trend_file_path, forcing_file, output_file)
    
    print(f"✓ {ensemble_name} processing completed successfully!")
    
    return result

def process_all_ensembles(ensemble_parent_dir, trend_file_path, ensemble_members=None):
    """Process multiple ensemble members automatically."""
    
    if ensemble_members is None:
        ensemble_members = ["SSP585-EM1", "SSP585-EM2", "SSP585-EM4", "SSP585-EM6", "SSP585-EM8"]
    
    print("=== SSP585 Trend + AISLENS Forcing Combination Script (Multi-Ensemble Python) ===")
    print(f"Parent directory: {ensemble_parent_dir}")
    print(f"Trend file: {trend_file_path}")
    print(f"Processing {len(ensemble_members)} ensemble members:")
    for ensemble in ensemble_members:
        print(f"  - {ensemble}")
    print()
    
    successful_count = 0
    failed_count = 0
    failed_ensembles = []
    
    for ensemble_name in ensemble_members:
        try:
            # Extract ensemble number
            ensemble_num = ensemble_name.replace("SSP585-EM", "")
            
            # Process this ensemble
            process_single_ensemble(ensemble_parent_dir, trend_file_path, ensemble_name, ensemble_num)
            successful_count += 1
            print(f"✓ {ensemble_name}: SUCCESS")
            
        except Exception as e:
            print(f"✗ {ensemble_name}: FAILED - {e}")
            failed_count += 1
            failed_ensembles.append(ensemble_name)
    
    print()
    print("==========================================")
    print("FINAL SUMMARY")
    print("==========================================")
    print(f"Total ensemble members processed: {len(ensemble_members)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    
    if failed_count > 0:
        print()
        print("Failed ensemble members:")
        for failed_ensemble in failed_ensembles:
            print(f"  - {failed_ensemble}")
        print()
        print("Please check the error messages above for details.")
        return False
    else:
        print()
        print("All ensemble members processed successfully!")
        return True

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Combine SSP585 trend with AISLENS forcing files using Python/xarray')
    
    # Single file processing
    parser.add_argument('--trend-file', type=str, required=True,
                        help='Path to SSP585 trend file')
    parser.add_argument('--forcing-file', type=str,
                        help='Path to AISLENS forcing file (for single file processing)')
    parser.add_argument('--output-file', type=str,
                        help='Path for output combined file (for single file processing)')
    
    # Multi-ensemble processing
    parser.add_argument('--ensemble-dir', type=str,
                        help='Parent directory containing ensemble subdirectories')
    parser.add_argument('--ensemble-members', type=str, nargs='+',
                        default=["SSP585-EM1", "SSP585-EM2", "SSP585-EM4", "SSP585-EM6", "SSP585-EM8"],
                        help='List of ensemble member names to process')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.trend_file).exists():
        raise FileNotFoundError(f"Trend file not found: {args.trend_file}")
    
    if args.ensemble_dir:
        # Multi-ensemble processing
        if not Path(args.ensemble_dir).exists():
            raise FileNotFoundError(f"Ensemble directory not found: {args.ensemble_dir}")
        
        success = process_all_ensembles(args.ensemble_dir, args.trend_file, args.ensemble_members)
        exit(0 if success else 1)
        
    elif args.forcing_file and args.output_file:
        # Single file processing
        if not Path(args.forcing_file).exists():
            raise FileNotFoundError(f"Forcing file not found: {args.forcing_file}")
        
        combine_ssp585_forcing_xarray(args.trend_file, args.forcing_file, args.output_file)
        
    else:
        print("Error: Must specify either:")
        print("  1. --forcing-file and --output-file for single file processing, OR")
        print("  2. --ensemble-dir for multi-ensemble processing")
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()
