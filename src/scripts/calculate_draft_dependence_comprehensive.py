# This script calculates comprehensive draft dependence parameters using changepoint detection
# and piecewise linear models for Antarctic ice shelves.
# 
# This enhanced version creates all 5 draft dependence parameters:
#   - draftDepenBasalMelt_minDraft: threshold draft value (0 for noisy shelves)
#   - draftDepenBasalMelt_constantMeltValue: constant melt rate for shallow areas  
#   - draftDepenBasalMelt_paramType: selector (0 for linear, 1 for constant)
#   - draftDepenBasalMeltAlpha0: intercept (0 for noisy shelves)
#   - draftDepenBasalMeltAlpha1: slope (0 for noisy shelves)
#
# Run this script after running prepare_data.py

from aislens.config import config
from aislens.utils import write_crs, merge_catchment_data
from aislens.dataprep import dedraft_catchment_comprehensive
import xarray as xr
import geopandas as gpd
import numpy as np
from pathlib import Path
import traceback

# Check for optional dependencies
try:
    import ruptures
    RUPTURES_AVAILABLE = True
    print(f"✓ ruptures library available (version: {ruptures.__version__})")
except ImportError:
    RUPTURES_AVAILABLE = False
    print(" WARNING: ruptures library not available - changepoint detection will fail!")
    print("   Install with: pip install ruptures")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print(" WARNING: pandas library not available - summary creation will fail!")

def calculate_draft_dependence_comprehensive(icems, satobs, config,
                                           n_bins=50, min_points_per_bin=5,
                                           ruptures_method='pelt', ruptures_penalty=1.0,
                                           min_r2_threshold=0.05, min_correlation=0.1,
                                           noisy_fallback='zero', model_selection='best'):
    """
    Calculate comprehensive draft dependence parameters for all ice shelf regions.

    Args:
        icems: GeoDataFrame with ice shelf masks
        satobs: xarray Dataset with satellite observations
        config: Configuration object
        n_bins: Number of bins for draft binning (default: 50)
        min_points_per_bin: Minimum points required per bin (default: 5)
        ruptures_method: Ruptures method ('pelt', 'binseg', 'window') (default: 'pelt')
        ruptures_penalty: Penalty parameter for ruptures (default: 1.0)
        min_r2_threshold: Minimum R² for meaningful relationship (default: 0.1)
        min_correlation: Minimum correlation for meaningful relationship (default: 0.3)
        noisy_fallback: For noisy data ('zero' or 'mean') (default: 'zero')
        model_selection: Which model to use ('best', 'zero_shallow', 'mean_shallow', 'threshold_intercept') (default: 'best')
    """

    print("CALCULATING COMPREHENSIVE DRAFT DEPENDENCE PARAMETERS...")
    print(f"Settings: method={ruptures_method}, penalty={ruptures_penalty}")
    print(f"Quality thresholds: R²≥{min_r2_threshold}, |corr|≥{min_correlation}")
    print(f"Model selection: {model_selection}, noisy fallback: {noisy_fallback}")
    print(f"Processing ice shelves starting from index 33 (Abbott Ice Shelf)")

    # Debug information
    print(f"Total ice shelves to process: {len(list(icems.name.values[33:]))}")  # Count from index 33 onwards
    print(f"icems dataframe length: {len(icems)}")
    print(f"Satellite data shape: {satobs[config.SATOBS_FLUX_VAR].shape}")

    # Check ice shelf names for first few
    print("Sample ice shelf names (starting from index 33):")
    for i, name in enumerate(list(icems.name.values[33:40])):  # Show first 7 from index 33
        actual_index = i + 33
        print(f"  Index {actual_index}: {name}")

    if len(icems) <= 40:
        print(f"⚠️  WARNING: icems only has {len(icems)} rows, so only {max(0, len(icems) - 33)} ice shelves will be processed")
    else:
        expected_count = len(icems) - 33
        print(f"Expected to process ~{expected_count} ice shelves (from index 33 to {len(icems)-1})")

    # Create save directory for comprehensive results
    save_dir_comprehensive = config.DIR_ICESHELF_DEDRAFT_SATOBS / "comprehensive"
    save_dir_comprehensive.mkdir(parents=True, exist_ok=True)

    # Store results for each ice shelf
    all_results = {}
    all_draft_params = {}

    # Process each ice shelf - use sequential processing like notebook
    processed_count = 0
    skipped_count = 0
    error_details = {}  # Track specific error types

    # Get ice shelf names starting from index 33 (like notebook does with [33:])
    shelf_names = list(icems.name.values[33:])  # Start from Abbott Ice Shelf

    for i, shelf_name in enumerate(shelf_names):
        actual_index = i + 33  # Convert back to actual DataFrame index
        try:
            print(f"\nProcessing ice shelf {actual_index} ({i+1}/{len(shelf_names)}): {shelf_name}...")

            # Additional checks before processing
            try:
                # Check if the ice shelf geometry is valid
                ice_shelf_geom = icems.iloc[actual_index].geometry
                if ice_shelf_geom is None or ice_shelf_geom.is_empty:
                    print(f"✗ Skipping {shelf_name}: Empty or invalid geometry")
                    skipped_count += 1
                    error_details[actual_index] = "Empty geometry"
                    continue

                print(f"  Geometry area: {ice_shelf_geom.area:.2e} (projection units)")

            except Exception as geom_error:
                print(f"✗ Skipping {shelf_name}: Geometry error - {geom_error}")
                skipped_count += 1
                error_details[actual_index] = f"Geometry error: {geom_error}"
                continue

            # Check if output files already exist for this ice shelf - skip processing if they do
            config_param_names = [
                'draftDepenBasalMelt_minDraft',
                'draftDepenBasalMelt_constantMeltValue',
                'draftDepenBasalMelt_paramType',
                'draftDepenBasalMeltAlpha0',
                'draftDepenBasalMeltAlpha1'
            ]
            
            # Check if all parameter files exist for this ice shelf
            all_files_exist = True
            for param_name in config_param_names:
                param_file = save_dir_comprehensive / f"{param_name}_{shelf_name}.nc"
                if not param_file.exists():
                    all_files_exist = False
                    break
            
            if all_files_exist:
                print(f"  ✓ All output files already exist for {shelf_name}, skipping analysis...")
                
                # Still need to track this ice shelf for summary - load basic info from files
                try:
                    # Load the paramType to determine if it's linear or constant
                    param_type_file = save_dir_comprehensive / f"draftDepenBasalMelt_paramType_{shelf_name}.nc"
                    param_type_ds = xr.open_dataset(param_type_file)
                    param_type_value = param_type_ds.draftDepenBasalMelt_paramType.values
                    # Get the most common non-NaN value (use scipy.stats.mode)
                    from scipy.stats import mode
                    valid_values = param_type_value[~np.isnan(param_type_value)]
                    param_type_mode = mode(valid_values)[0][0] if len(valid_values) > 0 else 1
                    
                    # Create mock results for summary tracking
                    mock_result = {
                        'is_meaningful': param_type_mode == 0,  # Linear = meaningful
                        'correlation': np.nan,  # Don't have correlation from saved files
                        'r2': np.nan,          # Don't have R² from saved files
                        'threshold': np.nan,    # Don't have threshold from saved files
                        'slope': 0.0,
                        'shallow_mean': 0.0,
                        'melt_vals': []
                    }
                    
                    mock_params = {
                        'minDraft': np.nan,
                        'constantValue': 0.0,
                        'paramType': int(param_type_mode),
                        'alpha0': 0.0,
                        'alpha1': 0.0
                    }
                    
                    all_results[shelf_name] = mock_result
                    all_draft_params[shelf_name] = mock_params
                    processed_count += 1
                    
                    print(f"  ✓ Loaded existing results for {shelf_name}: paramType={int(param_type_mode)}")
                    
                except Exception as load_error:
                    print(f"  Warning: Could not load existing files for {shelf_name}: {load_error}")
                    print(f"  Will reprocess this ice shelf...")
                    all_files_exist = False  # Force reprocessing
                
                if all_files_exist:
                    continue  # Skip to next ice shelf

            print(f"  Starting comprehensive analysis...")
            result = dedraft_catchment_comprehensive(
                actual_index, icems, satobs, config,  # Use actual_index instead of i
                save_dir=save_dir_comprehensive,
                weights=None,  # Don't use weights - set to None instead of False
                weight_power=0.25,
                save_pred=True,
                save_coefs=True,
                n_bins=n_bins,
                min_points_per_bin=min_points_per_bin,
                ruptures_method=ruptures_method,
                ruptures_penalty=ruptures_penalty,
                min_r2_threshold=min_r2_threshold,
                min_correlation=min_correlation,
                noisy_fallback=noisy_fallback,
                model_selection=model_selection
            )
            print(f"  Analysis completed successfully!")

            # Check if result has expected structure
            if not isinstance(result, dict) or 'full_results' not in result or 'draft_params' not in result:
                print(f"✗ Invalid result structure for {shelf_name}: {type(result)}")
                skipped_count += 1
                error_details[actual_index] = "Invalid result structure"
                continue

            all_results[shelf_name] = result['full_results']
            all_draft_params[shelf_name] = result['draft_params']
            processed_count += 1

            print(f"✓ Processed {shelf_name}: "
                  f"meaningful={result['full_results']['is_meaningful']}, "
                  f"paramType={result['draft_params']['paramType']}")

        except Exception as e:
            error_msg = str(e)
            print(f"✗ Error processing {shelf_name} (index {actual_index}): {error_msg}")

            # Categorize common error types
            if "ruptures" in error_msg.lower():
                error_type = "Ruptures library error"
            elif "memory" in error_msg.lower() or "allocation" in error_msg.lower():
                error_type = "Memory error"
            elif "data" in error_msg.lower() or "empty" in error_msg.lower():
                error_type = "Data error"
            elif "coordinate" in error_msg.lower() or "dimension" in error_msg.lower():
                error_type = "Coordinate error"
            else:
                error_type = "Other error"

            error_details[actual_index] = f"{error_type}: {error_msg[:100]}..."  # Truncate long errors

            traceback.print_exc()
            skipped_count += 1
            continue

    print(f"\nProcessing Summary:")
    print(f"  Successfully processed: {processed_count} ice shelves")
    print(f"  Skipped/Failed: {skipped_count} ice shelves")
    print(f"  Expected total: {len(shelf_names)} ice shelves")
    print(f"  Success rate: {processed_count/(processed_count+skipped_count)*100:.1f}%")

    # Detailed error breakdown
    if error_details:
        print(f"\nError Breakdown:")
        error_types = {}
        for idx, error in error_details.items():
            error_type = error.split(":")[0] if ":" in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count} ice shelves")

        # Show first few specific errors for debugging
        print(f"\nFirst few specific errors:")
        for idx, error in list(error_details.items())[:5]:
            shelf_name = icems.name.values[idx] if idx < len(icems) else f"Index_{idx}"
            print(f"  {idx} ({shelf_name}): {error}")

        if len(error_details) > 5:
            print(f"  ... and {len(error_details)-5} more errors")

    if processed_count == 0:
        print(f"\n⚠️  WARNING: No ice shelves were processed successfully!")
        print(f"   This suggests a systematic issue. Common causes:")
        print(f"   1. Missing dependencies (ruptures library)")
        print(f"   2. Data format/coordinate issues")
        print(f"   3. Index range problems")
        print(f"   4. Insufficient data in ice shelf regions")
        return {}, {}  # Return empty results to avoid downstream errors

    # Create comprehensive summary
    create_comprehensive_summary(all_results, all_draft_params, save_dir_comprehensive)

    # Merge parameters into ice sheet grids
    merge_comprehensive_parameters(all_draft_params, icems, satobs, config, save_dir_comprehensive)

    print("COMPREHENSIVE DRAFT DEPENDENCE PARAMETERS CALCULATED AND SAVED.")

    return all_results, all_draft_params

def create_comprehensive_summary(all_results, all_draft_params, save_dir):
    """Create summary statistics and save to CSV."""
    if not PANDAS_AVAILABLE:
        print("⚠️  Cannot create summary - pandas not available")
        return

    import pandas as pd

    summary_data = []

    for shelf_name, result in all_results.items():
        draft_params = all_draft_params[shelf_name]

        summary_data.append({
            'shelf_name': shelf_name,
            'is_meaningful': result['is_meaningful'],
            'correlation': result.get('correlation', np.nan),
            'r2': result.get('r2', np.nan),
            'threshold_draft': result.get('threshold', np.nan),
            'slope': result.get('slope', 0.0),
            'shallow_mean': result.get('shallow_mean', 0.0),
            'n_points': len(result.get('melt_vals', [])),
            'minDraft': draft_params['minDraft'],
            'constantValue': draft_params['constantValue'],
            'paramType': draft_params['paramType'],
            'alpha0': draft_params['alpha0'],
            'alpha1': draft_params['alpha1']
        })

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_file = save_dir / "comprehensive_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

    # Print key statistics
    meaningful_count = summary_df['is_meaningful'].sum()
    total_count = len(summary_df)
    print(f"\nSummary Statistics:")
    print(f"  Total shelves processed: {total_count}")
    print(f"  Meaningful relationships: {meaningful_count} ({meaningful_count/total_count*100:.1f}%)")
    print(f"  Linear parameterization (paramType=0): {(summary_df['paramType']==0).sum()}")
    print(f"  Constant parameterization (paramType=1): {(summary_df['paramType']==1).sum()}")
    print(f"  Mean correlation (meaningful only): {summary_df[summary_df['is_meaningful']]['correlation'].mean():.3f}")
    print(f"  Mean R² (meaningful only): {summary_df[summary_df['is_meaningful']]['r2'].mean():.3f}")

def merge_comprehensive_parameters(all_draft_params, icems, satobs, config, save_dir):
    """
    Merge individual ice shelf parameters into full ice sheet grids.
    
    Uses the simple and effective approach from the original calculate_draft_dependence.py:
    Just use xr.merge() to combine all the individual NetCDF files that were saved
    by dedraft_catchment_comprehensive().
    
    The order of merging matters for overlapping spatial points - the last ice shelf
    merged takes priority for all parameters at that location.
    """
    print("Merging comprehensive draft dependence parameters...")
    
    # Parameter names to merge
    config_param_names = [
        'draftDepenBasalMelt_minDraft',
        'draftDepenBasalMelt_constantMeltValue', 
        'draftDepenBasalMelt_paramType',
        'draftDepenBasalMeltAlpha0',
        'draftDepenBasalMeltAlpha1'
    ]
    
    # PRIORITY ORDERING OPTIONS FOR MERGING
    # The order determines which ice shelf takes priority at overlapping spatial points.
    # Choose one of the following approaches:
    
    # Option 1: Manual priority list (currently active)
    # Specify ice shelves in order of priority (first = lowest priority, last = highest priority)
    manual_priority_shelves = [
        # Add specific ice shelf names here if you want manual control
        # Example (uncomment and modify as needed):
        # 'Abbot Ice Shelf',              # Low priority
        # 'George VI Ice Shelf',
        # 'Larsen C Ice Shelf', 
        # 'Amery Ice Shelf',
        # 'Filchner Ice Shelf',
        # 'Ronne Ice Shelf',
        # 'Ross Ice Shelf',               # High priority
        # Currently empty - will use alphabetical order as fallback
    ]
    
    # Option 2: Size-based priority (commented out)
    # Larger ice shelves get higher priority (merged last)
    # Uncomment the following code block to use size-based ordering:
    """
    size_priority_shelves = []
    for shelf_name in all_draft_params.keys():
        try:
            shelf_match = icems[icems.name == shelf_name]
            if len(shelf_match) > 0:
                area = shelf_match.iloc[0].geometry.area
                size_priority_shelves.append((shelf_name, area))
            else:
                # If shelf not found in icems, assign small area (low priority)
                size_priority_shelves.append((shelf_name, 0))
                print(f"  Warning: {shelf_name} not found in icems, assigned low priority")
        except Exception as e:
            print(f"  Warning: Could not get area for {shelf_name}: {e}")
            size_priority_shelves.append((shelf_name, 0))
    
    # Sort by area (smallest first = lowest priority, largest last = highest priority)
    size_priority_shelves.sort(key=lambda x: x[1])
    merge_order = [shelf[0] for shelf in size_priority_shelves]
    print(f"Size-based merge order - largest shelves have highest priority")
    """
    
    # Option 3: Distance-based priority (commented out) 
    # Ice shelves closer to a reference point get higher priority
    # Uncomment and modify the reference point as needed:
    """
    from shapely.geometry import Point
    reference_point = Point(-1500000, 0)  # Example: Antarctic center in projected coords
    distance_priority_shelves = []
    
    for shelf_name in all_draft_params.keys():
        try:
            shelf_match = icems[icems.name == shelf_name]
            if len(shelf_match) > 0:
                centroid = shelf_match.iloc[0].geometry.centroid
                distance = reference_point.distance(centroid)
                distance_priority_shelves.append((shelf_name, distance))
            else:
                # If shelf not found, assign large distance (low priority)
                distance_priority_shelves.append((shelf_name, float('inf')))
                print(f"  Warning: {shelf_name} not found in icems, assigned low priority")
        except Exception as e:
            print(f"  Warning: Could not calculate distance for {shelf_name}: {e}")
            distance_priority_shelves.append((shelf_name, float('inf')))
    
    # Sort by distance (farthest first = lowest priority, closest last = highest priority)
    distance_priority_shelves.sort(key=lambda x: x[1], reverse=True)
    merge_order = [shelf[0] for shelf in distance_priority_shelves]
    print(f"Distance-based merge order - closer shelves have highest priority")
    """
    
    # Determine final merge order
    if manual_priority_shelves:
        # Use manual priority list, with any unlisted shelves added alphabetically at the end
        unlisted_shelves = [s for s in all_draft_params.keys() if s not in manual_priority_shelves]
        merge_order = manual_priority_shelves + sorted(unlisted_shelves)
        print(f"Using manual priority order: {len(manual_priority_shelves)} prioritized + {len(unlisted_shelves)} alphabetical")
    # Note: If you uncomment size-based or distance-based options above, 
    # they will set merge_order directly and override this logic
    else:
        # Fallback to alphabetical order
        merge_order = sorted(all_draft_params.keys())
        print(f"Using alphabetical order for {len(merge_order)} ice shelves")
    
    print(f"Merge order (first=lowest priority, last=highest priority):")
    for i, shelf in enumerate(merge_order[:5]):  # Show first 5
        print(f"  {i+1:2d}. {shelf}")
    if len(merge_order) > 5:
        print(f"  ... and {len(merge_order)-5} more shelves")
        print(f"  Last: {merge_order[-1]} (highest priority)")
    print()

    # Create merged datasets for each parameter using the simple approach
    for config_param_name in config_param_names:
        print(f"Merging {config_param_name}...")
        
        # Start with empty dataset
        merged_dataset = xr.Dataset()
        files_merged = 0
        
        # Loop through ice shelves in determined merge order (priority matters!)
        for shelf_name in merge_order:
            param_file = save_dir / f"{config_param_name}_{shelf_name}.nc"
            
            if param_file.exists():
                try:
                    # Load and merge - use default merge behavior
                    shelf_ds = xr.open_dataset(param_file)
                    
                    if len(merged_dataset.data_vars) == 0:
                        # First file - no conflicts possible
                        merged_dataset = shelf_ds.copy()
                    else:
                        # Subsequent files - use default merge (no compat specified)
                        merged_dataset = xr.merge([merged_dataset, shelf_ds])
                    
                    files_merged += 1
                    
                    # Debug: Check how much data we have after each merge
                    if hasattr(merged_dataset, 'data_vars') and len(merged_dataset.data_vars) > 0:
                        var_name = list(merged_dataset.data_vars.keys())[0]
                        valid_points = (~merged_dataset[var_name].isnull()).sum().item()
                        print(f"    After merging {shelf_name}: {valid_points} valid points")
                    
                except Exception as e:
                    print(f"  Warning: Could not merge {shelf_name} for {config_param_name}: {e}")
            else:
                print(f"  Warning: File not found: {param_file}")
        
        print(f"  Successfully merged {files_merged} files for {config_param_name}")
        
        # Save individual parameter file 
        if len(merged_dataset.data_vars) > 0:
            output_file = config.DIR_PROCESSED / "draft_dependence_changepoint" / f"ruptures_{config_param_name}.nc"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            merged_dataset.to_netcdf(output_file)
            print(f"  Saved merged {config_param_name} to {output_file}")
        else:
            print(f"  Warning: No data to save for {config_param_name}")

    # Create combined dataset with all parameters (like original script)
    print("Creating combined parameter dataset...")
    combined_dataset = xr.Dataset()
    
    for config_param_name in config_param_names:
        individual_file = config.DIR_PROCESSED / "draft_dependence_changepoint" / f"ruptures_{config_param_name}.nc"
        if individual_file.exists():
            try:
                param_ds = xr.open_dataset(individual_file)
                
                if len(combined_dataset.data_vars) == 0:
                    # First parameter dataset
                    combined_dataset = param_ds.copy()
                else:
                    # Subsequent datasets - use default merge
                    combined_dataset = xr.merge([combined_dataset, param_ds])
                    
            except Exception as e:
                print(f"Warning: Could not add {config_param_name} to combined dataset: {e}")

    # Save combined file
    if len(combined_dataset.data_vars) > 0:
        combined_file = config.DIR_PROCESSED / "draft_dependence_changepoint" / "ruptures_draftDepenBasalMelt_parameters.nc"
        combined_dataset.to_netcdf(combined_file)
        print(f"Saved combined parameters to {combined_file}")
        print(f"Combined dataset variables: {list(combined_dataset.data_vars.keys())}")
        
        # Fill NaN values with 0 for compatibility (like original script)
        combined_dataset_filled = combined_dataset.fillna(0)
        combined_file_filled = config.DIR_PROCESSED / "draft_dependence_changepoint" / "ruptures_draftDepenBasalMelt_parameters_filled.nc"
        combined_dataset_filled.to_netcdf(combined_file_filled)
        print(f"Saved filled parameters to {combined_file_filled}")
    else:
        print("Warning: No combined data to save")

if __name__ == "__main__":
    # Load data
    print("Loading satellite observation data...")
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
    satobs = write_crs(satobs, config.CRS_TARGET)

    print("Loading ice shelf masks...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs(config.CRS_TARGET)

    # Run comprehensive analysis
    # Note: Now processes ice shelves sequentially starting from index 33 (Abbott Ice Shelf)
    # PERMISSIVE SETTINGS: Lower thresholds to get linear relationships for more ice shelves
    all_results, all_draft_params = calculate_draft_dependence_comprehensive(
        icems, satobs, config,
        n_bins=25,                    # Fewer bins = less noise, easier to detect patterns
        min_points_per_bin=3,         # Lower minimum = more bins kept for analysis
        ruptures_method='pelt',       # Keep PELT - good for detecting changepoints
        ruptures_penalty=0.5,         # LOWER penalty = more changepoints detected (was 1.0)
        min_r2_threshold=0.005,        # LOWER R² threshold = accept weaker relationships (was 0.1)
        min_correlation=-0.7,        # Accept both positive AND negative correlations ≥|0.7| (was 0.05)
        noisy_fallback='mean',        # Use mean melt rate for noisy shelves (other options: 'zero')
        model_selection='threshold_intercept'  # Force true piecewise function (other options: 'mean_shallow', 'zero_shallow', 'best')
    )

    print(f"\nProcessing complete! Processed {len(all_results)} ice shelves.")
    print("Output files saved to:")
    print(f"  - Individual files: {config.DIR_ICESHELF_DEDRAFT_SATOBS / 'comprehensive'}")
    print(f"  - Merged grids: {config.DIR_PROCESSED / 'draft_dependence_changepoint'}")