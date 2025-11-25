#!/usr/bin/env python3
"""
Calculate comprehensive draft dependence parameters using changepoint detection
and piecewise linear models for Antarctic ice shelves.

Creates all 5 draft dependence parameters:
  - draftDepenBasalMelt_minDraft: threshold draft value (0 for noisy shelves)
  - draftDepenBasalMelt_constantMeltValue: constant melt rate for shallow areas  
  - draftDepenBasalMelt_paramType: selector (0 for linear, 1 for constant)
  - draftDepenBasalMeltAlpha0: intercept (0 for noisy shelves)
  - draftDepenBasalMeltAlpha1: slope (0 for noisy shelves)

Prerequisites: Run prepare_data.py first

Usage: python calculate_draft_dependence_comprehensive.py
"""

import logging
from pathlib import Path
import numpy as np
import xarray as xr
import geopandas as gpd

from aislens.config import config
from aislens.utils import write_crs, merge_catchment_data
from aislens.dataprep import dedraft_catchment_comprehensive

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import ruptures
    RUPTURES_AVAILABLE = True
    logger.info(f"ruptures library available (version: {ruptures.__version__})")
except ImportError:
    RUPTURES_AVAILABLE = False
    logger.warning("ruptures library not available - changepoint detection will fail!")
    logger.warning("Install with: pip install ruptures")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas library not available - summary creation will fail!")

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
        min_r2_threshold: Minimum R2 for meaningful relationship (default: 0.1)
        min_correlation: Minimum correlation for meaningful relationship (default: 0.3)
        noisy_fallback: For noisy data ('zero' or 'mean') (default: 'zero')
        model_selection: Which model to use ('best', 'zero_shallow', 'mean_shallow', 'threshold_intercept') (default: 'best')
    """
    logger.info("CALCULATING COMPREHENSIVE DRAFT DEPENDENCE PARAMETERS...")
    logger.info(f"Settings: method={ruptures_method}, penalty={ruptures_penalty}")
    logger.info(f"Quality thresholds: R2>={min_r2_threshold}, |corr|>={min_correlation}")
    logger.info(f"Model selection: {model_selection}, noisy fallback: {noisy_fallback}")
    
    n_shelves = len(icems) - 33
    logger.info(f"Processing {n_shelves} ice shelves (starting from index 33: {icems.name.values[33]})")
    logger.debug(f"Satellite data shape: {satobs[config.SATOBS_FLUX_VAR].shape}")

    # Create save directory for comprehensive results
    save_dir_comprehensive = config.DIR_ICESHELF_DEDRAFT_SATOBS / "comprehensive"
    save_dir_comprehensive.mkdir(parents=True, exist_ok=True)

    # Store results for each ice shelf
    all_results = {}
    all_draft_params = {}
    processed_count = 0
    skipped_count = 0
    error_details = {}

    # Get ice shelf names starting from index 33 (Abbott Ice Shelf)
    shelf_names = list(icems.name.values[33:])

    for i, shelf_name in enumerate(shelf_names):
        actual_index = i + 33
        logger.info(f"\nProcessing ice shelf {actual_index} ({i+1}/{len(shelf_names)}): {shelf_name}...")

        # Check if the ice shelf geometry is valid
        ice_shelf_geom = icems.iloc[actual_index].geometry
        if ice_shelf_geom is None or ice_shelf_geom.is_empty:
            logger.warning(f"Skipping {shelf_name}: Empty or invalid geometry")
            skipped_count += 1
            error_details[actual_index] = "Empty geometry"
            continue

        logger.debug(f"  Geometry area: {ice_shelf_geom.area:.2e} (projection units)")

        # Check if output files already exist for this ice shelf
        config_param_names = ['draftDepenBasalMelt_minDraft', 'draftDepenBasalMelt_constantMeltValue',
                             'draftDepenBasalMelt_paramType', 'draftDepenBasalMeltAlpha0', 'draftDepenBasalMeltAlpha1']
        
        all_files_exist = all(
            (save_dir_comprehensive / f"{param_name}_{shelf_name}.nc").exists()
            for param_name in config_param_names
        )
        
        if all_files_exist:
            logger.info(f"  All output files exist for {shelf_name}, loading existing results...")
            try:
                # Load paramType to track if linear or constant
                param_type_file = save_dir_comprehensive / f"draftDepenBasalMelt_paramType_{shelf_name}.nc"
                param_type_ds = xr.open_dataset(param_type_file)
                param_type_value = param_type_ds.draftDepenBasalMelt_paramType.values
                
                from scipy.stats import mode
                valid_values = param_type_value[~np.isnan(param_type_value)]
                param_type_mode = mode(valid_values, keepdims=True)[0][0] if len(valid_values) > 0 else 1
                
                # Create mock results for summary tracking
                all_results[shelf_name] = {
                    'is_meaningful': param_type_mode == 0,
                    'correlation': np.nan, 'r2': np.nan, 'threshold': np.nan,
                    'slope': 0.0, 'shallow_mean': 0.0, 'melt_vals': []
                }
                all_draft_params[shelf_name] = {
                    'minDraft': np.nan, 'constantValue': 0.0, 'paramType': int(param_type_mode),
                    'alpha0': 0.0, 'alpha1': 0.0
                }
                processed_count += 1
                logger.debug(f"  Loaded existing: paramType={int(param_type_mode)}")
                continue
                
            except Exception as load_error:
                logger.warning(f"  Could not load existing files for {shelf_name}: {load_error}")
                logger.info(f"  Will reprocess this ice shelf...")

        # Run comprehensive analysis
        try:
            logger.info(f"  Starting comprehensive analysis...")
            result = dedraft_catchment_comprehensive(
                actual_index, icems, satobs, config, save_dir=save_dir_comprehensive,
                weights=None, weight_power=0.25, save_pred=True, save_coefs=True,
                n_bins=n_bins, min_points_per_bin=min_points_per_bin,
                ruptures_method=ruptures_method, ruptures_penalty=ruptures_penalty,
                min_r2_threshold=min_r2_threshold, min_correlation=min_correlation,
                noisy_fallback=noisy_fallback, model_selection=model_selection
            )
            logger.info(f"  Analysis completed successfully!")

            # Validate result structure
            if not isinstance(result, dict) or 'full_results' not in result or 'draft_params' not in result:
                logger.error(f"Invalid result structure for {shelf_name}: {type(result)}")
                skipped_count += 1
                error_details[actual_index] = "Invalid result structure"
                continue

            all_results[shelf_name] = result['full_results']
            all_draft_params[shelf_name] = result['draft_params']
            processed_count += 1

            logger.info(f"Processed {shelf_name}: meaningful={result['full_results']['is_meaningful']}, "
                       f"paramType={result['draft_params']['paramType']}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing {shelf_name} (index {actual_index}): {error_msg}")

            # Categorize error types
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

            error_details[actual_index] = f"{error_type}: {error_msg[:100]}"
            skipped_count += 1

    # Print processing summary
    logger.info(f"\nProcessing Summary:")
    logger.info(f"  Successfully processed: {processed_count} ice shelves")
    logger.info(f"  Skipped/Failed: {skipped_count} ice shelves")
    logger.info(f"  Expected total: {len(shelf_names)} ice shelves")
    logger.info(f"  Success rate: {processed_count/(processed_count+skipped_count)*100:.1f}%")

    # Detailed error breakdown
    if error_details:
        logger.info(f"\nError Breakdown:")
        error_types = {}
        for idx, error in error_details.items():
            error_type = error.split(":")[0] if ":" in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {error_type}: {count} ice shelves")

        # Show first few specific errors for debugging
        logger.debug(f"\nFirst few specific errors:")
        for idx, error in list(error_details.items())[:5]:
            shelf_name = icems.name.values[idx] if idx < len(icems) else f"Index_{idx}"
            logger.debug(f"  {idx} ({shelf_name}): {error}")
        if len(error_details) > 5:
            logger.debug(f"  ... and {len(error_details)-5} more errors")

    if processed_count == 0:
        logger.error("\nNo ice shelves were processed successfully!")
        logger.error("  This suggests a systematic issue. Common causes:")
        logger.error("  1. Missing dependencies (ruptures library)")
        logger.error("  2. Data format/coordinate issues")
        logger.error("  3. Index range problems")
        logger.error("  4. Insufficient data in ice shelf regions")
        return {}, {}

    # Create comprehensive summary
    create_comprehensive_summary(all_results, all_draft_params, save_dir_comprehensive)

    # Merge parameters into ice sheet grids
    merge_comprehensive_parameters(all_draft_params, icems, satobs, config, save_dir_comprehensive)

    logger.info("COMPREHENSIVE DRAFT DEPENDENCE PARAMETERS CALCULATED AND SAVED.")

    return all_results, all_draft_params

def create_comprehensive_summary(all_results, all_draft_params, save_dir):
    """Create summary statistics and save to CSV."""
    if not PANDAS_AVAILABLE:
        logger.warning("Cannot create summary - pandas not available")
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
    summary_file = save_dir / "comprehensive_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary saved to {summary_file}")

    # Print key statistics
    meaningful_count = summary_df['is_meaningful'].sum()
    total_count = len(summary_df)
    logger.info(f"\nSummary Statistics:")
    logger.info(f"  Total shelves processed: {total_count}")
    logger.info(f"  Meaningful relationships: {meaningful_count} ({meaningful_count/total_count*100:.1f}%)")
    logger.info(f"  Linear parameterization (paramType=0): {(summary_df['paramType']==0).sum()}")
    logger.info(f"  Constant parameterization (paramType=1): {(summary_df['paramType']==1).sum()}")
    logger.info(f"  Mean correlation (meaningful only): {summary_df[summary_df['is_meaningful']]['correlation'].mean():.3f}")
    logger.info(f"  Mean R2 (meaningful only): {summary_df[summary_df['is_meaningful']]['r2'].mean():.3f}")

def merge_comprehensive_parameters(all_draft_params, icems, satobs, config, save_dir):
    """Merge individual ice shelf parameters into full ice sheet grids."""
    logger.info("Merging comprehensive draft dependence parameters...")
    logger.info("Note: Each ice shelf file contains constant values for its area (typically 1-10 grid cells)")
    
    config_param_names = ['draftDepenBasalMelt_minDraft', 'draftDepenBasalMelt_constantMeltValue',
                          'draftDepenBasalMelt_paramType', 'draftDepenBasalMeltAlpha0', 'draftDepenBasalMeltAlpha1']
    
    # Determine merge order (alphabetical by default, last = highest priority)
    # Alternative ordering strategies (commented out - uncomment to use):
    
    # Option 2: Size-based priority (larger shelves have higher priority)
    # shelf_sizes = [(name, icems[icems.name == name].iloc[0].geometry.area) 
    #                for name in all_draft_params.keys() if len(icems[icems.name == name]) > 0]
    # merge_order = [name for name, _ in sorted(shelf_sizes, key=lambda x: x[1])]
    # logger.info(f"Using size-based merge order (largest shelves have highest priority)")
    
    # Option 3: Distance-based priority (shelves closer to reference point have higher priority)
    # from shapely.geometry import Point
    # ref_point = Point(-1500000, 0)  # Example: Antarctic center in projected coords
    # shelf_dists = [(name, ref_point.distance(icems[icems.name == name].iloc[0].geometry.centroid))
    #                for name in all_draft_params.keys() if len(icems[icems.name == name]) > 0]
    # merge_order = [name for name, _ in sorted(shelf_dists, key=lambda x: x[1], reverse=True)]
    # logger.info(f"Using distance-based merge order (closer shelves have highest priority)")
    
    merge_order = sorted(all_draft_params.keys())
    logger.info(f"Using alphabetical merge order for {len(merge_order)} ice shelves")
    logger.debug(f"First 3: {', '.join(merge_order[:3])} ... Last: {merge_order[-1]} (highest priority)")

    # Helper function to clean grid_mapping attributes
    def clean_encoding(ds):
        for var in list(ds.data_vars) + list(ds.coords):
            ds[var].attrs.pop('grid_mapping', None)
            if hasattr(ds[var], 'encoding'):
                ds[var].encoding.pop('grid_mapping', None)
        return ds

    # Merge each parameter
    for param_name in config_param_names:
        logger.info(f"Merging {param_name}...")
        merged_dataset = xr.Dataset()
        files_merged = 0
        
        for shelf_name in merge_order:
            param_file = save_dir / f"{param_name}_{shelf_name}.nc"
            if param_file.exists():
                try:
                    shelf_ds = clean_encoding(xr.open_dataset(param_file))
                    merged_dataset = shelf_ds.copy() if len(merged_dataset.data_vars) == 0 else xr.merge([merged_dataset, shelf_ds])
                    files_merged += 1
                except Exception as e:
                    logger.warning(f"  Could not merge {shelf_name}: {e}")
            else:
                logger.debug(f"  File not found: {param_file.name}")
        
        logger.info(f"  Merged {files_merged} files")
        
        # Save merged parameter
        if len(merged_dataset.data_vars) > 0:
            var_name = list(merged_dataset.data_vars.keys())[0]
            total_valid = (~merged_dataset[var_name].isnull()).sum().item()
            total_size = merged_dataset[var_name].size
            logger.info(f"  Coverage: {total_valid}/{total_size} cells ({total_valid/total_size*100:.1f}%)")
            
            merged_dataset = write_crs(clean_encoding(merged_dataset), config.CRS_TARGET)
            output_file = config.DIR_PROCESSED / "draft_dependence_changepoint" / f"ruptures_{param_name}.nc"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            merged_dataset.to_netcdf(output_file)
            logger.info(f"  Saved to {output_file.name}")
        else:
            logger.warning(f"  No data to save for {param_name}")

    # Create combined dataset with all parameters
    logger.info("Creating combined parameter dataset...")
    combined_dataset = xr.Dataset()
    
    for param_name in config_param_names:
        individual_file = config.DIR_PROCESSED / "draft_dependence_changepoint" / f"ruptures_{param_name}.nc"
        if individual_file.exists():
            try:
                param_ds = clean_encoding(xr.open_dataset(individual_file))
                combined_dataset = param_ds.copy() if len(combined_dataset.data_vars) == 0 else xr.merge([combined_dataset, param_ds])
            except Exception as e:
                logger.warning(f"Could not add {param_name} to combined dataset: {e}")

    # Save combined, filled, and prepped versions
    if len(combined_dataset.data_vars) > 0:
        combined_dataset = write_crs(clean_encoding(combined_dataset), config.CRS_TARGET)
        
        # Save main combined file
        combined_file = config.DIR_PROCESSED / "draft_dependence_changepoint" / "ruptures_draftDepenBasalMelt_parameters.nc"
        combined_dataset.to_netcdf(combined_file)
        logger.info(f"Saved combined parameters: {list(combined_dataset.data_vars.keys())}")
        
        # Save filled version (NaN -> 0)
        combined_filled = clean_encoding(combined_dataset.fillna(0))
        combined_file_filled = config.DIR_PROCESSED / "draft_dependence_changepoint" / "ruptures_draftDepenBasalMelt_parameters_filled.nc"
        combined_filled.to_netcdf(combined_file_filled)
        logger.info(f"Saved filled parameters")
        
        # Save interpolation-prepped version (x/y -> x1/y1)
        coord_rename = {k: f"{k}1" for k in ['x', 'y'] if k in combined_filled.coords}
        if coord_rename:
            combined_prepped = clean_encoding(combined_filled.rename(coord_rename))
            combined_file_prepped = config.DIR_PROCESSED / "draft_dependence_changepoint" / "ruptures_draftDepenBasalMelt_parameters_filled_prepped.nc"
            combined_prepped.to_netcdf(combined_file_prepped)
            logger.info(f"Saved interpolation-prepped parameters (renamed: {coord_rename})")
        else:
            logger.warning("No x/y coordinates found for interpolation prep")
    else:
        logger.warning("No combined data to save")


if __name__ == "__main__":
    from aislens.utils import setup_logging
    
    # Setup logging
    output_dir = config.DIR_ICESHELF_DEDRAFT_SATOBS / "comprehensive"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "calculate_draft_dependence_comprehensive")
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE DRAFT DEPENDENCE CALCULATION")
    logger.info("="*80)
    
    # Load data
    logger.info("Loading satellite observation data...")
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
    
    # Convert satellite melt from m/yr to kg/m2/s if needed
    if config.SATOBS_FLUX_VAR in satobs:
        if satobs.attrs.get('units', '') == 'm of ice per year':
            logger.info("  Converting satellite melt from m/yr to kg/m2/s...")
            satobs[config.SATOBS_FLUX_VAR] = satobs[config.SATOBS_FLUX_VAR] * (config.RHO_ICE / config.SECONDS_PER_YEAR)
            satobs[config.SATOBS_FLUX_VAR].attrs['units'] = 'kg m^-2 s^-1'
            satobs[config.SATOBS_DRAFT_VAR].attrs['units'] = 'm'
        else:
            logger.debug(f"  Satellite melt units: {satobs[config.SATOBS_FLUX_VAR].attrs.get('units', 'unknown')}")
    satobs = write_crs(satobs, config.CRS_TARGET)

    logger.info("Loading ice shelf masks...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS).to_crs(config.CRS_TARGET)

    # Run comprehensive analysis with permissive settings
    logger.info("\nRunning analysis with permissive settings (lower thresholds for more linear relationships)...")
    all_results, all_draft_params = calculate_draft_dependence_comprehensive(
        icems, satobs, config,
        n_bins=25, min_points_per_bin=3,
        ruptures_method='pelt', ruptures_penalty=0.5,
        min_r2_threshold=0.005, min_correlation=-0.7,
        noisy_fallback='mean', model_selection='threshold_intercept'
    )

    logger.info(f"\nProcessing complete! Processed {len(all_results)} ice shelves.")
    logger.info(f"Output files:")
    logger.info(f"  - Individual files: {config.DIR_ICESHELF_DEDRAFT_SATOBS / 'comprehensive'}")
    logger.info(f"  - Merged grids: {config.DIR_PROCESSED / 'draft_dependence_changepoint'}")
    logger.info("="*80)
