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

from aislens.dataprep import dedraft_catchment_comprehensive
from aislens.config import config
from aislens.utils import write_crs, merge_catchment_data
import xarray as xr
import geopandas as gpd
import numpy as np
from pathlib import Path

def calculate_draft_dependence_comprehensive(icems, satobs, config, 
                                           n_bins=50, min_points_per_bin=5,
                                           ruptures_method='pelt', ruptures_penalty=1.0,
                                           min_r2_threshold=0.1, min_correlation=0.3,
                                           noisy_fallback='zero', model_selection='best',
                                           ice_shelf_range=None):
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
        ice_shelf_range: Range of ice shelf indices to process (default: config.ICE_SHELF_REGIONS)
    """
    
    print("CALCULATING COMPREHENSIVE DRAFT DEPENDENCE PARAMETERS...")
    print(f"Settings: method={ruptures_method}, penalty={ruptures_penalty}")
    print(f"Quality thresholds: R²≥{min_r2_threshold}, |corr|≥{min_correlation}")
    print(f"Model selection: {model_selection}, noisy fallback: {noisy_fallback}")
    
    # Use provided range or default to all ice shelf regions
    if ice_shelf_range is None:
        ice_shelf_range = config.ICE_SHELF_REGIONS
    
    # Create save directory for comprehensive results
    save_dir_comprehensive = config.DIR_ICESHELF_DEDRAFT_SATOBS / "comprehensive"
    save_dir_comprehensive.mkdir(parents=True, exist_ok=True)
    
    # Store results for each ice shelf
    all_results = {}
    all_draft_params = {}
    
    # Process each ice shelf
    for i in ice_shelf_range:
        try:
            result = dedraft_catchment_comprehensive(
                i, icems, satobs, config,
                save_dir=save_dir_comprehensive,
                weights=True,
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
            
            catchment_name = result['catchment_name']
            all_results[catchment_name] = result['full_results']
            all_draft_params[catchment_name] = result['draft_params']
            
            print(f"✓ Processed {catchment_name}: "
                  f"meaningful={result['full_results']['is_meaningful']}, "
                  f"paramType={result['draft_params']['paramType']}")
                  
        except Exception as e:
            catchment_name = icems.name.values[i]
            print(f"✗ Error processing {catchment_name}: {e}")
            continue
    
    print(f"\nProcessed {len(all_results)} ice shelves successfully")
    
    # Create comprehensive summary
    create_comprehensive_summary(all_results, all_draft_params, save_dir_comprehensive)
    
    # Merge parameters into ice sheet grids
    merge_comprehensive_parameters(all_draft_params, icems, satobs, config, save_dir_comprehensive)
    
    print("COMPREHENSIVE DRAFT DEPENDENCE PARAMETERS CALCULATED AND SAVED.")
    
    return all_results, all_draft_params

def create_comprehensive_summary(all_results, all_draft_params, save_dir):
    """Create summary statistics and save to CSV."""
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
    """Merge individual ice shelf parameters into full ice sheet grids."""
    
    # Initialize empty datasets for each parameter
    param_names = ['minDraft', 'constantValue', 'paramType', 'alpha0', 'alpha1']
    config_param_names = [
        'draftDepenBasalMelt_minDraft',
        'draftDepenBasalMelt_constantMeltValue', 
        'draftDepenBasalMelt_paramType',
        'draftDepenBasalMeltAlpha0',
        'draftDepenBasalMeltAlpha1'
    ]
    
    merged_datasets = {}
    
    for param_name, config_param_name in zip(param_names, config_param_names):
        merged_datasets[config_param_name] = xr.Dataset()
    
    # Merge individual catchment files
    for shelf_name, draft_params in all_draft_params.items():
        for param_name, config_param_name in zip(param_names, config_param_names):
            try:
                # Look for individual parameter files
                param_file = save_dir / f"{config_param_name}_{shelf_name}.nc"
                if param_file.exists():
                    param_ds = xr.open_dataset(param_file)
                    merged_datasets[config_param_name] = xr.merge([
                        merged_datasets[config_param_name], param_ds
                    ])
            except Exception as e:
                print(f"Warning: Could not merge {config_param_name} for {shelf_name}: {e}")
    
    # Fill NaN values and save merged datasets
    for config_param_name, merged_ds in merged_datasets.items():
        if len(merged_ds.data_vars) > 0:
            # Fill NaN values with 0 (consistent with original approach)
            filled_ds = merged_ds.fillna(0)
            
            # Save individual parameter file
            output_file = config.DIR_PROCESSED / f"draft_dependence_changepoint" / f"ruptures_{config_param_name}.nc"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            filled_ds.to_netcdf(output_file)
            print(f"Saved {config_param_name} to {output_file}")
    
    # Create combined dataset with all parameters
    combined_ds = xr.Dataset()
    for config_param_name, merged_ds in merged_datasets.items():
        if len(merged_ds.data_vars) > 0:
            filled_ds = merged_ds.fillna(0)
            combined_ds = xr.merge([combined_ds, filled_ds])
    
    if len(combined_ds.data_vars) > 0:
        combined_file = config.DIR_PROCESSED / "draft_dependence_changepoint" / "ruptures_draftDepenBasalMelt_parameters.nc"
        combined_ds.to_netcdf(combined_file)
        print(f"Saved combined parameters to {combined_file}")

if __name__ == "__main__":
    # Load data
    print("Loading satellite observation data...")
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
    satobs = write_crs(satobs, config.CRS_TARGET)
    
    print("Loading ice shelf masks...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs({'init': config.CRS_TARGET})
    
    # Run comprehensive analysis
    # Note: Using config.ICE_SHELF_REGIONS which starts from index 33 (Abbott Ice Shelf)
    all_results, all_draft_params = calculate_draft_dependence_comprehensive(
        icems, satobs, config,
        n_bins=50,
        min_points_per_bin=5,
        ruptures_method='pelt',
        ruptures_penalty=1.0,
        min_r2_threshold=0.1,
        min_correlation=0.3,
        noisy_fallback='zero',
        model_selection='best',
        ice_shelf_range=config.ICE_SHELF_REGIONS  # This starts from 33 (Abbott)
    )
    
    print(f"\nProcessing complete! Processed {len(all_results)} ice shelves.")
    print("Output files saved to:")
    print(f"  - Individual files: {config.DIR_ICESHELF_DEDRAFT_SATOBS / 'comprehensive'}")
    print(f"  - Merged grids: {config.DIR_PROCESSED / 'draft_dependence_changepoint'}")
