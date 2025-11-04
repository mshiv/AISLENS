#!/usr/bin/env python3
"""
Parameter Testing Framework for Draft Dependence Analysis

This script allows you to test different parameter combinations for draft dependence
analysis without modifying the original code. It creates multiple output directories
with different parameter sets so you can compare results.

Usage:
    python draft_dependence_parameter_tester.py

Author: Generated for AISLENS project
Date: August 2025
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import xarray as xr
import geopandas as gpd

from aislens.config import config
from aislens.utils import write_crs

# Import the main calculation function from the scripts directory
sys.path.insert(0, str(Path(__file__).parent))
from calculate_draft_dependence_comprehensive import calculate_draft_dependence_comprehensive


def define_parameter_sets():
    """
    Define different parameter combinations to test.
    
    Each parameter set will generate a separate output directory.
    Modify these to test different combinations.
    """
    
    parameter_sets = {
        # Original parameters (as baseline)
        'original': {
            'min_r2_threshold': 0.1,
            'min_correlation': 0.2,
            'ruptures_penalty': 1.0,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'Original parameters from your script'
        },
        
        # More permissive - catch more shelves as linear
        'permissive': {
            'min_r2_threshold': 0.05,  # Lower R² threshold
            'min_correlation': 0.1,    # Lower correlation threshold  
            'ruptures_penalty': 1.0,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'More permissive thresholds - catch more linear relationships'
        },
        
        # Very permissive
        'very_permissive': {
            'min_r2_threshold': 0.02,  # Very low R² threshold
            'min_correlation': 0.05,   # Very low correlation threshold
            'ruptures_penalty': 1.0,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'Very permissive - catch almost all as linear'
        },
        
        # Different ruptures sensitivity
        'sensitive_changepoint': {
            'min_r2_threshold': 0.1,
            'min_correlation': 0.2,
            'ruptures_penalty': 0.5,   # More sensitive to changepoints
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'More sensitive changepoint detection'
        },
        
        # Less sensitive changepoint detection
        'robust_changepoint': {
            'min_r2_threshold': 0.1,
            'min_correlation': 0.2,
            'ruptures_penalty': 2.0,   # Less sensitive to changepoints
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'Less sensitive changepoint detection'
        },
        
        # Different binning strategy
        'fine_binning': {
            'min_r2_threshold': 0.1,
            'min_correlation': 0.2,
            'ruptures_penalty': 1.0,
            'n_bins': 100,             # More bins
            'min_points_per_bin': 3,   # Fewer points per bin required
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'Finer binning with more bins'
        },
        
        # Coarse binning
        'coarse_binning': {
            'min_r2_threshold': 0.1,
            'min_correlation': 0.2,
            'ruptures_penalty': 1.0,
            'n_bins': 25,              # Fewer bins
            'min_points_per_bin': 10,  # More points per bin required
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'Coarser binning with fewer bins'
        }
    }
    
    return parameter_sets

def run_parameter_tests(icems, satobs, config, parameter_sets, output_base_dir=None):
    """
    Run draft dependence analysis with different parameter combinations.
    
    Args:
        icems: GeoDataFrame with ice shelf masks
        satobs: xarray Dataset with satellite observations  
        config: Configuration object
        parameter_sets: Dictionary of parameter combinations to test
        output_base_dir: Base directory for outputs (optional)
    """
    
    if output_base_dir is None:
        output_base_dir = config.DIR_ICESHELF_DEDRAFT_SATOBS / "parameter_tests"
    
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameter sets for reference
    param_file = output_base_dir / "parameter_sets.json"
    with open(param_file, 'w') as f:
        json.dump(parameter_sets, f, indent=2, default=str)
    print(f"Parameter sets saved to: {param_file}")
    
    results_summary = {}
    
    for set_name, params in parameter_sets.items():
        print(f"\n{'='*60}")
        print(f"TESTING PARAMETER SET: {set_name}")
        print(f"Description: {params['description']}")
        print(f"{'='*60}")
        
        # Create output directory for this parameter set
        set_output_dir = output_base_dir / set_name
        set_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save parameters for this run
        params_file = set_output_dir / "parameters.json"
        run_info = {
            'parameter_set_name': set_name,
            'parameters': params,
            'timestamp': datetime.now().isoformat(),
            'description': params['description']
        }
        with open(params_file, 'w') as f:
            json.dump(run_info, f, indent=2, default=str)
        
        try:
            # Temporarily modify config to save to this parameter set's directory
            original_dir = config.DIR_ICESHELF_DEDRAFT_SATOBS
            config.DIR_ICESHELF_DEDRAFT_SATOBS = set_output_dir
            
            # Extract parameters (remove description as it's not a function parameter)
            function_params = {k: v for k, v in params.items() if k != 'description'}
            
            # Run analysis with this parameter set
            all_results, all_draft_params = calculate_draft_dependence_comprehensive(
                icems, satobs, config, **function_params
            )
            
            # Restore original config
            config.DIR_ICESHELF_DEDRAFT_SATOBS = original_dir
            
            # Store summary statistics
            meaningful_count = sum(1 for r in all_results.values() if r.get('is_meaningful', False))
            total_count = len(all_results)
            
            results_summary[set_name] = {
                'total_shelves': total_count,
                'meaningful_shelves': meaningful_count,
                'meaningful_percentage': meaningful_count / total_count * 100 if total_count > 0 else 0,
                'linear_param_count': sum(1 for p in all_draft_params.values() if p.get('paramType', 1) == 0),
                'constant_param_count': sum(1 for p in all_draft_params.values() if p.get('paramType', 1) == 1),
                'output_directory': str(set_output_dir),
                'status': 'completed'
            }
            
            print(f"✓ Completed {set_name}: {meaningful_count}/{total_count} meaningful relationships")
            
        except Exception as e:
            print(f"✗ Error in parameter set {set_name}: {e}")
            results_summary[set_name] = {
                'status': 'failed',
                'error': str(e),
                'output_directory': str(set_output_dir)
            }
            
            # Restore original config in case of error
            config.DIR_ICESHELF_DEDRAFT_SATOBS = original_dir
    
    # Save results summary
    summary_file = output_base_dir / "results_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("PARAMETER TESTING SUMMARY")
    print(f"{'='*60}")
    
    for set_name, summary in results_summary.items():
        if summary['status'] == 'completed':
            print(f"{set_name:20s}: {summary['meaningful_shelves']:3d}/{summary['total_shelves']:3d} meaningful "
                  f"({summary['meaningful_percentage']:5.1f}%), "
                  f"linear: {summary['linear_param_count']:3d}, "
                  f"constant: {summary['constant_param_count']:3d}")
        else:
            print(f"{set_name:20s}: FAILED - {summary['error']}")
    
    print(f"\nResults saved to: {output_base_dir}")
    print(f"Summary file: {summary_file}")
    
    return results_summary

def main():
    """Main function to run parameter testing."""
    
    print("DRAFT DEPENDENCE PARAMETER TESTING FRAMEWORK")
    print("=" * 50)
    
    # Load data
    print("Loading satellite observation data...")
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
    satobs = write_crs(satobs, config.CRS_TARGET)
    
    print("Loading ice shelf masks...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs(config.CRS_TARGET)
    
    # Define parameter sets to test
    parameter_sets = define_parameter_sets()
    
    print(f"\nWill test {len(parameter_sets)} parameter combinations:")
    for name, params in parameter_sets.items():
        print(f"  {name}: {params['description']}")
    
    # Run parameter tests
    results_summary = run_parameter_tests(icems, satobs, config, parameter_sets)
    
    print("\nParameter testing complete!")
    print("Next steps:")
    print("1. Review the results_summary.json file")
    print("2. Use the visualization script to compare outputs")
    print("3. Choose the parameter set that gives you the desired results")

if __name__ == "__main__":
    main()
