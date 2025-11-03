#!/usr/bin/env python3
"""
Example script showing how to run draft dependence analysis with permissive parameters.

This demonstrates exactly what parameter changes to make in your existing script
to get more ice shelves with linear (vs constant) parameterizations.
"""

import sys
from pathlib import Path
import geopandas as gpd
import xarray as xr

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from aislens.config import load_config
from scripts.calculate_draft_dependence_comprehensive import calculate_draft_dependence_comprehensive

def run_conservative_analysis(icems, satobs, config):
    """Original conservative parameters - baseline for comparison."""
    print("Running CONSERVATIVE analysis...")
    return calculate_draft_dependence_comprehensive(
        icems, satobs, config,
        min_r2_threshold=0.1,        # High R² requirement
        min_correlation=0.2,         # High correlation requirement
        ruptures_penalty=1.0,        # Standard changepoint sensitivity
        n_bins=50,
        min_points_per_bin=5,
        noisy_fallback='zero',
        model_selection='best'
    )

def run_permissive_analysis(icems, satobs, config):
    """Recommended permissive parameters - more linear parameterizations."""
    print("Running PERMISSIVE analysis...")
    return calculate_draft_dependence_comprehensive(
        icems, satobs, config,
        min_r2_threshold=0.03,       # MUCH LOWER: 3x more permissive
        min_correlation=0.1,         # MUCH LOWER: 2x more permissive
        ruptures_penalty=0.8,        # MORE SENSITIVE: detects more changepoints
        n_bins=50,                   # Same as before
        min_points_per_bin=4,        # SLIGHTLY LOWER: allows sparser data
        noisy_fallback='zero',
        model_selection='best'
    )

def run_very_permissive_analysis(icems, satobs, config):
    """Very permissive parameters - maximum linear coverage."""
    print("Running VERY PERMISSIVE analysis...")
    return calculate_draft_dependence_comprehensive(
        icems, satobs, config,
        min_r2_threshold=0.02,       # VERY LOW: minimal R² requirement
        min_correlation=0.08,        # VERY LOW: minimal correlation requirement
        ruptures_penalty=0.6,        # VERY SENSITIVE: finds subtle changepoints
        n_bins=40,                   # FEWER BINS: better for sparse data
        min_points_per_bin=3,        # VERY LOW: minimal data per bin
        noisy_fallback='zero',
        model_selection='best'
    )

def compare_results(conservative_params, permissive_params, very_permissive_params):
    """Compare the results from different parameter sets."""
    
    def count_linear(params_dict):
        return sum(1 for p in params_dict.values() if p.get('paramType') == 0)
    
    conservative_linear = count_linear(conservative_params)
    permissive_linear = count_linear(permissive_params)
    very_permissive_linear = count_linear(very_permissive_params)
    
    total = len(conservative_params)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Parameter Set':<20} {'Linear':<8} {'Constant':<10} {'% Linear':<10}")
    print("-"*60)
    print(f"{'Conservative':<20} {conservative_linear:<8} {total-conservative_linear:<10} {conservative_linear/total*100:<8.1f}%")
    print(f"{'Permissive':<20} {permissive_linear:<8} {total-permissive_linear:<10} {permissive_linear/total*100:<8.1f}%")
    print(f"{'Very Permissive':<20} {very_permissive_linear:<8} {total-very_permissive_linear:<10} {very_permissive_linear/total*100:<8.1f}%")
    
    print(f"\nImprovements:")
    print(f"  Conservative → Permissive: +{permissive_linear - conservative_linear} ice shelves")
    print(f"  Permissive → Very Permissive: +{very_permissive_linear - permissive_linear} ice shelves")
    print(f"  Total improvement: +{very_permissive_linear - conservative_linear} ice shelves")

def main():
    print("Draft Dependence Parameter Comparison Script")
    print("=" * 50)
    
    # Load configuration and data
    config = load_config()
    
    print("Loading ice shelf masks...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs({'init': config.CRS_TARGET})
    print(f"Loaded {len(icems)} ice shelf masks")
    
    print("Loading satellite observations...")
    satobs = xr.open_dataset(config.FILE_SATOBS)
    print(f"Loaded satellite data with variables: {list(satobs.data_vars)}")
    
    # Run all three analyses
    print("\nRunning analyses with different parameter sets...")
    
    # Conservative (baseline)
    conservative_results, conservative_params = run_conservative_analysis(icems, satobs, config)
    
    # Permissive (recommended)
    permissive_results, permissive_params = run_permissive_analysis(icems, satobs, config)
    
    # Very permissive (maximum coverage)
    very_permissive_results, very_permissive_params = run_very_permissive_analysis(icems, satobs, config)
    
    # Compare results
    compare_results(conservative_params, permissive_params, very_permissive_params)
    
    print("\nRecommendations:")
    print("- For balanced quality/coverage: Use PERMISSIVE parameters")
    print("- For maximum linear coverage: Use VERY PERMISSIVE parameters")
    print("- For highest quality only: Use CONSERVATIVE parameters")
    
    print("\nNext steps:")
    print("1. Run: python visualize_draft_dependence.py")
    print("2. Run: python inspect_draft_dependence.py")
    print("3. Check individual ice shelf results in output directories")

if __name__ == "__main__":
    main()
