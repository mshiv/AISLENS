#!/usr/bin/env python3
"""
Script to run draft dependence analysis with more permissive parameters.

This script provides several parameter sets that progressively increase the number
of ice shelves receiving linear (vs constant) parameterizations.

Usage:
    python run_permissive_draft_dependence.py --parameter_set moderate
    python run_permissive_draft_dependence.py --parameter_set permissive  
    python run_permissive_draft_dependence.py --parameter_set very_permissive
    python run_permissive_draft_dependence.py --parameter_set ultra_permissive
"""

import sys
import argparse
from pathlib import Path
import geopandas as gpd
import xarray as xr

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from aislens.config import load_config
from scripts.calculate_draft_dependence_comprehensive import calculate_draft_dependence_comprehensive

# Parameter sets with increasing permissiveness
PARAMETER_SETS = {
    'conservative': {
        'min_r2_threshold': 0.1,
        'min_correlation': 0.2,
        'ruptures_penalty': 1.0,
        'n_bins': 50,
        'min_points_per_bin': 5,
        'description': 'Original conservative thresholds (~30-40% linear)'
    },
    
    'moderate': {
        'min_r2_threshold': 0.05,  # Lower R² requirement
        'min_correlation': 0.15,   # Lower correlation requirement
        'ruptures_penalty': 1.0,
        'n_bins': 50,
        'min_points_per_bin': 5,
        'description': 'Moderately relaxed thresholds (~40-55% linear)'
    },
    
    'permissive': {
        'min_r2_threshold': 0.03,  # Much lower R² 
        'min_correlation': 0.1,    # Much lower correlation
        'ruptures_penalty': 0.8,   # More sensitive changepoint detection
        'n_bins': 50,
        'min_points_per_bin': 4,   # Allow sparser bins
        'description': 'Permissive thresholds (~55-70% linear)'
    },
    
    'very_permissive': {
        'min_r2_threshold': 0.02,  # Very low R²
        'min_correlation': 0.08,   # Very low correlation  
        'ruptures_penalty': 0.6,   # More sensitive changepoints
        'n_bins': 40,              # Fewer bins (better for sparse data)
        'min_points_per_bin': 3,   # Allow very sparse bins
        'description': 'Very permissive thresholds (~65-80% linear)'
    },
    
    'ultra_permissive': {
        'min_r2_threshold': 0.01,  # Minimal R² requirement
        'min_correlation': 0.05,   # Minimal correlation requirement
        'ruptures_penalty': 0.4,   # Very sensitive changepoints  
        'n_bins': 30,              # Coarser binning
        'min_points_per_bin': 2,   # Minimal data requirement
        'description': 'Ultra permissive thresholds (~75-90% linear, may include noisy fits)'
    },
    
    # Special parameter sets for specific use cases
    'small_shelves': {
        'min_r2_threshold': 0.05,
        'min_correlation': 0.12,
        'ruptures_penalty': 1.2,   # Less sensitive (fewer spurious changepoints)
        'n_bins': 25,              # Fewer bins for small shelves
        'min_points_per_bin': 3,   # Sparse data tolerance
        'description': 'Optimized for small ice shelves with limited data'
    },
    
    'large_shelves': {
        'min_r2_threshold': 0.08,
        'min_correlation': 0.15,
        'ruptures_penalty': 0.5,   # More sensitive (detect complex patterns)
        'n_bins': 100,             # High resolution for large shelves
        'min_points_per_bin': 10,  # Require dense data
        'description': 'Optimized for large ice shelves with dense data'
    },
    
    'high_quality_only': {
        'min_r2_threshold': 0.15,  # Higher standards
        'min_correlation': 0.3,    # Higher standards
        'ruptures_penalty': 1.5,   # Conservative changepoint detection
        'n_bins': 50,
        'min_points_per_bin': 8,   # Require denser data
        'description': 'High quality fits only (~20-35% linear, but very reliable)'
    }
}

def main():
    parser = argparse.ArgumentParser(description='Run draft dependence analysis with various parameter sets')
    parser.add_argument('--parameter_set', 
                       choices=list(PARAMETER_SETS.keys()),
                       default='permissive',
                       help='Parameter set to use (default: permissive)')
    parser.add_argument('--list_sets', action='store_true',
                       help='List available parameter sets and exit')
    parser.add_argument('--output_suffix', type=str, default=None,
                       help='Suffix to add to output filenames')
    
    args = parser.parse_args()
    
    if args.list_sets:
        print("Available parameter sets:")
        print("=" * 60)
        for name, params in PARAMETER_SETS.items():
            print(f"\n{name.upper()}:")
            print(f"  {params['description']}")
            print(f"  R² threshold: {params['min_r2_threshold']}")
            print(f"  Correlation threshold: {params['min_correlation']}")
            print(f"  Ruptures penalty: {params['ruptures_penalty']}")
            print(f"  Bins: {params['n_bins']}, Min points/bin: {params['min_points_per_bin']}")
        return
    
    # Get parameter set
    param_set = PARAMETER_SETS[args.parameter_set]
    print(f"Running with parameter set: {args.parameter_set.upper()}")
    print(f"Description: {param_set['description']}")
    print("Parameters:")
    for key, value in param_set.items():
        if key != 'description':
            print(f"  {key}: {value}")
    
    # Load configuration and data
    print("\nLoading configuration and data...")
    config = load_config()
    
    # Load ice shelf masks
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs({'init': config.CRS_TARGET})
    print(f"Loaded {len(icems)} ice shelf masks")
    
    # Load satellite observations
    satobs = xr.open_dataset(config.FILE_SATOBS)
    print(f"Loaded satellite observations: {list(satobs.data_vars)}")
    
    # Run analysis with selected parameters
    print(f"\nRunning comprehensive analysis with {args.parameter_set} parameters...")
    
    # Remove description from params before passing to function
    run_params = {k: v for k, v in param_set.items() if k != 'description'}
    
    all_results, all_draft_params = calculate_draft_dependence_comprehensive(
        icems, satobs, config,
        **run_params,
        noisy_fallback='zero',
        model_selection='best'
    )
    
    print(f"\nAnalysis complete!")
    print(f"Results saved with parameter set: {args.parameter_set}")
    
    # Count linear vs constant parameterizations
    if all_draft_params:
        linear_count = sum(1 for params in all_draft_params.values() if params.get('paramType') == 0)
        total_count = len(all_draft_params)
        linear_percentage = linear_count / total_count * 100 if total_count > 0 else 0
        
        print(f"\nParameterization Results:")
        print(f"  Linear parameterizations: {linear_count}/{total_count} ({linear_percentage:.1f}%)")
        print(f"  Constant parameterizations: {total_count - linear_count}/{total_count} ({100 - linear_percentage:.1f}%)")
        
        print(f"\nExpected range for {args.parameter_set}: {param_set['description'].split('(')[1].split(')')[0]}")
    
    # Suggest next steps
    print(f"\nNext steps:")
    print(f"1. Check visualization: python visualize_draft_dependence.py")
    print(f"2. Inspect results: python inspect_draft_dependence.py --parameter_set {args.parameter_set}")
    print(f"3. Compare with other parameter sets")
    
    if linear_percentage < 50:
        print(f"\nSuggestion: Try a more permissive parameter set (e.g., 'very_permissive') for more linear parameterizations")
    elif linear_percentage > 85:
        print(f"\nWarning: Very high linear percentage - check visualization for noisy fits")

if __name__ == "__main__":
    main()
