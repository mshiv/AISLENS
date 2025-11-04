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
import logging
import traceback
from datetime import datetime

# Add parent directory to path to import calculate_draft_dependence_comprehensive
sys.path.insert(0, str(Path(__file__).parent))
from calculate_draft_dependence_comprehensive import calculate_draft_dependence_comprehensive

# Import config (already instantiated)
from aislens.config import config

# ===== LOGGING CONFIGURATION =====
logger = logging.getLogger(__name__)

def setup_logging(output_dir, verbose=False):
    """
    Configure logging for the permissive draft dependence script.
    
    Args:
        output_dir: Directory to save log file
        verbose: If True, set logging level to DEBUG
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = Path(output_dir) / f'permissive_draft_dependence_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configure logger
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return log_file

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
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Run draft dependence analysis with various parameter sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter Set Descriptions:
  conservative      - Original thresholds (~30-40% linear)
  moderate          - Moderately relaxed (~40-55% linear)
  permissive        - Permissive thresholds (~55-70% linear)
  very_permissive   - Very permissive (~65-80% linear)
  ultra_permissive  - Minimal requirements (~75-90% linear, may include noisy fits)
  small_shelves     - Optimized for small ice shelves
  large_shelves     - Optimized for large ice shelves
  high_quality_only - High quality fits only (~20-35% linear)

Examples:
  # Run with permissive parameter set
  python run_permissive_draft_dependence.py --parameter_set permissive
  
  # List all available parameter sets
  python run_permissive_draft_dependence.py --list_sets
  
  # Run with custom output suffix and verbose logging
  python run_permissive_draft_dependence.py --parameter_set moderate --output_suffix test1 --verbose
        """
    )
    parser.add_argument('--parameter_set', 
                       choices=list(PARAMETER_SETS.keys()),
                       default='permissive',
                       help='Parameter set to use (default: permissive)')
    parser.add_argument('--list_sets', action='store_true',
                       help='List available parameter sets and exit')
    parser.add_argument('--output_suffix', type=str, default=None,
                       help='Suffix to add to output filenames')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose debug logging')
    
    args = parser.parse_args()
    
    # List parameter sets if requested
    if args.list_sets:
        print("\nAvailable parameter sets:")
        print("=" * 80)
        for name, params in PARAMETER_SETS.items():
            print(f"\n{name.upper()}:")
            print(f"  {params['description']}")
            print(f"  R² threshold: {params['min_r2_threshold']}")
            print(f"  Correlation threshold: {params['min_correlation']}")
            print(f"  Ruptures penalty: {params['ruptures_penalty']}")
            print(f"  Bins: {params['n_bins']}, Min points/bin: {params['min_points_per_bin']}")
        print("")
        return
    
    # Setup logging
    output_dir = config.DIR_ICESHELF_DEDRAFT_SATOBS
    log_file = setup_logging(output_dir, verbose=args.verbose)
    
    logger.info("="*80)
    logger.info("PERMISSIVE DRAFT DEPENDENCE ANALYSIS")
    logger.info("="*80)
    
    # Get parameter set
    param_set = PARAMETER_SETS[args.parameter_set]
    logger.info(f"Running with parameter set: {args.parameter_set.upper()}")
    logger.info(f"Description: {param_set['description']}")
    logger.info("\nParameters:")
    for key, value in param_set.items():
        if key != 'description':
            logger.info(f"  {key}: {value}")
    
    try:
        # Load ice shelf masks
        logger.info("\nLoading ice shelf masks...")
        try:
            icems = gpd.read_file(config.FILE_ICESHELFMASKS)
            icems = icems.to_crs(config.CRS_TARGET)
            logger.info(f"Loaded {len(icems)} ice shelf masks")
        except FileNotFoundError:
            logger.error(f"Ice shelf mask file not found: {config.FILE_ICESHELFMASKS}")
            logger.error("Please check file path in config")
            sys.exit(1)
        
        # Load satellite observations
        logger.info("Loading satellite observations...")
        try:
            satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
            logger.info(f"Loaded satellite observations: {list(satobs.data_vars)}")
            logger.debug(f"Satellite data dimensions: {satobs.dims}")
        except FileNotFoundError:
            logger.error(f"Satellite observation file not found: {config.FILE_PAOLO23_SATOBS_PREPARED}")
            logger.error("Please run data preparation script first")
            sys.exit(1)
        
        # Run analysis with selected parameters
        logger.info(f"\nRunning comprehensive analysis with {args.parameter_set} parameters...")
        
        # Remove description from params before passing to function
        run_params = {k: v for k, v in param_set.items() if k != 'description'}
        
        all_results, all_draft_params = calculate_draft_dependence_comprehensive(
            icems, satobs, config,
            **run_params,
            noisy_fallback='zero',
            model_selection='best'
        )
        
        logger.info("\nAnalysis complete!")
        logger.info(f"Results saved with parameter set: {args.parameter_set}")
        
        # Count linear vs constant parameterizations
        if all_draft_params:
            linear_count = sum(1 for params in all_draft_params.values() if params.get('paramType') == 0)
            total_count = len(all_draft_params)
            linear_percentage = linear_count / total_count * 100 if total_count > 0 else 0
            
            logger.info("\n" + "="*80)
            logger.info("PARAMETERIZATION RESULTS")
            logger.info("="*80)
            logger.info(f"Total ice shelves processed: {total_count}")
            logger.info(f"Linear parameterizations: {linear_count} ({linear_percentage:.1f}%)")
            logger.info(f"Constant parameterizations: {total_count - linear_count} ({100 - linear_percentage:.1f}%)")
            
            # Extract expected range from description
            if '(' in param_set['description'] and ')' in param_set['description']:
                expected_range = param_set['description'].split('(')[1].split(')')[0]
                logger.info(f"\nExpected range for {args.parameter_set}: {expected_range}")
            
            # Provide recommendations
            if linear_percentage < 50:
                logger.info("\n💡 Suggestion: Try a more permissive parameter set (e.g., 'very_permissive') for more linear parameterizations")
            elif linear_percentage > 85:
                logger.warning("\n⚠️  Warning: Very high linear percentage - check visualization for noisy fits")
                logger.warning("   Consider using 'high_quality_only' parameter set for more reliable fits")
        else:
            logger.warning("No parameterization results generated")
        
        # Suggest next steps
        logger.info("\n" + "="*80)
        logger.info("NEXT STEPS")
        logger.info("="*80)
        logger.info(f"1. Visualize results:")
        logger.info(f"   python visualize_draft_dependence.py --parameter_set {args.parameter_set}")
        logger.info(f"2. Compare with other parameter sets:")
        logger.info(f"   python visualize_draft_dependence.py --create_summary")
        logger.info(f"3. Test multiple parameter sets:")
        logger.info(f"   python draft_dependence_parameter_tester.py")
        
        logger.info("\n" + "="*80)
        logger.info("Analysis complete!")
        logger.info("="*80)
        
    except ValueError as e:
        logger.error(f"Invalid value or parameter: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
