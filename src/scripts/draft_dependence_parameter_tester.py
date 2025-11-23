#!/usr/bin/env python3
"""
Parameter Testing Framework for Draft Dependence Analysis

This script allows you to test different parameter combinations for draft dependence
analysis without modifying the original code. It creates multiple output directories
with different parameter sets so you can compare results.

Usage:
    python draft_dependence_parameter_tester.py [--parameter_sets SET1 SET2 ...] [--output_dir PATH]

"""

import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import xarray as xr
import geopandas as gpd
import argparse
import logging
import traceback

from aislens.config import config
from aislens.utils import write_crs, setup_logging

sys.path.insert(0, str(Path(__file__).parent))
from calculate_draft_dependence_comprehensive import calculate_draft_dependence_comprehensive

logger = logging.getLogger(__name__)


def define_parameter_sets():
    """
    Define different parameter combinations to test.
    
    Each parameter set will generate a separate output directory.
    Modify these to test different combinations.
    """
    
    parameter_sets = {
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
        
        'permissive': {
            'min_r2_threshold': 0.05,
            'min_correlation': 0.1,
            'ruptures_penalty': 1.0,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'More permissive thresholds - catch more linear relationships'
        },
        
        'very_permissive': {
            'min_r2_threshold': 0.02,
            'min_correlation': 0.05,
            'ruptures_penalty': 1.0,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'Very permissive - catch almost all as linear'
        },
        
        'sensitive_changepoint': {
            'min_r2_threshold': 0.1,
            'min_correlation': 0.2,
            'ruptures_penalty': 0.5,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'More sensitive changepoint detection'
        },
        
        'robust_changepoint': {
            'min_r2_threshold': 0.1,
            'min_correlation': 0.2,
            'ruptures_penalty': 2.0,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'Less sensitive changepoint detection'
        },
        
        'fine_binning': {
            'min_r2_threshold': 0.1,
            'min_correlation': 0.2,
            'ruptures_penalty': 1.0,
            'n_bins': 100,
            'min_points_per_bin': 3,
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'Finer binning with more bins'
        },
        
        'coarse_binning': {
            'min_r2_threshold': 0.1,
            'min_correlation': 0.2,
            'ruptures_penalty': 1.0,
            'n_bins': 25,
            'min_points_per_bin': 10,
            'noisy_fallback': 'zero',
            'model_selection': 'best',
            'description': 'Coarser binning with fewer bins'
        },
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
    
    Returns:
        Dictionary with results summary for each parameter set
    """
    
    if output_base_dir is None:
        output_base_dir = config.DIR_ICESHELF_DEDRAFT_SATOBS / "parameter_tests"
    
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_base_dir}")
    
    param_file = output_base_dir / "parameter_sets.json"
    with open(param_file, 'w') as f:
        json.dump(parameter_sets, f, indent=2, default=str)
    logger.info(f"Parameter sets saved to: {param_file}")
    
    results_summary = {}
    successful_sets = 0
    failed_sets = 0
    
    for set_name, params in parameter_sets.items():
        logger.info(f"TESTING PARAMETER SET: {set_name}")
        logger.info(f"Description: {params['description']}")
        
        set_output_dir = output_base_dir / set_name
        set_output_dir.mkdir(parents=True, exist_ok=True)
        
        params_file = set_output_dir / "parameters.json"
        run_info = {
            'parameter_set_name': set_name,
            'parameters': params,
            'timestamp': datetime.now().isoformat(),
            'description': params['description']
        }
        with open(params_file, 'w') as f:
            json.dump(run_info, f, indent=2, default=str)
        
        logger.debug(f"Parameters saved to: {params_file}")
        
        try:
            original_dir = config.DIR_ICESHELF_DEDRAFT_SATOBS
            config.DIR_ICESHELF_DEDRAFT_SATOBS = set_output_dir
            
            function_params = {k: v for k, v in params.items() if k != 'description'}
            
            logger.info(f"Running analysis with parameters: {function_params}")
            
            all_results, all_draft_params = calculate_draft_dependence_comprehensive(
                icems, satobs, config, **function_params
            )
            
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
            
            logger.info(f"  Completed {set_name}: {meaningful_count}/{total_count} meaningful relationships")
            logger.info(f"  Linear parameterizations: {results_summary[set_name]['linear_param_count']}")
            logger.info(f"  Constant parameterizations: {results_summary[set_name]['constant_param_count']}")
            successful_sets += 1
            
        except FileNotFoundError as e:
            logger.error(f"File not found in parameter set {set_name}: {e}")
            results_summary[set_name] = {
                'status': 'failed',
                'error': f"File not found: {e}",
                'output_directory': str(set_output_dir)
            }
            failed_sets += 1
            
        except ValueError as e:
            logger.error(f"  Invalid value in parameter set {set_name}: {e}")
            results_summary[set_name] = {
                'status': 'failed',
                'error': f"Invalid value: {e}",
                'output_directory': str(set_output_dir)
            }
            failed_sets += 1
            
        except Exception as e:
            logger.error(f"  Error in parameter set {set_name}: {e}")
            logger.debug(traceback.format_exc())
            results_summary[set_name] = {
                'status': 'failed',
                'error': str(e),
                'output_directory': str(set_output_dir)
            }
            failed_sets += 1
        
        finally:
            config.DIR_ICESHELF_DEDRAFT_SATOBS = original_dir
    
    summary_file = output_base_dir / "results_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info("PARAMETER TESTING SUMMARY")
    logger.info(f"Total parameter sets: {len(parameter_sets)}")
    logger.info(f"Successful: {successful_sets}")
    logger.info(f"Failed: {failed_sets}")
    logger.info("")
    
    for set_name, summary in results_summary.items():
        if summary['status'] == 'completed':
            logger.info(f"{set_name:20s}: {summary['meaningful_shelves']:3d}/{summary['total_shelves']:3d} meaningful "
                  f"({summary['meaningful_percentage']:5.1f}%), "
                  f"linear: {summary['linear_param_count']:3d}, "
                  f"constant: {summary['constant_param_count']:3d}")
        else:
            logger.error(f"{set_name:20s}: FAILED - {summary['error']}")
    
    logger.info(f"\nResults saved to: {output_base_dir}")
    logger.info(f"Summary file: {summary_file}")
    
    return results_summary

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Parameter testing framework for draft dependence analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--parameter_sets', nargs='+', default=None,
                        help='Specific parameter set names to test (default: test all)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Base output directory for parameter testing results')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show parameter sets without running analysis')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose debug logging')
    
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.DIR_ICESHELF_DEDRAFT_SATOBS / "parameter_tests"
    
    setup_logging(output_dir, "parameter_testing")
    
    logger.info("DRAFT DEPENDENCE PARAMETER TESTING FRAMEWORK")
    
    all_parameter_sets = define_parameter_sets()
    if args.parameter_sets:
        logger.info(f"Filtering to requested parameter sets: {args.parameter_sets}")
        parameter_sets = {name: params for name, params in all_parameter_sets.items() 
                         if name in args.parameter_sets}
        
        invalid_names = set(args.parameter_sets) - set(all_parameter_sets.keys())
        if invalid_names:
            logger.error(f"Invalid parameter set names: {invalid_names}")
            logger.error(f"Available sets: {list(all_parameter_sets.keys())}")
            sys.exit(1)
    else:
        parameter_sets = all_parameter_sets
    
    logger.info(f"\nWill test {len(parameter_sets)} parameter combination(s):")
    for name, params in parameter_sets.items():
        logger.info(f"  {name}: {params['description']}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN MODE] - Parameter sets defined:")
        for name, params in parameter_sets.items():
            logger.info(f"\n{name}:")
            for key, value in params.items():
                if key != 'description':
                    logger.info(f"  {key}: {value}")
        logger.info("\nDry run complete. No analysis was performed.")
        return
    
    try:
        logger.info("\nLoading satellite observation data...")
        satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
        satobs = write_crs(satobs, config.CRS_TARGET)
        logger.info(f"Loaded satellite data with shape: {satobs.dims}")
        
        logger.info("Loading ice shelf masks...")
        icems = gpd.read_file(config.FILE_ICESHELFMASKS)
        icems = icems.to_crs(config.CRS_TARGET)
        logger.info(f"Loaded {len(icems)} ice shelf masks")
        
    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}")
        logger.error("Please ensure all input files are available")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        results_summary = run_parameter_tests(icems, satobs, config, parameter_sets, output_dir)
        
        logger.info("Parameter testing complete!")
        logger.info("\nNext steps:")
        logger.info("1. Review the results_summary.json file")
        logger.info("2. Use visualize_draft_dependence.py to compare outputs:")
        logger.info(f"   python visualize_draft_dependence.py --parameter_set <set_name>")
        logger.info("3. Choose the parameter set that gives you the desired results")
        
    except Exception as e:
        logger.error(f"Parameter testing failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
