#!/usr/bin/env python3
"""
Compare draft dependence analysis results across different parameter sets.

Tests conservative, permissive, and very permissive parameters to show how
parameter choices affect linear vs constant parameterization classifications.

Usage: python compare_parameter_sets.py
"""

import logging
import sys
from pathlib import Path
import geopandas as gpd
import xarray as xr

sys.path.append(str(Path(__file__).parent.parent))

from aislens.config import config
from aislens.utils import setup_logging
from scripts.calculate_draft_dependence_comprehensive import calculate_draft_dependence_comprehensive

logger = logging.getLogger(__name__)

def run_conservative_analysis(icems, satobs, config):
    """Original conservative parameters - baseline for comparison."""
    logger.info("Running CONSERVATIVE analysis...")
    return calculate_draft_dependence_comprehensive(
        icems, satobs, config,
        min_r2_threshold=0.1, min_correlation=0.2, ruptures_penalty=1.0,
        n_bins=50, min_points_per_bin=5, noisy_fallback='zero', model_selection='best'
    )

def run_permissive_analysis(icems, satobs, config):
    """Recommended permissive parameters - more linear parameterizations."""
    logger.info("Running PERMISSIVE analysis...")
    return calculate_draft_dependence_comprehensive(
        icems, satobs, config,
        min_r2_threshold=0.03, min_correlation=0.1, ruptures_penalty=0.8,
        n_bins=50, min_points_per_bin=4, noisy_fallback='zero', model_selection='best'
    )

def run_very_permissive_analysis(icems, satobs, config):
    """Very permissive parameters - maximum linear coverage."""
    logger.info("Running VERY PERMISSIVE analysis...")
    return calculate_draft_dependence_comprehensive(
        icems, satobs, config,
        min_r2_threshold=0.02, min_correlation=0.08, ruptures_penalty=0.6,
        n_bins=40, min_points_per_bin=3, noisy_fallback='zero', model_selection='best'
    )

def compare_results(conservative_params, permissive_params, very_permissive_params):
    """Compare the results from different parameter sets."""
    count_linear = lambda p: sum(1 for params in p.values() if params.get('paramType') == 0)
    
    cons_lin = count_linear(conservative_params)
    perm_lin = count_linear(permissive_params)
    vperm_lin = count_linear(very_permissive_params)
    total = len(conservative_params)
    
    logger.info("\n" + "="*60)
    logger.info("COMPARISON RESULTS")
    logger.info("="*60)
    logger.info(f"{'Parameter Set':<20} {'Linear':<8} {'Constant':<10} {'% Linear':<10}")
    logger.info("-"*60)
    logger.info(f"{'Conservative':<20} {cons_lin:<8} {total-cons_lin:<10} {cons_lin/total*100:<8.1f}%")
    logger.info(f"{'Permissive':<20} {perm_lin:<8} {total-perm_lin:<10} {perm_lin/total*100:<8.1f}%")
    logger.info(f"{'Very Permissive':<20} {vperm_lin:<8} {total-vperm_lin:<10} {vperm_lin/total*100:<8.1f}%")
    logger.info(f"\nImprovements:")
    logger.info(f"  Conservative → Permissive: +{perm_lin - cons_lin} shelves")
    logger.info(f"  Permissive → Very Permissive: +{vperm_lin - perm_lin} shelves")
    logger.info(f"  Total improvement: +{vperm_lin - cons_lin} shelves")

if __name__ == "__main__":
    output_dir = Path(config.DIR_ICESHELF_DEDRAFT_SATOBS) / "parameter_tests" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "compare_parameter_sets")
    
    logger.info("=" * 60)
    logger.info("DRAFT DEPENDENCE PARAMETER COMPARISON")
    logger.info("=" * 60)
    
    logger.info("Loading ice shelf masks...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS).to_crs({'init': config.CRS_TARGET})
    logger.info(f"Loaded {len(icems)} ice shelf masks")
    
    logger.info("Loading satellite observations...")
    satobs = xr.open_dataset(config.FILE_SATOBS)
    logger.info(f"Loaded data with variables: {list(satobs.data_vars)}")
    
    logger.info("\nRunning analyses with different parameter sets...")
    conservative_results, conservative_params = run_conservative_analysis(icems, satobs, config)
    permissive_results, permissive_params = run_permissive_analysis(icems, satobs, config)
    very_permissive_results, very_permissive_params = run_very_permissive_analysis(icems, satobs, config)
    
    compare_results(conservative_params, permissive_params, very_permissive_params)
    
    logger.info("\nRecommendations:")
    logger.info("  Balanced quality/coverage → PERMISSIVE parameters")
    logger.info("  Maximum linear coverage → VERY PERMISSIVE parameters")
    logger.info("  Highest quality only → CONSERVATIVE parameters")
    
    logger.info("\nNext steps:")
    logger.info("  1. Run: python visualize_draft_dependence.py")
    logger.info("  2. Run: python inspect_draft_dependence.py")
    logger.info("  3. Check individual results in output directories")
    
    logger.info("=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 60)
