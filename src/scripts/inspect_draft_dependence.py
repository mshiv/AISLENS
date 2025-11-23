#!/usr/bin/env python3
"""
Inspect draft dependence analysis results to understand ice shelf classifications.

Analyzes which ice shelves are classified as linear vs constant, shows classification
criteria, and suggests parameter modifications to achieve target classifications.

Usage:
    python inspect_draft_dependence.py --parameter_set original
    python inspect_draft_dependence.py --parameter_set original --shelf_name "Pine Island"
    python inspect_draft_dependence.py --parameter_set original --create_sensitivity_plot
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from aislens.config import config
from aislens.utils import setup_logging

logger = logging.getLogger(__name__)

def load_analysis_results(parameter_set_name, parameter_test_dir=None):
    """Load analysis results for a specific parameter set."""
    if parameter_test_dir is None:
        parameter_test_dir = config.DIR_ICESHELF_DEDRAFT_SATOBS / "parameter_tests"
    
    summary_file = Path(parameter_test_dir) / parameter_set_name / "comprehensive" / "comprehensive_summary.csv"
    
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        logger.info(f"Loaded results for {len(summary_df)} ice shelves from {summary_file}")
        return summary_df
    else:
        logger.warning(f"Summary file not found: {summary_file}")
        return None

def _log_shelf_examples(df, mask, cols, label, threshold_info=""):
    """Helper to log ice shelf examples matching criteria."""
    count = mask.sum()
    logger.info(f"\n{label} {threshold_info}: {count} shelves")
    if count > 0:
        for _, row in df[mask][cols].head(10).iterrows():
            metrics = ", ".join([f"{col}={row[col]:.3f}" if col in ['r2', 'correlation'] 
                                else f"{col}={row[col]}" for col in cols if col != 'shelf_name'])
            logger.info(f"  {row['shelf_name']}: {metrics}")

def analyze_classification_criteria(summary_df, min_r2=0.1, min_corr=0.2):
    """Analyze why ice shelves were classified as linear vs constant."""
    logger.info("\nCLASSIFICATION ANALYSIS")
    logger.info("=" * 50)
    
    total = len(summary_df)
    meaningful = summary_df['is_meaningful'].sum()
    logger.info(f"Total: {total} | Meaningful: {meaningful} ({meaningful/total*100:.1f}%)")
    logger.info(f"Linear: {(summary_df['paramType']==0).sum()} | Constant: {(summary_df['paramType']==1).sum()}")
    logger.info(f"\nThresholds - R²: {min_r2}, Correlation: {min_corr}")
    
    # Analyze different categories
    cols = ['shelf_name', 'r2', 'correlation', 'paramType']
    low_r2 = (summary_df['r2'] < min_r2) & (~summary_df['r2'].isna())
    low_corr = (np.abs(summary_df['correlation']) < min_corr) & (~summary_df['correlation'].isna())
    borderline = (((summary_df['r2'] >= min_r2) & (summary_df['r2'] < min_r2*2)) | 
                  ((np.abs(summary_df['correlation']) >= min_corr) & (np.abs(summary_df['correlation']) < min_corr*2))
                 ) & (~summary_df['r2'].isna()) & (~summary_df['correlation'].isna())
    
    _log_shelf_examples(summary_df, low_r2, cols, "Below R²", f"({min_r2})")
    _log_shelf_examples(summary_df, low_corr, cols, "Below correlation", f"({min_corr})")
    _log_shelf_examples(summary_df, borderline, cols + ['is_meaningful'], "Borderline")

def suggest_parameter_modifications(summary_df, target_pct=None):
    """Suggest parameter modifications to achieve desired classification results."""
    logger.info("\nPARAMETER MODIFICATION SUGGESTIONS")
    logger.info("=" * 50)
    
    current_pct = (summary_df['paramType'] == 0).sum() / len(summary_df) * 100
    logger.info(f"Current linear: {current_pct:.1f}%" + (f" | Target: {target_pct:.1f}%" if target_pct else ""))
    
    if target_pct and target_pct > current_pct:
        candidates = summary_df[summary_df['paramType'] == 1].sort_values(
            ['r2', 'correlation'], ascending=False, na_position='last')
        n_convert = int((target_pct - current_pct) / 100 * len(summary_df))
        
        logger.info(f"\nTo increase: convert ~{n_convert} shelves from constant → linear")
        if len(candidates) > 0:
            logger.info("  Top candidates:")
            for i, (_, row) in enumerate(candidates.head(min(n_convert + 5, len(candidates))).iterrows()):
                marker = "***" if i < n_convert else "   "
                logger.info(f"  {marker} {row['shelf_name']}: R²={row['r2']:.3f}, corr={row['correlation']:.3f}")
            
            if n_convert > 0:
                top = candidates.head(n_convert)
                min_r2 = top['r2'].min() if not top['r2'].isna().all() else 0
                min_corr = top['correlation'].abs().min() if not top['correlation'].isna().all() else 0
                logger.info(f"\n  Suggested: min_r2={min_r2:.3f}, min_corr={min_corr:.3f}")
    
    # Very low metrics
    for metric, thresh, col in [('R²', 0.01, 'r2'), ('correlation', 0.05, 'correlation')]:
        low = summary_df[summary_df[col].abs() < thresh if col == 'correlation' 
                        else summary_df[col] < thresh]['shelf_name'].tolist()
        if low:
            logger.info(f"\nVery low {metric} (<{thresh}): {len(low)} - {', '.join(low[:5])}")

def inspect_specific_shelf(summary_df, shelf_name):
    """Inspect results for a specific ice shelf (case-insensitive partial match)."""
    matches = summary_df[summary_df['shelf_name'].str.contains(shelf_name, case=False, na=False)]
    
    if len(matches) == 0:
        logger.warning(f"No match for '{shelf_name}' | Available: {', '.join(summary_df['shelf_name'].tolist()[:10])}...")
        return
    if len(matches) > 1:
        logger.warning(f"Multiple matches: {', '.join(matches['shelf_name'].tolist())}")
        return
    
    r = matches.iloc[0]
    logger.info(f"\nDETAILED ANALYSIS: {r['shelf_name']}")
    logger.info("=" * 50)
    logger.info(f"{'Meaningful' if r['is_meaningful'] else 'Not meaningful'} | "
               f"{'Linear' if r['paramType']==0 else 'Constant'} parameterization")
    logger.info(f"R²={r['r2']:.4f}, corr={r['correlation']:.4f}, n={r['n_points']}")
    logger.info(f"Threshold draft={r['threshold_draft']:.2f}m, slope={r['slope']:.6f}, shallow_mean={r['shallow_mean']:.4f}m/yr")
    logger.info(f"Parameters: minDraft={r['minDraft']:.2f}m, constantValue={r['constantValue']:.4f}m/yr, "
               f"alpha0={r['alpha0']:.6f}, alpha1={r['alpha1']:.6f}")

def create_threshold_sensitivity_plot(summary_df, output_dir=None):
    """Create contour plot showing how classification changes with different thresholds."""
    r2_vals = np.arange(0.01, 0.3, 0.01)
    corr_vals = np.arange(0.05, 0.5, 0.02)
    X, Y = np.meshgrid(corr_vals, r2_vals)
    
    # Calculate linear % for each threshold combination
    linear_pct = np.array([[
        (((summary_df['r2'] >= r2) | summary_df['r2'].isna()) & 
         ((np.abs(summary_df['correlation']) >= corr) | summary_df['correlation'].isna())).sum() / len(summary_df) * 100
        for corr in corr_vals] for r2 in r2_vals])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, linear_pct, levels=20, cmap='RdYlBu_r')
    ax.clabel(ax.contour(X, Y, linear_pct, levels=10, colors='black', alpha=0.4, linewidths=0.5),
             inline=True, fontsize=8, fmt='%.0f%%')
    ax.plot(0.2, 0.1, 'ro', markersize=10, label='Current')
    ax.set_xlabel('Correlation Threshold')
    ax.set_ylabel('R² Threshold')
    ax.set_title('% Ice Shelves with Linear Parameterization vs Quality Thresholds')
    plt.colorbar(contour, ax=ax, label='% Linear')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_dir:
        output_file = Path(output_dir) / "threshold_sensitivity.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved → {output_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect draft dependence analysis results')
    parser.add_argument('--parameter_set', required=True,
                       help='Parameter set name to inspect')
    parser.add_argument('--parameter_test_dir', type=Path,
                       help='Directory with parameter test results')
    parser.add_argument('--shelf_name',
                       help='Specific ice shelf to inspect')
    parser.add_argument('--target_linear_pct', type=float,
                       help='Target percentage of linear parameterizations')
    parser.add_argument('--create_sensitivity_plot', action='store_true',
                       help='Create threshold sensitivity plot')
    parser.add_argument('--output_dir', type=Path,
                       help='Directory to save plots')
    args = parser.parse_args()
    
    output_dir = args.output_dir or Path(config.DIR_ICESHELF_DEDRAFT_SATOBS) / "parameter_tests" / args.parameter_set
    setup_logging(output_dir, "inspect_draft_dependence")
    
    logger.info("=" * 60)
    logger.info("DRAFT DEPENDENCE ANALYSIS INSPECTOR")
    logger.info("=" * 60)
    
    summary_df = load_analysis_results(args.parameter_set, args.parameter_test_dir)
    if summary_df is None:
        logger.error("Could not load analysis results")
        exit(1)
    
    if args.shelf_name:
        inspect_specific_shelf(summary_df, args.shelf_name)
    else:
        analyze_classification_criteria(summary_df)
        suggest_parameter_modifications(summary_df, args.target_linear_pct)
    
    if args.create_sensitivity_plot:
        create_threshold_sensitivity_plot(summary_df, args.output_dir)
    
    logger.info("=" * 60)
    logger.info("INSPECTION COMPLETE")
    logger.info("=" * 60)
