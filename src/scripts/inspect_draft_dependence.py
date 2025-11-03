#!/usr/bin/env python3
"""
Draft Dependence Analysis Inspector

This script helps you inspect the results of draft dependence analysis to understand
which ice shelves are classified as linear vs constant/noisy, and why.

Usage:
    python inspect_draft_dependence.py --parameter_set original
    python inspect_draft_dependence.py --parameter_set original --shelf_name "Pine Island Glacier"

Author: Generated for AISLENS project
Date: August 2025
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_analysis_results(parameter_set_name, parameter_test_dir=None):
    """
    Load analysis results for a specific parameter set.
    
    Args:
        parameter_set_name: Name of parameter set
        parameter_test_dir: Directory containing parameter test results
        
    Returns:
        Dictionary with results data
    """
    
    if parameter_test_dir is None:
        from aislens.config import config
        parameter_test_dir = config.DIR_ICESHELF_DEDRAFT_SATOBS / "parameter_tests"
    
    parameter_test_dir = Path(parameter_test_dir)
    param_set_dir = parameter_test_dir / parameter_set_name
    
    # Load results summary
    summary_file = param_set_dir / "comprehensive" / "comprehensive_summary.csv"
    
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        print(f"Loaded results for {len(summary_df)} ice shelves")
        return summary_df
    else:
        print(f"Warning: Summary file not found: {summary_file}")
        return None

def analyze_classification_criteria(summary_df, min_r2_threshold=0.1, min_correlation=0.2):
    """
    Analyze why ice shelves were classified as linear vs constant.
    
    Args:
        summary_df: DataFrame with analysis results
        min_r2_threshold: R² threshold used in analysis
        min_correlation: Correlation threshold used in analysis
    """
    
    print(f"\nCLASSIFICATION ANALYSIS")
    print("=" * 50)
    
    # Count classifications
    total_shelves = len(summary_df)
    meaningful_shelves = summary_df['is_meaningful'].sum()
    linear_shelves = (summary_df['paramType'] == 0).sum()
    constant_shelves = (summary_df['paramType'] == 1).sum()
    
    print(f"Total ice shelves: {total_shelves}")
    print(f"Meaningful relationships: {meaningful_shelves} ({meaningful_shelves/total_shelves*100:.1f}%)")
    print(f"Linear parameterization (paramType=0): {linear_shelves} ({linear_shelves/total_shelves*100:.1f}%)")
    print(f"Constant parameterization (paramType=1): {constant_shelves} ({constant_shelves/total_shelves*100:.1f}%)")
    
    # Analyze thresholds
    print(f"\nTHRESHOLD ANALYSIS")
    print(f"R² threshold: {min_r2_threshold}")
    print(f"Correlation threshold: {min_correlation}")
    
    # Ice shelves that failed R² threshold
    low_r2_mask = (summary_df['r2'] < min_r2_threshold) & (~summary_df['r2'].isna())
    print(f"\nIce shelves below R² threshold ({min_r2_threshold}):")
    print(f"  Count: {low_r2_mask.sum()}")
    if low_r2_mask.sum() > 0:
        low_r2_shelves = summary_df[low_r2_mask][['shelf_name', 'r2', 'correlation', 'paramType']]
        print(f"  Examples:")
        for _, row in low_r2_shelves.head(10).iterrows():
            print(f"    {row['shelf_name']}: R²={row['r2']:.3f}, corr={row['correlation']:.3f}, type={row['paramType']}")
    
    # Ice shelves that failed correlation threshold
    low_corr_mask = (np.abs(summary_df['correlation']) < min_correlation) & (~summary_df['correlation'].isna())
    print(f"\nIce shelves below correlation threshold ({min_correlation}):")
    print(f"  Count: {low_corr_mask.sum()}")
    if low_corr_mask.sum() > 0:
        low_corr_shelves = summary_df[low_corr_mask][['shelf_name', 'r2', 'correlation', 'paramType']]
        print(f"  Examples:")
        for _, row in low_corr_shelves.head(10).iterrows():
            print(f"    {row['shelf_name']}: R²={row['r2']:.3f}, corr={row['correlation']:.3f}, type={row['paramType']}")
    
    # Ice shelves that are borderline (close to thresholds)
    borderline_r2_mask = (summary_df['r2'] >= min_r2_threshold) & (summary_df['r2'] < min_r2_threshold * 2) & (~summary_df['r2'].isna())
    borderline_corr_mask = (np.abs(summary_df['correlation']) >= min_correlation) & (np.abs(summary_df['correlation']) < min_correlation * 2) & (~summary_df['correlation'].isna())
    borderline_mask = borderline_r2_mask | borderline_corr_mask
    
    print(f"\nBorderline ice shelves (might benefit from lower thresholds):")
    print(f"  Count: {borderline_mask.sum()}")
    if borderline_mask.sum() > 0:
        borderline_shelves = summary_df[borderline_mask][['shelf_name', 'r2', 'correlation', 'paramType', 'is_meaningful']]
        print(f"  Examples:")
        for _, row in borderline_shelves.head(10).iterrows():
            print(f"    {row['shelf_name']}: R²={row['r2']:.3f}, corr={row['correlation']:.3f}, meaningful={row['is_meaningful']}, type={row['paramType']}")

def suggest_parameter_modifications(summary_df, target_linear_percentage=None):
    """
    Suggest parameter modifications to achieve desired classification results.
    
    Args:
        summary_df: DataFrame with analysis results
        target_linear_percentage: Desired percentage of linear parameterizations
    """
    
    print(f"\nPARAMETER MODIFICATION SUGGESTIONS")
    print("=" * 50)
    
    current_linear_pct = (summary_df['paramType'] == 0).sum() / len(summary_df) * 100
    print(f"Current linear percentage: {current_linear_pct:.1f}%")
    
    if target_linear_percentage is not None:
        print(f"Target linear percentage: {target_linear_percentage:.1f}%")
        
        if target_linear_percentage > current_linear_pct:
            print("\nTo increase linear classifications:")
            
            # Find ice shelves with constant classification but reasonable R² or correlation
            constant_mask = (summary_df['paramType'] == 1)
            candidates = summary_df[constant_mask].copy()
            
            # Sort by best R² and correlation among constant shelves
            candidates = candidates.sort_values(['r2', 'correlation'], ascending=False, na_position='last')
            
            n_to_convert = int((target_linear_percentage - current_linear_pct) / 100 * len(summary_df))
            print(f"  Need to convert ~{n_to_convert} shelves from constant to linear")
            
            if len(candidates) > 0:
                print(f"  Top candidates (currently constant but with reasonable metrics):")
                for i, (_, row) in enumerate(candidates.head(min(n_to_convert + 5, len(candidates))).iterrows()):
                    marker = "***" if i < n_to_convert else "   "
                    print(f"  {marker} {row['shelf_name']}: R²={row['r2']:.3f}, corr={row['correlation']:.3f}")
                
                # Suggest threshold changes
                if n_to_convert > 0:
                    top_candidates = candidates.head(n_to_convert)
                    min_r2_needed = top_candidates['r2'].min() if not top_candidates['r2'].isna().all() else 0
                    min_corr_needed = top_candidates['correlation'].abs().min() if not top_candidates['correlation'].isna().all() else 0
                    
                    print(f"\n  Suggested threshold changes:")
                    print(f"    Lower min_r2_threshold to: {min_r2_needed:.3f} (currently ~0.1)")
                    print(f"    Lower min_correlation to: {min_corr_needed:.3f} (currently ~0.2)")
    
    # Identify potential problem shelves (very low metrics)
    very_low_r2 = summary_df[summary_df['r2'] < 0.01]['shelf_name'].tolist()
    very_low_corr = summary_df[np.abs(summary_df['correlation']) < 0.05]['shelf_name'].tolist()
    
    if very_low_r2:
        print(f"\nIce shelves with very low R² (<0.01): {len(very_low_r2)}")
        print(f"  Examples: {', '.join(very_low_r2[:5])}")
        
    if very_low_corr:
        print(f"\nIce shelves with very low correlation (<0.05): {len(very_low_corr)}")
        print(f"  Examples: {', '.join(very_low_corr[:5])}")

def inspect_specific_shelf(summary_df, shelf_name):
    """
    Inspect results for a specific ice shelf.
    
    Args:
        summary_df: DataFrame with analysis results
        shelf_name: Name of ice shelf to inspect
    """
    
    # Find the ice shelf (case-insensitive partial match)
    matches = summary_df[summary_df['shelf_name'].str.contains(shelf_name, case=False, na=False)]
    
    if len(matches) == 0:
        print(f"No ice shelf found matching '{shelf_name}'")
        print(f"Available shelves: {summary_df['shelf_name'].tolist()[:10]}...")
        return
    
    if len(matches) > 1:
        print(f"Multiple matches found for '{shelf_name}':")
        for _, row in matches.iterrows():
            print(f"  {row['shelf_name']}")
        print("Please be more specific.")
        return
    
    row = matches.iloc[0]
    
    print(f"\nDETAILED ANALYSIS: {row['shelf_name']}")
    print("=" * 50)
    print(f"Classification: {'Meaningful' if row['is_meaningful'] else 'Not meaningful'}")
    print(f"Parameterization: {'Linear' if row['paramType'] == 0 else 'Constant'}")
    print(f"R² value: {row['r2']:.4f}")
    print(f"Correlation: {row['correlation']:.4f}")
    print(f"Number of data points: {row['n_points']}")
    print(f"Threshold draft: {row['threshold_draft']:.2f} m")
    print(f"Slope (alpha1): {row['slope']:.6f}")
    print(f"Shallow mean: {row['shallow_mean']:.4f} m/yr")
    
    print(f"\nDraft dependence parameters:")
    print(f"  minDraft: {row['minDraft']:.2f} m")
    print(f"  constantValue: {row['constantValue']:.4f} m/yr") 
    print(f"  paramType: {row['paramType']} ({'Linear' if row['paramType'] == 0 else 'Constant'})")
    print(f"  alpha0 (intercept): {row['alpha0']:.6f}")
    print(f"  alpha1 (slope): {row['alpha1']:.6f}")

def create_threshold_sensitivity_plot(summary_df, output_dir=None):
    """
    Create a plot showing how classification changes with different thresholds.
    
    Args:
        summary_df: DataFrame with analysis results
        output_dir: Directory to save plot
    """
    
    # Test different threshold combinations
    r2_thresholds = np.arange(0.01, 0.3, 0.01)
    corr_thresholds = np.arange(0.05, 0.5, 0.02)
    
    linear_percentages = np.zeros((len(r2_thresholds), len(corr_thresholds)))
    
    for i, r2_thresh in enumerate(r2_thresholds):
        for j, corr_thresh in enumerate(corr_thresholds):
            # Simulate classification with these thresholds
            meets_r2 = (summary_df['r2'] >= r2_thresh) | summary_df['r2'].isna()
            meets_corr = (np.abs(summary_df['correlation']) >= corr_thresh) | summary_df['correlation'].isna()
            would_be_meaningful = meets_r2 & meets_corr
            
            # Count how many would be linear (meaningful shelves get paramType=0)
            linear_count = would_be_meaningful.sum()
            linear_percentages[i, j] = linear_count / len(summary_df) * 100
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    X, Y = np.meshgrid(corr_thresholds, r2_thresholds)
    contour = ax.contourf(X, Y, linear_percentages, levels=20, cmap='RdYlBu_r')
    
    # Add contour lines
    contour_lines = ax.contour(X, Y, linear_percentages, levels=10, colors='black', alpha=0.4, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f%%')
    
    # Mark current thresholds
    ax.plot(0.2, 0.1, 'ro', markersize=10, label='Current thresholds')
    
    ax.set_xlabel('Correlation Threshold')
    ax.set_ylabel('R² Threshold')
    ax.set_title('Percentage of Ice Shelves with Linear Parameterization\nvs Quality Thresholds')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Percentage Linear (%)')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "threshold_sensitivity.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Threshold sensitivity plot saved to: {output_file}")
    
    plt.show()

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Inspect draft dependence analysis results')
    parser.add_argument('--parameter_set', type=str, required=True,
                        help='Name of parameter set to inspect')
    parser.add_argument('--parameter_test_dir', type=str, default=None,
                        help='Directory containing parameter test results')
    parser.add_argument('--shelf_name', type=str, default=None,
                        help='Specific ice shelf to inspect')
    parser.add_argument('--target_linear_pct', type=float, default=None,
                        help='Target percentage of linear parameterizations')
    parser.add_argument('--create_sensitivity_plot', action='store_true',
                        help='Create threshold sensitivity plot')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    print("DRAFT DEPENDENCE ANALYSIS INSPECTOR")
    print("=" * 50)
    
    # Load results
    summary_df = load_analysis_results(args.parameter_set, args.parameter_test_dir)
    
    if summary_df is None:
        print("Could not load analysis results.")
        return
    
    if args.shelf_name:
        # Inspect specific shelf
        inspect_specific_shelf(summary_df, args.shelf_name)
    else:
        # General analysis
        analyze_classification_criteria(summary_df)
        suggest_parameter_modifications(summary_df, args.target_linear_pct)
    
    if args.create_sensitivity_plot:
        create_threshold_sensitivity_plot(summary_df, args.output_dir)
    
    print("\nInspection complete!")

if __name__ == "__main__":
    main()
