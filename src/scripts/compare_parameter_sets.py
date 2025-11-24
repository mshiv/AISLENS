#!/usr/bin/env python3

#!/usr/bin/env python3
"""
Compare draft dependence results across multiple parameter sets.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import geopandas as gpd
from typing import List, Dict
import warnings

from aislens.config import config
from aislens.utils import write_crs, setup_logging

logger = logging.getLogger(__name__)

# --- Function definitions ---

def load_summary_data(parameter_sets: List[str], base_dir: Path) -> Dict[str, pd.DataFrame]:
    summaries = {}
    for param_set in parameter_sets:
        summary_file = base_dir / param_set / "comprehensive" / "comprehensive_summary.csv"
        if not summary_file.exists():
            logger.warning(f"Summary file not found for {param_set}: {summary_file}")
            continue
        df = pd.read_csv(summary_file)
        summaries[param_set] = df
        logger.info(f"Loaded {len(df)} shelves for parameter set '{param_set}'")
    return summaries

def load_parameter_grids(parameter_sets: List[str], base_dir: Path) -> Dict[str, xr.Dataset]:
    param_grids = {}
    for param_set in parameter_sets:
        grid_file = base_dir / param_set / "ruptures_draftDepenBasalMelt_parameters_filled.nc"
        if not grid_file.exists():
            logger.warning(f"Parameter grid not found for {param_set}: {grid_file}")
            continue
        ds = xr.open_dataset(grid_file)
        param_grids[param_set] = ds
        logger.info(f"Loaded parameter grid for '{param_set}'")
    return param_grids

def create_comparison_table(summaries: Dict[str, pd.DataFrame], output_dir: Path):
    comparison_data = []
    for param_set, df in summaries.items():
        total = len(df)
        meaningful = df['is_meaningful'].sum() if 'is_meaningful' in df.columns else 0
        linear = df['is_linear'].sum() if 'is_linear' in df.columns else 0
        constant = df['is_constant'].sum() if 'is_constant' in df.columns else 0
        avg_r2 = df[df['is_meaningful']]['r2'].mean() if meaningful > 0 else np.nan
        avg_corr = df[df['is_meaningful']]['correlation'].mean() if meaningful > 0 else np.nan
        comparison_data.append({
            'Parameter Set': param_set,
            'Total Shelves': total,
            'Meaningful': meaningful,
            'Linear': linear,
            'Constant': constant,
            'Avg R²': f"{avg_r2:.4f}" if not np.isnan(avg_r2) else "N/A",
            'Avg Correlation': f"{avg_corr:.4f}" if not np.isnan(avg_corr) else "N/A"
        })
    comp_df = pd.DataFrame(comparison_data)
    output_file = output_dir / "parameter_set_comparison.csv"
    comp_df.to_csv(output_file, index=False)
    logger.info(f"Saved comparison table to {output_file}")
    return comp_df

def create_shelf_comparison_plot(shelf_name: str, shelf_idx: int, 
                                 parameter_sets: List[str],
                                 summaries: Dict[str, pd.DataFrame],
                                 param_grids: Dict[str, xr.Dataset],
                                 icems: gpd.GeoDataFrame,
                                 satobs: xr.Dataset,
                                 output_dir: Path,
                                 base_dir: Path):
    ice_shelf_geom = icems.iloc[shelf_idx].geometry
    if ice_shelf_geom is None or ice_shelf_geom.is_empty:
        return None
    satobs_clipped = satobs.rio.clip([ice_shelf_geom], crs=config.CRS_TARGET, drop=False)
    draft_data = satobs_clipped[config.SATOBS_DRAFT_VAR]
    melt_data = satobs_clipped[config.SATOBS_FLUX_VAR]
    # Use time mean if available
    if 'Time' in draft_data.dims:
        draft_data = draft_data.mean(dim='Time')
    if 'Time' in melt_data.dims:
        melt_data = melt_data.mean(dim='Time')
    draft_flat = draft_data.values.flatten()
    melt_flat = melt_data.values.flatten()
    valid_mask = ~np.isnan(draft_flat) & ~np.isnan(melt_flat) & (draft_flat > 0)
    if valid_mask.sum() == 0:
        return None
    obs_draft = draft_flat[valid_mask]
    obs_melt = melt_flat[valid_mask]
    n_sets = len(parameter_sets)
    fig = plt.figure(figsize=(6*n_sets, 5))
    gs = gridspec.GridSpec(1, n_sets, figure=fig, wspace=0.3)
    for idx, param_set in enumerate(parameter_sets):
        ax = fig.add_subplot(gs[0, idx])
        # Plot observed data: melt vs draft, draft on y-axis
        n_plot = min(2000, len(obs_draft))
        if len(obs_draft) > n_plot:
            plot_idx = np.random.choice(len(obs_draft), n_plot, replace=False)
            plot_draft = obs_draft[plot_idx]
            plot_melt = obs_melt[plot_idx]
        else:
            plot_draft = obs_draft
            plot_melt = obs_melt
        ax.scatter(plot_melt, plot_draft, c='black', s=2, alpha=0.6, label='Observed')
        # Plot predictions if available
        if param_set in summaries:
            shelf_data = summaries[param_set][summaries[param_set]['shelf_name'] == shelf_name]
            # Load per-shelf parameter file
            param_file = (param_grids[param_set].encoding.get('source') if param_set in param_grids else None)
            shelf_param_file = None
            # Try to find the per-shelf file in processed_dir
            processed_dir = param_grids[param_set].encoding.get('source') if param_set in param_grids else None
            debug_candidates = []
            if processed_dir:
                processed_dir = Path(processed_dir).parent
                candidate = processed_dir / f'draftDepenBasalMelt_comprehensive_{shelf_name}.nc'
                debug_candidates.append(str(candidate))
                if candidate.exists():
                    shelf_param_file = candidate
            if not shelf_param_file:
                # Fallback: always use the provided base_dir argument
                candidate = Path(base_dir) / param_set / 'comprehensive' / f'draftDepenBasalMelt_comprehensive_{shelf_name}.nc'
                debug_candidates.append(str(candidate))
                if candidate.exists():
                    shelf_param_file = candidate
            if not shelf_param_file:
                logger.warning(f"Parameter file not found for shelf '{shelf_name}' in parameter set '{param_set}'. Candidates checked: {debug_candidates}")
                ax.set_title(f"{param_set}\nNo parameter file", fontsize=9, pad=10)
            elif not shelf_data.empty:
                row = shelf_data.iloc[0]
                is_meaningful = row.get('is_meaningful', False)
                is_linear = row.get('is_linear', False)
                ds = xr.open_dataset(shelf_param_file)
                min_draft = float(ds['draftDepenBasalMelt_minDraft'].values)
                constant_val = float(ds['draftDepenBasalMelt_constantMeltValue'].values)
                alpha0 = float(ds['draftDepenBasalMeltAlpha0'].values)
                alpha1 = float(ds['draftDepenBasalMeltAlpha1'].values)
                # Predicted melt: match the optimized script logic
                pred_melt = np.full_like(plot_draft, constant_val)
                if is_linear:
                    deep_mask = plot_draft >= min_draft
                    pred_melt[deep_mask] = alpha0 + alpha1 * plot_draft[deep_mask]
                # Convert predicted melt from kg/m²/s to m/yr for visualization
                pred_melt_vis = pred_melt * 31536000 / 917
                # Plot predicted melt vs draft
                ax.scatter(pred_melt_vis, plot_draft, c='orange', s=2, alpha=0.8, label='Predicted')
                # Add threshold line for linear parameterization
                if is_linear and min_draft > 0:
                    ax.axhline(min_draft, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                # Metrics
                valid_mask_pred = ~np.isnan(plot_melt) & ~np.isnan(pred_melt_vis)
                if np.sum(valid_mask_pred) > 0:
                    melt_valid = plot_melt[valid_mask_pred]
                    pred_valid = pred_melt_vis[valid_mask_pred]
                    mse = np.mean((melt_valid - pred_valid)**2)
                    var_obs = np.var(melt_valid)
                    r2 = 1 - np.sum((melt_valid - pred_valid)**2) / np.sum((melt_valid - np.mean(melt_valid))**2) if var_obs > 0 else 0.0
                    if is_linear:
                        param_info = f"MSE: {mse:.2e}, R²: {r2:.3f}\nThreshold: {min_draft:.1f}m"
                    else:
                        param_info = f"MSE: {mse:.2e}, R²: {r2:.3f}\nConstant: {constant_val:.3f}"
                else:
                    param_info = "No valid predictions"
                ax.set_title(f"{param_set}\n{param_info}", fontsize=9, pad=10)
            else:
                ax.set_title(f"{param_set}\nNo predictions", fontsize=9, pad=10)
        else:
            ax.set_title(f"{param_set}", fontsize=9, pad=10)
        ax.set_xlabel('Melt Rate (kg m⁻² s⁻¹)', fontsize=8)
        ax.set_ylabel('Draft (m)', fontsize=8)
        # Axis limits and formatting
        if len(plot_melt) > 0 and len(plot_draft) > 0:
            melt_range = np.ptp(plot_melt)
            draft_range_val = np.ptp(plot_draft)
            if melt_range > 0:
                melt_margin = 0.1 * melt_range
                ax.set_xlim(plot_melt.min() - melt_margin, plot_melt.max() + melt_margin)
            else:
                margin = 0.1 * abs(plot_melt.mean()) + 0.01
                ax.set_xlim(plot_melt.mean() - margin, plot_melt.mean() + margin)
            if draft_range_val > 0:
                draft_margin = 0.1 * draft_range_val
                ax.set_ylim(plot_draft.min() - draft_margin, plot_draft.max() + draft_margin)
            else:
                margin = 0.1 * abs(plot_draft.mean()) + 10
                ax.set_ylim(plot_draft.mean() - margin, plot_draft.mean() + margin)
            ax.invert_yaxis()
        else:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-100, 100)
            ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if param_set in summaries and param_set in param_grids:
            ax.legend(fontsize=7, loc='upper right')
    fig.suptitle(f'{shelf_name} - Parameter Set Comparison', fontsize=14, fontweight='bold')
    output_file = output_dir / f"{shelf_name}_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    return output_file

def create_metric_comparison_plots(summaries: Dict[str, pd.DataFrame], output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    param_sets = list(summaries.keys())
    meaningful_counts = [summaries[ps]['is_meaningful'].sum() for ps in param_sets]
    linear_counts = [summaries[ps]['is_linear'].sum() if 'is_linear' in summaries[ps].columns else 0 for ps in param_sets]
    constant_counts = [summaries[ps]['is_constant'].sum() if 'is_constant' in summaries[ps].columns else 0 for ps in param_sets]
    ax = axes[0, 0]
    x = np.arange(len(param_sets))
    width = 0.25
    ax.bar(x - width, meaningful_counts, width, label='Meaningful', alpha=0.8)
    ax.bar(x, linear_counts, width, label='Linear', alpha=0.8)
    ax.bar(x + width, constant_counts, width, label='Constant', alpha=0.8)
    ax.set_xlabel('Parameter Set')
    ax.set_ylabel('Count')
    ax.set_title('Parameterization Types by Parameter Set')
    ax.set_xticks(x)
    ax.set_xticklabels(param_sets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    r2_data = [summaries[ps][summaries[ps]['is_meaningful']]['r2'].values for ps in param_sets]
    bp = ax.boxplot(r2_data, labels=param_sets, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel('Parameter Set')
    ax.set_ylabel('R²')
    ax.set_title('R² Distribution (Meaningful Shelves)')
    ax.set_xticklabels(param_sets, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    corr_data = [summaries[ps][summaries[ps]['is_meaningful']]['correlation'].values for ps in param_sets]
    bp = ax.boxplot(corr_data, labels=param_sets, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    ax.set_xlabel('Parameter Set')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation Distribution (Meaningful Shelves)')
    ax.set_xticklabels(param_sets, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    for ps in param_sets:
        df = summaries[ps]
        meaningful_df = df[df['is_meaningful']]
        if len(meaningful_df) > 0:
            ax.scatter(meaningful_df['r2'], meaningful_df['correlation'], 
                      label=ps, alpha=0.6, s=50)
    ax.set_xlabel('R²')
    ax.set_ylabel('Correlation')
    ax.set_title('R² vs Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = output_dir / "metric_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metric comparison plot to {output_file}")
    return output_file

def find_best_parameter_set(summaries: Dict[str, pd.DataFrame], 
                            criterion: str = 'r2') -> pd.DataFrame:
    all_shelves = set()
    for df in summaries.values():
        all_shelves.update(df['shelf_name'].values)
    results = []
    for shelf_name in sorted(all_shelves):
        best_set = None
        best_value = -np.inf if criterion in ['r2', 'correlation'] else np.inf
        best_is_meaningful = False
        for param_set, df in summaries.items():
            shelf_data = df[df['shelf_name'] == shelf_name]
            if not shelf_data.empty:
                row = shelf_data.iloc[0]
                is_meaningful = row.get('is_meaningful', False)
                if is_meaningful:
                    value = row.get(criterion, np.nan)
                    if not np.isnan(value):
                        if (criterion in ['r2', 'correlation'] and value > best_value) or \
                           (criterion not in ['r2', 'correlation'] and value < best_value):
                            best_set = param_set
                            best_value = value
                            best_is_meaningful = True
        results.append({
            'shelf_name': shelf_name,
            'best_parameter_set': best_set if best_is_meaningful else 'None',
            f'best_{criterion}': best_value if best_is_meaningful else np.nan
        })
    return pd.DataFrame(results)

# --- Main block ---

def main():
    parser = argparse.ArgumentParser(
        description='Compare draft dependence results across parameter sets'
    )
    parser.add_argument('--parameter-sets', nargs='+', required=True,
                        help='Parameter set names to compare (e.g., standard permissive strict)')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory containing parameter set subdirectories')
    parser.add_argument('--processed-dir', type=str, default=None,
                        help='Processed directory containing merged parameter grids')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save comparison outputs')
    parser.add_argument('--plot-shelves', nargs='+', default=None,
                        help='Specific shelf names to plot (default: top 10 by data availability)')
    parser.add_argument('--plot-all-shelves', action='store_true',
                        help='Plot all shelves (overrides --plot-shelves and default)')
    parser.add_argument('--start-index', type=int, default=33,
                        help='Starting ice shelf index')
    args = parser.parse_args()
    base_dir = Path(args.base_dir) if args.base_dir else config.DIR_ICESHELF_DEDRAFT_SATOBS
    processed_dir = Path(args.processed_dir) if args.processed_dir else config.DIR_PROCESSED / "draft_dependence_changepoint"
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "comparison_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "compare_parameter_sets")
    logger.info("="*80)
    logger.info("PARAMETER SET COMPARISON")
    logger.info(f"Parameter sets: {', '.join(args.parameter_sets)}")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)
    logger.info("Loading summary data...")
    summaries = load_summary_data(args.parameter_sets, base_dir)
    if not summaries:
        logger.error("No summary data found for any parameter set")
        return
    logger.info("Creating comparison table...")
    comp_table = create_comparison_table(summaries, output_dir)
    print("\nParameter Set Comparison:")
    print(comp_table.to_string(index=False))
    logger.info("Creating metric comparison plots...")
    create_metric_comparison_plots(summaries, output_dir)
    logger.info("Finding best parameter set for each shelf...")
    best_params = find_best_parameter_set(summaries, criterion='r2')
    best_params_file = output_dir / "best_parameter_sets_by_shelf.csv"
    best_params.to_csv(best_params_file, index=False)
    logger.info(f"Saved best parameter sets to {best_params_file}")
    logger.info("Loading parameter grids...")
    param_grids = load_parameter_grids(args.parameter_sets, processed_dir)
    if param_grids:
        logger.info("Loading observational data...")
        satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
        satobs = write_crs(satobs, config.CRS_TARGET)
        icems = gpd.read_file(config.FILE_ICESHELFMASKS).to_crs(config.CRS_TARGET)
        if args.plot_shelves:
            shelves_to_plot = args.plot_shelves
        elif args.plot_all_shelves:
            first_summary = list(summaries.values())[0]
            shelves_to_plot = first_summary['shelf_name'].tolist()
        else:
            first_summary = list(summaries.values())[0]
            meaningful_shelves = first_summary[first_summary['is_meaningful']].sort_values('r2', ascending=False)
            shelves_to_plot = meaningful_shelves['shelf_name'].head(10).tolist()
        logger.info(f"Creating comparison plots for {len(shelves_to_plot)} shelves...")
        shelf_plots_dir = output_dir / "shelf_comparisons"
        shelf_plots_dir.mkdir(exist_ok=True)
        for shelf_name in shelves_to_plot:
            shelf_idx = icems[icems['name'] == shelf_name].index[0]
            try:
                output_file = create_shelf_comparison_plot(
                    shelf_name, shelf_idx, args.parameter_sets,
                    summaries, param_grids, icems, satobs, shelf_plots_dir, base_dir
                )
                if output_file:
                    logger.info(f"  Created: {output_file.name}")
            except Exception as e:
                logger.warning(f"  Failed to plot {shelf_name}: {e}")
    logger.info("="*80)
    logger.info("Comparison complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    main()