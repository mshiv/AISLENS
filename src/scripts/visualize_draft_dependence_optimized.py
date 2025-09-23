#!/usr/bin/env python3
"""
Draft Dependence Visualization Script with Optimizations

This version includes parallel processing, optimized I/O operations, and vectorized computations
for faster execution.

Usage:
    python visualize_draft_dependence_optimized.py --parameter_set original
    python visualize_draft_dependence_optimized.py --parameter_set permissive --output_dir /path/to/output

Author: Generated for AISLENS project
Date: August 2025
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Import your AISLENS modules
from aislens.config import config
from aislens.utils import write_crs

# Global variables for caching loaded datasets
_satobs_cache = None
_icems_cache = None
_param_ds_cache = None

@lru_cache(maxsize=1)
def load_datasets():
    """Load and cache datasets for reuse across processes."""
    global _satobs_cache, _icems_cache
    
    if _satobs_cache is None:
        print("Loading satellite observation data...")
        _satobs_cache = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
        _satobs_cache = write_crs(_satobs_cache, config.CRS_TARGET)
    
    if _icems_cache is None:
        print("Loading ice shelf masks...")
        _icems_cache = gpd.read_file(config.FILE_ICESHELFMASKS).to_crs(config.CRS_TARGET)
    
    return _satobs_cache, _icems_cache

def load_ice_shelf_data(ice_shelf_index, icems, satobs, config):
    """
    Load observational data for a specific ice shelf.
    Optimized with vectorized operations.
    """
    try:
        ice_shelf_geom = icems.iloc[ice_shelf_index].geometry
        shelf_name = icems.iloc[ice_shelf_index].name

        if ice_shelf_geom is None or ice_shelf_geom.is_empty:
            return None

        satobs_clipped = satobs.rio.clip([ice_shelf_geom], crs=config.CRS_TARGET, drop=False)
        draft_var = config.SATOBS_DRAFT_VAR
        flux_var = config.SATOBS_FLUX_VAR

        draft_data = satobs_clipped[draft_var].mean(dim=config.TIME_DIM) if config.TIME_DIM in satobs_clipped[draft_var].dims else satobs_clipped[draft_var]
        melt_data = satobs_clipped[flux_var].mean(dim=config.TIME_DIM) if config.TIME_DIM in satobs_clipped[flux_var].dims else satobs_clipped[flux_var]

        draft_flat = draft_data.values.flatten()
        melt_flat = melt_data.values.flatten()

        valid_mask = ~np.isnan(draft_flat) & ~np.isnan(melt_flat) & (draft_flat > 0)

        if valid_mask.sum() == 0:
            return None

        return {
            'draft': draft_flat[valid_mask],
            'melt': melt_flat[valid_mask],
            'shelf_name': shelf_name
        }

    except Exception as e:
        return None

def load_predicted_data(shelf_name, ice_shelf_geom, parameter_set_dir, config):
    """
    Load predicted melt rates for an ice shelf from merged parameter grid files.
    Optimized for better I/O handling with fallback mechanism.
    """
    try:
        combined_file = parameter_set_dir / "ruptures_draftDepenBasalMelt_parameters_filled.nc"
        if not combined_file.exists():
            return None

        # Cache dataset if not already loaded
        global _param_ds_cache
        if _param_ds_cache is None:
            _param_ds_cache = xr.open_dataset(combined_file)
            if not hasattr(_param_ds_cache, 'rio') or _param_ds_cache.rio.crs is None:
                _param_ds_cache = _param_ds_cache.rio.write_crs(config.CRS_TARGET)
        
        ds = _param_ds_cache

        # Try clipping first
        try:
            clipped = ds.rio.clip([ice_shelf_geom], crs=config.CRS_TARGET, drop=False)
        except Exception:
            # Fallback to spatial bounds
            bounds = ice_shelf_geom.bounds
            x_mask = (ds.x >= bounds[0]) & (ds.x <= bounds[2])
            y_mask = (ds.y >= bounds[1]) & (ds.y <= bounds[3])
            if x_mask.sum() > 0 and y_mask.sum() > 0:
                clipped = ds.isel(x=x_mask, y=y_mask)
            else:
                return None
        
        # Vectorized parameter extraction
        param_vars = {
            'minDraft': 'draftDepenBasalMelt_minDraft',
            'constantValue': 'draftDepenBasalMelt_constantMeltValue',
            'paramType': 'draftDepenBasalMelt_paramType',
            'alpha0': 'draftDepenBasalMeltAlpha0',
            'alpha1': 'draftDepenBasalMeltAlpha1'
        }

        params = {}
        for param_name, var_name in param_vars.items():
            if var_name in clipped.data_vars:
                data = clipped[var_name].values.flatten()
                valid_values = data[~np.isnan(data) & (data != 0)]
                params[param_name] = float(valid_values[0]) if len(valid_values) > 0 else 0.0
            else:
                params[param_name] = 0.0

        return params

    except Exception:
        return None

def create_draft_melt_prediction(draft_range, params):
    """
    Create draft-melt prediction curve from parameters.
    Vectorized for better performance.
    """
    min_draft = params['minDraft'] 
    constant_value = params['constantValue']
    param_type = params['paramType']
    alpha0 = params['alpha0']
    alpha1 = params['alpha1']
    
    predicted_melt = np.full_like(draft_range, constant_value)
    
    if param_type == 0:  # Linear parameterization
        # Vectorized operation: apply linear relationship to deep areas
        deep_mask = draft_range >= min_draft
        predicted_melt[deep_mask] = alpha0 + alpha1 * draft_range[deep_mask]
        
    return predicted_melt

def plot_ice_shelf_comparison(obs_data, pred_params, shelf_name, ax):
    """
    Create scatter plot for a single ice shelf showing observations and predictions.
    Optimized with vectorized operations and reduced plotting overhead.
    """
    if obs_data is None or len(obs_data['draft']) == 0:
        ax.text(0.5, 0.5, f"{shelf_name}\nNo data", 
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return

    # Vectorized subsampling for plotting
    n_plot = min(2000, len(obs_data['draft']))
    if len(obs_data['draft']) > n_plot:
        plot_idx = np.random.choice(len(obs_data['draft']), n_plot, replace=False)
        plot_draft = obs_data['draft'][plot_idx]
        plot_melt = obs_data['melt'][plot_idx]
    else:
        plot_draft = obs_data['draft']
        plot_melt = obs_data['melt']
    
    # Unit conversion for consistent plotting
    # Both observed and predicted data are in kg/m²/s (SI units)
    # Convert both to m/yr for better visualization
    melt_units = 'm/yr'
    plot_melt = plot_melt * 31536000 / 917  # Convert from kg/m²/s to m/yr
    
    # Plot observed data
    ax.scatter(plot_melt, plot_draft, c='black', s=2, alpha=0.6, label='Observed')
    
    # Plot predictions if available
    if pred_params is not None:
        pred_melt = create_draft_melt_prediction(plot_draft, pred_params)
        
        # DEBUG: Print parameter values and prediction ranges
        print(f"    DEBUG {shelf_name}: Parameters - alpha0={pred_params['alpha0']:.6f}, alpha1={pred_params['alpha1']:.6f}, minDraft={pred_params['minDraft']:.2f}")
        print(f"    DEBUG {shelf_name}: Draft range [{plot_draft.min():.1f}, {plot_draft.max():.1f}]")
        print(f"    DEBUG {shelf_name}: Predicted melt range (raw) [{pred_melt.min():.6f}, {pred_melt.max():.6f}]")
        print(f"    DEBUG {shelf_name}: Observed melt range (after conversion) [{plot_melt.min():.6f}, {plot_melt.max():.6f}]")
        
        # Convert predicted melt rates to match observed units (both should be in m/yr)
        print(f"    DEBUG {shelf_name}: Converting predicted melt from kg/m²/s to m/yr")
        pred_melt = pred_melt * 31536000 / 917
        print(f"    DEBUG {shelf_name}: Predicted melt range (after conversion) [{pred_melt.min():.6f}, {pred_melt.max():.6f}]")
        
        # Determine colors and calculate metrics vectorized
        is_meaningful = True  # Could add correlation-based logic here
        pred_color = 'orange' if is_meaningful else 'red'
        title_color = 'black' if is_meaningful else 'red'
        
        if not is_meaningful:
            ax.collections[0].set_color('gray')
        
        ax.scatter(pred_melt, plot_draft, c=pred_color, s=2, alpha=0.8, label='Predicted')
        
        # Add threshold line for linear parameterization
        param_type = pred_params['paramType']
        if param_type == 0 and is_meaningful:
            min_draft = pred_params['minDraft']
            if min_draft > 0:
                ax.axhline(min_draft, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Vectorized metrics calculation
        valid_mask = ~np.isnan(plot_melt) & ~np.isnan(pred_melt)
        if np.sum(valid_mask) > 0:
            melt_valid = plot_melt[valid_mask]
            pred_valid = pred_melt[valid_mask]
            
            mse = np.mean((melt_valid - pred_valid)**2)
            var_obs = np.var(melt_valid)
            r2 = 1 - np.sum((melt_valid - pred_valid)**2) / np.sum((melt_valid - np.mean(melt_valid))**2) if var_obs > 0 else 0.0
            
            if param_type == 0:
                min_draft = pred_params['minDraft']
                param_info = f"MSE: {mse:.2e}, R²: {r2:.3f}\nThreshold: {min_draft:.1f}m"
                if not is_meaningful:
                    param_info = f"{shelf_name} (NOISY)\n" + param_info
            else:
                param_info = f"MSE: {mse:.2e}, R²: {r2:.3f}\nConstant: {pred_params['constantValue']:.3f}"
        else:
            param_info = "No valid predictions"
    else:
        param_info = "No predictions"
        title_color = 'black'
    
    # Set labels and title
    ax.set_xlabel(f'Melt Rate ({melt_units})', fontsize=8)
    ax.set_ylabel('Draft (m)', fontsize=8)
    ax.set_title(f"{shelf_name}\n{param_info}", fontsize=9, pad=10, color=title_color)
    
    # Vectorized axis limit calculation
    if len(plot_melt) > 0 and len(plot_draft) > 0:
        melt_range = np.ptp(plot_melt)  # peak-to-peak (vectorized range)
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
    
    # Add grid and formatting
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    
    # Add legend if predictions exist
    if pred_params is not None:
        ax.legend(fontsize=7, loc='upper right')

def process_and_plot_ice_shelf(args):
    """
    Process a single ice shelf for visualization AND create the plot immediately.
    Used for parallel processing - each worker handles both analysis and plotting.
    """
    ice_shelf_index, shelf_name, icems, satobs, param_set_dir, config, individual_plots_dir, parameter_set_name = args
    try:
        ice_shelf_geom = icems.iloc[ice_shelf_index].geometry
        
        # Analyze the ice shelf data
        obs_data = load_ice_shelf_data(ice_shelf_index, icems, satobs, config)
        pred_params = load_predicted_data(shelf_name, ice_shelf_geom, param_set_dir, config)
        
        # Create plot immediately after analysis
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        if obs_data is not None:
            plot_ice_shelf_comparison(obs_data, pred_params, shelf_name, ax)
            has_data = True
        else:
            ax.text(0.5, 0.5, f"{shelf_name}\nNo data", 
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            has_data = False
        
        # Save plot immediately
        safe_shelf_name = shelf_name.replace(" ", "_").replace("/", "_")
        output_file = individual_plots_dir / f"{safe_shelf_name}_{parameter_set_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        
        # Save high-res version
        #output_file_hires = individual_plots_dir / f"{safe_shelf_name}_{parameter_set_name}_hires.png"
        #plt.savefig(output_file_hires, dpi=300, bbox_inches='tight')
        
        plt.close()  # Important: close the figure to free memory
        
        return (shelf_name, output_file, has_data)
        
    except Exception as e:
        # Return error info if something goes wrong
        return (shelf_name, None, False)

def create_draft_dependence_visualization(parameter_set_name, parameter_test_dir=None, output_dir=None, max_shelves=None, start_index=33):
    """
    Create individual scatter plots for each ice shelf comparing observations and predictions.
    Optimized with parallel processing.
    """
    if parameter_test_dir is None:
        parameter_test_dir = config.DIR_PROCESSED / "draft_dependence_changepoint"
    parameter_test_dir = Path(parameter_test_dir)

    if output_dir is None:
        output_dir = parameter_test_dir.parent / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for individual ice shelf plots
    individual_plots_dir = output_dir / f"individual_plots_{parameter_set_name}"
    individual_plots_dir.mkdir(parents=True, exist_ok=True)

    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
    satobs = write_crs(satobs, config.CRS_TARGET)
    icems = gpd.read_file(config.FILE_ICESHELFMASKS).to_crs(config.CRS_TARGET)

    shelf_names = list(icems.name.values[start_index:])
    if max_shelves is not None:
        shelf_names = shelf_names[:max_shelves]

    n_shelves = len(shelf_names)
    args_list = [(i + start_index, shelf_name, icems, satobs, parameter_test_dir, config, individual_plots_dir, parameter_set_name) for i, shelf_name in enumerate(shelf_names)]

    # Use parallel processing with optimized worker count
    max_workers = min(mp.cpu_count(), 8, n_shelves)  # Don't use more workers than shelves
    print(f"Using {max_workers} parallel workers")
    print(f"Processing and plotting {n_shelves} ice shelves...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_and_plot_ice_shelf, args_list))

    # Count successful processing and collect output files
    processed_count = 0
    output_files = []
    
    for shelf_name, output_file, has_data in results:
        if has_data:
            processed_count += 1
        if output_file is not None:
            output_files.append(output_file)
    
    print(f"Processed {processed_count} ice shelves with data out of {len(shelf_names)} total")
    print(f"Individual plots saved to: {individual_plots_dir}")
    
    return output_files

def create_summary_comparison(parameter_test_dir=None, output_dir=None):
    """
    Create a summary plot comparing different parameter sets.
    Optimized version with vectorized operations.
    """
    if parameter_test_dir is None:
        parameter_test_dir = config.DIR_ICESHELF_DEDRAFT_SATOBS / "parameter_tests"
    parameter_test_dir = Path(parameter_test_dir)
    
    if output_dir is None:
        output_dir = parameter_test_dir / "visualizations"  
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results summary
    summary_file = parameter_test_dir / "results_summary.json"
    if not summary_file.exists():
        print(f"Error: Results summary not found: {summary_file}")
        return
        
    with open(summary_file, 'r') as f:
        results_summary = json.load(f)
    
    # Vectorized data extraction
    completed_summaries = {name: summary for name, summary in results_summary.items() 
                          if summary['status'] == 'completed'}
    
    if not completed_summaries:
        print("No completed parameter sets found")
        return
    
    # Use numpy arrays for vectorized operations
    param_names = list(completed_summaries.keys())
    meaningful_counts = np.array([completed_summaries[name]['meaningful_shelves'] for name in param_names])
    linear_counts = np.array([completed_summaries[name]['linear_param_count'] for name in param_names])
    constant_counts = np.array([completed_summaries[name]['constant_param_count'] for name in param_names])
    
    # Create comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x_pos = np.arange(len(param_names))
    
    # Plot 1: Meaningful relationships
    ax1.bar(x_pos, meaningful_counts, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Parameter Set')
    ax1.set_ylabel('Number of Meaningful Relationships')
    ax1.set_title('Meaningful Draft-Melt Relationships by Parameter Set')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(param_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Vectorized label addition
    for i, count in enumerate(meaningful_counts):
        ax1.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    # Plot 2: Parameterization types
    width = 0.35
    ax2.bar(x_pos - width/2, linear_counts, width, label='Linear (paramType=0)', alpha=0.7, color='lightgreen')
    ax2.bar(x_pos + width/2, constant_counts, width, label='Constant (paramType=1)', alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Parameter Set')
    ax2.set_ylabel('Number of Ice Shelves')
    ax2.set_title('Parameterization Types by Parameter Set')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(param_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    output_file = output_dir / "parameter_set_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Parameter set comparison saved to: {output_file}")
    
    plt.close()
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Visualize draft dependence analysis results (Optimized)')
    parser.add_argument('--parameter_set', type=str, required=True, help='Name of parameter set to visualize')
    parser.add_argument('--parameter_test_dir', type=str, default=None, help='Directory containing parameter test results')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save visualization plots')
    parser.add_argument('--max_shelves', type=int, default=None, help='Maximum number of shelves to plot (for testing)')
    parser.add_argument('--start_index', type=int, default=33, help='Starting index for ice shelves (default: 33)')
    parser.add_argument('--create_summary', action='store_true', help='Also create summary comparison plot')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of parallel workers (default: auto-detect)')

    args = parser.parse_args()
    
    # Set number of workers for parallel processing
    if args.n_workers is None:
        args.n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    print("DRAFT DEPENDENCE VISUALIZATION (OPTIMIZED)")
    print("=" * 50)
    print(f"Using {args.n_workers} parallel workers")
    
    # Create main visualization
    output_file = create_draft_dependence_visualization(
        parameter_set_name=args.parameter_set,
        parameter_test_dir=args.parameter_test_dir,
        output_dir=args.output_dir,
        max_shelves=args.max_shelves,
        start_index=args.start_index
    )
    
    # Create summary comparison if requested
    if args.create_summary:
        summary_file = create_summary_comparison(
            parameter_test_dir=args.parameter_test_dir,
            output_dir=args.output_dir
        )
        if summary_file:
            print(f"Summary comparison saved to: {summary_file}")
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()