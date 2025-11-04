#!/usr/bin/env python3
"""
Draft Dependence Visualization Script

Creates scatter plots showing observed melt rates vs draft depth for each ice shelf,
along with the predicted melt rates from the draft dependence parameterization.
Displays all ice shelves in a grid layout for easy comparison.

Usage:
    python visualize_draft_dependence.py --parameter_set original
    python visualize_draft_dependence.py --parameter_set permissive --output_dir /path/to/output

Author: Generated for AISLENS project  
Date: August 2025
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xarray as xr
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import your AISLENS modules
from aislens.config import config
from aislens.utils import write_crs

# Try to import scipy for mode calculation
try:
    from scipy.stats import mode
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using mean instead of mode for parameter extraction")

def load_ice_shelf_data(ice_shelf_index, icems, satobs, config):
    """
    Load observational data for a specific ice shelf.
    
    Args:
        ice_shelf_index: Index of ice shelf in icems dataframe
        icems: GeoDataFrame with ice shelf masks
        satobs: xarray Dataset with satellite observations
        config: Configuration object
        
    Returns:
        dict with 'draft', 'melt', 'shelf_name' keys, or None if no data
    """
    try:
        # Get ice shelf geometry and name
        ice_shelf_geom = icems.iloc[ice_shelf_index].geometry
        shelf_name = icems.iloc[ice_shelf_index].name
        
        if ice_shelf_geom is None or ice_shelf_geom.is_empty:
            print(f"  {shelf_name}: Empty geometry")
            return None
            
        # Clip satellite data to ice shelf region
        satobs_clipped = satobs.rio.clip([ice_shelf_geom], crs=config.CRS_TARGET, drop=False)
        
        # Extract draft and melt rate data
        draft_var = config.SATOBS_DRAFT_VAR
        flux_var = config.SATOBS_FLUX_VAR
        
        print(f"  {shelf_name}: Looking for variables '{draft_var}' and '{flux_var}'")
        print(f"  Available variables: {list(satobs_clipped.data_vars.keys())}")
        
        # Get draft data (take time mean if time dimension exists)
        if config.TIME_DIM in satobs_clipped[draft_var].dims:
            draft_data = satobs_clipped[draft_var].mean(dim=config.TIME_DIM)
        else:
            draft_data = satobs_clipped[draft_var]
            
        # Get melt rate data (take time mean if time dimension exists) 
        if config.TIME_DIM in satobs_clipped[flux_var].dims:
            melt_data = satobs_clipped[flux_var].mean(dim=config.TIME_DIM)
        else:
            melt_data = satobs_clipped[flux_var]
        
        # Flatten and remove NaN values
        draft_flat = draft_data.values.flatten()
        melt_flat = melt_data.values.flatten()
        
        print(f"  {shelf_name}: Raw data shapes - draft: {draft_flat.shape}, melt: {melt_flat.shape}")
        print(f"  {shelf_name}: Draft range: [{np.nanmin(draft_flat):.2f}, {np.nanmax(draft_flat):.2f}]")
        print(f"  {shelf_name}: Melt range: [{np.nanmin(melt_flat):.2e}, {np.nanmax(melt_flat):.2e}]")
        
        # Create mask for valid data points
        valid_mask = ~np.isnan(draft_flat) & ~np.isnan(melt_flat) & (draft_flat > 0)
        
        print(f"  {shelf_name}: Valid points: {valid_mask.sum()} out of {len(valid_mask)}")
        
        if valid_mask.sum() == 0:
            print(f"  {shelf_name}: No valid data points found")
            return None
            
        return {
            'draft': draft_flat[valid_mask],
            'melt': melt_flat[valid_mask], 
            'shelf_name': shelf_name
        }
        
    except Exception as e:
        print(f"Warning: Could not load data for ice shelf {ice_shelf_index}: {e}")
        return None

def load_predicted_data(shelf_name, ice_shelf_geom, parameter_set_dir, config):
    """
    Load predicted melt rates for an ice shelf from merged parameter grid files.
    Simplified version with better error handling and debugging.
    """
    try:
        # Use the combined parameter file instead of individual files
        combined_file = parameter_set_dir / "ruptures_draftDepenBasalMelt_parameters_filled.nc"
        
        if not combined_file.exists():
            print(f"Warning: Combined parameter file not found: {combined_file}")
            return None
            
        # Load the combined dataset
        ds = xr.open_dataset(combined_file)
        
        # Ensure CRS is set
        if not hasattr(ds, 'rio') or ds.rio.crs is None:
            ds = ds.rio.write_crs(config.CRS_TARGET)
        
        # Try to clip to ice shelf geometry
        try:
            clipped = ds.rio.clip([ice_shelf_geom], crs=config.CRS_TARGET, drop=False)
            
            # Extract parameter values - take first valid value from clipped region
            params = {}
            param_vars = {
                'minDraft': 'draftDepenBasalMelt_minDraft',
                'constantValue': 'draftDepenBasalMelt_constantMeltValue',
                'paramType': 'draftDepenBasalMelt_paramType', 
                'alpha0': 'draftDepenBasalMeltAlpha0',
                'alpha1': 'draftDepenBasalMeltAlpha1'
            }
            
            for param_name, var_name in param_vars.items():
                if var_name in clipped.data_vars:
                    data = clipped[var_name].values.flatten()
                    valid_values = data[~np.isnan(data) & (data != 0)]
                    if len(valid_values) > 0:
                        params[param_name] = float(valid_values[0])
                    else:
                        params[param_name] = 0.0
                else:
                    params[param_name] = 0.0
                    
            print(f"  Extracted parameters for {shelf_name}: {params}")
            return params
            
        except Exception as clip_error:
            print(f"  Clipping failed for {shelf_name}, trying fallback method: {clip_error}")
            
            # Fallback: use spatial bounds to find relevant data
            try:
                bounds = ice_shelf_geom.bounds  # (minx, miny, maxx, maxy)
                
                # Select data within approximate bounds
                x_mask = (ds.x >= bounds[0]) & (ds.x <= bounds[2])
                y_mask = (ds.y >= bounds[1]) & (ds.y <= bounds[3])
                
                if x_mask.sum() > 0 and y_mask.sum() > 0:
                    subset = ds.isel(x=x_mask, y=y_mask)
                    
                    params = {}
                    param_vars = {
                        'minDraft': 'draftDepenBasalMelt_minDraft',
                        'constantValue': 'draftDepenBasalMelt_constantMeltValue',
                        'paramType': 'draftDepenBasalMelt_paramType',
                        'alpha0': 'draftDepenBasalMeltAlpha0', 
                        'alpha1': 'draftDepenBasalMeltAlpha1'
                    }
                    
                    for param_name, var_name in param_vars.items():
                        if var_name in subset.data_vars:
                            data = subset[var_name].values.flatten()
                            valid_values = data[~np.isnan(data) & (data != 0)]
                            if len(valid_values) > 0:
                                params[param_name] = float(valid_values[0])
                            else:
                                params[param_name] = 0.0
                        else:
                            params[param_name] = 0.0
                            
                    print(f"  Fallback extraction for {shelf_name}: {params}")
                    return params
                else:
                    print(f"  No data found in bounds for {shelf_name}")
                    return None
                    
            except Exception as fallback_error:
                print(f"  Fallback failed for {shelf_name}: {fallback_error}")
                return None
        
    except Exception as e:
        print(f"Error loading predicted data for {shelf_name}: {e}")
        return None

def create_draft_melt_prediction(draft_range, params):
    """
    Create draft-melt prediction curve from parameters.
    
    Args:
        draft_range: Array of draft values to predict for
        params: Dictionary of draft dependence parameters
        
    Returns:
        Array of predicted melt rates
    """
    min_draft = params['minDraft'] 
    constant_value = params['constantValue']
    param_type = params['paramType']
    alpha0 = params['alpha0']
    alpha1 = params['alpha1']
    
    predicted_melt = np.zeros_like(draft_range)
    
    if param_type == 0:  # Linear parameterization
        # Shallow areas (below threshold): constant value
        shallow_mask = draft_range < min_draft
        predicted_melt[shallow_mask] = constant_value
        
        # Deep areas (above threshold): linear relationship
        deep_mask = draft_range >= min_draft
        predicted_melt[deep_mask] = alpha0 + alpha1 * draft_range[deep_mask]
        
    else:  # Constant parameterization (param_type == 1)
        predicted_melt[:] = constant_value
        
    return predicted_melt

def plot_ice_shelf_comparison(obs_data, pred_params, shelf_name, ax):
    """
    Create scatter plot for a single ice shelf showing observations and predictions.
    Following the notebook plotting style: melt on X-axis, draft on Y-axis.
    
    Args:
        obs_data: Observational data dict with 'draft' and 'melt' keys
        pred_params: Prediction parameters dict, or None if no predictions
        shelf_name: Name of ice shelf
        ax: Matplotlib axis to plot on
    """
    
    # Plot observational data
    if obs_data is not None and len(obs_data['draft']) > 0:
        # Debug: Print data ranges
        print(f"  {shelf_name}: Draft range [{obs_data['draft'].min():.1f}, {obs_data['draft'].max():.1f}], "
              f"Melt range [{obs_data['melt'].min():.2e}, {obs_data['melt'].max():.2e}], "
              f"N points: {len(obs_data['draft'])}")
        
        # Follow notebook style: melt on X-axis, draft on Y-axis
        # Subsample for plotting if too many points (like in notebook)
        n_plot = min(2000, len(obs_data['draft']))
        if len(obs_data['draft']) > n_plot:
            plot_idx = np.random.choice(len(obs_data['draft']), n_plot, replace=False)
            plot_draft = obs_data['draft'][plot_idx]
            plot_melt = obs_data['melt'][plot_idx]
        else:
            plot_draft = obs_data['draft']
            plot_melt = obs_data['melt']
        
        # Convert melt units if needed (from kg/m²/s to more reasonable units)
        # The notebook data likely uses different units, so let's convert to kg/m²/s
        # Check if values are very small (likely in kg/m²/s) and convert for visibility
        melt_units = 'kg/m²/s'
        if np.abs(plot_melt).max() < 0.01:  # If very small values, convert to more visible units
            # Convert from kg/m²/s to m/yr (assuming ice density ~917 kg/m³)
            # 1 kg/m²/s * (1 m³/917 kg) * (31536000 s/yr) = ~34394 m/yr
            plot_melt = plot_melt * 31536000 / 917  # Convert to m/yr
            melt_units = 'm/yr'
            print(f"    Converted melt units to {melt_units}, new range: [{plot_melt.min():.3f}, {plot_melt.max():.3f}]")
        
        # Plot observed data (black points like in notebook) with smaller scatter points
        ax.scatter(plot_melt, plot_draft, c='black', s=2, alpha=0.6, label='Observed')
        
        # Determine if relationship should be considered meaningful
        is_meaningful = True  # Default assumption, could add logic here based on correlation
        
        # Plot predictions if available
        if pred_params is not None:
            print(f"    {shelf_name}: Creating predictions with parameters: {pred_params}")
            
            # Create prediction for the same draft values
            pred_melt = create_draft_melt_prediction(plot_draft, pred_params)
            
            print(f"    {shelf_name}: Predicted melt range (raw, already in m/yr): [{pred_melt.min():.3f}, {pred_melt.max():.3f}]")
            
            # Predicted values are already in m/yr (from parameter units), so no conversion needed
            # Only convert observations from SI units to m/yr for comparison
            
            # Choose colors based on meaningfulness (like notebook)
            if is_meaningful:
                # Meaningful relationships: black observed, orange predicted
                pred_color = 'orange'
                title_color = 'black'
            else:
                # Non-meaningful relationships: gray observed, red predicted
                pred_color = 'red'
                title_color = 'red'
                ax.collections[0].set_color('gray')  # Change observed points to gray
            
            # Plot predicted data with smaller scatter points
            print(f"    {shelf_name}: Plotting {len(pred_melt)} predicted points in {pred_color}")
            ax.scatter(pred_melt, plot_draft, c=pred_color, s=2, alpha=0.8, label='Predicted')
            
            # Add threshold line if it's a linear parameterization
            param_type = pred_params['paramType']
            if param_type == 0:  # Linear parameterization
                min_draft = pred_params['minDraft']
                if min_draft > 0 and is_meaningful:
                    ax.axhline(min_draft, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                
                if is_meaningful:
                    param_info = f"MSE: N/A, R²: N/A\nThreshold: {min_draft:.1f}m"
                else:
                    param_info = f"{shelf_name} (NOISY)\nMSE: N/A, R²: N/A\nThreshold: {min_draft:.1f}m"
            else:
                param_info = f"Constant ({pred_params['constantValue']:.3f})"
                
            # Calculate metrics like in notebook
            valid_mask = ~np.isnan(plot_melt) & ~np.isnan(pred_melt)
            if np.sum(valid_mask) > 0:
                melt_valid = plot_melt[valid_mask]
                pred_valid = pred_melt[valid_mask]
                
                mse = np.mean((melt_valid - pred_valid)**2)
                if np.var(melt_valid) > 0:
                    r2 = 1 - np.sum((melt_valid - pred_valid)**2) / np.sum((melt_valid - np.mean(melt_valid))**2)
                else:
                    r2 = 0.0
                    
                if param_type == 0:
                    min_draft = pred_params['minDraft']
                    if is_meaningful:
                        param_info = f"MSE: {mse:.2e}, R²: {r2:.3f}\nThreshold: {min_draft:.1f}m"
                    else:
                        param_info = f"{shelf_name} (NOISY)\nMSE: {mse:.2e}, R²: {r2:.3f}\nThreshold: {min_draft:.1f}m"
                else:
                    param_info = f"MSE: {mse:.2e}, R²: {r2:.3f}\nConstant: {pred_params['constantValue']:.3f}"
        else:
            param_info = "No predictions"
            title_color = 'black'
            
        # Set labels and title (following notebook style)
        ax.set_xlabel(f'Melt Rate ({melt_units})', fontsize=8)
        ax.set_ylabel('Draft (m)', fontsize=8)
        ax.set_title(f"{shelf_name}\n{param_info}", fontsize=9, pad=10, color=title_color)
        
        # Set axis limits with some padding
        if len(plot_melt) > 0 and len(plot_draft) > 0:
            melt_min, melt_max = plot_melt.min(), plot_melt.max()
            melt_range = melt_max - melt_min
            if melt_range > 0:
                x_min = melt_min - 0.1*melt_range
                x_max = melt_max + 0.1*melt_range
                ax.set_xlim(x_min, x_max)
            else:
                # Handle case where all melt values are the same
                x_min = melt_min - 0.1*abs(melt_min) - 0.01
                x_max = melt_max + 0.1*abs(melt_max) + 0.01
                ax.set_xlim(x_min, x_max)
            
            draft_min, draft_max = plot_draft.min(), plot_draft.max()
            draft_range_val = draft_max - draft_min  
            if draft_range_val > 0:
                y_min = draft_min - 0.1*draft_range_val
                y_max = draft_max + 0.1*draft_range_val
                ax.set_ylim(y_min, y_max)
            else:
                # Handle case where all draft values are the same
                y_min = draft_min - 0.1*abs(draft_min) - 10
                y_max = draft_max + 0.1*abs(draft_max) + 10
                ax.set_ylim(y_min, y_max)
            
            print(f"    Axis bounds: X=[{ax.get_xlim()[0]:.3f}, {ax.get_xlim()[1]:.3f}], Y=[{ax.get_ylim()[0]:.1f}, {ax.get_ylim()[1]:.1f}]")
            
            # Invert Y-axis like in notebook (deeper drafts at bottom)
            ax.invert_yaxis()
        else:
            # Set default limits if no data
            ax.set_xlim(-1, 1)
            ax.set_ylim(-100, 100)
            ax.invert_yaxis()
            print(f"    Using default axis bounds: X=[-1, 1], Y=[-100, 100]")
        
        # Add grid and formatting
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        # Add legend to each subplot if there are both observed and predicted data
        if pred_params is not None:
            ax.legend(fontsize=7, loc='upper right')
        
    else:
        # No observational data
        ax.text(0.5, 0.5, f"{shelf_name}\nNo data", 
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

def create_draft_dependence_visualization(parameter_set_name, parameter_test_dir=None, 
                                        output_dir=None, max_shelves=None, start_index=33):
    """
    Create grid of scatter plots comparing observations and predictions for all ice shelves.
    
    Args:
        parameter_set_name: Name of parameter set to visualize
        parameter_test_dir: Directory containing parameter test results
        output_dir: Directory to save plots  
        max_shelves: Maximum number of shelves to plot (for testing)
        start_index: Starting index for ice shelves (default: 33 for Abbott Ice Shelf)
    """
    
    print(f"Creating draft dependence visualization for parameter set: {parameter_set_name}")
    
    try:
        # Set up directories
        if parameter_test_dir is None:
            parameter_test_dir = config.DIR_PROCESSED / "draft_dependence_changepoint"
        parameter_test_dir = Path(parameter_test_dir)
        
        if output_dir is None:
            output_dir = parameter_test_dir.parent / "visualizations"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("Loading satellite observation data...")
        satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
        satobs = write_crs(satobs, config.CRS_TARGET)
        print(f"  Loaded satellite data with shape: {satobs.dims}")
        
        print("Loading ice shelf masks...")
        icems = gpd.read_file(config.FILE_ICESHELFMASKS)
        icems = icems.to_crs(config.CRS_TARGET)
        print(f"  Loaded {len(icems)} ice shelf masks")
        
        # For merged parameter grids, the parameter set directory IS the data directory
        param_set_dir = parameter_test_dir
        if not param_set_dir.exists():
            print(f"Error: Parameter set directory not found: {param_set_dir}")
            return
            
        # Get ice shelf names starting from specified index
        shelf_names = list(icems.name.values[start_index:])
        if max_shelves is not None:
            shelf_names = shelf_names[:max_shelves]
            
        print(f"Processing {len(shelf_names)} ice shelves starting from index {start_index}")
        
        # Calculate grid layout
        n_shelves = len(shelf_names)
        n_cols = min(6, n_shelves)  # Max 6 columns
        n_rows = int(np.ceil(n_shelves / n_cols))
        
        print(f"Creating {n_rows} x {n_cols} grid for {n_shelves} ice shelves")
        
        # Create figure with subplots (like in the notebook)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Handle single subplot case
        if n_shelves == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = np.array(axes).flatten()
        
        # Process each ice shelf
        processed_count = 0
        for i, shelf_name in enumerate(shelf_names):
            actual_index = i + start_index
            
            ax = axes[i]
            
            print(f"Processing {shelf_name} ({i+1}/{len(shelf_names)})...")
            
            try:
                # Get ice shelf geometry
                ice_shelf_geom = icems.iloc[actual_index].geometry
                
                # Load observational data
                obs_data = load_ice_shelf_data(actual_index, icems, satobs, config)
                
                # Load prediction parameters from merged grids
                pred_params = load_predicted_data(shelf_name, ice_shelf_geom, param_set_dir, config)
                
                # Create plot
                plot_ice_shelf_comparison(obs_data, pred_params, shelf_name, ax)
                
                if obs_data is not None:
                    processed_count += 1
                    
            except Exception as shelf_error:
                print(f"  Error processing {shelf_name}: {shelf_error}")
                # Still create a placeholder plot
                ax.text(0.5, 0.5, f"{shelf_name}\nError: {str(shelf_error)[:50]}...", 
                        transform=ax.transAxes, ha='center', va='center', fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
        # Add overall title (positioned to avoid overlap)
        #fig.suptitle(f'Draft Dependence Analysis: {parameter_set_name}\n'
        #            f'Observations (black) vs Predictions (orange)', fontsize=14, y=0.95)
        
        # Remove the overall figure legend - keep individual subplot legends only
        # (Individual legends are created in plot_ice_shelf_comparison function)
        
        # Hide unused subplots
        for i in range(n_shelves, len(axes)):
            axes[i].set_visible(False)
        
        # Save plot
        output_file = output_dir / f"draft_dependence_comparison_{parameter_set_name}.png"
        print(f"Saving plot to: {output_file}")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
        
        # Save high-res version  
        output_file_hires = output_dir / f"draft_dependence_comparison_{parameter_set_name}_hires.png"
        plt.savefig(output_file_hires, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        print(f"Processed {processed_count} ice shelves with data out of {len(shelf_names)} total")
        
        return output_file
        
    except Exception as e:
        print(f"Error in create_draft_dependence_visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_summary_comparison(parameter_test_dir=None, output_dir=None):
    """
    Create a summary plot comparing different parameter sets.
    
    Args:
        parameter_test_dir: Directory containing parameter test results
        output_dir: Directory to save plots
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
    
    # Create comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data for plotting
    param_names = []
    meaningful_counts = []
    linear_counts = []
    constant_counts = []
    
    for name, summary in results_summary.items():
        if summary['status'] == 'completed':
            param_names.append(name)
            meaningful_counts.append(summary['meaningful_shelves'])
            linear_counts.append(summary['linear_param_count'])
            constant_counts.append(summary['constant_param_count'])
    
    x_pos = np.arange(len(param_names))
    
    # Plot 1: Meaningful relationships
    ax1.bar(x_pos, meaningful_counts, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Parameter Set')
    ax1.set_ylabel('Number of Meaningful Relationships')
    ax1.set_title('Meaningful Draft-Melt Relationships by Parameter Set')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(param_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
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
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Visualize draft dependence analysis results')
    parser.add_argument('--parameter_set', type=str, required=True,
                        help='Name of parameter set to visualize')
    parser.add_argument('--parameter_test_dir', type=str, default=None,
                        help='Directory containing parameter test results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualization plots')
    parser.add_argument('--max_shelves', type=int, default=None,
                        help='Maximum number of shelves to plot (for testing)')
    parser.add_argument('--start_index', type=int, default=33,
                        help='Starting index for ice shelves (default: 33)')
    parser.add_argument('--create_summary', action='store_true',
                        help='Also create summary comparison plot')
    
    args = parser.parse_args()
    
    print("DRAFT DEPENDENCE VISUALIZATION")
    print("=" * 40)
    
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
        print(f"Summary comparison saved to: {summary_file}")
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()
