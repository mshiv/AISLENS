#!/usr/bin/env python3
"""
Draft Dependence Map Visualization Script

Creates a large Antarctica ice sheet map with colored ice shelf regions and 
scatter plot insets positioned near their corresponding ice shelves showing 
observed vs predicted melt rates from draft dependence parameterization.

Usage:
    python visualize_draft_dependence_map.py --parameter_set original
    python visualize_draft_dependence_map.py --parameter_set permissive --output_dir /path/to/output

Shiva Muruganandham
August 28, 2025
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
    """
    try:
        # Use the combined parameter file
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
            
            # Extract parameter values
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
            return None
        
    except Exception as e:
        print(f"Error loading predicted data for {shelf_name}: {e}")
        return None

def create_draft_melt_prediction(draft_range, params):
    """
    Create draft-melt prediction curve from parameters.
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

def get_ice_shelf_centroid(ice_shelf_geom, target_crs='EPSG:3031'):
    """
    Get the centroid of an ice shelf geometry in target CRS coordinates.
    
    Args:
        ice_shelf_geom: Shapely geometry of ice shelf
        target_crs: Target coordinate reference system
        
    Returns:
        (x, y) coordinates of centroid
    """
    try:
        centroid = ice_shelf_geom.centroid
        return centroid.x, centroid.y
    except Exception as e:
        print(f"Error getting centroid: {e}")
        return 0, 0

def calculate_inset_position(centroid_x, centroid_y, map_bounds, inset_size=0.08):
    """
    Calculate inset position as fraction of figure coordinates.
    
    Args:
        centroid_x, centroid_y: Ice shelf centroid in map coordinates
        map_bounds: (xmin, xmax, ymin, ymax) of map extent
        inset_size: Size of inset as fraction of figure
        
    Returns:
        (left, bottom, width, height) in figure coordinates
    """
    xmin, xmax, ymin, ymax = map_bounds
    
    # Convert to normalized coordinates (0-1)
    norm_x = (centroid_x - xmin) / (xmax - xmin)
    norm_y = (centroid_y - ymin) / (ymax - ymin)
    
    # Offset to position inset near but not exactly on centroid
    offset_x = 0.02  # Small offset to avoid overlap
    offset_y = 0.02
    
    # Calculate inset position (centered on offset location)
    left = norm_x + offset_x - inset_size/2
    bottom = norm_y + offset_y - inset_size/2
    
    # Keep insets within figure bounds
    left = max(0.01, min(0.99 - inset_size, left))
    bottom = max(0.01, min(0.99 - inset_size, bottom))
    
    return left, bottom, inset_size, inset_size

def plot_ice_shelf_inset(obs_data, pred_params, shelf_name, color, inset_ax):
    """
    Create scatter plot inset for a single ice shelf.
    
    Args:
        obs_data: Observational data dict with 'draft' and 'melt' keys
        pred_params: Prediction parameters dict, or None if no predictions
        shelf_name: Name of ice shelf
        color: Color for ice shelf (for title and border)
        inset_ax: Matplotlib axis for the inset
    """
    
    # Plot observational data if available
    if obs_data is not None and len(obs_data['draft']) > 0:
        # Subsample for plotting if too many points
        n_plot = min(500, len(obs_data['draft']))  # Smaller for insets
        if len(obs_data['draft']) > n_plot:
            plot_idx = np.random.choice(len(obs_data['draft']), n_plot, replace=False)
            plot_draft = obs_data['draft'][plot_idx]
            plot_melt = obs_data['melt'][plot_idx]
        else:
            plot_draft = obs_data['draft']
            plot_melt = obs_data['melt']
        
        # Convert melt units if needed
        melt_units = 'kg/m²/s'
        if np.abs(plot_melt).max() < 0.01:
            plot_melt = plot_melt * 31536000 / 917  # Convert to m/yr
            melt_units = 'm/yr'
        
        # Plot observed data with very small points for insets
        inset_ax.scatter(plot_melt, plot_draft, c='black', s=0.5, alpha=0.6)
        
        # Plot predictions if available
        if pred_params is not None:
            pred_melt = create_draft_melt_prediction(plot_draft, pred_params)
            
            # Determine if relationship is meaningful
            is_meaningful = True  # Could add logic here
            pred_color = 'orange' if is_meaningful else 'red'
            
            # Plot predicted data
            inset_ax.scatter(pred_melt, plot_draft, c=pred_color, s=0.5, alpha=0.8)
            
            # Add threshold line if linear parameterization
            param_type = pred_params['paramType']
            if param_type == 0 and pred_params['minDraft'] > 0:
                inset_ax.axhline(pred_params['minDraft'], color='red', 
                               linestyle='--', linewidth=0.8, alpha=0.8)
        
        # Set axis limits with padding
        if len(plot_melt) > 0 and len(plot_draft) > 0:
            melt_min, melt_max = plot_melt.min(), plot_melt.max()
            draft_min, draft_max = plot_draft.min(), plot_draft.max()
            
            melt_range = melt_max - melt_min
            draft_range = draft_max - draft_min
            
            if melt_range > 0:
                x_min = melt_min - 0.1*melt_range
                x_max = melt_max + 0.1*melt_range
                inset_ax.set_xlim(x_min, x_max)
            
            if draft_range > 0:
                y_min = draft_min - 0.1*draft_range
                y_max = draft_max + 0.1*draft_range
                inset_ax.set_ylim(y_min, y_max)
            
            # Invert Y-axis (deeper drafts at bottom)
            inset_ax.invert_yaxis()
        
        # Add title with ice shelf color
        inset_ax.set_title(shelf_name, fontsize=6, color=color, fontweight='bold')
        
        # Minimal axis formatting for insets
        inset_ax.tick_params(labelsize=4)
        inset_ax.grid(True, alpha=0.3, linewidth=0.3)
        
        # Set inset border color to match ice shelf color
        for spine in inset_ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)
        
    else:
        # No data available
        inset_ax.text(0.5, 0.5, 'No data', transform=inset_ax.transAxes, 
                     ha='center', va='center', fontsize=5, color=color)
        inset_ax.set_xlim(0, 1)
        inset_ax.set_ylim(0, 1)
        inset_ax.set_title(shelf_name, fontsize=6, color=color, fontweight='bold')

def create_draft_dependence_map_visualization(parameter_set_name, parameter_test_dir=None, 
                                            output_dir=None, max_shelves=None, start_index=33):
    """
    Create Antarctica map with colored ice shelf regions and scatter plot insets.
    
    Args:
        parameter_set_name: Name of parameter set to visualize
        parameter_test_dir: Directory containing parameter test results
        output_dir: Directory to save plots  
        max_shelves: Maximum number of shelves to plot (for testing)
        start_index: Starting index for ice shelves (default: 33)
    """
    
    print(f"Creating draft dependence map visualization for parameter set: {parameter_set_name}")
    
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
        
        print("Loading ice shelf masks...")
        icems = gpd.read_file(config.FILE_ICESHELFMASKS)
        icems = icems.to_crs(config.CRS_TARGET)
        
        # Parameter set directory
        param_set_dir = parameter_test_dir
        if not param_set_dir.exists():
            print(f"Error: Parameter set directory not found: {param_set_dir}")
            return
        
        # Get ice shelf subset
        ice_shelf_indices = list(range(start_index, len(icems)))
        if max_shelves is not None:
            ice_shelf_indices = ice_shelf_indices[:max_shelves]
        
        print(f"Processing {len(ice_shelf_indices)} ice shelves starting from index {start_index}")
        
        # Create color map for ice shelves
        n_shelves = len(ice_shelf_indices)
        colors = plt.cm.tab20(np.linspace(0, 1, n_shelves))
        if n_shelves > 20:
            # Use multiple colormaps if more than 20 shelves
            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
            colors2 = plt.cm.Set3(np.linspace(0, 1, n_shelves - 20))
            colors = np.vstack([colors1, colors2])
        
        # Create main figure with Antarctica map
        fig = plt.figure(figsize=(16, 16))
        
        # Main map axis with South Polar Stereographic projection
        ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
        
        # Set map extent for Antarctica
        ax_map.set_extent([-180, 180, -60, -90], ccrs.PlateCarree())
        
        # Add map features
        ax_map.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        ax_map.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax_map.coastlines(resolution='10m', linewidth=0.5)
        ax_map.gridlines(draw_labels=False, alpha=0.3)
        
        # Plot all ice shelf boundaries first (background)
        ice_shelf_subset = icems.iloc[start_index:start_index+len(ice_shelf_indices)]
        ice_shelf_subset.boundary.plot(ax=ax_map, linewidth=0.3, color='lightsteelblue', 
                                     transform=ccrs.PlateCarree())
        
        # Get map bounds for inset positioning
        ax_map_bounds = ax_map.get_extent()
        
        # Process each ice shelf and create insets
        processed_count = 0
        for i, ice_shelf_idx in enumerate(ice_shelf_indices):
            shelf_name = icems.iloc[ice_shelf_idx].name
            ice_shelf_geom = icems.iloc[ice_shelf_idx].geometry
            shelf_color = colors[i]
            
            print(f"Processing {shelf_name} ({i+1}/{len(ice_shelf_indices)})...")
            
            try:
                # Plot colored ice shelf on main map
                icems.iloc[[ice_shelf_idx]].plot(ax=ax_map, color=shelf_color, 
                                               linewidth=0.4, transform=ccrs.PlateCarree())
                
                # Load observational and prediction data
                obs_data = load_ice_shelf_data(ice_shelf_idx, icems, satobs, config)
                pred_params = load_predicted_data(shelf_name, ice_shelf_geom, param_set_dir, config)
                
                # Skip inset if no observational data
                if obs_data is None:
                    print(f"  {shelf_name}: Skipping inset - no observational data")
                    continue
                
                # Get ice shelf centroid for inset positioning
                centroid_x, centroid_y = get_ice_shelf_centroid(ice_shelf_geom)
                
                # Convert centroid to figure coordinates for inset positioning
                # This is simplified - in practice you might need more sophisticated positioning
                # to avoid overlaps and place insets optimally
                
                # Calculate inset position (this is a simplified approach)
                # You may want to manually adjust positions for better layout
                inset_size = 0.06  # Size of each inset
                
                # Simple grid-like positioning with offset based on index
                row = i // 8  # 8 insets per row
                col = i % 8
                left = 0.05 + col * 0.12
                bottom = 0.85 - row * 0.15
                
                # Ensure insets stay within figure bounds
                if bottom < 0.05:
                    bottom = 0.05 + (row % 5) * 0.15  # Wrap to avoid going off figure
                
                # Create inset axes
                inset_ax = fig.add_axes([left, bottom, inset_size, inset_size])
                
                # Plot scatter plot in inset
                plot_ice_shelf_inset(obs_data, pred_params, shelf_name, shelf_color, inset_ax)
                
                processed_count += 1
                
            except Exception as shelf_error:
                print(f"  Error processing {shelf_name}: {shelf_error}")
                continue
        
        # Add title and labels
        ax_map.set_title(f'Draft Dependence Analysis: {parameter_set_name}\n'
                        f'Antarctica Ice Shelves with Melt Rate vs Draft Depth Insets\n'
                        f'Black: Observed, Orange: Predicted', 
                        fontsize=14, pad=20)
        
        # Add a small legend explaining the inset plots
        legend_text = ('Inset plots show melt rate (x-axis) vs draft depth (y-axis)\n'
                      'Black dots: Observed data\n'
                      'Orange dots: Predicted data\n'
                      'Red dashed line: Draft threshold (if applicable)')
        
        fig.text(0.02, 0.02, legend_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        output_file = output_dir / f"draft_dependence_map_{parameter_set_name}.png"
        print(f"Saving plot to: {output_file}")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Map visualization saved to: {output_file}")
        
        # Save high-res version  
        output_file_hires = output_dir / f"draft_dependence_map_{parameter_set_name}_hires.png"
        plt.savefig(output_file_hires, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        print(f"Processed {processed_count} ice shelves with data out of {len(ice_shelf_indices)} total")
        
        return output_file
        
    except Exception as e:
        print(f"Error in create_draft_dependence_map_visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Visualize draft dependence analysis on Antarctica map')
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
    
    args = parser.parse_args()
    
    print("DRAFT DEPENDENCE MAP VISUALIZATION")
    print("=" * 45)
    
    # Create map visualization
    output_file = create_draft_dependence_map_visualization(
        parameter_set_name=args.parameter_set,
        parameter_test_dir=args.parameter_test_dir,
        output_dir=args.output_dir,
        max_shelves=args.max_shelves,
        start_index=args.start_index
    )
    
    print("Map visualization complete!")

if __name__ == "__main__":
    main()