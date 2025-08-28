#!/usr/bin/env python3
"""
Draft Dependence Map Visualization

Creates a massive Antarctica ice sheet map with colored ice shelf regions and 
scatter plot insets positioned near their corresponding ice shelves showing 
observed vs predicted melt rates from draft dependence parameterization.

Shiva Muruganandham
August 2025
"""

import argparse
import json
import pickle
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
from shapely.geometry import mapping
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

# Import your AISLENS modules
from aislens.config import config
from aislens.utils import write_crs

def load_ice_shelf_data(ice_shelf_index, icems, satobs, config):
    """
    Load observational data for a specific ice shelf.
    """
    try:
        # Get ice shelf geometry and name
        ice_shelf_geom = icems.iloc[ice_shelf_index].geometry
        shelf_name = icems.iloc[ice_shelf_index].name
        
        if ice_shelf_geom is None or ice_shelf_geom.is_empty:
            print(f"  {shelf_name}: Empty geometry")
            return None
            
        # Create mask for ice shelf using the same method as notebook
        ice_shelf_mask = icems.loc[[ice_shelf_index], 'geometry'].apply(mapping)
        
        # Clip satellite data to ice shelf region
        satobs_clipped = satobs.rio.clip(ice_shelf_mask, icems.crs)
        
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

def load_predicted_data(shelf_name, ice_shelf_index, icems, parameter_set_dir, config):
    """
    Load predicted melt rates using the same approach as the notebook.
    """
    try:
        # Use the combined parameter file (similar to notebook approach)
        combined_file = parameter_set_dir / "ruptures_draftDepenBasalMelt_parameters_filled.nc"
        
        if not combined_file.exists():
            print(f"Warning: Combined parameter file not found: {combined_file}")
            return None
            
        # Load the combined dataset
        ds = xr.open_dataset(combined_file)
        
        # Ensure CRS is set
        if not hasattr(ds, 'rio') or ds.rio.crs is None:
            ds = ds.rio.write_crs(config.CRS_TARGET)
        
        # Create mask using same method as notebook
        ice_shelf_mask = icems.loc[[ice_shelf_index], 'geometry'].apply(mapping)
        
        try:
            clipped = ds.rio.clip(ice_shelf_mask, icems.crs)
            
            # Extract parameter values using notebook approach
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
            print(f"  Clipping failed for {shelf_name}: {clip_error}")
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

def get_ice_shelf_centroid(ice_shelf_geom):
    """
    Get the centroid of an ice shelf geometry.
    """
    try:
        centroid = ice_shelf_geom.centroid
        return centroid.x, centroid.y
    except Exception as e:
        print(f"Error getting centroid: {e}")
        return 0, 0

def calculate_inset_positions(ice_shelf_centroids, n_shelves):
    """
    Calculate smart inset positions to avoid overlaps.
    Uses a combination of geographic position and grid layout.
    """
    positions = []
    
    # Define regions of Antarctica and their preferred inset locations
    # This is a simplified approach - you could make this more sophisticated
    
    # Create a grid of potential positions
    n_cols = 6
    n_rows = int(np.ceil(n_shelves / n_cols))
    
    # Start positions around the edges of the figure
    left_start = 0.02
    bottom_start = 0.02
    width_spacing = 0.15
    height_spacing = 0.12
    inset_size = 0.08
    
    for i in range(n_shelves):
        row = i // n_cols
        col = i % n_cols
        
        left = left_start + col * width_spacing
        bottom = bottom_start + row * height_spacing
        
        # Ensure we don't go off the figure
        if left + inset_size > 0.98:
            left = 0.98 - inset_size
        if bottom + inset_size > 0.98:
            bottom = 0.98 - inset_size
            
        positions.append((left, bottom, inset_size, inset_size))
    
    return positions

def plot_ice_shelf_inset(obs_data, pred_params, shelf_name, color, inset_ax):
    """
    Create scatter plot inset for a single ice shelf (similar to notebook approach).
    """
    
    # Plot observational data if available
    if obs_data is not None and len(obs_data['draft']) > 0:
        # Subsample for plotting if too many points
        n_plot = min(500, len(obs_data['draft']))
        if len(obs_data['draft']) > n_plot:
            plot_idx = np.random.choice(len(obs_data['draft']), n_plot, replace=False)
            plot_draft = obs_data['draft'][plot_idx]
            plot_melt = obs_data['melt'][plot_idx]
        else:
            plot_draft = obs_data['draft']
            plot_melt = obs_data['melt']
        
        # Convert melt units if needed (similar to notebook)
        melt_units = 'kg/m²/s'
        if np.abs(plot_melt).max() < 0.01:
            plot_melt = plot_melt * 31536000 / 917  # Convert to m/yr
            melt_units = 'm/yr'
        
        # Plot observed data with small points for insets
        inset_ax.scatter(plot_melt, plot_draft, c='black', s=1, alpha=0.6, marker='x')
        
        # Plot predictions if available
        if pred_params is not None:
            pred_melt = create_draft_melt_prediction(plot_draft, pred_params)
            
            # Plot predicted data
            inset_ax.scatter(pred_melt, plot_draft, c=color, s=1, alpha=0.8)
            
            # Add threshold line if linear parameterization
            param_type = pred_params['paramType']
            if param_type == 0 and pred_params['minDraft'] > 0:
                inset_ax.axhline(pred_params['minDraft'], color='red', 
                               linestyle='--', linewidth=0.8, alpha=0.8)
        
        # Set axis limits with padding (similar to notebook approach)
        if len(plot_melt) > 0 and len(plot_draft) > 0:
            # Set y-axis from deepest to shallowest (invert)
            inset_ax.set_ylim(plot_draft.max() * 1.1, 0)
            
            # Set x-axis limits
            melt_min, melt_max = plot_melt.min(), plot_melt.max()
            melt_range = melt_max - melt_min
            if melt_range > 0:
                x_min = melt_min - 0.1*melt_range
                x_max = melt_max + 0.1*melt_range
                inset_ax.set_xlim(x_min, x_max)
        
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

def process_and_cache_data(parameter_set_name, parameter_test_dir, cache_file, 
                          max_shelves=None, start_index=33):
    """
    Process all ice shelf data and cache results for faster subsequent runs.
    """
    print(f"Processing data for parameter set: {parameter_set_name}")
    
    # Set up directories
    if parameter_test_dir is None:
        parameter_test_dir = config.DIR_PROCESSED / "draft_dependence_changepoint"
    parameter_test_dir = Path(parameter_test_dir)
    
    # Load data
    print("Loading satellite observation data...")
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
    satobs = write_crs(satobs, config.CRS_TARGET)
    
    print("Loading ice shelf masks...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs(config.CRS_TARGET)
    
    # Parameter set directory
    param_set_dir = parameter_test_dir
    
    # Get ice shelf subset (ice shelves from index 33 onwards, like in notebook)
    ice_shelf_indices = list(range(start_index, len(icems)))
    if max_shelves is not None:
        ice_shelf_indices = ice_shelf_indices[:max_shelves]
    
    print(f"Processing {len(ice_shelf_indices)} ice shelves starting from index {start_index}")
    
    # Process each ice shelf
    processed_data = {
        'ice_shelves': [],
        'obs_data': [],
        'pred_params': [],
        'geometries': [],
        'centroids': []
    }
    
    for ice_shelf_idx in ice_shelf_indices:
        shelf_name = icems.iloc[ice_shelf_idx].name
        ice_shelf_geom = icems.iloc[ice_shelf_idx].geometry
        
        print(f"Processing {shelf_name}...")
        
        try:
            # Load observational and prediction data
            obs_data = load_ice_shelf_data(ice_shelf_idx, icems, satobs, config)
            pred_params = load_predicted_data(shelf_name, ice_shelf_idx, icems, param_set_dir, config)
            
            # Get centroid
            centroid_x, centroid_y = get_ice_shelf_centroid(ice_shelf_geom)
            
            # Store data
            processed_data['ice_shelves'].append(shelf_name)
            processed_data['obs_data'].append(obs_data)
            processed_data['pred_params'].append(pred_params)
            processed_data['geometries'].append(ice_shelf_geom)
            processed_data['centroids'].append((centroid_x, centroid_y))
            
        except Exception as e:
            print(f"Error processing {shelf_name}: {e}")
            continue
    
    # Cache the processed data
    print(f"Caching processed data to: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    return processed_data

def load_cached_data(cache_file):
    """
    Load previously processed and cached data.
    """
    print(f"Loading cached data from: {cache_file}")
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def create_draft_dependence_map_visualization(parameter_set_name, parameter_test_dir=None, 
                                            output_dir=None, max_shelves=None, start_index=33,
                                            use_cache=False, cache_file=None):
    """
    Create Antarctica map with colored ice shelf regions and scatter plot insets.
    """
    
    print(f"Creating draft dependence map visualization for parameter set: {parameter_set_name}")
    
    # Set up directories
    if parameter_test_dir is None:
        parameter_test_dir = config.DIR_PROCESSED / "draft_dependence_changepoint"
    parameter_test_dir = Path(parameter_test_dir)
    
    if output_dir is None:
        output_dir = parameter_test_dir.parent / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set cache file
    if cache_file is None:
        cache_file = output_dir / f"processed_data_{parameter_set_name}.pkl"
    
    # Load or process data
    if use_cache and Path(cache_file).exists():
        processed_data = load_cached_data(cache_file)
    else:
        processed_data = process_and_cache_data(
            parameter_set_name, parameter_test_dir, cache_file, max_shelves, start_index
        )
    
    # Load ice shelf masks for plotting
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs(config.CRS_TARGET)
    
    # Filter to ice shelves we have data for
    n_shelves = len(processed_data['ice_shelves'])
    print(f"Creating visualization for {n_shelves} ice shelves")
    
    # Create color map for ice shelves (use distinctive colors like in notebook)
    if n_shelves <= 5:
        colors = ['r', 'b', 'g', 'y', 'm'][:n_shelves]
    else:
        # Use colormap for more ice shelves
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_shelves)))
        if n_shelves > 20:
            colors2 = plt.cm.Set3(np.linspace(0, 1, n_shelves - 20))
            colors = np.vstack([colors[:20], colors2])
    
    # Create main figure with Antarctica map
    fig = plt.figure(figsize=(20, 16))
    
    # Main map axis with South Polar Stereographic projection
    ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    
    # Set map extent for Antarctica
    ax_map.set_extent([-180, 180, -60, -90], ccrs.PlateCarree())
    
    # Plot base ice shelf regions (like in notebook)
    icems[start_index:start_index+100].plot(ax=ax_map, color='aliceblue', linewidth=0, zorder=1,
                                           transform=ccrs.PlateCarree())
    icems[start_index:start_index+100].boundary.plot(ax=ax_map, linewidth=0.3, color='lightsteelblue',
                                                    transform=ccrs.PlateCarree(), zorder=2)
    
    # Add coastlines
    ax_map.coastlines(resolution='10m', linewidth=0.5)
    
    # Find indices of our ice shelves in the full icems dataframe
    ice_shelf_indices = []
    for shelf_name in processed_data['ice_shelves']:
        try:
            idx = icems[icems['name'] == shelf_name].index[0]
            ice_shelf_indices.append(idx)
        except:
            print(f"Warning: Could not find index for {shelf_name}")
            ice_shelf_indices.append(None)
    
    # Plot colored ice shelves on main map (like in notebook)
    for i, (shelf_name, idx) in enumerate(zip(processed_data['ice_shelves'], ice_shelf_indices)):
        if idx is not None:
            shelf_color = colors[i] if isinstance(colors[0], str) else colors[i]
            icems.loc[[idx]].plot(ax=ax_map, color=shelf_color, linewidth=0.4, 
                                transform=ccrs.PlateCarree(), zorder=3)
    
    # Calculate inset positions
    inset_positions = calculate_inset_positions(processed_data['centroids'], n_shelves)
    
    # Create inset scatter plots
    processed_count = 0
    for i, (shelf_name, obs_data, pred_params, position) in enumerate(
        zip(processed_data['ice_shelves'], processed_data['obs_data'], 
            processed_data['pred_params'], inset_positions)
    ):
        
        if obs_data is None:
            print(f"  {shelf_name}: Skipping inset - no observational data")
            continue
        
        shelf_color = colors[i] if isinstance(colors[0], str) else colors[i]
        
        # Create inset axes
        left, bottom, width, height = position
        inset_ax = fig.add_axes([left, bottom, width, height])
        
        # Plot scatter plot in inset
        plot_ice_shelf_inset(obs_data, pred_params, shelf_name, shelf_color, inset_ax)
        
        processed_count += 1
    
    # Add title and labels
    ax_map.set_title(f'Draft Dependence Analysis: {parameter_set_name}\n'
                    f'Antarctica Ice Shelves with Melt Rate vs Draft Depth Insets\n'
                    f'Black crosses: Observed, Colored dots: Predicted', 
                    fontsize=16, pad=20)
    
    # Add a legend explaining the inset plots
    legend_text = ('Inset plots show melt rate (x-axis) vs draft depth (y-axis)\n'
                  'Black crosses: Observed data\n'
                  'Colored dots: Predicted data\n'
                  'Red dashed line: Draft threshold (if applicable)')
    
    fig.text(0.02, 0.02, legend_text, fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    output_file = output_dir / f"draft_dependence_map_{parameter_set_name}.png"
    print(f"Saving plot to: {output_file}")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Map visualization saved to: {output_file}")
    
    # Save high-res version  
    output_file_hires = output_dir / f"draft_dependence_map_{parameter_set_name}_hires.png"
    plt.savefig(output_file_hires, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Processed {processed_count} ice shelves with data out of {n_shelves} total")
    
    return output_file

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
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached processed data if available')
    parser.add_argument('--cache_file', type=str, default=None,
                        help='Path to cache file for processed data')
    
    args = parser.parse_args()
    
    print("DRAFT DEPENDENCE MAP VISUALIZATION")
    print("=" * 45)
    
    # Create map visualization
    output_file = create_draft_dependence_map_visualization(
        parameter_set_name=args.parameter_set,
        parameter_test_dir=args.parameter_test_dir,
        output_dir=args.output_dir,
        max_shelves=args.max_shelves,
        start_index=args.start_index,
        use_cache=args.use_cache,
        cache_file=args.cache_file
    )
    
    print("Map visualization complete!")

if __name__ == "__main__":
    main()