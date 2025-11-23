#!/usr/bin/env python3
"""
Create Antarctica map with ice shelf insets showing draft dependence analysis.

Visualizes observed vs predicted melt rates from draft dependence parameterization
with scatter plot insets positioned geographically near their ice shelves.

Usage:
    python visualize_draft_dependence_map.py --parameter_set original
    python visualize_draft_dependence_map.py --parameter_set original --use_cache
"""

import argparse
import logging
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

from aislens.config import config
from aislens.utils import setup_logging, write_crs

logger = logging.getLogger(__name__)

def load_ice_shelf_data(ice_shelf_index, icems, satobs, config):
    """Load observational data for a specific ice shelf."""
    try:
        ice_shelf_geom = icems.iloc[ice_shelf_index].geometry
        shelf_name = icems.iloc[ice_shelf_index].name
        
        if ice_shelf_geom is None or ice_shelf_geom.is_empty:
            logger.debug(f"{shelf_name}: Empty geometry")
            return None
            
        ice_shelf_mask = icems.loc[[ice_shelf_index], 'geometry'].apply(mapping)
        satobs_clipped = satobs.rio.clip(ice_shelf_mask, icems.crs)
        
        draft_var, flux_var = config.SATOBS_DRAFT_VAR, config.SATOBS_FLUX_VAR
        logger.debug(f"{shelf_name}: Variables '{draft_var}', '{flux_var}'")
        
        # Get time-averaged draft and melt data
        draft_data = (satobs_clipped[draft_var].mean(dim=config.TIME_DIM) if config.TIME_DIM in satobs_clipped[draft_var].dims 
                     else satobs_clipped[draft_var])
        melt_data = (satobs_clipped[flux_var].mean(dim=config.TIME_DIM) if config.TIME_DIM in satobs_clipped[flux_var].dims 
                    else satobs_clipped[flux_var])
        
        draft_flat, melt_flat = draft_data.values.flatten(), melt_data.values.flatten()
        logger.debug(f"{shelf_name}: Draft [{np.nanmin(draft_flat):.2f}, {np.nanmax(draft_flat):.2f}], "
                    f"Melt [{np.nanmin(melt_flat):.2e}, {np.nanmax(melt_flat):.2e}]")
        
        valid_mask = ~np.isnan(draft_flat) & ~np.isnan(melt_flat) & (draft_flat > 0)
        logger.debug(f"{shelf_name}: {valid_mask.sum()}/{len(valid_mask)} valid points")
        
        if valid_mask.sum() == 0:
            logger.debug(f"{shelf_name}: No valid data")
            return None
            
        return {'draft': draft_flat[valid_mask], 'melt': melt_flat[valid_mask], 'shelf_name': shelf_name}
        
    except Exception as e:
        logger.warning(f"Could not load data for shelf {ice_shelf_index}: {e}")
        return None

def load_predicted_data(shelf_name, ice_shelf_index, icems, parameter_set_dir, config):
    """
    Load predicted melt rates using the same approach as the notebook.
    """
    try:
        # Use the combined parameter file (similar to notebook approach)
        combined_file = parameter_set_dir / "ruptures_draftDepenBasalMelt_parameters_filled.nc"
        
        if not combined_file.exists():
            logger.warning(f"Combined parameter file not found: {combined_file}")
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
                    params[param_name] = float(valid_values[0]) if len(valid_values) > 0 else 0.0
                else:
                    params[param_name] = 0.0
                    
            logger.debug(f"Extracted parameters for {shelf_name}: {params}")
            return params
            
        except Exception as clip_error:
            logger.warning(f"Clipping failed for {shelf_name}: {clip_error}")
            return None
        
    except Exception as e:
        logger.warning(f"Error loading predicted data for {shelf_name}: {e}")
        return None

def create_draft_melt_prediction(draft_range, params):
    """
    Create draft-melt prediction curve from parameters (following visualize_draft_dependence.py).
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
        logger.error(f"Error getting centroid: {e}")
        return 0, 0

def calculate_geographic_inset_positions(ice_shelf_centroids, ice_shelf_names, n_shelves):
    """
    Calculate inset positions based on geographic proximity and regions.
    Ice shelves close together on the map should have adjacent insets.
    """
    logger.info(f"Calculating geographic inset positions for {n_shelves} ice shelves")
    
    # Convert centroids to arrays for easier processing
    centroids = np.array(ice_shelf_centroids)
    x_coords = centroids[:, 0]
    y_coords = centroids[:, 1]
    
    logger.info(f"Centroid ranges: X=[{x_coords.min():.0f}, {x_coords.max():.0f}], Y=[{y_coords.min():.0f}, {y_coords.max():.0f}]")
    
    # Define Antarctic regions based on coordinates (rough approximation)
    # These are approximate regions in Antarctic Polar Stereographic coordinates
    regions = {
        'West Antarctic Pacific': lambda x, y: x < -1000000 and y < 0,  # Pacific sector
        'West Antarctic Atlantic': lambda x, y: x > 1000000 and y < 0,  # Atlantic sector  
        'East Antarctic Indian': lambda x, y: x < 0 and y > 0,          # Indian sector
        'East Antarctic Pacific': lambda x, y: x > 0 and y > 0,         # Pacific sector
        'Central/Ross': lambda x, y: x < -500000 and y > -500000,       # Ross Sea area
        'Other': lambda x, y: True  # Catch-all
    }
    
    # Assign ice shelves to regions
    shelf_regions = {}
    for i, (name, (x, y)) in enumerate(zip(ice_shelf_names, ice_shelf_centroids)):
        assigned = False
        for region_name, region_func in regions.items():
            if region_func(x, y) and region_name != 'Other':
                shelf_regions[i] = region_name
                assigned = True
                break
        if not assigned:
            shelf_regions[i] = 'Other'
    
    logger.info("Regional assignments:")
    for i, region in shelf_regions.items():
        logger.debug(f"{ice_shelf_names[i]}: {region}")
    
    # Group ice shelves by region
    region_groups = {}
    for i, region in shelf_regions.items():
        if region not in region_groups:
            region_groups[region] = []
        region_groups[region].append(i)
    
    # Sort ice shelves within each region by proximity (simple clustering)
    ordered_indices = []
    for region_name, shelf_indices in region_groups.items():
        if len(shelf_indices) == 1:
            ordered_indices.extend(shelf_indices)
        else:
            # Sort by x-coordinate within region for simplicity
            region_centroids = [(i, ice_shelf_centroids[i]) for i in shelf_indices]
            sorted_by_x = sorted(region_centroids, key=lambda x: x[1][0])
            ordered_indices.extend([i for i, _ in sorted_by_x])
    
    logger.info(f"Geographic ordering: {[ice_shelf_names[i] for i in ordered_indices]}")
    
    # Create grid layout for insets
    n_cols = 6
    n_rows = int(np.ceil(n_shelves / n_cols))
    
    # Define inset grid parameters
    left_start = 0.02
    bottom_start = 0.02
    width_spacing = 0.15
    height_spacing = 0.12
    inset_size = 0.08
    
    # Calculate positions for ordered ice shelves
    positions = []
    for i, shelf_idx in enumerate(ordered_indices):
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
    
    # Create mapping from original order to geographic order
    position_mapping = {}
    for i, shelf_idx in enumerate(ordered_indices):
        position_mapping[shelf_idx] = positions[i]
    
    # Return positions in original order
    final_positions = []
    for i in range(n_shelves):
        final_positions.append(position_mapping[i])
    
    return final_positions

def plot_ice_shelf_inset(obs_data, pred_params, shelf_name, shelf_index, color, inset_ax):
    """
    Create scatter plot inset for a single ice shelf (following visualize_draft_dependence.py approach).
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
        
        # Convert melt units consistently (following visualize_draft_dependence.py)
        melt_units = 'kg/m²/s'
        
        # Always convert to m/yr for consistency
        plot_melt_display = plot_melt * 31536000 / 917  # Convert to m/yr
        melt_units = 'm/yr'
        
        # Plot observed data with small points for insets
        inset_ax.scatter(plot_melt_display, plot_draft, c='black', s=1, alpha=0.6, marker='x')
        
        # Plot predictions if available
        if pred_params is not None:
            logger.debug(f"Creating predictions for {shelf_name} with parameters: {pred_params}")
            
            # Create predictions using original draft values
            pred_melt = create_draft_melt_prediction(plot_draft, pred_params)
            
            # The predicted values should already be in m/yr from the parameters
            # But let's check the range and convert if needed
            if np.abs(pred_melt).max() < 0.01:
                # If predictions are very small, they might be in kg/m²/s
                pred_melt_display = pred_melt * 31536000 / 917
                logger.debug(f"Converted predictions from kg/m²/s to m/yr")
            else:
                pred_melt_display = pred_melt
            
            logger.debug(f"Predicted melt range: [{pred_melt_display.min():.3f}, {pred_melt_display.max():.3f}] m/yr")
            logger.debug(f"Observed melt range: [{plot_melt_display.min():.3f}, {plot_melt_display.max():.3f}] m/yr")
            
            # Plot predicted data
            inset_ax.scatter(pred_melt_display, plot_draft, c=color, s=1, alpha=0.8)
            
            # Add threshold line if linear parameterization
            param_type = pred_params['paramType']
            if param_type == 0 and pred_params['minDraft'] > 0:
                inset_ax.axhline(pred_params['minDraft'], color='red', 
                               linestyle='--', linewidth=0.8, alpha=0.8)
        
        # Set axis limits with padding
        if len(plot_melt_display) > 0 and len(plot_draft) > 0:
            # Set y-axis from deepest to shallowest (invert)
            inset_ax.set_ylim(plot_draft.max() * 1.1, 0)
            
            # Combine observed and predicted data for x-axis limits
            all_melt = plot_melt_display
            if pred_params is not None and 'pred_melt_display' in locals():
                all_melt = np.concatenate([plot_melt_display, pred_melt_display])
            
            # Set x-axis limits
            melt_min, melt_max = all_melt.min(), all_melt.max()
            melt_range = melt_max - melt_min
            if melt_range > 0:
                x_min = melt_min - 0.1*melt_range
                x_max = melt_max + 0.1*melt_range
                inset_ax.set_xlim(x_min, x_max)
            else:
                # Handle case where all melt values are the same
                x_min = melt_min - 0.1*abs(melt_min) - 0.01
                x_max = melt_max + 0.1*abs(melt_max) + 0.01
                inset_ax.set_xlim(x_min, x_max)
        
        # Add title with ice shelf index and name
        inset_ax.set_title(f"{shelf_index}: {shelf_name}", fontsize=6, color=color, fontweight='bold')
        
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
        inset_ax.set_title(f"{shelf_index}: {shelf_name}", fontsize=6, color=color, fontweight='bold')

def process_and_cache_data(parameter_set_name, parameter_test_dir, cache_file, 
                          max_shelves=None, start_index=33):
    """
    Process all ice shelf data and cache results for faster subsequent runs.
    """
    logger.info(f"Processing data for parameter set: {parameter_set_name}")
    
    # Set up directories
    if parameter_test_dir is None:
        parameter_test_dir = config.DIR_PROCESSED / "draft_dependence_changepoint"
    parameter_test_dir = Path(parameter_test_dir)
    
    # Load data
    logger.info("Loading satellite observation data...")
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
    satobs = write_crs(satobs, config.CRS_TARGET)
    
    logger.info("Loading ice shelf masks...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS)
    icems = icems.to_crs(config.CRS_TARGET)
    
    # Parameter set directory
    param_set_dir = parameter_test_dir
    
    # Get ice shelf subset (ice shelves from index 33 onwards, like in notebook)
    ice_shelf_indices = list(range(start_index, len(icems)))
    if max_shelves is not None:
        ice_shelf_indices = ice_shelf_indices[:max_shelves]
    
    logger.info(f"Processing {len(ice_shelf_indices)} ice shelves starting from index {start_index}")
    
    # Process each ice shelf
    processed_data = {
        'ice_shelves': [],
        'ice_shelf_indices': [],  # Store original indices
        'obs_data': [],
        'pred_params': [],
        'geometries': [],
        'centroids': []
    }
    
    for ice_shelf_idx in ice_shelf_indices:
        shelf_name = icems.iloc[ice_shelf_idx].name
        ice_shelf_geom = icems.iloc[ice_shelf_idx].geometry
        
        logger.info(f"Processing {shelf_name}...")
        
        try:
            # Load observational and prediction data
            obs_data = load_ice_shelf_data(ice_shelf_idx, icems, satobs, config)
            pred_params = load_predicted_data(shelf_name, ice_shelf_idx, icems, param_set_dir, config)
            
            # Get centroid
            centroid_x, centroid_y = get_ice_shelf_centroid(ice_shelf_geom)
            
            # Store data
            processed_data['ice_shelves'].append(shelf_name)
            processed_data['ice_shelf_indices'].append(ice_shelf_idx)  # Store original index
            processed_data['obs_data'].append(obs_data)
            processed_data['pred_params'].append(pred_params)
            processed_data['geometries'].append(ice_shelf_geom)
            processed_data['centroids'].append((centroid_x, centroid_y))
            
        except Exception as e:
            logger.error(f"Error processing {shelf_name}: {e}")
            continue
    
    # Cache the processed data
    logger.info(f"Caching processed data to: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    return processed_data

def load_cached_data(cache_file):
    """
    Load previously processed and cached data.
    """
    logger.info(f"Loading cached data from: {cache_file}")
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def create_draft_dependence_map_visualization(parameter_set_name, parameter_test_dir=None, 
                                            output_dir=None, max_shelves=None, start_index=33,
                                            use_cache=False, cache_file=None):
    """
    Create Antarctica map with colored ice shelf regions and scatter plot insets.
    """
    
    logger.info(f"Creating draft dependence map visualization for parameter set: {parameter_set_name}")
    
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
    logger.info(f"Creating visualization for {n_shelves} ice shelves")
    
    # Create color map for ice shelves (fixed color generation)
    if n_shelves <= 5:
        colors = ['r', 'b', 'g', 'y', 'm'][:n_shelves]
    else:
        # Use colormap for more ice shelves - ensure proper format
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_shelves)))
        if n_shelves > 20:
            colors2 = plt.cm.Set3(np.linspace(0, 1, n_shelves - 20))
            colors = list(colors) + list(colors2)
    
    # Create main figure with Antarctica map
    fig = plt.figure(figsize=(20, 16))
    
    # Main map axis with South Polar Stereographic projection
    ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    
    # Set map extent for Antarctica
    ax_map.set_extent([-180, 180, -60, -90], ccrs.PlateCarree())
    
    # Plot base ice shelf regions (corrected - no transform needed, proper range)
    icems[33:133].plot(ax=ax_map, color='antiquewhite', linewidth=0, zorder=1)
    icems[33:133].boundary.plot(ax=ax_map, linewidth=0.3, color='lightsteelblue', zorder=2)
    
    # Add coastlines
    ax_map.coastlines(resolution='10m', linewidth=0.5)
    
    # Use the stored ice shelf indices directly (with backwards compatibility)
    if 'ice_shelf_indices' in processed_data:
        ice_shelf_indices = processed_data['ice_shelf_indices']
        logger.info(f"Using stored ice shelf indices: {ice_shelf_indices}")
    else:
        # Backwards compatibility: reconstruct indices for old cache files
        logger.info("Old cache format detected, reconstructing ice shelf indices...")
        ice_shelf_indices = []
        for i, shelf_name in enumerate(processed_data['ice_shelves']):
            try:
                idx = icems[icems['name'] == shelf_name].index[0]
                ice_shelf_indices.append(idx)
            except:
                logger.warning(f"Could not find index for {shelf_name}")
                # Calculate based on start_index + position
                calculated_idx = start_index + i
                ice_shelf_indices.append(calculated_idx)
        logger.info(f"Reconstructed ice shelf indices: {ice_shelf_indices}")
    
    # Calculate geographically-aware inset positions
    inset_positions = calculate_geographic_inset_positions(
        processed_data['centroids'], processed_data['ice_shelves'], n_shelves
    )
    
    # Create inset scatter plots AND plot colored ice shelves
    processed_count = 0
    plotted_ice_shelves = []  # Track which ice shelves actually get plotted
    
    for i, (shelf_name, obs_data, pred_params, position) in enumerate(
        zip(processed_data['ice_shelves'], processed_data['obs_data'], 
            processed_data['pred_params'], inset_positions)
    ):
        
        if obs_data is None:
            logger.debug(f"{shelf_name}: Skipping inset - no observational data")
            continue
        
        shelf_color = colors[processed_count] if isinstance(colors[0], str) else colors[processed_count]
        
        # Plot the colored ice shelf on main map (only for shelves with data)
        idx = ice_shelf_indices[i]
        if idx is not None:
            icems.loc[[idx]].plot(ax=ax_map, color=shelf_color, linewidth=0.4, zorder=3)
            plotted_ice_shelves.append(shelf_name)
        
        # Create inset axes
        left, bottom, width, height = position
        inset_ax = fig.add_axes([left, bottom, width, height])
        
        # Plot scatter plot in inset (with index in title)
        plot_ice_shelf_inset(obs_data, pred_params, shelf_name, idx, shelf_color, inset_ax)
        
        processed_count += 1
    
    logger.info(f"Plotted colored ice shelves: {plotted_ice_shelves}")
    
    # Add title and labels
    ax_map.set_title(f'Draft Dependence Analysis: {parameter_set_name}\n'
                    f'Antarctica Ice Shelves with Melt Rate vs Draft Depth Insets\n'
                    f'Black crosses: Observed, Colored dots: Predicted', 
                    fontsize=16, pad=20)
    
    # Add a legend explaining the inset plots
    legend_text = ('Inset plots show melt rate (x-axis) vs draft depth (y-axis)\n'
                  'Black crosses: Observed data\n'
                  'Colored dots: Predicted data\n'
                  'Red dashed line: Draft threshold (if applicable)\n'
                  'Geographically proximate ice shelves have adjacent insets')
    
    fig.text(0.02, 0.02, legend_text, fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    output_file = output_dir / f"draft_dependence_map_{parameter_set_name}.png"
    logger.info(f"Saving plot to: {output_file}")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    logger.info(f"Map visualization saved to: {output_file}")
    
    # Save high-res version  
    output_file_hires = output_dir / f"draft_dependence_map_{parameter_set_name}_hires.png"
    plt.savefig(output_file_hires, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    logger.info(f"Processed {processed_count} ice shelves with data out of {n_shelves} total")
    
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
    
    logger.info("DRAFT DEPENDENCE MAP VISUALIZATION")
    
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
    
    logger.info("Map visualization complete!")

if __name__ == "__main__":
    main()