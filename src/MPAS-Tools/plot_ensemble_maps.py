#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble snapshot map plotting for MALI.
Plots min, max, mean, std, range for specified variables at specified years.
Overlays grounding lines for all runs in ensemble.
"""

import numpy as np
from netCDF4 import Dataset
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="Ensemble map plots for MALI.")
parser.add_argument("--ensemble_files", required=True, help="Comma-separated ensemble stats NetCDF files (one per year).")
parser.add_argument("--years", required=True, help="Comma-separated years.")
parser.add_argument("--variables", required=True, help="Comma-separated variables.")
parser.add_argument("--mesh_files", required=True, help="Comma-separated mesh files for GL overlays (one per run, start year).")
parser.add_argument("--run_names", required=True, help="Comma-separated run names (for legend).")
parser.add_argument("--save_base", required=True, help="Base filename for saving figures.")

args = parser.parse_args()
ensemble_files = args.ensemble_files.split(',')
years = [int(y) for y in args.years.split(',')]
variables = args.variables.split(',')
mesh_files = args.mesh_files.split(',')
run_names = args.run_names.split(',')
save_base = args.save_base

print(f"Processing {len(ensemble_files)} ensemble files for years: {years}")
print(f"Variables: {variables}")
print(f"Using {len(mesh_files)} mesh files for grounding line overlays")

stat_types = ["mean", "min", "max", "range", "std"]
defaultColors = {'thickness': 'Blues', 'surfaceSpeed': 'plasma', 'dhdt': 'RdBu'}
sec_per_year = 60. * 60. * 24. * 365.

# Define bit mask values
groundingLineValue = 256
initialExtentValue = 1
floatValue = 4

def dist(i1, i2, xCell, yCell):
    """Helper function to calculate distance between cells"""
    return ((xCell[i1]-xCell[i2])**2 + (yCell[i1]-yCell[i2])**2)**0.5

def create_custom_colormap():
    """Create custom diverging colormap"""
    colors = ['Navajowhite', 'Darkorange', 'Darkred', 'white', 
              'Lightsteelblue', 'Royalblue', 'Navy']
    return LinearSegmentedColormap.from_list("custom", colors, N=200)

# Load mesh info and grounding lines for GL overlays
print("Loading grounding line information...")
grounding_lines = []
gl_colors = cm.tab10(np.linspace(0, 1, len(mesh_files)))  # Generate colors for each run

for i, mesh_fn in enumerate(mesh_files):
    if not mesh_fn.strip():
        continue
    try:
        print(f"Loading mesh file: {mesh_fn}")
        m = Dataset(mesh_fn, 'r')
        
        # Handle different dimension structures
        xCell = m.variables["xCell"][0] if m.variables["xCell"].ndim > 1 else m.variables["xCell"][:]
        yCell = m.variables["yCell"][0] if m.variables["yCell"].ndim > 1 else m.variables["yCell"][:]
        
        if "cellMask" in m.variables:
            cellMask = m.variables["cellMask"][:]
            if cellMask.ndim > 1:
                cellMask = cellMask[0]  # Take first time step
            gl_mask = (cellMask & groundingLineValue) // groundingLineValue
            initial_extent_mask = (cellMask & initialExtentValue) // initialExtentValue
        else:
            gl_mask = None
            initial_extent_mask = None
            print(f"WARNING: No cellMask found in {mesh_fn}")
        
        grounding_lines.append({
            'x': xCell, 
            'y': yCell, 
            'gl_mask': gl_mask,
            'extent_mask': initial_extent_mask,
            'color': gl_colors[i],
            'run_name': run_names[i] if i < len(run_names) else f'Run_{i+1}'
        })
        m.close()
        
    except Exception as e:
        print(f"ERROR loading {mesh_fn}: {e}")
        continue

print(f"Successfully loaded {len(grounding_lines)} grounding line datasets")

# Process each variable
for variable in variables:
    print(f"\nProcessing variable: {variable}")
    
    for stat in stat_types:
        print(f"  Processing statistic: {stat}")
        
        # Create figure for this variable/stat across all years
        fig_width = min(20, 5*len(years))  # Cap at reasonable width
        fig = plt.figure(figsize=(fig_width, 8))
        
        if len(years) == 1:
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 0.2])
        else:
            gs = gridspec.GridSpec(1, len(years) + 1, width_ratios=[1]*len(years) + [0.1])
        
        axs = []
        vdata = []
        units = "unknown"
        successful_loads = 0
        
        # Load data for all years
        for i, (stats_file, year) in enumerate(zip(ensemble_files, years)):
            if not stats_file.strip():
                vdata.append(None)
                axs.append(plt.subplot(gs[i]) if len(years) > 1 else plt.subplot(gs[0]))
                continue
                
            try:
                print(f"    Loading {stats_file}")
                f = Dataset(stats_file, 'r')
                varname = f"{variable}_{stat}"
                
                if varname not in f.variables:
                    print(f"    WARNING: {varname} not found in {stats_file}")
                    vdata.append(None)
                    axs.append(plt.subplot(gs[i]) if len(years) > 1 else plt.subplot(gs[0]))
                    f.close()
                    continue
                
                arr = f.variables[varname][:]
                
                # Get units
                if 'units' in f.variables[varname].ncattrs():
                    units = f.variables[varname].units
                elif variable == 'surfaceSpeed':
                    units = 'm yr^{-1}'
                elif variable == 'thickness':
                    units = 'm'
                elif variable == 'dhdt':
                    units = 'm yr^{-1}'
                else:
                    units = "unknown"
                
                # Handle dimensionality
                if arr.ndim == 1:
                    arr = arr.reshape((1, np.shape(arr)[0]))
                
                # Convert speed units if needed
                if 'Speed' in variable and units != 'm yr^{-1}':
                    arr *= sec_per_year
                    units = 'm yr^{-1}'
                
                vdata.append(arr[0,:])
                axs.append(plt.subplot(gs[i]) if len(years) > 1 else plt.subplot(gs[0]))
                successful_loads += 1
                f.close()
                
            except Exception as e:
                print(f"    ERROR loading {stats_file}: {e}")
                vdata.append(None)
                axs.append(plt.subplot(gs[i]) if len(years) > 1 else plt.subplot(gs[0]))
                continue
        
        if successful_loads == 0:
            print(f"    No data loaded for {variable}_{stat}, skipping...")
            plt.close(fig)
            continue
        
        # Determine colormap and normalization
        all_valid = [a for a in vdata if a is not None]
        if not all_valid:
            print(f"    No valid data for {variable}_{stat}, skipping...")
            plt.close(fig)
            continue
            
        all_flat = np.concatenate(all_valid)
        all_flat = all_flat[~np.isnan(all_flat)]
        
        if len(all_flat) == 0:
            print(f"    All data is NaN for {variable}_{stat}, skipping...")
            plt.close(fig)
            continue
        
        vmin = np.nanquantile(all_flat, 0.01)
        vmax = np.nanquantile(all_flat, 0.99)
        
        # Choose colormap and normalization
        if variable == 'dhdt' or stat == 'range':
            # Use diverging colormap centered at zero
            max_abs = max(abs(vmin), abs(vmax))
            vmin = -max_abs
            vmax = max_abs
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
            cmap = create_custom_colormap()
        elif variable == 'surfaceSpeed':
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap('plasma')
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(defaultColors.get(variable, 'viridis'))
        
        print(f"    Color range: {vmin:.3f} to {vmax:.3f}")
        
        # Plot panels for each year
        mappables = []
        for i, (arr, ax, year) in enumerate(zip(vdata, axs, years)):
            if arr is None:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=14)
                ax.set_title(f"Year {year}")
                continue
            
            try:
                # Load mesh information for this year
                m = Dataset(ensemble_files[i], 'r')
                xCell = m.variables["xCell"][0] if m.variables["xCell"].ndim > 1 else m.variables["xCell"][:]
                yCell = m.variables["yCell"][0] if m.variables["yCell"].ndim > 1 else m.variables["yCell"][:]
                dcEdge = m.variables["dcEdge"][0] if m.variables["dcEdge"].ndim > 1 else m.variables["dcEdge"][:]
                
                # Create triangulation
                triang = tri.Triangulation(xCell, yCell)
                triMask = np.zeros(len(triang.triangles), dtype=bool)
                maxDist = np.max(dcEdge) * 2.0
                
                for t in range(len(triang.triangles)):
                    thisTri = triang.triangles[t, :]
                    if dist(thisTri[0], thisTri[1], xCell, yCell) > maxDist:
                        triMask[t] = True
                    if dist(thisTri[1], thisTri[2], xCell, yCell) > maxDist:
                        triMask[t] = True
                    if dist(thisTri[0], thisTri[2], xCell, yCell) > maxDist:
                        triMask[t] = True
                
                triang.set_mask(triMask)
                
                # Plot the main variable field
                h = ax.tripcolor(triang, arr, cmap=cmap, shading='flat', norm=norm)
                mappables.append(h)
                
                # Overlay grounding lines from all runs
                for gl_info in grounding_lines:
                    if gl_info['gl_mask'] is not None:
                        try:
                            # Create triangulation for this grounding line
                            gl_triang = tri.Triangulation(gl_info['x'], gl_info['y'])
                            
                            # Apply distance-based mask
                            gl_triMask = np.zeros(len(gl_triang.triangles), dtype=bool)
                            for t in range(len(gl_triang.triangles)):
                                thisTri = gl_triang.triangles[t, :]
                                if (dist(thisTri[0], thisTri[1], gl_info['x'], gl_info['y']) > maxDist or
                                    dist(thisTri[1], thisTri[2], gl_info['x'], gl_info['y']) > maxDist or
                                    dist(thisTri[0], thisTri[2], gl_info['x'], gl_info['y']) > maxDist):
                                    gl_triMask[t] = True
                            
                            gl_triang.set_mask(gl_triMask)
                            
                            # Plot grounding line
                            ax.tricontour(gl_triang, gl_info['gl_mask'], 
                                        levels=[0.9999], colors=[gl_info['color']], 
                                        linestyles='solid', linewidths=1.5)
                        except Exception as e:
                            print(f"    WARNING: Could not plot grounding line for {gl_info['run_name']}: {e}")
                
                ax.set_title(f"{variable} [{stat}]\nYear {year}")
                ax.set_aspect('equal')
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                
                m.close()
                
            except Exception as e:
                print(f"    ERROR plotting year {year}: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=10)
                continue
        
        # Add colorbar
        if mappables:
            if len(years) == 1:
                cbar_ax = plt.subplot(gs[1])
            else:
                cbar_ax = plt.subplot(gs[-1])
            cbar = Colorbar(ax=cbar_ax, mappable=mappables[0], orientation='vertical')
            cbar.set_label(f"{variable} [{stat}] ({units})", rotation=270, labelpad=20)
        
        # Add legend for grounding lines
        if len(grounding_lines) > 0 and axs:
            legend_elements = []
            for gl_info in grounding_lines:
                legend_elements.append(plt.Line2D([0], [0], color=gl_info['color'], 
                                           lw=1.5, label=f"GL {gl_info['run_name']}"))