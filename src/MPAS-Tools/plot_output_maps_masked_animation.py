#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 9, 2022

@author: Trevor Hillebrand, Matthew Hoffman

Modified script to plot snapshot maps of MALI output for all available years
with consistent colorbars across all years for animation purposes.
"""
import numpy as np
from netCDF4 import Dataset
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import os
import re


print("** Gathering information.  (Invoke with --help for more details. All arguments are optional)")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-r", dest="runs", default=None, metavar="FILENAME",
                    help="path to .nc file or dir containing output.nc \
                          file (strings separated by commas; no spaces)")
parser.add_argument("-t", dest="timeLevels", default="-1",
                    help="integer time levels at which to plot \
                          (int separated by commas; no spaces)")
parser.add_argument("-v", dest="variables", default='thickness',
                    help="variable(s) to plot (list separated by commas; no spaces)")
parser.add_argument("-l", dest="log_plot", default=None,
                    help="Whether to plot the log10 of each variable \
                          (True or False list separated by commas; no spaces)")
parser.add_argument("-c", dest="colormaps", default=None,
                    help="colormaps to use for plotting (list separated by commas \
                          , no spaces). This overrides default colormaps.")
parser.add_argument("--vmin", dest="vmin", default=None,
                    help="minimum value(s) for colorbar(s) \
                          (list separated by commas; no spaces)")
parser.add_argument("--vmax", dest="vmax", default=None,
                    help="maximum value(s) for colorbar(s) \
                          (list separated by commas; no spaces)")
parser.add_argument("-m", dest="mesh", default=None, metavar="FILENAME",
                    help="Optional input file(s) containing mesh variables. This \
                          is useful when plotting from files that have no mesh \
                          variables to limit file size. Define either one mesh file \
                          to be applied to all run files, or one mesh file per \
                          run file (list separated by commas; no spaces)")
parser.add_argument("-m2", dest="mesh2", default=None, metavar="FILENAME",
                    help="Second input file(s) containing mesh variables for end years.")
parser.add_argument("-s", dest="saveNames", default=None, metavar="FILENAME",
                    help="filename base for saving. If empty or None, will plot \
                          to screen instead of saving.")
parser.add_argument("--all_years", dest="all_years", default=None,
                    help="List of all years being processed (space-separated)")

args = parser.parse_args()

# Parse arguments
runs = args.runs.split(',') if args.runs else []
variables = args.variables.split(',')
timeLevs = args.timeLevels.split(',')
timeLevs = [int(i) for i in timeLevs]

# Parse years if provided
if args.all_years:
    all_years = [int(year) for year in args.all_years.split()]
else:
    all_years = []

sec_per_year = 60. * 60. * 24. * 365.
rhoi = 910.
rhosw = 1028.

if args.vmin is not None:
    vmins = args.vmin.split(',')
else:
    vmins = [None] * len(variables)

if args.vmax is not None:
    vmaxs = args.vmax.split(',')
else:
    vmaxs = [None] * len(variables)

if args.log_plot is not None:
    log_plot = args.log_plot.split(',')
else:
    log_plot = [False] * len(variables)

if args.colormaps is not None:
    colormaps = args.colormaps.split(',')
else:
    colormaps = ['viridis'] * len(variables)

# Parse mesh files
if args.mesh is not None:
    mesh = args.mesh.split(',')
    if len(mesh) == 1 and len(runs) > 1:
        mesh *= len(runs)
    assert len(mesh) == len(runs), (f"Define either one master mesh file, "
                                   f"or one mesh file per run file. "
                                   f"You defined {len(mesh)} files and "
                                   f"{len(runs)} run files.")
else:
    mesh = runs

# Parse mesh2 files for end years
if args.mesh2 is not None:
    mesh2_files = args.mesh2.split(',')
else:
    print("Error: Second mesh files (-m2) are required.")
    exit(1)

# Set bitmask values
initialExtentValue = 1
dynamicValue = 2
floatValue = 4
groundingLineValue = 256

# Set up a dictionary of default colormaps for common variables
defaultColors = {'thickness': 'Blues',
                 'surfaceSpeed': 'plasma',
                 'basalSpeed': 'plasma',
                 'bedTopography': 'BrBG',
                 'floatingBasalMassBalApplied': 'cividis',
                 'dhdt': 'RdBu_r'}

if args.colormaps is not None:
    colormaps = args.colormaps.split(',')
else:
    colormaps = []
    for variable in variables:
        if variable in defaultColors.keys():
            colormaps.append(defaultColors[variable])
        else:
            colormaps.append('viridis')

divColorMaps = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

def dist(i1, i2, xCell, yCell):
    """Helper distance function"""
    dist = ((xCell[i1]-xCell[i2])**2 + (yCell[i1]-yCell[i2])**2)**0.5
    return dist

def create_custom_cmap(low_colors, high_colors):
    """Create custom diverging colormap"""
    low_colors = list(reversed(low_colors))
    colors = low_colors + ['white'] + high_colors
    n_bins = 200
    return LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

# First pass: Calculate global min/max for consistent colorbars
print("First pass: Calculating global min/max values for consistent colorbars...")
global_vmins = {}
global_vmaxs = {}

all_var_data = {}
for variable in variables:
    all_var_data[variable] = []

# Collect all data for each variable across all files
for ii, run in enumerate(runs):
    try:
        f = Dataset(run, 'r')
        f.set_auto_mask(False)
        
        # Get variable data
        for variable in variables:
            if variable == 'observedSpeed':
                var_to_plot = np.sqrt(f.variables['observedSurfaceVelocityX'][:]**2 +
                                      f.variables['observedSurfaceVelocityY'][:]**2)
            else:
                var_to_plot = f.variables[variable][:]
            
            if len(np.shape(var_to_plot)) == 1:
                var_to_plot = var_to_plot.reshape((1, np.shape(var_to_plot)[0]))
            
            if 'Speed' in variable:
                var_to_plot *= sec_per_year
            
            # Apply log if requested
            log_this_var = log_plot[variables.index(variable)]
            if log_this_var == 'True':
                var_to_plot = np.log10(var_to_plot)
                var_to_plot[np.isinf(var_to_plot)] = np.nan
            
            all_var_data[variable].append(var_to_plot[timeLevs, :])
        
        f.close()
    except Exception as e:
        print(f"Error processing {run} for global min/max: {e}")
        continue

# Calculate global min/max for each variable
for vi, variable in enumerate(variables):
    if all_var_data[variable]:
        all_data = np.concatenate(all_var_data[variable], axis=1)
        
        if vmins[vi] in ['None', None]:
            global_vmin = np.nanquantile(all_data, 0.01)
            if 'Speed' in variable and log_plot[vi] == 'True':
                global_vmin = max(global_vmin, -1.)
        else:
            global_vmin = float(vmins[vi])
            
        if vmaxs[vi] in ['None', None]:
            global_vmax = np.nanquantile(all_data, 0.99)
        else:
            global_vmax = float(vmaxs[vi])
        
        global_vmins[variable] = global_vmin
        global_vmaxs[variable] = global_vmax
        
        print(f"Global range for {variable}: {global_vmin:.4f} to {global_vmax:.4f}")

print("Second pass: Creating individual plots with consistent colorbars...")

# Second pass: Create individual plots for each file
for ii, run in enumerate(runs):
    try:
        f = Dataset(run, 'r')
        if 'daysSinceStart' in f.variables.keys():
            yr = f.variables['daysSinceStart'][:] / 365.
        else:
            yr = [0.]
        
        f.set_auto_mask(False)
        
        # Get mesh geometry
        if args.mesh is not None:
            m = Dataset(mesh[ii], 'r')
        else:
            m = f
        
        # Load mesh2 file for this run
        m2 = Dataset(mesh2_files[ii], 'r')
        
        # Extract year from filename for title
        year_match = re.search(r'dhdt_(\d+)yr', run)
        if year_match:
            nyears = int(year_match.group(1))
            end_year = 2000 + nyears
        else:
            end_year = "unknown"
        
        xCell = m.variables["xCell"][0] if len(m.variables["xCell"].shape) > 1 else m.variables["xCell"][:]
        yCell = m.variables["yCell"][0] if len(m.variables["yCell"].shape) > 1 else m.variables["yCell"][:]
        dcEdge = m.variables["dcEdge"][0] if len(m.variables["dcEdge"].shape) > 1 else m.variables["dcEdge"][:]
        
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
        
        # Set up figure
        fig = plt.figure(figsize=(12, 8))
        nRows = len(variables)
        nCols = len(timeLevs) + 1
        
        gs = gridspec.GridSpec(nRows, nCols,
                              height_ratios=[1] * nRows,
                              width_ratios=[1] * (nCols - 1) + [0.1])
        
        axs = []
        cbar_axs = []
        for row in np.arange(0, nRows):
            cbar_axs.append(plt.subplot(gs[row, -1]))
            for col in np.arange(0, nCols-1):
                if axs == []:
                    axs.append(plt.subplot(gs[row, col]))
                else:
                    axs.append(plt.subplot(gs[row, col], sharex=axs[0], sharey=axs[0]))
        
        # Calculate masks
        calc_mask = False
        if 'cellMask' in m.variables.keys() and 'cellMask' in m2.variables.keys():
            calc_mask = True
            cellMask_2000 = m.variables["cellMask"][:]
            cellMask_end = m2.variables["cellMask"][:]
            
            # Create plot mask
            grounded_end = (cellMask_end & floatValue) == 0
            floating_2000 = (cellMask_2000 & floatValue) == floatValue
            floating_end = (cellMask_end & floatValue) == floatValue
            plot_mask = grounded_end | (floating_end & ~floating_2000)
            
            groundingLineMask_2000 = (cellMask_2000 & groundingLineValue) // groundingLineValue
            groundingLineMask_end = (cellMask_end & groundingLineValue) // groundingLineValue
            initialExtentMask = (cellMask_2000 & initialExtentValue) // initialExtentValue
        
        # Process each variable
        for row, variable in enumerate(variables):
            if variable == 'observedSpeed':
                var_to_plot = np.sqrt(f.variables['observedSurfaceVelocityX'][:]**2 +
                                      f.variables['observedSurfaceVelocityY'][:]**2)
            else:
                var_to_plot = f.variables[variable][:]
            
            if len(np.shape(var_to_plot)) == 1:
                var_to_plot = var_to_plot.reshape((1, np.shape(var_to_plot)[0]))
            
            if 'Speed' in variable:
                units = 'm yr^{-1}'
                var_to_plot *= sec_per_year
            else:
                try:
                    units = f.variables[variable].units
                except AttributeError:
                    units = 'no-units'
            
            log_this_var = log_plot[row]
            if log_this_var == 'True':
                var_to_plot = np.log10(var_to_plot)
                var_to_plot[np.isinf(var_to_plot)] = np.nan
                colorbar_label_prefix = 'log10 '
            else:
                colorbar_label_prefix = ''
            
            # Use global min/max values
            vmin = global_vmins[variable]
            vmax = global_vmaxs[variable]
            
            # Create custom colormap
            custom_cmap = create_custom_cmap(['Navajowhite', 'Darkorange', 'Darkred'], 
                                           ['Lightsteelblue', 'Royalblue', 'Navy'])
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            
            # Plot each time level
            varPlots = []
            for col, timeLev in enumerate(timeLevs):
                index = row * (nCols - 1) + col
                
                # Apply mask if available
                if calc_mask:
                    tri_plot_mask = plot_mask[timeLev, triang.triangles].any(axis=1)
                    current_triMask = triMask | ~tri_plot_mask
                    triang.set_mask(current_triMask)
                
                # Create the plot
                varPlot = axs[index].tripcolor(triang, var_to_plot[timeLev, :], 
                                             cmap=custom_cmap, shading='flat', norm=norm)
                varPlots.append(varPlot)
                
                # Add contours
                if calc_mask:
                    axs[index].tricontour(triang, groundingLineMask_2000[timeLev, :],
                                        levels=[0.9999], colors='darkgreen',
                                        linestyles='solid', linewidths=0.9)
                    axs[index].tricontour(triang, groundingLineMask_end[timeLev, :],
                                        levels=[0.9999], colors='darkviolet',
                                        linestyles='solid', linewidths=0.9)
                    axs[index].tricontour(triang, initialExtentMask[timeLev, :],
                                        levels=[0.9999], colors='black',
                                        linestyles='solid', linewidths=0.8)
                
                axs[index].set_aspect('equal')
                axs[index].set_title(f'year = {yr[timeLev]:0.2f}')
            
            # Add colorbar for this variable
            cbar = Colorbar(ax=cbar_axs[row], mappable=varPlots[0], orientation='vertical',
                           label=f'{colorbar_label_prefix}{variable} (${units}$)')
            
            # Add legend to the last subplot of each row
            if calc_mask:
                legend_elements = [
                    plt.Line2D([0], [0], color='darkgreen', lw=1.5, label='2000 GL'),
                    plt.Line2D([0], [0], color='darkviolet', lw=1.5, label=f'{end_year} GL')
                ]
                last_ax_in_row = axs[row * (nCols - 1) + (nCols - 2)]
                last_ax_in_row.legend(handles=legend_elements, loc='lower right', fontsize='small')
        
        fig.tight_layout()
        
        # Save figure with year-specific filename
        if args.saveNames is not None:
            if end_year != "unknown":
                save_filename = f'{args.saveNames}_{end_year}.png'
            else:
                save_filename = f'{args.saveNames}_{ii:04d}.png'
            fig.savefig(save_filename, dpi=400, bbox_inches='tight')
            print(f"Saved: {save_filename}")
        
        f.close()
        m.close()
        m2.close()
        plt.close(fig)  # Close figure to free memory
        
    except Exception as e:
        print(f"Error processing {run}: {e}")
        continue

print("All plots created successfully!")
if not args.saveNames:
    plt.show()