#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 9, 2022

@author: Trevor Hillebrand, Matthew Hoffman

Script to plot snapshot maps of MALI output for an arbitrary number of files,
variables, and output times. There is no requirement for all output files
to be on the same mesh. Each output file gets its own figure, each
variable gets its own row, and each time gets its own column. Three contours
are automatically plotted, showing intial ice extent (black), initial
grounding-line position (grey), and grounding-line position at the desired
time (white).

// TODO - test todo

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
                    help="Second input file containing mesh variables for year 2300.")
parser.add_argument("-s", dest="saveNames", default=None, metavar="FILENAME",
                    help="filename for saving. If empty or None, will plot \
                          to screen instead of saving.")

args = parser.parse_args()
runs = args.runs.split(',') # split run directories into list
variables = args.variables.split(',')
if args.vmin is not None:
    vmins = args.vmin.split(',')
else:
    vmins = [None] * len(variables)

if args.vmax is not None:
    vmaxs = args.vmax.split(',')
else:
    vmaxs = [None] * len(variables)

timeLevs = args.timeLevels.split(',')  # split time levels into list
# convert timeLevs to list of ints
timeLevs = [int(i) for i in timeLevs]
sec_per_year = 60. * 60. * 24. * 365.
rhoi = 910.
rhosw = 1028.

if args.log_plot is not None:
    log_plot = args.log_plot.split(',')
else:
    log_plot = [False] * len(variables)

if args.colormaps is not None:
    colormaps = args.colormaps.split(',')
else:
    colormaps = ['viridis'] * len(variables)

# If separate mesh file(s) specified, use those.
# Otherwise, get mesh variables from runs files.
# If -m is used, there will either be one 'master'
# mesh file that is used for all run files, or
# there will be one mesh file per run file.
if args.mesh is not None:
   mesh = args.mesh.split(',')
   if len(mesh) == 1 and len(runs) > 1:
      mesh *= len(runs)
   assert len(mesh) == len(runs), ("Define either one master mesh file, "
                                   "or one mesh file per run file. "
                                   f"You defined {len(mesh)} files and "
                                   f"{len(runs)} run files.")
else:
   mesh = runs

if args.mesh2 is not None:
    m2 = Dataset(args.mesh2, 'r')
else:
    print("Error: Second mesh file (-m2) is required.")
    exit(1)



if args.saveNames is not None:
    saveNames = args.saveNames.split(',')

# Set up a dictionary of default colormaps for common variables.
# These can be overridden by the -c flag.
defaultColors = {'thickness' : 'Blues',
                 'surfaceSpeed' : 'plasma',
                 'basalSpeed' : 'plasma',
                 'bedTopography' : 'BrBG',
                 'floatingBasalMassBalApplied' : 'cividis'
                }

if args.colormaps is not None:
    colormaps = args.colormaps.split(',')
else:
    colormaps = []
    for variable in variables:
        if variable in defaultColors.keys():
            colormaps.append(defaultColors[variable])
        else:
            # All other variables default to viridis
            colormaps.append('viridis')

# Set bitmask values
initialExtentValue = 1
dynamicValue = 2
floatValue = 4
groundingLineValue = 256


# List of diverging colormaps for use in plotting bedTopography.
# I don't see a way around hard-coding this.
divColorMaps = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

def dist(i1, i2, xCell, yCell):  # helper distance fn
    dist = ((xCell[i1]-xCell[i2])**2 + (yCell[i1]-yCell[i2])**2)**0.5
    return dist

# Loop over runs
# Each run gets its own figure
# Each variable gets its own row
# Each time level gets its own column
varPlot = {}
figs = {}
gs = {}
for ii, run in enumerate(runs):
    if '.nc' not in run:
        run = run + '/output_flux_all_timesteps_{}.nc'.format(ii+2000)
    f = Dataset(run, 'r')
    if 'daysSinceStart' in f.variables.keys():
        yr = f.variables['daysSinceStart'][:] / 365.
    else:
        yr = [0.]

    f.set_auto_mask(False)

    # Get mesh geometry and calculate triangulation. 
    # It would be more efficient to do this outside
    # this loop if all runs are on the same mesh, but we
    # want this to be as general as possible.
    if args.mesh is not None:
       m = Dataset(mesh[ii], 'r')
    else:
       m = f  # use run file for mesh variables

    xCell = m.variables["xCell"][:]
    yCell = m.variables["yCell"][:]
    dcEdge = m.variables["dcEdge"][:]
    # The above should work but an error to do with how NETCDF reads record Time dimension occurs, or the way the file is created (using nco)
    # Temporary fix below: (TODO: Clean up)
    xCell = m.variables["xCell"][0]
    yCell = m.variables["yCell"][0]
    dcEdge = m.variables["dcEdge"][0]
    #if args.mesh is not None:
    #   m.close()

    triang = tri.Triangulation(xCell, yCell)
    triMask = np.zeros(len(triang.triangles), dtype=bool)
    # Maximum distance in m of edges between points.
    # Make twice dcEdge to be safe
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
    print(f"Unique values in triMask: {triMask}")

    # set up figure for this run
    figs[run] = plt.figure()
    figs[run].suptitle(run)
    nRows = len(variables)
    nCols = len(timeLevs) + 1

    # last column is for colorbars
    gs[run] = gridspec.GridSpec(nRows, nCols,
                           height_ratios=[1] * nRows,
                           width_ratios=[1] * (nCols - 1) + [0.1])
    axs = []
    cbar_axs = []
    for row in np.arange(0, nRows):
        cbar_axs.append(plt.subplot(gs[run][row,-1]))
        for col in np.arange(0, nCols-1):
            if axs == []:
                axs.append(plt.subplot(gs[run][row, col]))
            else:
                axs.append(plt.subplot(gs[run][row, col], sharex=axs[0], sharey=axs[0]))

    varPlot[run] = {}  # is a dict of dicts too complicated?
    cbars = []
    # Loop over variables
    for row, (variable, log, colormap, cbar_ax, vmin, vmax) in enumerate(
        zip(variables, log_plot, colormaps, cbar_axs, vmins, vmaxs)):
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
                units='no-units'

        if log == 'True':
            var_to_plot = np.log10(var_to_plot)
            # Get rid of +/- inf values that ruin vmin and vmax
            # calculations below.
            var_to_plot[np.isinf(var_to_plot)] = np.nan
            colorbar_label_prefix = 'log10 '
        else:
            colorbar_label_prefix = ''

        varPlot[run][variable] = []

        # Set lower and upper bounds for plotting
        if vmin in ['None', None]:
            # 0.1 m/yr is a pretty good lower bound for speed
            first_quant = np.nanquantile(var_to_plot[timeLevs, :], 0.01)
            if 'Speed' in variable and log == 'True':
                vmin = max(first_quant, -1.)
            else:
                vmin = first_quant
        if vmax in ['None', None]:
            vmax = np.nanquantile(var_to_plot[timeLevs, :], 0.99)
        # Plot bedTopography on an asymmetric colorbar if appropriate
        if ( (variable == 'bedTopography') and
             (np.nanquantile(var_to_plot[timeLevs, :], 0.99) > 0.) and
             (colormap in divColorMaps) ):
            norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0.)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)


        if 'cellMask' in m.variables.keys() and 'cellMask' in m2.variables.keys():
            calc_mask = True
            cellMask_2000 = m.variables["cellMask"][:]
            cellMask_2300 = m2.variables["cellMask"][:]
            # Print unique values of cellMask_2000 and cellMask_2300
            unique_values_2000 = np.unique(cellMask_2000)
            print(f"Unique values in cellMask_2000: {unique_values_2000}")

            unique_values_2300 = np.unique(cellMask_2300)
            print(f"Unique values in cellMask_2300: {unique_values_2300}")

            mask = (cellMask_2000 == 1) | ((cellMask_2000 & floatValue) == 0) & ((cellMask_2300 & floatValue) == floatValue)
            mask = mask.astype(bool)
            unique_values_mask = np.unique(mask)
            print(f"Unique values in mask: {unique_values_mask}")

            # Create masks for grounded ice and floating ice
            grounded_2000 = (cellMask_2000 & floatValue) == 0
            grounded_2300 = (cellMask_2300 & floatValue) == 0
            floating_2000 = (cellMask_2000 & floatValue) == floatValue
            floating_2300 = (cellMask_2300 & floatValue) == floatValue

            # Create a mask for areas to plot: grounded ice in 2300 or newly formed floating ice
            plot_mask = grounded_2300 | (floating_2300 & ~floating_2000)
            #plot_mask = floating_2300 & ~floating_2000

            groundingLineMask_2000 = (cellMask_2000 & groundingLineValue) // groundingLineValue
            groundingLineMask_2300 = (cellMask_2300 & groundingLineValue) // groundingLineValue
            initialExtentMask = (cellMask_2000 & initialExtentValue) // initialExtentValue

        else:
            print(f'cellMask not present in both mesh files; skipping mask calculation.')
            calc_mask = False

        # Loop over time levels
        for col, timeLev in enumerate(timeLevs):
            index = row * (nCols - 1) + col
            
            masked_var = np.ma.masked_where(~mask[timeLev, :], var_to_plot[timeLev, :])

            # Create a mask for newly formed floating ice
            #newFloatingMask = (floatMask[timeLev, :] == 0) & (initialFloatingMask == 1)
            # Mask both grounded ice and initial floating ice
            #masked_var = np.ma.masked_where(~newFloatingMask, var_to_plot[timeLev, :])

            # plot initial grounding line position, initial extent, and GL position at t=timeLev
            if calc_mask:
                gl_2000 = axs[index].tricontour(triang, groundingLineMask_2000[timeLev, :],
                                              levels=[0.9999], colors='darkgreen',
                                              linestyles='solid', linewidths=0.9)
                gl_2300 = axs[index].tricontour(triang, groundingLineMask_2300[timeLev, :],
                                              levels=[0.9999], colors='darkviolet',
                                              linestyles='solid', linewidths=0.9)
                axs[index].tricontour(triang, initialExtentMask[timeLev, :],
                                      levels=[0.9999], colors='black',
                                      linestyles='solid', linewidths=0.8)                

            # Plot 2D field at each desired time. Use quantile range of 0.01-0.99 to cut out
            # outliers. Could improve on this by accounting for areaCell, as currently all cells
            # are weighted equally in determining vmin and vmax.

            #triang = tri.Triangulation(xCell, yCell)
            #tri_float_mask = mask[timeLev, triang.triangles].any(axis=1)
            #triMask = tri_float_mask | triMask


            tri_plot_mask = plot_mask[timeLev, triang.triangles].any(axis=1)
            triMask = triMask | ~tri_plot_mask  # Invert because triMask True means "do not plot"


            # Maximum distance in m of edges between points.
            # Make twice dcEdge to be safe
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
            print(f"Unique values in triMask: {tri_plot_mask}")

            #if vmin in ['None', None]:
            
            #    first_quant = np.nanquantile(var_to_plot[timeLevs, :], 0.01)
            #if 'Speed' in variable and log == 'True':
            #    vmin = max(first_quant, -1.)
            #else:
            #    vmin = first_quant
            #if vmax in ['None', None]:
            #    vmax = np.nanquantile(var_to_plot[timeLevs, :], 0.99)

            def create_diverging_cmap(low_color, high_color):
                colors = [low_color, 'white', high_color]
                n_bins = 200  # Number of color bins
                return LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

            def create_custom_cmap(low_colors, high_colors):
                # Reverse the low_colors list so that the darkest color is closest to white
                sns.set_palette("mako")
                low_colors = list(reversed(low_colors))
                # Combine the color lists with white in the center
                colors = low_colors + ['white'] + high_colors
                
                n_bins = 200  # Number of color bins
                return LinearSegmentedColormap.from_list("custom", colors, N=n_bins)


            #if variable in ['bedTopography', 'dhdt']:  # Add any variables that should use the diverging colormap
            # Create custom diverging colormap
            # custom_cmap = create_diverging_cmap(')

            custom_cmap = create_custom_cmap(['Navajowhite', 'Darkorange', 'Darkred'], ['Lightsteelblue', 'Royalblue', 'Navy'])

            # Find the maximum absolute value for symmetry
            #max_abs_val = max(abs(vmin), abs(vmax))

            # Create a TwoSlopeNorm with white at center
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            

            varPlot[run][variable].append(
                              axs[index].tripcolor(
                                 triang, var_to_plot[timeLev, :], cmap=custom_cmap,
                                 shading='flat', norm=norm))
            
            #floating_ice = axs[index].tripcolor(triang, mask[timeLev, :])

            axs[index].set_aspect('equal')
            axs[index].set_title(f'year = {yr[timeLev]:0.2f}')

        cbars.append(Colorbar(ax=cbar_ax, mappable=varPlot[run][variable][0], orientation='vertical',
                 label=f'{colorbar_label_prefix}{variable} (${units}$)'))
        #cbars.append(Colorbar(ax=cbar_ax, mappable=floating_ice, orientation='vertical', label='floating ice'))

        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=1.5, label='2000 GL'),
            plt.Line2D([0], [0], color='purple', lw=1.5, label='2300 GL')
            ]
    
        # Add legend to the last subplot of each row
        last_ax_in_row = axs[row * (nCols - 1) + (nCols - 2)]
        last_ax_in_row.legend(handles=legend_elements, loc='lower right', fontsize='small')
        
    figs[run].tight_layout()
    if args.saveNames is not None:
        figs[run].savefig(f'{saveNames[ii]}_{ii+2000}.png', dpi=400, bbox_inches='tight')
    
    f.close()

plt.show()
