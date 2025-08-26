import numpy as np
from netCDF4 import Dataset
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, TwoSlopeNorm
import os

# This script plots a single variable from a single MPAS NetCDF file on a map.
# It can plot the variable at specified time levels.

print("** Gathering information. (Invoke with --help for more details.)")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-f", dest="filename", required=True, metavar="FILENAME",
                    help="path to .nc file to plot")
parser.add_argument("-t", dest="timeLevels", default="-1",
                    help="integer time levels at which to plot \
                          (int separated by commas; no spaces)")
parser.add_argument("-v", dest="variable", default='thickness',
                    help="variable to plot")
parser.add_argument("-c", dest="colormap", default=None,
                    help="colormap to use for plotting. This overrides default colormap.")
parser.add_argument("-s", dest="save", action='store_true',
                    help="save figure as PNG file")
parser.add_argument("-o", dest="output", default=None,
                    help="output filename for saved figure (if -s is used)")
parser.add_argument("--vmin", dest="vmin", type=float, default=None,
                    help="minimum value for colorbar (overrides automatic percentile)")
parser.add_argument("--vmax", dest="vmax", type=float, default=None,
                    help="maximum value for colorbar (overrides automatic percentile)")

args = parser.parse_args()
filename = args.filename
variable = args.variable
timeLevs = args.timeLevels.split(',')  # split time levels into list
# convert timeLevs to list of ints
timeLevs = [int(i) for i in timeLevs]
save_fig = args.save
output_filename = args.output
vmin_manual = args.vmin
vmax_manual = args.vmax

# Constants
sec_per_year = 60. * 60. * 24. * 365.
rhoi = 910.
rhosw = 1028.

# Set up a dictionary of default colormaps for common variables.
defaultColors = {'thickness' : 'Blues',
                 'surfaceSpeed' : 'plasma',
                 'basalSpeed' : 'plasma',
                 'bedTopography' : 'BrBG',
                 'floatingBasalMassBalApplied' : 'cividis'
                }

if args.colormap is not None:
    colormap = args.colormap
else:
    if variable in defaultColors.keys():
        colormap = defaultColors[variable]
    else:
        # All other variables default to viridis
        colormap = 'viridis'

# Set bitmask values
initialExtentValue = 1
dynamicValue = 2
floatValue = 4
groundingLineValue = 256

# List of diverging colormaps for use in plotting bedTopography.
divColorMaps = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

def dist(i1, i2, xCell, yCell):  # helper distance fn
    dist = ((xCell[i1]-xCell[i2])**2 + (yCell[i1]-yCell[i2])**2)**0.5
    return dist

# Check if file exists
if not os.path.exists(filename):
    raise FileNotFoundError(f"File {filename} not found")

# Open the NetCDF file
f = Dataset(filename, 'r')

# Get time information
if 'daysSinceStart' in f.variables.keys():
    yr = f.variables['daysSinceStart'][:] / 365.
else:
    yr = np.arange(len(timeLevs))  # Use indices if no time variable

f.set_auto_mask(False)

# Get mesh geometry and calculate triangulation
xCell = f.variables["xCell"][:]
yCell = f.variables["yCell"][:]
dcEdge = f.variables["dcEdge"][:]

triang = tri.Triangulation(xCell, yCell)
triMask = np.zeros(len(triang.triangles))
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

# Get the variable to plot
if variable == 'observedSpeed':
    var_to_plot = np.sqrt(f.variables['observedSurfaceVelocityX'][:]**2 +
                          f.variables['observedSurfaceVelocityY'][:]**2)
else:
    var_to_plot = f.variables[variable][:]

if len(np.shape(var_to_plot)) == 1:
   var_to_plot = var_to_plot.reshape((1, np.shape(var_to_plot)[0]))

# Handle units and scaling
if 'Speed' in variable:
    units = 'm yr^{-1}'
    var_to_plot *= sec_per_year
else:
    try:
        units = f.variables[variable].units
    except AttributeError:
        units='no-units'

# Set up figure
nCols = len(timeLevs) + 1  # +1 for colorbar
fig = plt.figure(figsize=(4*len(timeLevs) + 1, 4))
gs = gridspec.GridSpec(1, nCols,
                       width_ratios=[1] * (nCols - 1) + [0.1])

# Create subplots
axs = []
cbar_ax = plt.subplot(gs[0, -1])
for col in range(nCols-1):
    if axs == []:
        axs.append(plt.subplot(gs[0, col]))
    else:
        axs.append(plt.subplot(gs[0, col], sharex=axs[0], sharey=axs[0]))

# Set lower and upper bounds for plotting
if vmin_manual is not None:
    vmin = vmin_manual
    print(f"Using manual vmin: {vmin}")
else:
    vmin = np.nanquantile(var_to_plot[timeLevs, :], 0.01)
    print(f"Using automatic vmin (1st percentile): {vmin:.3e}")

if vmax_manual is not None:
    vmax = vmax_manual
    print(f"Using manual vmax: {vmax}")
else:
    vmax = np.nanquantile(var_to_plot[timeLevs, :], 0.99)
    print(f"Using automatic vmax (99th percentile): {vmax:.3e}")

# Plot bedTopography on an asymmetric colorbar if appropriate
if ( (variable == 'bedTopography') and
     (np.nanquantile(var_to_plot[timeLevs, :], 0.99) > 0.) and
     (colormap in divColorMaps) ):
    norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0.)
else:
    norm = Normalize(vmin=vmin, vmax=vmax)

# Calculate masks for grounding line and ice extent
if 'cellMask' in f.variables.keys():
    calc_mask = True
    cellMask = f.variables["cellMask"][:]
    floatMask = (cellMask & floatValue) // floatValue
    dynamicMask = (cellMask & dynamicValue) // dynamicValue
    groundingLineMask = (cellMask & groundingLineValue) // groundingLineValue
    initialExtentMask = (cellMask & initialExtentValue) // initialExtentValue
elif ( 'cellMask' not in f.variables.keys() and
     'thickness' in f.variables.keys() and 
     'bedTopography' in f.variables.keys() ):
    print(f'cellMask is not present in output file {filename}; calculating masks from ice thickness')
    calc_mask = True
    groundedMask = (f.variables['thickness'][:] > (-rhosw / rhoi * f.variables['bedTopography'][:]))
    groundingLineMask = groundedMask.copy()  # This isn't technically correct, but works for plotting
    initialExtentMask = (f.variables['thickness'][:] > 0.)
else:
    print(f'cellMask and thickness and/or bedTopography not present in output file {filename};'
           ' Skipping mask calculation.')
    calc_mask = False

# Plot at each time level
mappables = []
for col, timeLev in enumerate(timeLevs):
    # plot initial grounding line position, initial extent, and GL position at t=timeLev
    if calc_mask:
        axs[col].tricontour(triang, groundingLineMask[0, :],
                          levels=[0.9999], colors='grey',
                          linestyles='solid')
        axs[col].tricontour(triang, groundingLineMask[timeLev, :],
                          levels=[0.9999], colors='white',
                          linestyles='solid')
        axs[col].tricontour(triang, initialExtentMask[timeLev, :],
                          levels=[0.9999], colors='black',
                          linestyles='solid')

    # Plot 2D field at each desired time. Use quantile range of 0.01-0.99 to cut out
    # outliers. Could improve on this by accounting for areaCell, as currently all cells
    # are weighted equally in determining vmin and vmax.
    mappable = axs[col].tripcolor(triang, var_to_plot[timeLev, :], cmap=colormap,
                                  shading='flat', norm=norm)
    mappables.append(mappable)
    axs[col].set_aspect('equal')
    if len(yr) > timeLev:
        axs[col].set_title(f'year = {yr[timeLev]:0.2f}')
    else:
        axs[col].set_title(f'time level = {timeLev}')

# Add colorbar
cbar = Colorbar(ax=cbar_ax, mappable=mappables[0], orientation='vertical',
                label=f'{variable} (${units}$)')

# Set main title
(path, inFileName) = os.path.split(filename)
fig.suptitle(f'{variable} from {inFileName}')

plt.tight_layout()

# Save figure if requested
if save_fig:
    if output_filename is None:
        output_filename = f'{variable}_{os.path.splitext(inFileName)[0]}.png'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {output_filename}")

f.close()
plt.show()
