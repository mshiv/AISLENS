#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots any variable/statistic (thickness, dhdt, surfaceSpeed, basalSpeed, bedTopography, floatingBasalMassBalApplied, etc.).
Each ensembleStats file used as data contains multiple data variables: e.g. thickness_mean, thickness_min, thickness_max, thickness_range, thickness_std.
Also can plot diff files if provided.
"""

import numpy as np
from netCDF4 import Dataset
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap

def dist(i1, i2, xCell, yCell):
    return ((xCell[i1]-xCell[i2])**2 + (yCell[i1]-yCell[i2])**2)**0.5

print("** Gathering information.  (Invoke with --help for more details. All arguments are optional)")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-r", dest="runs", default=None, metavar="FILENAME",
                    help="Comma-separated paths to ensembleStats .nc files")
parser.add_argument("-v", dest="variable", default='thickness',
                    help="Variable to plot")
parser.add_argument("-m", dest="mesh", default=None, metavar="FILENAME",
                    help="Mesh file for reference year (usually 2000)")
parser.add_argument("-m2", dest="mesh2", default=None, metavar="FILENAME",
                    help="Comma-separated mesh files for each run/stat file (end years)")
parser.add_argument("-s", dest="saveNames", default=None, metavar="FILENAME",
                    help="Base filename for saving plots (stat/variable/year will be appended)")
parser.add_argument("--stat_labels", dest="stat_labels", default=None, help="Comma-separated list of statistic labels")
parser.add_argument("--year_labels", dest="year_labels", default=None, help="Comma-separated list of years for each file")
parser.add_argument("--diff_files", dest="diff_files", default=None, help="Comma-separated diff files for each year (optional)")
parser.add_argument("-l", dest="log_plot", default="False", help="Whether to plot the log10 of this variable (True/False)")
parser.add_argument("-c", dest="colormap", default=None, help="Colormap to use for plotting. Overrides defaults.")
parser.add_argument("--vmin", dest="vmin", default=None, help="Minimum value for colorbar")
parser.add_argument("--vmax", dest="vmax", default=None, help="Maximum value for colorbar")

args = parser.parse_args()

runs = args.runs.split(',') if args.runs is not None else []
stat_labels = args.stat_labels.split(',') if args.stat_labels is not None else ["" for _ in runs]
year_labels = args.year_labels.split(',') if args.year_labels is not None else ["" for _ in runs]
mesh2_files = args.mesh2.split(',') if args.mesh2 is not None else []
diff_files = args.diff_files.split(',') if args.diff_files is not None else []

variable = args.variable
save_base = args.saveNames if args.saveNames is not None else "ensemble_stat"
log_plot = args.log_plot if args.log_plot is not None else "False"
colormap = args.colormap if args.colormap is not None else None
vmin = float(args.vmin) if args.vmin is not None else None
vmax = float(args.vmax) if args.vmax is not None else None

defaultColors = {
    'thickness': 'Blues',
    'surfaceSpeed': 'plasma',
    'basalSpeed': 'plasma',
    'bedTopography': 'BrBG',
    'floatingBasalMassBalApplied': 'cividis',
    'dhdt': 'RdBu'
}
divColorMaps = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

floatValue = 4
groundingLineValue = 256
initialExtentValue = 1

def create_custom_cmap(low_colors, high_colors):
    low_colors = list(reversed(low_colors))
    colors = low_colors + ['white'] + high_colors
    n_bins = 200
    return LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

def get_colormap(variable, user_cmap=None):
    if user_cmap is not None:
        return user_cmap
    if variable in defaultColors.keys():
        return defaultColors[variable]
    return 'viridis'

if args.mesh is not None:
    m_ref = Dataset(args.mesh, 'r')
    xCell_ref = m_ref.variables["xCell"][0] if m_ref.variables["xCell"].ndim > 1 else m_ref.variables["xCell"][:]
    yCell_ref = m_ref.variables["yCell"][0] if m_ref.variables["yCell"].ndim > 1 else m_ref.variables["yCell"][:]
    dcEdge_ref = m_ref.variables["dcEdge"][0] if m_ref.variables["dcEdge"].ndim > 1 else m_ref.variables["dcEdge"][:]
    cellMask_2000 = m_ref.variables["cellMask"][:] if "cellMask" in m_ref.variables else None
else:
    xCell_ref = yCell_ref = dcEdge_ref = cellMask_2000 = None

# All possible stats inside ensembleStats file
stats_keys = ["mean", "min", "max", "range", "std"]

for idx, (run_file, mesh_file, stat_label, year_label) in enumerate(zip(runs, mesh2_files, stat_labels, year_labels)):
    print(f"Processing: {run_file} | mesh: {mesh_file} | year: {year_label}")

    f = Dataset(run_file, 'r')
    m = Dataset(mesh_file, 'r')
    xCell = m.variables["xCell"][0] if m.variables["xCell"].ndim > 1 else m.variables["xCell"][:]
    yCell = m.variables["yCell"][0] if m.variables["yCell"].ndim > 1 else m.variables["yCell"][:]
    dcEdge = m.variables["dcEdge"][0] if m.variables["dcEdge"].ndim > 1 else m.variables["dcEdge"][:]
    cellMask_end = m.variables["cellMask"][:] if "cellMask" in m.variables else None

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

    # For each stat in the file, plot a separate figure
    for stat in stats_keys:
        varname = f"{variable}_{stat}"
        if varname not in f.variables:
            continue
        var_to_plot = f.variables[varname][:]
        if var_to_plot.ndim == 1:
            var_to_plot = var_to_plot.reshape((1, np.shape(var_to_plot)[0]))

        units = f.variables[varname].units if 'units' in f.variables[varname].ncattrs() else 'no-units'

        # Log scaling
        if log_plot == 'True':
            var_to_plot = np.log10(var_to_plot)
            var_to_plot[np.isinf(var_to_plot)] = np.nan
            colorbar_label_prefix = 'log10 '
        else:
            colorbar_label_prefix = ''

        cmap = get_colormap(variable, colormap)
        # Use diverging norm for dhdt/range
        if variable == 'dhdt' or stat in ['range', 'diff_2000'] or cmap in divColorMaps:
            custom_cmap = create_custom_cmap(['Navajowhite', 'Darkorange', 'Darkred'], ['Lightsteelblue', 'Royalblue', 'Navy'])
        else:
            custom_cmap = plt.get_cmap(cmap)

        # vmin/vmax
        vmin_stat = vmin if vmin is not None else np.nanquantile(var_to_plot, 0.01)
        vmax_stat = vmax if vmax is not None else np.nanquantile(var_to_plot, 0.99)

        if variable == 'dhdt' or stat in ['range', 'diff_2000']:
            norm = TwoSlopeNorm(vmin=vmin_stat, vcenter=0., vmax=vmax_stat)
        else:
            norm = Normalize(vmin=vmin_stat, vmax=vmax_stat)

        fig, ax = plt.subplots(figsize=(9,8))
        h = ax.tripcolor(triang, var_to_plot[0, :], cmap=custom_cmap, shading='flat', norm=norm)

        # Overlay grounding lines if mask available
        if cellMask_2000 is not None and cellMask_end is not None:
            gl_mask_2000 = (cellMask_2000 & groundingLineValue) // groundingLineValue
            gl_mask_end = (cellMask_end & groundingLineValue) // groundingLineValue
            ax.tricontour(triang, gl_mask_2000, levels=[0.9999], colors='darkgreen', linestyles='solid', linewidths=1.2, label='2000 GL')
            ax.tricontour(triang, gl_mask_end, levels=[0.9999], colors='darkviolet', linestyles='solid', linewidths=1.2, label=f'{year_label} GL')

        ax.set_title(f"{variable} [{stat}] year={year_label}")
        plt.colorbar(h, ax=ax, orientation='vertical', label=f'{colorbar_label_prefix}{variable} [{stat}] (${units}$)')
        legend_elements = [
            plt.Line2D([0], [0], color='darkgreen', lw=1.5, label='2000 GL'),
            plt.Line2D([0], [0], color='darkviolet', lw=1.5, label=f'{year_label} GL')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize='small')
        fig.tight_layout()
        png_name = f"{save_base}_{variable}_{stat}_{year_label}.png"
        fig.savefig(png_name, dpi=350, bbox_inches='tight')
        print(f"Saved {png_name}")
        plt.close(fig)

    # Optionally plot diff file if available
    if idx < len(diff_files):
        diff_file = diff_files[idx]
        if diff_file and diff_file != 'None':
            fdiff = Dataset(diff_file, 'r')
            # Assume variable name is the same as in original
            if variable in fdiff.variables:
                var_to_plot = fdiff.variables[variable][:]
            else:
                # Fallback: try variable_diff, etc.
                for v in fdiff.variables:
                    if variable in v and 'diff' in v:
                        var_to_plot = fdiff.variables[v][:]
                        break
            if var_to_plot.ndim == 1:
                var_to_plot = var_to_plot.reshape((1, np.shape(var_to_plot)[0]))
            fig, ax = plt.subplots(figsize=(9,8))
            # Use diverging colormap
            custom_cmap = create_custom_cmap(['Navajowhite', 'Darkorange', 'Darkred'], ['Lightsteelblue', 'Royalblue', 'Navy'])
            vmin_stat = vmin if vmin is not None else np.nanquantile(var_to_plot, 0.01)
            vmax_stat = vmax if vmax is not None else np.nanquantile(var_to_plot, 0.99)
            norm = TwoSlopeNorm(vmin=vmin_stat, vcenter=0., vmax=vmax_stat)
            h = ax.tripcolor(triang, var_to_plot[0, :], cmap=custom_cmap, shading='flat', norm=norm)
            ax.set_title(f"{variable} [diff_2000] year={year_label}")
            plt.colorbar(h, ax=ax, orientation='vertical', label=f'{variable} [diff_2000]')
            legend_elements = [
                plt.Line2D([0], [0], color='darkgreen', lw=1.5, label='2000 GL'),
                plt.Line2D([0], [0], color='darkviolet', lw=1.5, label=f'{year_label} GL')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize='small')
            fig.tight_layout()
            png_name = f"{save_base}_{variable}_diff_2000_{year_label}.png"
            fig.savefig(png_name, dpi=350, bbox_inches='tight')
            print(f"Saved {png_name}")
            plt.close(fig)
            fdiff.close()

    f.close()
    m.close()