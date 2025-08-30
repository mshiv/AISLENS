#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble snapshot map plotting for MALI.
Plots min, max, mean, std, range for specified variables and years.
Overlays grounding lines for all runs in ensemble, with adjustable line thickness.
Saves one figure per (variable, stat, year).
"""

import os
import numpy as np
from netCDF4 import Dataset
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="Ensemble map plots for MALI.")
parser.add_argument("--ensemble_files", required=True, help="Comma-separated ensemble stats NetCDF files (one per year).")
parser.add_argument("--years", required=True, help="Comma-separated years.")
parser.add_argument("--variables", required=True, help="Comma-separated variables.")
parser.add_argument("--mesh_files", required=True, help="Comma-separated mesh files for GL overlays (one per run, start year).")
parser.add_argument("--run_names", required=True, help="Comma-separated run names (for legend).")
parser.add_argument("--save_base", required=False, default=None, help="Path to directory for saving figures (if not provided, figures are not saved).")
parser.add_argument("--gl_linewidth", required=False, default=0.7, type=float, help="Linewidth for grounding lines (default: 0.7)")

args = parser.parse_args()
ensemble_files = args.ensemble_files.split(',')
years = [int(y) for y in args.years.split(',')]
variables = args.variables.split(',')
mesh_files = args.mesh_files.split(',')
run_names = args.run_names.split(',')
save_base = args.save_base
gl_linewidth = args.gl_linewidth

print(f"Processing {len(ensemble_files)} ensemble files for years: {years}")
print(f"Variables: {variables}")
print(f"Using {len(mesh_files)} mesh files for grounding line overlays")
print(f"Grounding line linewidth: {gl_linewidth}")

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

# Ensure save_base directory exists if provided
if save_base is not None and save_base != "":
    os.makedirs(save_base, exist_ok=True)

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

# Main plotting loop: one file per variable/stat/year
for variable in variables:
    print(f"\nProcessing variable: {variable}")
    for stat in stat_types:
        print(f"  Processing statistic: {stat}")
        # For each year, plot and save a separate file
        for i, (stats_file, year) in enumerate(zip(ensemble_files, years)):
            print(f"    Year {year}: Loading {stats_file}")
            if not stats_file.strip():
                print(f"      Skipping empty stats file for year {year}")
                continue
            try:
                f = Dataset(stats_file, 'r')
                varname = f"{variable}_{stat}"
                if varname not in f.variables:
                    print(f"      WARNING: {varname} not found in {stats_file}")
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
                arr = arr[0, :]
                # Get mesh info for this year
                xCell = f.variables["xCell"][0] if f.variables["xCell"].ndim > 1 else f.variables["xCell"][:]
                yCell = f.variables["yCell"][0] if f.variables["yCell"].ndim > 1 else f.variables["yCell"][:]
                dcEdge = f.variables["dcEdge"][0] if f.variables["dcEdge"].ndim > 1 else f.variables["dcEdge"][:]
                # Triangulation and mask
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
                f.close()
            except Exception as e:
                print(f"      ERROR loading or preparing data for {year}: {e}")
                continue

            # Color scale
            if np.all(np.isnan(arr)):
                print(f"      All data is NaN for {variable}_{stat}, year {year}.")
                continue
            vmin = np.nanquantile(arr, 0.01)
            vmax = np.nanquantile(arr, 0.99)
            # Diverging colormap for dhdt/range
            if variable == 'dhdt' or stat == 'range':
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
            print(f"      Color range: {vmin:.3f} to {vmax:.3f}")

            # Setup figure for this year
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)
            h = ax.tripcolor(triang, arr, cmap=cmap, shading='flat', norm=norm)
            ax.set_title(f"{variable} [{stat}] Year {year}", fontsize=13)
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')

            # Overlay grounding lines for all runs
            legend_elements = []
            for gl_info in grounding_lines:
                if gl_info['gl_mask'] is not None:
                    try:
                        gl_triang = tri.Triangulation(gl_info['x'], gl_info['y'])
                        gl_triMask = np.zeros(len(gl_triang.triangles), dtype=bool)
                        for t in range(len(gl_triang.triangles)):
                            thisTri = gl_triang.triangles[t, :]
                            if (dist(thisTri[0], thisTri[1], gl_info['x'], gl_info['y']) > maxDist or
                                dist(thisTri[1], thisTri[2], gl_info['x'], gl_info['y']) > maxDist or
                                dist(thisTri[0], thisTri[2], gl_info['x'], gl_info['y']) > maxDist):
                                gl_triMask[t] = True
                        gl_triang.set_mask(gl_triMask)
                        ax.tricontour(gl_triang, gl_info['gl_mask'],
                                      levels=[0.9999],
                                      colors=[gl_info['color']],
                                      linestyles='solid',
                                      linewidths=gl_linewidth)
                        legend_elements.append(plt.Line2D([0], [0], color=gl_info['color'],
                                                          lw=gl_linewidth, label=f"GL {gl_info['run_name']}"))
                    except Exception as e:
                        print(f"      WARNING: Could not plot grounding line for {gl_info['run_name']}: {e}")

            if legend_elements:
                ax.legend(handles=legend_elements, loc='lower right', fontsize='small')

            # Add colorbar
            cbar = fig.colorbar(h, ax=ax, orientation='vertical', fraction=0.035, pad=0.03)
            cbar.set_label(f"{variable} [{stat}] ({units})", rotation=270, labelpad=20)

            fig.tight_layout()
            # Save the figure if save_base provided
            if save_base is not None and save_base != "":
                out_png = os.path.join(save_base, f"ensemble_{variable}_{stat}_{year}.png")
                fig.savefig(out_png, dpi=400, bbox_inches='tight')
                print(f"      Saved {out_png}")
            plt.close(fig)

print("All ensemble plots complete.")