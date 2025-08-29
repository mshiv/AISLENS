#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-panel snapshot maps for MALI ensemble.
Rows: variables (thickness, surfaceSpeed, dhdt)
Columns: years at 25-year increments
Handles dhdt as dedicated file if available.
"""
import numpy as np
from netCDF4 import Dataset
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap

parser = argparse.ArgumentParser(description="Snapshot map plotter for MALI ensemble.")
parser.add_argument("-r", dest="run_files", required=True, help="Comma-separated list of .nc files for each year.")
parser.add_argument("-d", dest="dhdt_files", required=False, default=None, help="Comma-separated list of dhdt files (use '' if not present).")
parser.add_argument("-v", dest="variables", required=True, help="Comma-separated variable list (thickness,surfaceSpeed,dhdt).")
parser.add_argument("--years", dest="years", required=True, help="Comma-separated list of years.")
parser.add_argument("-m", dest="meshes", required=True, help="Comma-separated mesh files (one per run file).")
parser.add_argument("-s", dest="save_base", required=False, default=None, help="Base filename for saving figures.")

args = parser.parse_args()

run_files = args.run_files.split(',')
dhdt_files = args.dhdt_files.split(',') if args.dhdt_files else [''] * len(run_files)
variables = args.variables.split(',')
years = args.years.split(',')
meshes = args.meshes.split(',')
save_base = args.save_base

nRows = len(variables)
nCols = len(run_files)

defaultColors = {'thickness': 'Blues', 'surfaceSpeed': 'plasma', 'dhdt': 'RdBu'}
sec_per_year = 60. * 60. * 24. * 365.

def dist(i1, i2, xCell, yCell):
    return ((xCell[i1]-xCell[i2])**2 + (yCell[i1]-yCell[i2])**2)**0.5

fig = plt.figure(figsize=(4*nCols, 4*nRows))
gs = gridspec.GridSpec(nRows, nCols+1, height_ratios=[1]*nRows, width_ratios=[1]*nCols+[0.12])
axs = []
cbar_axs = []
for row in range(nRows):
    cbar_axs.append(plt.subplot(gs[row,-1]))
    for col in range(nCols):
        axs.append(plt.subplot(gs[row, col]))

for row, variable in enumerate(variables):
    all_data = []
    # Collect all data for colorbar scaling
    for col, (run_file, mesh_file, dhdt_file) in enumerate(zip(run_files, meshes, dhdt_files)):
        if variable == 'dhdt' and dhdt_file and dhdt_file != '':
            try:
                f = Dataset(dhdt_file, 'r')
                arr = f.variables['dhdt'][:]
                f.close()
            except Exception as e:
                print(f"Error reading dhdt file {dhdt_file}: {e}")
                arr = np.nan * np.ones_like(arr)
        else:
            try:
                f = Dataset(run_file, 'r')
                arr = f.variables[variable][:]
                if variable == 'surfaceSpeed':
                    arr *= sec_per_year
                f.close()
            except Exception as e:
                print(f"Error reading {variable} from {run_file}: {e}")
                arr = np.nan * np.ones_like(arr)
        if arr.ndim == 1:
            arr = arr.reshape((1, np.shape(arr)[0]))
        all_data.append(arr[0,:])
    # Colorbar limits: 1st/99th percentile
    vmin = np.nanquantile(np.concatenate(all_data), 0.01)
    vmax = np.nanquantile(np.concatenate(all_data), 0.99)
    # Colormap/norm
    if variable == 'dhdt':
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
        cmap = LinearSegmentedColormap.from_list("custom", ['Navajowhite', 'Darkorange', 'Darkred','white','Lightsteelblue', 'Royalblue', 'Navy'], N=200)
    elif variable == 'surfaceSpeed':
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('plasma')
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('Blues')

    # Plot each year/column
    for col, (run_file, mesh_file, dhdt_file, year) in enumerate(zip(run_files, meshes, dhdt_files, years)):
        index = row * nCols + col
        try:
            m = Dataset(mesh_file, 'r')
            xCell = m.variables["xCell"][0] if m.variables["xCell"].ndim > 1 else m.variables["xCell"][:]
            yCell = m.variables["yCell"][0] if m.variables["yCell"].ndim > 1 else m.variables["yCell"][:]
            dcEdge = m.variables["dcEdge"][0] if m.variables["dcEdge"].ndim > 1 else m.variables["dcEdge"][:]
            cellMask = m.variables["cellMask"][:] if "cellMask" in m.variables else None
            m.close()
        except Exception as e:
            print(f"Error reading mesh file {mesh_file}: {e}")
            continue

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

        # Read variable
        if variable == 'dhdt' and dhdt_file and dhdt_file != '':
            try:
                f = Dataset(dhdt_file, 'r')
                arr = f.variables['dhdt'][:]
                units = f.variables['dhdt'].units if 'units' in f.variables['dhdt'].ncattrs() else 'm/yr'
                f.close()
            except Exception as e:
                print(f"Error reading dhdt file {dhdt_file}: {e}")
                arr = np.nan * np.ones_like(xCell)
                units = 'm/yr'
        else:
            try:
                f = Dataset(run_file, 'r')
                arr = f.variables[variable][:]
                if variable == 'surfaceSpeed':
                    arr *= sec_per_year
                    units = 'm/yr'
                else:
                    units = f.variables[variable].units if 'units' in f.variables[variable].ncattrs() else 'unknown'
                f.close()
            except Exception as e:
                print(f"Error reading {variable} from {run_file}: {e}")
                arr = np.nan * np.ones_like(xCell)
                units = 'unknown'
        if arr.ndim == 1:
            arr = arr.reshape((1, np.shape(arr)[0]))

        h = axs[index].tripcolor(triang, arr[0, :], cmap=cmap, shading='flat', norm=norm)
        axs[index].set_title(f"{variable} year={year}")
        axs[index].set_aspect('equal')

    # Add colorbar for this variable
    Colorbar(ax=cbar_axs[row], mappable=h, orientation='vertical', label=f'{variable} ({units})')

fig.tight_layout()
if save_base:
    fig.savefig(f'{save_base}_panel.png', dpi=400, bbox_inches='tight')
    print(f"Saved {save_base}_panel.png")
else:
    plt.show()