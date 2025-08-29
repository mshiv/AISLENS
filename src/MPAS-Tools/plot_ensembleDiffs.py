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

stat_types = ["mean", "min", "max", "range", "std"]
defaultColors = {'thickness': 'Blues', 'surfaceSpeed': 'plasma', 'dhdt': 'RdBu'}
sec_per_year = 60. * 60. * 24. * 365.

# Load mesh info for GL overlays
grounding_lines = []
for mesh_fn in mesh_files:
    m = Dataset(mesh_fn, 'r')
    xCell = m.variables["xCell"][0] if m.variables["xCell"].ndim > 1 else m.variables["xCell"][:]
    yCell = m.variables["yCell"][0] if m.variables["yCell"].ndim > 1 else m.variables["yCell"][:]
    cellMask = m.variables["cellMask"][:] if "cellMask" in m.variables else None
    groundingLineValue = 256
    gl_mask = (cellMask & groundingLineValue) // groundingLineValue if cellMask is not None else None
    grounding_lines.append({'x': xCell, 'y': yCell, 'mask': gl_mask})
    m.close()

def dist(i1, i2, xCell, yCell):
    return ((xCell[i1]-xCell[i2])**2 + (yCell[i2]-yCell[i1])**2)**0.5

for variable in variables:
    for stat in stat_types:
        # Create figure for this variable/stat across all years
        fig = plt.figure(figsize=(5*len(years), 7))
        gs = gridspec.GridSpec(1, len(years), width_ratios=[1]*len(years))
        axs = []
        vdata = []
        units = "unknown"
        for i, (stats_file, year) in enumerate(zip(ensemble_files, years)):
            f = Dataset(stats_file, 'r')
            varname = f"{variable}_{stat}"
            if varname not in f.variables:
                print(f"Skipping {varname} in {stats_file}")
                vdata.append(None)
                axs.append(plt.subplot(gs[i]))
                continue
            arr = f.variables[varname][:]
            units = f.variables[varname].units if 'units' in f.variables[varname].ncattrs() else "unknown"
            if arr.ndim == 1:
                arr = arr.reshape((1, np.shape(arr)[0]))
            vdata.append(arr[0,:])
            axs.append(plt.subplot(gs[i]))
            f.close()

        # Determine colormap and normalization
        all_flat = np.concatenate([a for a in vdata if a is not None])
        vmin = np.nanquantile(all_flat, 0.01)
        vmax = np.nanquantile(all_flat, 0.99)
        if variable == 'dhdt' or stat == 'range':
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
            cmap = LinearSegmentedColormap.from_list("custom", ['Navajowhite', 'Darkorange', 'Darkred','white','Lightsteelblue', 'Royalblue', 'Navy'], N=200)
        elif variable == 'surfaceSpeed':
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap('plasma')
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap('Blues')

        # Plot panels for each year
        for i, (arr, ax, year) in enumerate(zip(vdata, axs, years)):
            if arr is None: continue
            m = Dataset(ensemble_files[i], 'r')
            xCell = m.variables["xCell"][0] if m.variables["xCell"].ndim > 1 else m.variables["xCell"][:]
            yCell = m.variables["yCell"][0] if m.variables["yCell"].ndim > 1 else m.variables["yCell"][:]
            dcEdge = m.variables["dcEdge"][0] if m.variables["dcEdge"].ndim > 1 else m.variables["dcEdge"][:]
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
            h = ax.tripcolor(triang, arr, cmap=cmap, shading='flat', norm=norm)
            ax.set_title(f"{variable} [{stat}] Year {year}")
            ax.set_aspect('equal')
            # Overlay GLs for all runs
            for gl, name in zip(grounding_lines, run_names):
                if gl['mask'] is not None:
                    ax.tricontour(triang, gl['mask'], levels=[0.9999], colors='darkviolet', linestyles='solid', linewidths=1.2, label=f'GL {name}')
            m.close()
        # Colorbar and legend
        cbar = fig.colorbar(h, ax=axs, orientation='vertical', fraction=0.025, pad=0.03, label=f"{variable} [{stat}] ({units})")
        legend_elements = [plt.Line2D([0], [0], color='darkviolet', lw=1.2, label=f'GL {name}') for name in run_names]
        axs[-1].legend(handles=legend_elements, loc='lower right', fontsize='small')
        fig.suptitle(f"MALI Ensemble: {variable} [{stat}]")
        fig.tight_layout()
        out_png = f"{save_base}_{variable}_{stat}.png"
        fig.savefig(out_png, dpi=400, bbox_inches='tight')
        print(f"Saved {out_png}")

print("All ensemble plots complete.")