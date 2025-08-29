#!/usr/bin/env python3
"""
Enhanced MALI ensemble difference plotting script.
- Supports arbitrary variables (default: thickness, dhdt)
- Computes pairwise differences, ensemble min/max/range/mean/stddev
- Overlays 2000 and target year grounding lines for each run
- Saves one PNG per variable/year/comparison
"""

import numpy as np
from netCDF4 import Dataset
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap
import os
import itertools

parser = argparse.ArgumentParser(description="Ensemble difference plots for MALI output")
parser.add_argument('--runs', required=True, help='Comma-separated list of output .nc files for all runs (same year)')
parser.add_argument('--meshes', required=True, help='Comma-separated list of mesh files for all runs (same year)')
parser.add_argument('--mesh2000', required=True, help='Comma-separated list of mesh files for all runs (year 2000)')
parser.add_argument('--variable', required=True, help='Variable to plot (e.g. thickness, dhdt, etc.)')
parser.add_argument('--year', required=True, help='Year being compared (string, e.g., "2150")')
parser.add_argument('--outdir', required=True, help='Directory to save output figures')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

run_files = args.runs.split(',')
mesh_files = args.meshes.split(',')
mesh2000_files = args.mesh2000.split(',')
variable = args.variable
year = args.year

# Helper for mesh vars
def load_mesh_vars(mesh_path):
    m = Dataset(mesh_path, 'r')
    xCell = m.variables["xCell"][:]
    yCell = m.variables["yCell"][:]
    dcEdge = m.variables["dcEdge"][:]
    # Try to correct for record dimension
    if len(xCell.shape) > 1: xCell = xCell[0]
    if len(yCell.shape) > 1: yCell = yCell[0]
    if len(dcEdge.shape) > 1: dcEdge = dcEdge[0]
    cellMask = m.variables["cellMask"][:] if "cellMask" in m.variables else None
    m.close()
    return xCell, yCell, dcEdge, cellMask

floatValue = 4
groundingLineValue = 256

def plot_grounding_lines(ax, xCell, yCell, cellMask_2000, cellMask_end, color_2000, color_end, label_2000, label_end):
    if cellMask_2000 is not None and cellMask_end is not None:
        gl_mask_2000 = (cellMask_2000 & groundingLineValue) // groundingLineValue
        gl_mask_end = (cellMask_end & groundingLineValue) // groundingLineValue
        triang = tri.Triangulation(xCell, yCell)
        ax.tricontour(triang, gl_mask_2000, levels=[0.9999], colors=color_2000, linestyles='solid', linewidths=1.5, label=label_2000)
        ax.tricontour(triang, gl_mask_end, levels=[0.9999], colors=color_end, linestyles='solid', linewidths=2, label=label_end)

def create_custom_cmap(low_colors, high_colors):
    colors = list(reversed(low_colors)) + ['white'] + high_colors
    n_bins = 200
    return LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

custom_cmap = create_custom_cmap(['Navajowhite', 'Darkorange', 'Darkred'], ['Lightsteelblue', 'Royalblue', 'Navy'])

# Load all runs
data = []
meshes = []
mesh2000s = []
run_labels = []
for i, (run_path, mesh_path, mesh2000_path) in enumerate(zip(run_files, mesh_files, mesh2000_files)):
    try:
        f = Dataset(run_path, 'r')
        arr = f.variables[variable][:]
        if arr.ndim == 2:
            arr = arr[0]  # Use first time index if present
        data.append(arr)
        meshes.append(load_mesh_vars(mesh_path))
        mesh2000s.append(load_mesh_vars(mesh2000_path))
        run_labels.append(os.path.basename(os.path.dirname(run_path)))
        f.close()
    except Exception as e:
        print(f"Failed to load {variable} from {run_path}: {e}")

data = np.array(data)  # Shape: (n_runs, n_cells)

# --- Ensemble statistics ---
ens_mean = np.nanmean(data, axis=0)
ens_std = np.nanstd(data, axis=0)
ens_min = np.nanmin(data, axis=0)
ens_max = np.nanmax(data, axis=0)
ens_range = ens_max - ens_min

# --- Pairwise differences ---
pairs = list(itertools.combinations(range(len(data)), 2))
pair_diffs = []
for (i, j) in pairs:
    diff = data[i] - data[j]
    pair_diffs.append((i, j, diff))

# --- Plotting ensemble stats ---
stats_to_plot = {
    'mean': ens_mean,
    'stddev': ens_std,
    'min': ens_min,
    'max': ens_max,
    'range': ens_range
}

for statname, arr in stats_to_plot.items():
    xCell, yCell, dcEdge, cellMask = meshes[0]
    tri_mesh = tri.Triangulation(xCell, yCell)
    # Mask triangles
    triMask = np.zeros(len(tri_mesh.triangles), dtype=bool)
    maxDist = np.max(dcEdge) * 2.0
    for t in range(len(tri_mesh.triangles)):
        thisTri = tri_mesh.triangles[t, :]
        if ((xCell[thisTri[0]] - xCell[thisTri[1]])**2 + (yCell[thisTri[0]] - yCell[thisTri[1]])**2)**0.5 > maxDist:
            triMask[t] = True
        if ((xCell[thisTri[1]] - xCell[thisTri[2]])**2 + (yCell[thisTri[1]] - yCell[thisTri[2]])**2)**0.5 > maxDist:
            triMask[t] = True
        if ((xCell[thisTri[0]] - xCell[thisTri[2]])**2 + (yCell[thisTri[0]] - yCell[thisTri[2]])**2)**0.5 > maxDist:
            triMask[t] = True
    tri_mesh.set_mask(triMask)
    fig, ax = plt.subplots(figsize=(8,7))
    vmax = np.nanquantile(np.abs(arr), 0.99)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) if statname in ['mean', 'range'] else None
    h = ax.tripcolor(tri_mesh, arr, cmap=custom_cmap, shading='flat', norm=norm)
    # Overlay grounding lines for all runs
    for idx, (mesh, mesh2000) in enumerate(zip(meshes, mesh2000s)):
        plot_grounding_lines(ax, mesh[0], mesh[1], mesh2000[3], mesh[3],
                             color_2000='darkgreen', color_end='darkviolet',
                             label_2000=f'{run_labels[idx]} 2000 GL', label_end=f'{run_labels[idx]} {year} GL')
    ax.set_title(f'{variable} ensemble {statname} ({year})')
    plt.colorbar(h, ax=ax, orientation='vertical', label=f'{variable} {statname}')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fname = os.path.join(args.outdir, f'{variable}_{statname}_{year}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

# --- Pairwise difference plots ---
for (i, j, diff) in pair_diffs:
    xCell, yCell, dcEdge, cellMask = meshes[i]
    tri_mesh = tri.Triangulation(xCell, yCell)
    triMask = np.zeros(len(tri_mesh.triangles), dtype=bool)
    maxDist = np.max(dcEdge) * 2.0
    for t in range(len(tri_mesh.triangles)):
        thisTri = tri_mesh.triangles[t, :]
        if ((xCell[thisTri[0]] - xCell[thisTri[1]])**2 + (yCell[thisTri[0]] - yCell[thisTri[1]])**2)**0.5 > maxDist:
            triMask[t] = True
        if ((xCell[thisTri[1]] - xCell[thisTri[2]])**2 + (yCell[thisTri[1]] - yCell[thisTri[2]])**2)**0.5 > maxDist:
            triMask[t] = True
        if ((xCell[thisTri[0]] - xCell[thisTri[2]])**2 + (yCell[thisTri[0]] - yCell[thisTri[2]])**2)**0.5 > maxDist:
            triMask[t] = True
    tri_mesh.set_mask(triMask)
    fig, ax = plt.subplots(figsize=(8,7))
    vmax = np.nanquantile(np.abs(diff), 0.99)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    h = ax.tripcolor(tri_mesh, diff, cmap=custom_cmap, shading='flat', norm=norm)
    # Overlay grounding lines for both runs
    plot_grounding_lines(ax, meshes[i][0], meshes[i][1], mesh2000s[i][3], meshes[i][3],
                         color_2000='darkgreen', color_end='darkviolet',
                         label_2000=f'{run_labels[i]} 2000 GL', label_end=f'{run_labels[i]} {year} GL')
    plot_grounding_lines(ax, meshes[j][0], meshes[j][1], mesh2000s[j][3], meshes[j][3],
                         color_2000='black', color_end='red',
                         label_2000=f'{run_labels[j]} 2000 GL', label_end=f'{run_labels[j]} {year} GL')
    ax.set_title(f'{variable} diff ({run_labels[i]} - {run_labels[j]}) ({year})')
    plt.colorbar(h, ax=ax, orientation='vertical', label=f'{variable} diff ({run_labels[i]} - {run_labels[j]})')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fname = os.path.join(args.outdir, f'{variable}_diff_{run_labels[i]}_vs_{run_labels[j]}_{year}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"Saved all plots to {args.outdir}")