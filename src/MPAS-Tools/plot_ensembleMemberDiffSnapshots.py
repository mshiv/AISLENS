#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare thickness and dhdt difference maps between two runs at a specified year.
Overlay grounding lines for year 2000 and for the year being compared (for both runs).
Requires thickness and dhdt variables in the input .nc files.
Also requires mesh files for both runs at year 2000 and at the target year.

Modified from plot_maps.py by Trevor Hillebrand, Matt Hoffman

Shiva Muruganandham
June 2025
"""

import numpy as np
from netCDF4 import Dataset
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run_a', required=True, help='Path to output .nc for run A at target year')
parser.add_argument('--run_b', required=True, help='Path to output .nc for run B at target year')
parser.add_argument('--mesh_a', required=True, help='Path to mesh file for run A at target year')
parser.add_argument('--mesh_b', required=True, help='Path to mesh file for run B at target year')
parser.add_argument('--mesh_2000_a', required=True, help='Path to mesh file for run A at year 2000')
parser.add_argument('--mesh_2000_b', required=True, help='Path to mesh file for run B at year 2000')
parser.add_argument('--outdir', required=True, help='Directory to save output figures')
parser.add_argument('--year', required=True, help='Year being compared (string, e.g., "2050")')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

def load_mesh_vars(mesh_path):
    m = Dataset(mesh_path, 'r')
    xCell = m.variables["xCell"][:]
    yCell = m.variables["yCell"][:]
    dcEdge = m.variables["dcEdge"][:]
    # Try to correct for record dimension if needed
    if len(xCell.shape) > 1: xCell = xCell[0]
    if len(yCell.shape) > 1: yCell = yCell[0]
    if len(dcEdge.shape) > 1: dcEdge = dcEdge[0]
    cellMask = m.variables["cellMask"][:] if "cellMask" in m.variables else None
    m.close()
    return xCell, yCell, dcEdge, cellMask

# Load thickness and dhdt for both runs
fA = Dataset(args.run_a, 'r')
fB = Dataset(args.run_b, 'r')
thickA = fA.variables["thickness"][:]
thickB = fB.variables["thickness"][:]
dhdtA = fA.variables["dhdt"][:] if "dhdt" in fA.variables else None
dhdtB = fB.variables["dhdt"][:] if "dhdt" in fB.variables else None

# Load mesh variables for both runs (target year and 2000)
xA, yA, dcA, cellMaskA = load_mesh_vars(args.mesh_a)
xB, yB, dcB, cellMaskB = load_mesh_vars(args.mesh_b)
x2000A, y2000A, dc2000A, cellMask2000A = load_mesh_vars(args.mesh_2000_a)
x2000B, y2000B, dc2000B, cellMask2000B = load_mesh_vars(args.mesh_2000_b)

# Use first time index if 3D
if thickA.ndim == 2: thickA = thickA[0]
if thickB.ndim == 2: thickB = thickB[0]
if dhdtA is not None and dhdtA.ndim == 2: dhdtA = dhdtA[0]
if dhdtB is not None and dhdtB.ndim == 2: dhdtB = dhdtB[0]

# Compute difference maps
thickness_diff = thickA - thickB
dhdt_diff = dhdtA - dhdtB if dhdtA is not None and dhdtB is not None else None

# Triangulations
triA = tri.Triangulation(xA, yA)
triB = tri.Triangulation(xB, yB)

def mask_triangles(triang, xCell, yCell, dcEdge):
    triMask = np.zeros(len(triang.triangles), dtype=bool)
    maxDist = np.max(dcEdge) * 2.0
    for t in range(len(triang.triangles)):
        thisTri = triang.triangles[t, :]
        if ((xCell[thisTri[0]] - xCell[thisTri[1]])**2 + (yCell[thisTri[0]] - yCell[thisTri[1]])**2)**0.5 > maxDist:
            triMask[t] = True
        if ((xCell[thisTri[1]] - xCell[thisTri[2]])**2 + (yCell[thisTri[1]] - yCell[thisTri[2]])**2)**0.5 > maxDist:
            triMask[t] = True
        if ((xCell[thisTri[0]] - xCell[thisTri[2]])**2 + (yCell[thisTri[0]] - yCell[thisTri[2]])**2)**0.5 > maxDist:
            triMask[t] = True
    triang.set_mask(triMask)
    return triang

triA = mask_triangles(triA, xA, yA, dcA)
triB = mask_triangles(triB, xB, yB, dcB)

# Custom diverging colormap
def create_custom_cmap(low_colors, high_colors):
    # Combine reversed low_colors, white, high_colors
    colors = list(reversed(low_colors)) + ['white'] + high_colors
    n_bins = 200
    return LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

custom_cmap = create_custom_cmap(['Navajowhite', 'Darkorange', 'Darkred'], ['Lightsteelblue', 'Royalblue', 'Navy'])

floatValue = 4
groundingLineValue = 256

def plot_grounding_lines(ax, xCell, yCell, cellMask_2000, cellMask_end, color_2000, color_end, label_2000, label_end):
    # Grounding line mask
    if cellMask_2000 is not None and cellMask_end is not None:
        gl_mask_2000 = (cellMask_2000 & groundingLineValue) // groundingLineValue
        gl_mask_end = (cellMask_end & groundingLineValue) // groundingLineValue
        triang = tri.Triangulation(xCell, yCell)
        ax.tricontour(triang, gl_mask_2000, levels=[0.9999], colors=color_2000, linestyles='solid', linewidths=1.5, label=label_2000)
        ax.tricontour(triang, gl_mask_end, levels=[0.9999], colors=color_end, linestyles='solid', linewidths=2, label=label_end)

# Plot THICKNESS DIFF
fig, ax = plt.subplots(figsize=(8,7))
vmax = np.nanquantile(np.abs(thickness_diff), 0.99)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
h = ax.tripcolor(triA, thickness_diff, cmap=custom_cmap, shading='flat', norm=norm)
plot_grounding_lines(ax, xA, yA, cellMask2000A, cellMaskA, 'darkgreen', 'darkviolet', 'Run A 2000 GL', f'Run A {args.year} GL')
plot_grounding_lines(ax, xB, yB, cellMask2000B, cellMaskB, 'black', 'red', 'Run B 2000 GL', f'Run B {args.year} GL')
ax.set_title(f'Thickness Difference ({args.year}): Run A - Run B')
plt.colorbar(h, ax=ax, orientation='vertical', label='Thickness Difference (m)')
ax.legend(loc='lower right')
fig.tight_layout()
plt.savefig(os.path.join(args.outdir, f'thickness_diff_{args.year}.png'), dpi=300, bbox_inches='tight')

# Plot DHDT DIFF
if dhdt_diff is not None:
    fig, ax = plt.subplots(figsize=(8,7))
    vmax = np.nanquantile(np.abs(dhdt_diff), 0.99)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    h = ax.tripcolor(triA, dhdt_diff, cmap=custom_cmap, shading='flat', norm=norm)
    plot_grounding_lines(ax, xA, yA, cellMask2000A, cellMaskA, 'darkgreen', 'darkviolet', 'Run A 2000 GL', f'Run A {args.year} GL')
    plot_grounding_lines(ax, xB, yB, cellMask2000B, cellMaskB, 'black', 'red', 'Run B 2000 GL', f'Run B {args.year} GL')
    ax.set_title(f'dhdt Difference ({args.year}): Run A - Run B')
    plt.colorbar(h, ax=ax, orientation='vertical', label='dhdt Difference (m/yr)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, f'dhdt_diff_{args.year}.png'), dpi=300, bbox_inches='tight')

fA.close()
fB.close()
print(f"Plots saved to {args.outdir}")