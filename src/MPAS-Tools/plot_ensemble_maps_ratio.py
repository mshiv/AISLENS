#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone ensemble map plotting for MALI with ratio output
This script reads ensemble-stats NetCDF files (one file per year) and, for a
requested variable (typically `thickness`) computes the ratio:

    ratio = <variable>_range / dhdt_mean

and writes a PNG and a compact NetCDF with the ratio per year. If the
stats files don't contain the required variables the script will warn and
skip the ratio for that year.

d assumes the ensembleStats_*.nc files have mesh
variables (`xCell`, `yCell`, `dcEdge`) and the stats variables
(e.g. `thickness_range`, `dhdt_mean`).

Usage example:

python3 src/MPAS-Tools/plot_ensemble_maps_ratio.py \
  --stats_files /path/to/ensembleStats_2025.nc \
  --years 2025 \
  --variables thickness \
  --run_dirs /path/to/run1/output,/path/to/run2/output \
  --run_names run1,run2 \
  --save_base /path/to/save/dir

"""

import os
import sys
import argparse
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.cm as cm


def safe_flatten(var):
    """Return a 1-D numpy array for typical (1,nCells) or (nCells,) variables."""
    a = np.array(var)
    if a.ndim == 2 and a.shape[0] == 1:
        return a[0, :]
    if a.ndim == 1:
        return a
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    return a.ravel()


parser = argparse.ArgumentParser(description="Plot ensemble ratio: <variable>_range / dhdt_mean")
parser.add_argument("--stats_files", required=True,
                    help="Comma-separated ensemble stats NetCDF files (one per year).")
parser.add_argument("--years", required=True, help="Comma-separated years.")
parser.add_argument("--variables", required=True, help="Comma-separated variables (e.g. thickness)")
parser.add_argument("--run_dirs", required=True, help="Comma-separated run output directories (for grounding line overlays)")
parser.add_argument("--run_names", required=True, help="Comma-separated run names (for legend)")
parser.add_argument("--save_base", required=False, default=None, help="Directory to save PNG and NetCDF outputs")
parser.add_argument("--gl_linewidth", required=False, default=0.7, type=float, help="Grounding line linewidth")
parser.add_argument("--cmap", required=False, default="viridis", help="Colormap for ratio plots")

args = parser.parse_args()
stats_files = args.stats_files.split(',')
years = [int(y) for y in args.years.split(',')]
variables = args.variables.split(',')
run_dirs = args.run_dirs.split(',')
run_names = args.run_names.split(',')
save_base = args.save_base
gl_linewidth = args.gl_linewidth
cmap_name = args.cmap

if save_base:
    os.makedirs(save_base, exist_ok=True)

print(f"Plotting ratio for years={years} variables={variables}")

# simple grounding-line loader (only used for overlay aesthetic)

def load_grounding_for_year(run_dirs, run_names, year):
    entries = []
    colors = cm.tab10(np.linspace(0, 1, len(run_dirs)))
    for i, rd in enumerate(run_dirs):
        if not rd:
            continue
        mesh_file = os.path.join(rd, f"output_flux_all_timesteps_{year}_tAvg.nc")
        if not os.path.exists(mesh_file):
            continue
        try:
            m = Dataset(mesh_file)
            xCell = safe_flatten(m.variables['xCell'][:])
            yCell = safe_flatten(m.variables['yCell'][:])
            cellMask = None
            if 'cellMask' in m.variables:
                cmask = m.variables['cellMask'][:]
                if cmask.ndim > 1:
                    cmask = cmask[0]
                cellMask = cmask
            m.close()
            entries.append({'x': xCell, 'y': yCell, 'cellMask': cellMask, 'color': colors[i], 'name': run_names[i] if i < len(run_names) else f'run{i}'})
        except Exception as e:
            print(f"Warning: cannot load mesh {mesh_file}: {e}")
    return entries


for variable in variables:
    if variable != 'thickness':
        print(f"Note: this script is optimised for computing ratio for 'thickness' variable. Skipping {variable}.")
        continue

    for stats_file, year in zip(stats_files, years):
        print(f"Processing {stats_file} (year {year})")
        if not os.path.exists(stats_file):
            print(f"  WARNING: stats file not found: {stats_file}")
            continue
        try:
            f = Dataset(stats_file, 'r')
        except Exception as e:
            print(f"  ERROR opening {stats_file}: {e}")
            continue

        # Required names (conventional)
        range_name = f"{variable}_range"
        dhdt_name = 'dhdt_mean'

        if range_name not in f.variables:
            print(f"  WARNING: {range_name} not found in {stats_file}. Skipping.")
            f.close()
            continue
        if dhdt_name not in f.variables:
            print(f"  WARNING: {dhdt_name} not found in {stats_file}. Skipping.")
            f.close()
            continue

        try:
            arr_range = safe_flatten(f.variables[range_name][:])
            dhdt = safe_flatten(f.variables[dhdt_name][:])
        except Exception as e:
            print(f"  ERROR reading arrays: {e}")
            f.close()
            continue

        # mesh for plotting
        try:
            xCell = safe_flatten(f.variables['xCell'][:])
            yCell = safe_flatten(f.variables['yCell'][:])
            dcEdge = safe_flatten(f.variables['dcEdge'][:]) if 'dcEdge' in f.variables else None
        except Exception as e:
            print(f"  ERROR reading mesh from {stats_file}: {e}")
            f.close()
            continue

        f.close()

        # make sure arrays align
        if arr_range.shape != dhdt.shape:
            # attempt simple broadcast-friendly trimming to min length
            nmin = min(arr_range.size, dhdt.size)
            print(f"  WARNING: arr sizes differ (range={arr_range.size}, dhdt={dhdt.size}). Trimming to {nmin}.")
            arr_range = arr_range[:nmin]
            dhdt = dhdt[:nmin]
            xCell = xCell[:nmin]
            yCell = yCell[:nmin]

        # compute ratio, avoid division by zero
        tiny = 1e-12
        dhdt_safe = dhdt.astype(float)
        dhdt_safe[np.abs(dhdt_safe) < tiny] = np.nan
        ratio = arr_range / dhdt_safe

        # prepare triangulation
        triang = tri.Triangulation(xCell, yCell)
        if dcEdge is not None:
            maxDist = np.max(dcEdge) * 2.0
            triMask = np.zeros(len(triang.triangles), dtype=bool)
            for t in range(len(triang.triangles)):
                triIdx = triang.triangles[t, :]
                # compute simple edge length checks
                if np.max([np.hypot(xCell[triIdx[0]]-xCell[triIdx[1]], yCell[triIdx[0]]-yCell[triIdx[1]]),
                           np.hypot(xCell[triIdx[1]]-xCell[triIdx[2]], yCell[triIdx[1]]-yCell[triIdx[2]]),
                           np.hypot(xCell[triIdx[0]]-xCell[triIdx[2]], yCell[triIdx[0]]-yCell[triIdx[2]])]) > maxDist:
                    triMask[t] = True
            triang.set_mask(triMask)

        # plotting
        fig, ax = plt.subplots(figsize=(7, 7))
        cmap = plt.get_cmap(cmap_name)
        # color limits
        finite_mask = np.isfinite(ratio)
        if not np.any(finite_mask):
            print(f"  INFO: All ratio values are NaN for {stats_file}. Skipping plot.")
            plt.close(fig)
            continue
        vmin = np.nanquantile(ratio, 0.01)
        vmax = np.nanquantile(ratio, 0.99)
        norm = Normalize(vmin=vmin, vmax=vmax)
        tcol = ax.tripcolor(triang, ratio, cmap=cmap, shading='flat', norm=norm)
        ax.set_title(f"{variable} range / dhdt_mean — Year {year}")
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        # overlay grounding lines (best-effort)
        gls = load_grounding_for_year(run_dirs, run_names, year)
        for g in gls:
            if g.get('cellMask') is None:
                continue
            # overlay contour where grounding bit (256) set
            try:
                gl_tr = tri.Triangulation(g['x'], g['y'])
                mask = g['cellMask']
                ax.tricontour(gl_tr, mask, levels=[0.5], colors=[g['color']], linewidths=gl_linewidth)
            except Exception:
                continue

        cbar = fig.colorbar(tcol, ax=ax, orientation='vertical', fraction=0.035, pad=0.03)
        cbar.set_label(f"{variable}_range / dhdt_mean (units: derived)")

        if save_base:
            out_png = os.path.join(save_base, f"ensemble_ratio_{variable}_range_{year}.png")
            fig.savefig(out_png, dpi=300, bbox_inches='tight')
            print(f"  Saved plot {out_png}")
            # write compact NetCDF
            out_nc = os.path.join(save_base, f"ensemble_ratio_{variable}_range_{year}.nc")
            try:
                ncw = Dataset(out_nc, 'w')
                ncw.createDimension('nCells', xCell.size)
                xv = ncw.createVariable('xCell', 'f8', ('nCells',))
                yv = ncw.createVariable('yCell', 'f8', ('nCells',))
                rv = ncw.createVariable(f"{variable}_range_over_dhdt_mean", 'f8', ('nCells',), zlib=True)
                xv[:] = xCell
                yv[:] = yCell
                rv[:] = ratio
                rv.units = 'derived (thickness / (thickness/yr))'
                ncw.close()
                print(f"  Saved ratio NetCDF {out_nc}")
            except Exception as e:
                print(f"  WARNING: failed to write ratio NetCDF: {e}")
        plt.close(fig)

print('Done')
