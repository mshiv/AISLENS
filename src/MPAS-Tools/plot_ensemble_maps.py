#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble snapshot map plotting for MALI.
Plots min, max, mean, std, range for specified variables and years.
Overlays grounding lines for all runs in ensemble, with adjustable line thickness.
For each plotted year, grounding lines are extracted from that same year's data for each run.

Output modes:
  - Single-panel (default): One figure per (variable, stat, year)
  - Multi-panel (--multi-panel): One figure per (variable, stat) with all years side-by-side
"""

import os
import numpy as np
from netCDF4 import Dataset
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap, LogNorm
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="Ensemble map plots for MALI with year-specific grounding lines.")
parser.add_argument("--ensemble_files", required=True, help="Comma-separated ensemble stats NetCDF files (one per year).")
parser.add_argument("--years", required=True, help="Comma-separated years.")
parser.add_argument("--variables", required=True, help="Comma-separated variables.")
parser.add_argument("--run_dirs", required=True, help="Comma-separated run output directories.")
parser.add_argument("--run_names", required=True, help="Comma-separated run names (for legend).")
parser.add_argument("--save_base", required=False, default=None, help="Path to directory for saving figures (if not provided, figures are not saved).")
parser.add_argument("--gl_linewidth", required=False, default=1.25, type=float, help="Linewidth for grounding lines (default: 0.7)")
parser.add_argument("--single_gl", action="store_true", help="If set, plot only a single grounding line instead of all runs (default: False)")
parser.add_argument("--gl_run_index", required=False, type=int, default=None, help="If --single_gl, use this run index (0-based) to select which run's grounding line to plot")
parser.add_argument("--gl_run_name", required=False, default=None, help="If --single_gl, select grounding line by run name (matches run_names entry)")
parser.add_argument("--log_cbar", action="store_true", help="Use log scale for the colorbar (requires positive data).")
parser.add_argument("--multi_panel", action="store_true", help="Output multi-panel figures (all years on one plot) instead of single-year figures.")

args = parser.parse_args()
ensemble_files = args.ensemble_files.split(',')
years = [int(y) for y in args.years.split(',')]
variables = args.variables.split(',')
run_dirs = args.run_dirs.split(',')
run_names = args.run_names.split(',')
save_base = args.save_base
gl_linewidth = args.gl_linewidth
single_gl = args.single_gl
gl_run_index = args.gl_run_index
gl_run_name = args.gl_run_name
use_log_cbar = args.log_cbar
multi_panel = args.multi_panel

print(f"Processing {len(ensemble_files)} ensemble files for years: {years}")
print(f"Variables: {variables}")
print(f"Using {len(run_dirs)} run directories for year-specific grounding line overlays")
print(f"Grounding line linewidth: {gl_linewidth}")
print(f"Output mode: {'multi-panel' if multi_panel else 'single-panel'}")

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
    colors = ['Darkred', 'Darkorange', 'Navajowhite', 'white', 
              'Lightsteelblue', 'Royalblue', 'Navy']
    return LinearSegmentedColormap.from_list("custom", colors, N=200)

def get_colormap_and_norm(variable, stat, arr, use_log=False):
    """Determine colormap and normalization for a variable/stat combination."""
    pos_mask = np.isfinite(arr) & (arr > 0)
    
    # Check if log scale is appropriate
    actual_use_log = use_log or (stat == 'std')
    if actual_use_log and (variable == 'dhdt' or stat == 'range'):
        actual_use_log = False
    if actual_use_log and not np.any(pos_mask):
        actual_use_log = False
    
    # Compute bounds
    if actual_use_log:
        vmin = np.nanquantile(arr[pos_mask], 0.01)
        vmax = np.nanquantile(arr[pos_mask], 0.99)
    else:
        vmin = np.nanquantile(arr, 0.01)
        vmax = np.nanquantile(arr, 0.99)
    
    # Determine colormap and normalization
    if variable == 'dhdt' or stat == 'range':
        max_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -max_abs, max_abs
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
        cmap = create_custom_colormap()
    elif variable == 'surfaceSpeed':
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('plasma')
    else:
        if actual_use_log and vmin > 0 and vmax > 0:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(defaultColors.get(variable, 'viridis'))
    
    return cmap, norm, vmin, vmax

def load_grounding_lines_for_year(run_dirs, run_names, target_year):
    """Load grounding line information for a specific year from all runs"""
    grounding_lines = []
    gl_colors = cm.tab10(np.linspace(0, 1, len(run_dirs)))  # Generate colors for each run
    
    print(f"Loading grounding lines for year {target_year}...")
    
    for i, run_dir in enumerate(run_dirs):
        if not run_dir.strip():
            continue
        
        # Construct the mesh file path for this specific year
        mesh_file = os.path.join(run_dir, f"output_flux_all_timesteps_{target_year}_tAvg.nc")
        
        try:
            print(f"  Loading mesh file: {mesh_file}")
            if not os.path.exists(mesh_file):
                print(f"  WARNING: Mesh file does not exist: {mesh_file}")
                continue
                
            m = Dataset(mesh_file, 'r')
            
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
                print(f"  WARNING: No cellMask found in {mesh_file}")
            
            grounding_lines.append({
                'x': xCell, 
                'y': yCell, 
                'gl_mask': gl_mask,
                'extent_mask': initial_extent_mask,
                'color': gl_colors[i],
                'run_name': run_names[i] if i < len(run_names) else f'Run_{i+1}',
                'year': target_year
            })
            m.close()
            print(f"  Successfully loaded GL for {run_names[i] if i < len(run_names) else f'Run_{i+1}'}")
            
        except Exception as e:
            print(f"  ERROR loading {mesh_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(grounding_lines)} grounding line datasets for year {target_year}")
    return grounding_lines

# Ensure save_base directory exists if provided
if save_base is not None and save_base != "":
    os.makedirs(save_base, exist_ok=True)


def load_year_data(stats_file, variable, stat, year):
    """Load data for a single year/variable/stat combination.
    Returns dict with arr, xCell, yCell, dcEdge, units, triang, maxDist or None on failure."""
    if not stats_file.strip():
        print(f"      Skipping empty stats file for year {year}")
        return None
    try:
        f = Dataset(stats_file, 'r')
        varname = f"{variable}_{stat}"
        if varname not in f.variables:
            print(f"      WARNING: {varname} not found in {stats_file}")
            f.close()
            return None
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
        # Get mesh info
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
        return {'arr': arr, 'xCell': xCell, 'yCell': yCell, 'dcEdge': dcEdge,
                'units': units, 'triang': triang, 'maxDist': maxDist}
    except Exception as e:
        print(f"      ERROR loading or preparing data for {year}: {e}")
        return None


def filter_grounding_lines(grounding_lines, single_gl, gl_run_name, gl_run_index):
    """Filter grounding lines to a single entry if requested."""
    if single_gl and grounding_lines:
        sel = None
        if gl_run_name is not None:
            for g in grounding_lines:
                if g.get('run_name') == gl_run_name:
                    sel = g
                    break
        if sel is None and gl_run_index is not None:
            if 0 <= gl_run_index < len(grounding_lines):
                sel = grounding_lines[gl_run_index]
        if sel is None:
            sel = grounding_lines[0]
        return [sel]
    return grounding_lines


def plot_grounding_lines(ax, grounding_lines_to_plot, maxDist, gl_linewidth, year):
    """Plot grounding lines on an axis and return legend elements."""
    legend_elements = []
    for gl_info in grounding_lines_to_plot:
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
                                                  lw=gl_linewidth, label=f"GL {gl_info['run_name']} ({year})"))
            except Exception as e:
                print(f"      WARNING: Could not plot grounding line for {gl_info['run_name']}: {e}")
    return legend_elements


def plot_single_panel(variable, stat, years, ensemble_files, run_dirs, run_names,
                      save_base, gl_linewidth, single_gl, gl_run_index, gl_run_name, use_log_cbar):
    """Single-panel mode: one figure per (variable, stat, year)."""
    for i, (stats_file, year) in enumerate(zip(ensemble_files, years)):
        print(f"    Year {year}: Loading {stats_file}")
        data = load_year_data(stats_file, variable, stat, year)
        if data is None:
            continue

        arr = data['arr']
        if np.all(np.isnan(arr)):
            print(f"      All data is NaN for {variable}_{stat}, year {year}.")
            continue

        cmap, norm, vmin, vmax = get_colormap_and_norm(variable, stat, arr, use_log_cbar)
        print(f"      Color range: {vmin:.3f} to {vmax:.3f}")

        grounding_lines = load_grounding_lines_for_year(run_dirs, run_names, year)
        grounding_lines_to_plot = filter_grounding_lines(grounding_lines, single_gl, gl_run_name, gl_run_index)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        h = ax.tripcolor(data['triang'], arr, cmap=cmap, shading='flat', norm=norm)
        ax.set_title(f"{variable} [{stat}] Year {year}", fontsize=13)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        legend_elements = plot_grounding_lines(ax, grounding_lines_to_plot, data['maxDist'], gl_linewidth, year)
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower right', fontsize='small')

        cbar = fig.colorbar(h, ax=ax, orientation='vertical', fraction=0.035, pad=0.03)
        cbar.set_label(f"{variable} [{stat}] ({data['units']})", rotation=270, labelpad=20)

        fig.tight_layout()
        if save_base is not None and save_base != "":
            out_png = os.path.join(save_base, f"ensemble_{variable}_{stat}_{year}.png")
            fig.savefig(out_png, dpi=400, bbox_inches='tight')
            print(f"      Saved {out_png}")
        plt.close(fig)


def plot_multi_panel(variable, stat, years, ensemble_files, run_dirs, run_names,
                     save_base, gl_linewidth, single_gl, gl_run_index, gl_run_name, use_log_cbar):
    """Multi-panel mode: one figure per (variable, stat) with all years side-by-side."""
    import matplotlib.gridspec as gridspec

    # First pass: load all data and compute global color scale
    year_data = []
    all_values = []
    units = "unknown"

    for stats_file, year in zip(ensemble_files, years):
        print(f"    Year {year}: Loading {stats_file}")
        data = load_year_data(stats_file, variable, stat, year)
        if data is not None and not np.all(np.isnan(data['arr'])):
            year_data.append((year, data))
            all_values.append(data['arr'])
            units = data['units']
        else:
            year_data.append((year, None))
            if data is None:
                print(f"      Skipping year {year} (load failed)")
            else:
                print(f"      Skipping year {year} (all NaN)")

    if not all_values:
        print(f"    No valid data for {variable}_{stat}, skipping.")
        return

    # Compute unified colormap across all years
    all_flat = np.concatenate([a[~np.isnan(a)] for a in all_values])
    cmap, norm, vmin, vmax = get_colormap_and_norm(variable, stat, all_flat, use_log_cbar)
    print(f"    Unified color range: {vmin:.3f} to {vmax:.3f}")

    # Create multi-panel figure
    n_panels = len([yd for yd in year_data if yd[1] is not None])
    if n_panels == 0:
        return

    fig = plt.figure(figsize=(5 * n_panels, 6))
    gs = gridspec.GridSpec(1, n_panels, wspace=0.05)

    panel_idx = 0
    axes = []
    last_h = None

    for year, data in year_data:
        if data is None:
            continue

        ax = fig.add_subplot(gs[panel_idx])
        axes.append(ax)

        h = ax.tripcolor(data['triang'], data['arr'], cmap=cmap, shading='flat', norm=norm)
        last_h = h
        ax.set_title(f"Year {year}", fontsize=12)
        ax.set_aspect('equal')

        if panel_idx == 0:
            ax.set_ylabel('y (m)')
        else:
            ax.set_yticklabels([])

        ax.set_xlabel('x (m)')

        # Load and plot grounding lines for this year
        grounding_lines = load_grounding_lines_for_year(run_dirs, run_names, year)
        grounding_lines_to_plot = filter_grounding_lines(grounding_lines, single_gl, gl_run_name, gl_run_index)
        plot_grounding_lines(ax, grounding_lines_to_plot, data['maxDist'], gl_linewidth, year)

        panel_idx += 1

    # Add single colorbar for all panels
    if last_h is not None and axes:
        cbar = fig.colorbar(last_h, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label(f"{variable} [{stat}] ({units})", rotation=270, labelpad=20)

    # Add legend for grounding lines (from last panel's data)
    if grounding_lines_to_plot:
        legend_elements = [plt.Line2D([0], [0], color=gl['color'], lw=gl_linewidth,
                                      label=f"GL {gl['run_name']}") for gl in grounding_lines_to_plot]
        axes[-1].legend(handles=legend_elements, loc='lower right', fontsize='x-small')

    fig.suptitle(f"MALI Ensemble: {variable} [{stat}]", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_base is not None and save_base != "":
        out_png = os.path.join(save_base, f"ensemble_{variable}_{stat}_multipanel.png")
        fig.savefig(out_png, dpi=400, bbox_inches='tight')
        print(f"    Saved {out_png}")
    plt.close(fig)


# Main plotting loop
for variable in variables:
    print(f"\nProcessing variable: {variable}")
    for stat in stat_types:
        print(f"  Processing statistic: {stat}")
        if multi_panel:
            plot_multi_panel(variable, stat, years, ensemble_files, run_dirs, run_names,
                            save_base, gl_linewidth, single_gl, gl_run_index, gl_run_name, use_log_cbar)
        else:
            plot_single_panel(variable, stat, years, ensemble_files, run_dirs, run_names,
                             save_base, gl_linewidth, single_gl, gl_run_index, gl_run_name, use_log_cbar)

print("All ensemble plots complete.")