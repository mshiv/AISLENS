#!/usr/bin/env python3
"""
Script to analyze ensemble thickness variability and mean change.

Calculates:
1. Range of thickness across ensemble members at each time step
2. Ensemble mean thickness change from initial year
3. Ratio of range to mean change

Usage:
    python plot_ensemble_thickness_analysis.py --ensemble-dir /path/to/ensemble --experiments EM1,EM2,EM4 --time-index 120
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import glob

print("** Gathering information. (Invoke with --help for more details.)")
parser = OptionParser(description=__doc__)
parser.add_option("-r", "--root", dest="rootDataDir", help="Root data directory path", metavar="PATH")
parser.add_option("-b", "--base", dest="ensembleDir", help="Ensemble directory name", metavar="DIRNAME", required=True)
parser.add_option("-e", "--experiments", dest="experimentList", help="Comma-separated list of experiment names", metavar="EXP1,EXP2", required=True)
parser.add_option("-m", "--mesh", dest="meshFile", help="Mesh file path (optional - can use coordinates from output files)", metavar="FILENAME")
parser.add_option("-t", "--time-index", dest="timeIndex", help="Time index to analyze (default: -1 for last timestep)", default="-1", metavar="N")
parser.add_option("--var-name", dest="varName", help="Thickness variable name (default: thickness)", default="thickness", metavar="VARNAME")
parser.add_option("--output-dir", dest="outputDir", help="Output directory for plots", metavar="PATH")
parser.add_option("-s", "--save", dest="saveFigs", help="Save figures to files", action='store_true', default=False)
parser.add_option("--dpi", dest="dpi", help="DPI for saved figures (default: 300)", default="300", metavar="N")
parser.add_option("--file-pattern", dest="filePattern", 
                  help="File pattern to match (default: output*.nc)", 
                  default="output*.nc", metavar="PATTERN")

options, args = parser.parse_args()

# Parse inputs
experiment_names = [exp.strip() for exp in options.experimentList.split(',')]
num_experiments = len(experiment_names)
time_index = int(options.timeIndex)

print(f"Analyzing {num_experiments} experiments: {experiment_names}")
print(f"Using time index: {time_index}")

# Build file paths for each experiment
ensemble_base = os.path.join(options.rootDataDir, options.ensembleDir) if options.rootDataDir else options.ensembleDir

experiment_files = []
for exp_name in experiment_names:
    exp_dir = os.path.join(ensemble_base, exp_name)
    
    # Look for output files (adjust pattern as needed for your file naming)
    output_pattern = os.path.join(exp_dir, options.filePattern)
    files = sorted(glob.glob(output_pattern))
    
    if not files:
        print(f"Warning: No output files found for {exp_name} in {exp_dir}")
        continue
    
    # Use the last file or a specific pattern
    experiment_files.append((exp_name, files[-1]))  # Adjust if you need different file selection

    # OPTION 1: If you have specific output file naming pattern
    # output_pattern = os.path.join(exp_dir, "landice_output*.nc")
    
    # OPTION 2: If you have timestamped files
    # output_pattern = os.path.join(exp_dir, "*_restart.nc")
    
    # OPTION 3: If you have a specific output file name
    # output_file = os.path.join(exp_dir, "output.nc")
    # if os.path.exists(output_file):
    #     experiment_files.append((exp_name, output_file))
    
    # OPTION 4: If you need a specific time range file
    # output_pattern = os.path.join(exp_dir, "output.2000-01-01_00.00.00.nc")

if len(experiment_files) != num_experiments:
    print(f"Warning: Only found {len(experiment_files)} of {num_experiments} experiment files")

# Read mesh/coordinate information
if options.meshFile:
    print(f"Loading mesh from: {options.meshFile}")
    mesh_data = Dataset(options.meshFile, 'r')
    xCell = mesh_data.variables['xCell'][:]
    yCell = mesh_data.variables['yCell'][:]
    mesh_data.close()
    nCells = len(xCell)
    print(f"Mesh has {nCells} cells")
else:
    print("No mesh file specified - will try to read coordinates from output files")
    xCell = None
    yCell = None
    nCells = None

# Initialize arrays to store thickness data from all experiments
thickness_data = []
initial_thickness_data = []

print("\nReading thickness data from experiments...")
# When reading first file, get coordinates if not from mesh
for exp_name, file_path in experiment_files:
    print(f"  Reading {exp_name}: {file_path}")
    
    try:
        f = Dataset(file_path, 'r')
        
        # Get coordinates from file if not already loaded
        if xCell is None and 'xCell' in f.variables:
            print("    Reading coordinates from output file")
            xCell = f.variables['xCell'][:]
            yCell = f.variables['yCell'][:]
            nCells = len(xCell)
        
        # Get thickness variable
        if options.varName not in f.variables:
            print(f"    Warning: Variable '{options.varName}' not found in {exp_name}")
            f.close()
            continue
        
        thickness_var = f.variables[options.varName]
        
        # Read initial thickness (time index 0)
        initial_thickness = thickness_var[0, :]
        initial_thickness_data.append(initial_thickness)
        
        # Read thickness at specified time
        if time_index == -1:
            time_idx = thickness_var.shape[0] - 1
        else:
            time_idx = time_index
            
        thickness_at_time = thickness_var[time_idx, :]
        thickness_data.append(thickness_at_time)
        
        print(f"    Loaded thickness at time index {time_idx}")
        print(f"    Thickness range: {np.min(thickness_at_time):.2f} to {np.max(thickness_at_time):.2f} m")
        
        f.close()
        
    except Exception as e:
        print(f"    Error reading {exp_name}: {e}")
        continue

if len(thickness_data) == 0:
    sys.exit("ERROR: No thickness data loaded successfully")

# Convert to numpy arrays
thickness_data = np.array(thickness_data)  # Shape: (n_experiments, nCells)
initial_thickness_data = np.array(initial_thickness_data)

print(f"\nLoaded thickness data for {len(thickness_data)} experiments")

# Calculate metrics
print("\nCalculating ensemble metrics...")

# 1. Range across ensemble members at time t
thickness_range = np.max(thickness_data, axis=0) - np.min(thickness_data, axis=0)
print(f"  Thickness range: {np.min(thickness_range):.2f} to {np.max(thickness_range):.2f} m")

# 2. Ensemble mean thickness change from initial
ensemble_mean_thickness = np.mean(thickness_data, axis=0)
ensemble_mean_initial = np.mean(initial_thickness_data, axis=0)
ensemble_mean_change = ensemble_mean_thickness - ensemble_mean_initial
print(f"  Mean thickness change: {np.min(ensemble_mean_change):.2f} to {np.max(ensemble_mean_change):.2f} m")

# 3. Ratio of range to absolute mean change
# Add small epsilon to avoid division by zero
epsilon = 1e-6
abs_mean_change = np.abs(ensemble_mean_change)
ratio = thickness_range / (abs_mean_change + epsilon)

# Mask out areas with very small changes (noise)
threshold = 10.0  # meters - adjust as needed
ratio_masked = ma.masked_where(abs_mean_change < threshold, ratio)

print(f"  Ratio range: {np.min(ratio_masked):.2f} to {np.max(ratio_masked):.2f}")
print(f"  (masked where |mean change| < {threshold} m)")

# Check if we have coordinates for plotting
has_coordinates = (xCell is not None and yCell is not None)

if not has_coordinates:
    print("\nWarning: No spatial coordinates available")
    print("Will only generate statistics, not spatial maps")

# Create plots
print("\nGenerating plots...")

if has_coordinates:
    # Create full figure with spatial maps
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Thickness Range
    ax1 = fig.add_subplot(2, 2, 1)
    sc1 = ax1.scatter(xCell/1000, yCell/1000, c=thickness_range, s=1, cmap='viridis')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_title(f'Thickness Range Across {num_experiments} Ensemble Members')
    ax1.set_aspect('equal')
    plt.colorbar(sc1, ax=ax1, label='Thickness Range (m)')

    # Plot 2: Ensemble Mean Thickness Change
    ax2 = fig.add_subplot(2, 2, 2)
    sc2 = ax2.scatter(xCell/1000, yCell/1000, c=ensemble_mean_change, s=1, cmap='RdBu_r', 
                      vmin=-np.max(np.abs(ensemble_mean_change)), vmax=np.max(np.abs(ensemble_mean_change)))
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.set_title('Ensemble Mean Thickness Change from Initial')
    ax2.set_aspect('equal')
    plt.colorbar(sc2, ax=ax2, label='Thickness Change (m)')

    # Plot 3: Ratio (log scale)
    ax3 = fig.add_subplot(2, 2, 3)
    # Use log scale for better visualization
    ratio_plot = ma.log10(ratio_masked)
    sc3 = ax3.scatter(xCell/1000, yCell/1000, c=ratio_plot, s=1, cmap='plasma')
    ax3.set_xlabel('X (km)')
    ax3.set_ylabel('Y (km)')
    ax3.set_title('Log10(Range / |Mean Change|)')
    ax3.set_aspect('equal')
    cbar3 = plt.colorbar(sc3, ax=ax3, label='Log10(Ratio)')

    # Plot 4: Statistics histogram
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(ratio_masked.compressed(), bins=50, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Range / |Mean Change|')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Thickness Variability Ratio')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Ensemble: {options.ensembleDir}\n"
    stats_text += f"Experiments: {num_experiments}\n"
    stats_text += f"Time index: {time_index}\n\n"
    stats_text += f"Range stats:\n"
    stats_text += f"  Mean: {np.mean(thickness_range):.2f} m\n"
    stats_text += f"  Median: {np.median(thickness_range):.2f} m\n"
    stats_text += f"  Max: {np.max(thickness_range):.2f} m\n\n"
    stats_text += f"Mean change stats:\n"
    stats_text += f"  Mean: {np.mean(ensemble_mean_change):.2f} m\n"
    stats_text += f"  Median: {np.median(ensemble_mean_change):.2f} m\n"
    stats_text += f"  Max |change|: {np.max(np.abs(ensemble_mean_change)):.2f} m\n\n"
    stats_text += f"Ratio stats (where |change| > {threshold} m):\n"
    stats_text += f"  Mean: {np.mean(ratio_masked):.2f}\n"
    stats_text += f"  Median: {np.median(ratio_masked):.2f}\n"
    stats_text += f"  90th percentile: {np.percentile(ratio_masked.compressed(), 90):.2f}"

    ax4.text(1.15, 0.5, stats_text, transform=ax4.transAxes, 
             fontsize=8, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'Ensemble Thickness Analysis: {options.ensembleDir}', fontsize=14, fontweight='bold')
    plt.tight_layout()

else:
    # Create figure with only statistics (no spatial maps)
    print("Creating statistics-only plots (no spatial data)")
    fig = plt.figure(figsize=(12, 8))

    # Plot 1: Range histogram
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(thickness_range, bins=50, alpha=0.7, edgecolor='black', color='blue')
    ax1.set_xlabel('Thickness Range (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of Thickness Range\nAcross {num_experiments} Ensemble Members')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean change histogram
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(ensemble_mean_change, bins=50, alpha=0.7, edgecolor='black', color='red')
    ax2.set_xlabel('Mean Thickness Change (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Ensemble Mean Thickness Change')
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Ratio histogram
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(ratio_masked.compressed(), bins=50, alpha=0.7, edgecolor='black', color='purple')
    ax3.set_xlabel('Range / |Mean Change|')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Thickness Variability Ratio')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Statistics summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Add statistics text
    stats_text = f"Ensemble: {options.ensembleDir}\n"
    stats_text += f"Experiments: {num_experiments}\n"
    stats_text += f"Time index: {time_index}\n\n"
    stats_text += f"Range statistics:\n"
    stats_text += f"  Mean: {np.mean(thickness_range):.2f} m\n"
    stats_text += f"  Median: {np.median(thickness_range):.2f} m\n"
    stats_text += f"  Std dev: {np.std(thickness_range):.2f} m\n"
    stats_text += f"  Max: {np.max(thickness_range):.2f} m\n\n"
    stats_text += f"Mean change statistics:\n"
    stats_text += f"  Mean: {np.mean(ensemble_mean_change):.2f} m\n"
    stats_text += f"  Median: {np.median(ensemble_mean_change):.2f} m\n"
    stats_text += f"  Std dev: {np.std(ensemble_mean_change):.2f} m\n"
    stats_text += f"  Max |change|: {np.max(np.abs(ensemble_mean_change)):.2f} m\n\n"
    stats_text += f"Ratio statistics (|change| > {threshold} m):\n"
    stats_text += f"  Mean: {np.mean(ratio_masked):.2f}\n"
    stats_text += f"  Median: {np.median(ratio_masked):.2f}\n"
    stats_text += f"  25th percentile: {np.percentile(ratio_masked.compressed(), 25):.2f}\n"
    stats_text += f"  75th percentile: {np.percentile(ratio_masked.compressed(), 75):.2f}\n"
    stats_text += f"  90th percentile: {np.percentile(ratio_masked.compressed(), 90):.2f}\n\n"
    stats_text += f"Number of cells analyzed: {nCells}\n"
    stats_text += f"Cells with |change| > {threshold} m: {np.sum(abs_mean_change > threshold)}"

    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle(f'Ensemble Thickness Analysis: {options.ensembleDir}\n(Statistics Only - No Mesh File Provided)', 
                 fontsize=14, fontweight='bold')

# Save or show
if options.saveFigs:
    # Use specified output dir, or default to ensemble base directory
    output_dir = options.outputDir if options.outputDir else ensemble_base
    os.makedirs(output_dir, exist_ok=True)
    
    # PNG plot
    output_file = os.path.join(output_dir, f'thickness_analysis_t{time_index}.png')
    plt.savefig(output_file, dpi=int(options.dpi), bbox_inches='tight')
    print(f"\nFigure saved: {output_file}")
    
    # NetCDF data
    output_nc = os.path.join(output_dir, f'thickness_analysis_t{time_index}.nc')
    print(f"Saving analysis data to: {output_nc}")
    
    nc_out = Dataset(output_nc, 'w', format='NETCDF4')
    nc_out.createDimension('nCells', nCells)
    nc_out.createDimension('nExperiments', num_experiments)
    
    # Write variables
    var_range = nc_out.createVariable('thicknessRange', 'f8', ('nCells',))
    var_range[:] = thickness_range
    var_range.long_name = 'Range of thickness across ensemble members'
    var_range.units = 'm'
    
    var_mean_change = nc_out.createVariable('ensembleMeanChange', 'f8', ('nCells',))
    var_mean_change[:] = ensemble_mean_change
    var_mean_change.long_name = 'Ensemble mean thickness change from initial'
    var_mean_change.units = 'm'
    
    var_ratio = nc_out.createVariable('rangeToChangeRatio', 'f8', ('nCells',))
    var_ratio[:] = ratio
    var_ratio.long_name = 'Ratio of thickness range to absolute mean change'
    var_ratio.units = 'dimensionless'
    
    # Add metadata
    nc_out.ensemble_directory = options.ensembleDir
    nc_out.experiments = ','.join([exp for exp, _ in experiment_files])



    nc_out.masking_threshold = threshold    nc_out.time_index = time_index    nc_out.num_experiments = num_experiments    
    nc_out.close()
    print(f"Analysis data saved: {output_nc}")

plt.show()

print("\nAnalysis complete!")
