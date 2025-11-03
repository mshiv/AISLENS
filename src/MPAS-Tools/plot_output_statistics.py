#!/usr/bin/env python
'''
Script to plot probability distribution functions of variables from multiple 
landice globalStats files at each time step.

Based on plot_ensembleGlobalStats.py
Author: Shiva Muruganandham
Date: August 28, 2025
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
from optparse import OptionParser
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

rhoi = 910.0
rhosw = 1028.

print("** Gathering information.  (Invoke with --help for more details. All arguments are optional)")
parser = OptionParser(description=__doc__)
parser.add_option("-r", "--root", dest="rootDataDir", help="Root data directory path", metavar="PATH")
parser.add_option("-b", "--base", dest="ensembleBaseDir", help="Ensemble base directory (relative to root)", metavar="DIRNAME")
parser.add_option("-e", "--experiments", dest="experimentList", help="Comma-separated list of experiment run names (subdirectories of base)", metavar="EXP1,EXP2,EXP3")
parser.add_option("-f", "--filename", dest="statsFilename", help="Statistics filename to look for in each experiment directory", default="globalStats.nc", metavar="FILENAME")
parser.add_option("-u", dest="units", help="units for mass/volume: m3, kg, Gt", default="Gt", metavar="UNITS")
parser.add_option("-v", "--variable", dest="variable", help="Variable to analyze", default="volumeAboveFloatation", metavar="VARNAME")
parser.add_option("-t", "--timesteps", dest="timeSteps", help="Comma-separated list of time steps (years) to plot PDFs for", default="0,50,100,200,300", metavar="T1,T2,T3")
parser.add_option("-c", dest="plotChange", help="plot time series as absolute change from initial", action='store_true', default=False)
parser.add_option("-p", dest="plotPercentChange", help="plot time series as percentage change from initial", action='store_true', default=False)
parser.add_option("-s", dest="plotSave", help="save figure", metavar="FILENAME")
options, args = parser.parse_args()

# Check for conflicting options
if options.plotChange and options.plotPercentChange:
    sys.exit("ERROR: Cannot use both -c (absolute change) and -p (percentage change) options simultaneously")

print("Using ice density of {} kg/m3 if required for unit conversions".format(rhoi))

# Parse experiment list
if not options.experimentList:
    sys.exit("ERROR: Must specify experiment list with -e/--experiments option")

experiment_names = [exp.strip() for exp in options.experimentList.split(',')]
num_experiments = len(experiment_names)

# Parse time steps for PDF plotting
time_steps = [float(t.strip()) for t in options.timeSteps.split(',')]

print(f"Processing {num_experiments} experiments: {experiment_names}")
print(f"Analyzing variable: {options.variable}")
print(f"Will plot PDFs at time steps: {time_steps} years")

# Build file paths and validate
experiment_files = []
for exp_name in experiment_names:
    if options.rootDataDir and options.ensembleBaseDir:
        file_path = os.path.join(options.rootDataDir, options.ensembleBaseDir, exp_name, options.statsFilename)
    elif options.ensembleBaseDir:
        file_path = os.path.join(options.ensembleBaseDir, exp_name, options.statsFilename)
    else:
        sys.exit("ERROR: Must specify at least ensemble base directory (-b/--base)")
    
    if not os.path.exists(file_path):
        sys.exit(f"ERROR: File not found: {file_path}")
    
    experiment_files.append(file_path)
    print(f"Found file for {exp_name}: {file_path}")

# Set up units
if options.units == "m3":
   massUnit = "m$^3$"
   scaleVol = 1.
elif options.units == "kg":
   massUnit = "kg"
   scaleVol = 1./rhoi
elif options.units == "Gt":
   massUnit = "Gt"
   scaleVol = 1.0e12 / rhoi
else:
   sys.exit("Unknown mass/volume units")

print("Using volume/mass units of: ", massUnit)

def extract_variable_data(fname, exp_name):
    """Extract variable time series from a single experiment file"""
    print(f"Reading {options.variable} data from: {fname} for experiment: {exp_name}")
    
    f = Dataset(fname, 'r')
    
    # Check if variable exists
    if options.variable not in f.variables:
        f.close()
        raise ValueError(f"Variable '{options.variable}' not found in file {fname}")
    
    yr = f.variables['daysSinceStart'][:]/365.0
    yr = yr - yr[0]  # Start from year 0
    
    var_data = f.variables[options.variable][:]
    
    # Apply scaling for volume/mass variables
    if 'volume' in options.variable.lower() or 'vaf' in options.variable.lower():
        var_data = var_data / scaleVol
    elif 'area' in options.variable.lower():
        var_data = var_data / 1000.0**2  # Convert to km^2
    elif 'flux' in options.variable.lower() and options.units == "Gt":
        var_data = var_data / 1e12  # Convert to Gt/yr for flux variables
    
    # Apply change calculations if requested
    if options.plotChange:
        var_data = var_data - var_data[0]
    elif options.plotPercentChange:
        var_data = (var_data - var_data[0])*100/var_data[0]
    
    f.close()
    
    return yr, var_data

def interpolate_to_timesteps(years, var_data, target_times):
    """Interpolate variable data to specific time steps"""
    # Only interpolate within the available time range
    valid_times = [t for t in target_times if t >= years.min() and t <= years.max()]
    
    if len(valid_times) == 0:
        return np.array([]), np.array([])
    
    # Create interpolation function
    interp_func = interp1d(years, var_data, kind='linear', bounds_error=False, fill_value=np.nan)
    
    # Interpolate to target times
    interpolated_var = interp_func(valid_times)
    
    return np.array(valid_times), interpolated_var

# Collect variable data from all experiments
all_var_data = {}  # Dictionary: {time_step: [var_values]}
experiment_time_ranges = []

for exp_file, exp_name in zip(experiment_files, experiment_names):
    years, var_data = extract_variable_data(exp_file, exp_name)
    experiment_time_ranges.append((years.min(), years.max()))
    
    # Interpolate to target time steps
    interp_times, interp_var = interpolate_to_timesteps(years, var_data, time_steps)
    
    # Store interpolated values
    for i, time_step in enumerate(interp_times):
        if time_step not in all_var_data:
            all_var_data[time_step] = []
        all_var_data[time_step].append(interp_var[i])

# Remove time steps that don't have enough data points
min_experiments = max(3, int(0.5 * num_experiments))  # Need at least 3 or 50% of experiments
valid_time_steps = []
for time_step in sorted(all_var_data.keys()):
    valid_data = [v for v in all_var_data[time_step] if not np.isnan(v)]
    if len(valid_data) >= min_experiments:
        valid_time_steps.append(time_step)
        all_var_data[time_step] = valid_data
    else:
        print(f"Warning: Skipping time step {time_step} - only {len(valid_data)} valid experiments")

print(f"Valid time steps for PDF analysis: {valid_time_steps}")

# Create figure for PDF plots
n_plots = len(valid_time_steps)
ncols = min(3, n_plots)
nrows = int(np.ceil(n_plots / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), facecolor='w')
if n_plots == 1:
    axes = [axes]
elif nrows == 1:
    axes = axes
else:
    axes = axes.flatten()

# Plot PDFs for each valid time step
for i, time_step in enumerate(valid_time_steps):
    ax = axes[i]
    var_values = np.array(all_var_data[time_step])
    
    # Calculate skewness with error handling
    try:
        if np.std(var_values) > 1e-10:  # Check if there's meaningful variation
            skewness = stats.skew(var_values)
        else:
            skewness = 0.0  # No variation means symmetric
    except:
        skewness = 0.0

    # Plot histogram with automatic binning
    ax.hist(var_values, bins='auto', alpha=0.7, density=True, 
            color='skyblue', edgecolor='black', label='Data')

    # Fit and plot normal distribution
    if len(var_values) >= 3 and np.std(var_values) > 1e-10:
        try:
            mu, sigma = stats.norm.fit(var_values)
            x_range = np.linspace(var_values.min(), var_values.max(), 100)
            normal_pdf = stats.norm.pdf(x_range, mu, sigma)
            ax.plot(x_range, normal_pdf, 'r-', linewidth=2, 
                    label=f'Normal fit\n(μ={mu:.2f}, σ={sigma:.2f})\nSkewness={skewness:.2f}')
        except:
            # If fitting fails, just show the skewness
            ax.text(0.02, 0.98, f'Skewness: {skewness:.2f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    else:
        # For low-variation data, just show basic info
        ax.text(0.02, 0.98, f'Low variation\nSkewness: {skewness:.2f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Determine appropriate units based on variable type
    if 'volume' in options.variable.lower() or 'vaf' in options.variable.lower():
        if options.plotChange:
            unit_str = f' change ({massUnit})'
        elif options.plotPercentChange:
            unit_str = ' change (%)'
        else:
            unit_str = f' ({massUnit})'
    elif 'area' in options.variable.lower():
        if options.plotChange:
            unit_str = ' change (km²)'
        elif options.plotPercentChange:
            unit_str = ' change (%)'
        else:
            unit_str = ' (km²)'
    elif 'flux' in options.variable.lower():
        if options.units == "Gt":
            unit_str = ' (Gt/yr)'
        else:
            unit_str = ''
    else:
        unit_str = ''
    
    ax.set_xlabel(f'{options.variable}{unit_str}')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{options.variable} PDF at Year {time_step:.0f}\n({len(var_values)} experiments)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Remove empty subplots
for i in range(n_plots, len(axes)):
    fig.delaxes(axes[i])

# Overall title
title_str = f"{options.variable} Probability Distributions\n{num_experiments} Experiments"
fig.suptitle(title_str, fontsize=14)

# Create a separate plot for skewness evolution
fig_skew, ax_skew = plt.subplots(1, 1, figsize=(8, 6))

skewness_values = []
time_points = []

for time_step in valid_time_steps:
    var_values = np.array(all_var_data[time_step])
    
    # Calculate skewness with error handling
    try:
        if np.std(var_values) > 1e-10:
            skewness = stats.skew(var_values)
        else:
            skewness = 0.0
    except:
        skewness = 0.0
        
    skewness_values.append(skewness)
    time_points.append(time_step)

ax_skew.plot(time_points, skewness_values, 'o-', linewidth=2, markersize=8, color='purple')
ax_skew.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Symmetric (skew=0)')
ax_skew.set_xlabel('Time (years)')
ax_skew.set_ylabel('Skewness')
ax_skew.set_title(f'{options.variable} Distribution Skewness Evolution')
ax_skew.grid(True, alpha=0.3)
ax_skew.legend()

# Add interpretation text
ax_skew.text(0.02, 0.98, 'Positive: Right-skewed\nNegative: Left-skewed', 
             transform=ax_skew.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

if options.plotSave:
    skew_save_name = options.plotSave.replace('.png', '_skewness.png')
    fig_skew.savefig(skew_save_name, dpi=300, bbox_inches='tight')
    print(f"Skewness plot saved as: {skew_save_name}")

# Create a separate figure showing only smooth PDFs for all time steps
fig_pdf, ax_pdf = plt.subplots(1, 1, figsize=(10, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(valid_time_steps)))

# Determine the overall range for all data
all_values = np.concatenate([np.array(all_var_data[ts]) for ts in valid_time_steps])
data_range = all_values.max() - all_values.min()
x_min = all_values.min() - 0.1 * data_range
x_max = all_values.max() + 0.1 * data_range
x_range = np.linspace(x_min, x_max, 300)

for i, time_step in enumerate(valid_time_steps):
    var_values = np.array(all_var_data[time_step])
    
    # Check if data has sufficient variation for KDE
    if len(var_values) >= 3 and np.std(var_values) > 1e-10:
        try:
            # Kernel Density Estimation for smooth PDF
            kde = gaussian_kde(var_values)
            kde_pdf = kde(x_range)
            
            ax_pdf.plot(x_range, kde_pdf, color=colors[i], linewidth=2.5, 
                        label=f'Year {time_step:.0f}')
            
            # Add vertical line for mean
            mean_val = np.mean(var_values)
            ax_pdf.axvline(mean_val, color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
            
        except Exception as e:
            print(f"Warning: Could not create KDE for time step {time_step}: {str(e)}")
            # Fallback: just plot vertical line at the mean value
            mean_val = np.mean(var_values)
            ax_pdf.axvline(mean_val, color=colors[i], linestyle='-', alpha=0.8, linewidth=3,
                          label=f'Year {time_step:.0f} (low var.)')
    else:
        print(f"Warning: Skipping KDE for time step {time_step} - insufficient variation or data points")
        # Just plot vertical line for constant/near-constant data
        mean_val = np.mean(var_values)
        ax_pdf.axvline(mean_val, color=colors[i], linestyle='-', alpha=0.8, linewidth=3,
                      label=f'Year {time_step:.0f} (const.)')

ax_pdf.set_xlabel(f'{options.variable}{unit_str}')
ax_pdf.set_ylabel('Probability Density')
ax_pdf.set_title(f'{options.variable} PDF Evolution Over Time\n({num_experiments} experiments)')
ax_pdf.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_pdf.grid(True, alpha=0.3)

# Add text box with interpretation
#textstr = 'Dashed lines show means\nColors: Early → Late time'
#props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
#ax_pdf.text(0.02, 0.98, textstr, transform=ax_pdf.transAxes, fontsize=10,
#            verticalalignment='top', bbox=props)

plt.tight_layout()

if options.plotSave:
    pdf_save_name = options.plotSave.replace('.png', '_pdf_evolution.png')
    fig_pdf.savefig(pdf_save_name, dpi=300, bbox_inches='tight')
    print(f"PDF evolution plot saved as: {pdf_save_name}")

plt.tight_layout()

# Save or show
if options.plotSave:
    fig.savefig(options.plotSave, dpi=300, bbox_inches='tight')
    print(f"Figure saved as: {options.plotSave}")

plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
for time_step in valid_time_steps:
    var_values = np.array(all_var_data[time_step])
    mean_val = np.mean(var_values)
    std_val = np.std(var_values)
    median_val = np.median(var_values)
    q25, q75 = np.percentile(var_values, [25, 75])
    
    # Calculate skewness and kurtosis with error handling
    try:
        if std_val > 1e-10:
            skewness = stats.skew(var_values)
            kurtosis = stats.kurtosis(var_values)
        else:
            skewness = 0.0
            kurtosis = 0.0
    except:
        skewness = 0.0
        kurtosis = 0.0
    
    print(f"Year {time_step:.0f}: N={len(var_values)}, Mean={mean_val:.2f}, Std={std_val:.2f}, "
          f"Median={median_val:.2f}, IQR=[{q25:.2f}, {q75:.2f}], Skew={skewness:.2f}, Kurt={kurtosis:.2f}")