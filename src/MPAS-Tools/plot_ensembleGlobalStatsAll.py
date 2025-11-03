#!/usr/bin/env python
'''
Script to plot common time-series from one or more landice globalStats files.

Modified version that takes multiple ensemble directory structures and automatically
searches for experiments across different ensemble bases (e.g., ISMIP6, etc.).

Original by Matt Hoffman, 8/23/2022
--
Shiva Muruganandham, 8/27/2024
Updated: 8/28/2025
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import glob
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
from optparse import OptionParser
import matplotlib.pyplot as plt

rhoi = 910.0
rhosw = 1028.

print("** Gathering information.  (Invoke with --help for more details. All arguments are optional)")
parser = OptionParser(description=__doc__)
parser.add_option("-r", "--root", dest="rootDataDir", help="Root data directory path", metavar="PATH")
parser.add_option("-b", "--base", dest="ensembleBaseDirs", help="Comma-separated list of ensemble base directories (relative to root)", metavar="DIRNAME1,DIRNAME2,DIRNAME3")
parser.add_option("-e", "--experiments", dest="experimentList", help="Comma-separated list of experiment run names (will search across all base directories)", metavar="EXP1,EXP2,EXP3")
parser.add_option("-f", "--filename", dest="statsFilename", help="Statistics filename to look for in each experiment directory", default="globalStats.nc", metavar="FILENAME")
parser.add_option("-u", dest="units", help="units for mass/volume: m3, kg, Gt", default="Gt", metavar="UNITS")
parser.add_option("-c", dest="plotChange", help="plot time series as absolute change from initial.  (not applied to GL flux or calving flux)  Without this option, the full magnitude of time series is used", action='store_true', default=False)
parser.add_option("-p", dest="plotPercentChange", help="plot time series as percentage change from initial.  (not applied to GL flux or calving flux)  Without this option, the full magnitude of time series is used", action='store_true', default=False)
parser.add_option("-s", dest="plotSave", help="save figure")
parser.add_option("--verbose", dest="verbose", help="print detailed search information", action='store_true', default=False)
options, args = parser.parse_args()

# Check for conflicting options
if options.plotChange and options.plotPercentChange:
    sys.exit("ERROR: Cannot use both -c (absolute change) and -p (percentage change) options simultaneously")

print("Using ice density of {} kg/m3 if required for unit conversions".format(rhoi))

# Parse ensemble base directories
if not options.ensembleBaseDirs:
    sys.exit("ERROR: Must specify at least one ensemble base directory with -b/--base option")

ensemble_base_dirs = [base.strip() for base in options.ensembleBaseDirs.split(',')]
print(f"Ensemble base directories: {ensemble_base_dirs}")

# Parse experiment list
if not options.experimentList:
    sys.exit("ERROR: Must specify experiment list with -e/--experiments option")

experiment_names = [exp.strip() for exp in options.experimentList.split(',')]
print(f"Looking for experiments: {experiment_names}")

def find_experiment_file(exp_name, base_dirs, root_dir=None, filename="globalStats.nc", verbose=False):
    """
    Search for an experiment file across multiple ensemble base directories.
    Returns (file_path, base_dir_used) if found, (None, None) if not found.
    """
    for base_dir in base_dirs:
        if root_dir:
            search_path = os.path.join(root_dir, base_dir, exp_name, filename)
            search_dir = os.path.join(root_dir, base_dir, exp_name)
        else:
            search_path = os.path.join(base_dir, exp_name, filename)
            search_dir = os.path.join(base_dir, exp_name)
        
        if verbose:
            print(f"  Checking: {search_path}")
        
        if os.path.exists(search_path):
            return search_path, base_dir
        
        # Also try searching for subdirectories within the experiment directory
        # This handles cases like ISMIP6/SSP126/exp_name or ISMIP6/SSP585/exp_name
        if os.path.exists(search_dir):
            # Look for the file in any subdirectory
            pattern = os.path.join(search_dir, "**", filename)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                if verbose:
                    print(f"  Found via recursive search: {matches[0]}")
                return matches[0], base_dir
    
    return None, None

# Build file paths by searching across ensemble directories
experiment_files = []
experiment_sources = []  # Track which base directory each experiment came from
found_experiments = []

print("\nSearching for experiment files...")
for exp_name in experiment_names:
    if options.verbose:
        print(f"\nSearching for experiment: {exp_name}")
    
    file_path, base_dir_used = find_experiment_file(
        exp_name, 
        ensemble_base_dirs, 
        options.rootDataDir, 
        options.statsFilename,
        options.verbose
    )
    
    if file_path:
        experiment_files.append(file_path)
        experiment_sources.append(base_dir_used)
        found_experiments.append(exp_name)
        print(f"✓ Found {exp_name}: {file_path}")
    else:
        print(f"✗ Could not find {exp_name} in any ensemble directory")

if not experiment_files:
    sys.exit("ERROR: No experiment files found!")

num_experiments = len(experiment_files)
print(f"\nSuccessfully found {num_experiments} experiment files")

# Create axes to plot into
fig = plt.figure(1, figsize=(15, 12), facecolor='w')

nrow = 3
ncol = 3

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

if options.plotChange:
    plotChangeStr = ' change'
    plotChangeUnit = f' ({massUnit})'
elif options.plotPercentChange:
    plotChangeStr = ' change'
    plotChangeUnit = ' (%)'
else:
    plotChangeStr = ''
    plotChangeUnit = f' ({massUnit})'

axVol = fig.add_subplot(nrow, ncol, 1)
plt.xlabel('Year')
plt.ylabel(f'volume{plotChangeStr}{plotChangeUnit}')
plt.grid()
axX = axVol

axVAF = fig.add_subplot(nrow, ncol, 2, sharex=axX)
plt.xlabel('Year')
plt.ylabel(f'VAF{plotChangeStr}{plotChangeUnit}')
plt.grid()

axVolGround = fig.add_subplot(nrow, ncol, 3, sharex=axX)
plt.xlabel('Year')
plt.ylabel(f'grounded volume{plotChangeStr}{plotChangeUnit}')
plt.grid()

axVolFloat = fig.add_subplot(nrow, ncol, 4, sharex=axX)
plt.xlabel('Year')
plt.ylabel(f'floating volume{plotChangeStr}{plotChangeUnit}')
plt.grid()

axGrdArea = fig.add_subplot(nrow, ncol, 5, sharex=axX)
plt.xlabel('Year')
if options.plotChange:
    plt.ylabel(f'grounded area change (km$^2$)')
elif options.plotPercentChange:
    plt.ylabel(f'grounded area change (%)')
else:
    plt.ylabel(f'grounded area (km$^2$)')
plt.grid()

axFltArea = fig.add_subplot(nrow, ncol, 6, sharex=axX)
plt.xlabel('Year')
if options.plotChange:
    plt.ylabel(f'floating area change (km$^2$)')
elif options.plotPercentChange:
    plt.ylabel(f'floating area change (%)')
else:
    plt.ylabel(f'floating area (km$^2$)')
plt.grid()

axGLflux = fig.add_subplot(nrow, ncol, 7, sharex=axX)
plt.xlabel('Year')
plt.ylabel('GL flux (kg/yr)')
plt.grid()

axCalvFlux = fig.add_subplot(nrow, ncol, 8, sharex=axX)
plt.xlabel('Year')
plt.ylabel('calving flux (kg/yr)')
plt.grid()

axTotalBMB = fig.add_subplot(nrow, ncol, 9, sharex=axX)
plt.xlabel('Year')
plt.ylabel('total floating BMB (Gt/yr)')
plt.grid()

def VAF2seaLevel(vol):
    return vol * scaleVol / 3.62e14 * rhoi / rhosw * 1000.

def seaLevel2VAF(vol):
    return vol / scaleVol * 3.62e14 * rhosw / rhoi / 1000. 

def addSeaLevAx(axName):
    seaLevAx = axName.secondary_yaxis('right', functions=(VAF2seaLevel, seaLevel2VAF))
    seaLevAx.set_ylabel('Sea-level\nequivalent (mm)')

# Create a color map that distinguishes between different ensemble sources
unique_sources = list(set(experiment_sources))
source_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_sources)))
source_color_map = {source: color for source, color in zip(unique_sources, source_colors)}

# Create a line style map for different experiments within the same source
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

def plotStat(fname, exp_name, source_dir, color, linestyle):
    print("Reading and plotting file: {} for experiment: {} from {}".format(fname, exp_name, source_dir))

    f = Dataset(fname,'r')
    yr = f.variables['daysSinceStart'][:]/365.0
    yr = yr-yr[0]
    dt = f.variables['deltat'][:]/3.15e7
    print(f"{exp_name} time span: {yr.max():.1f} years")

    # Create label with source information
    label = f"{exp_name} ({source_dir})"

    vol = f.variables['totalIceVolume'][:] / scaleVol
    if options.plotChange:
        vol = vol - vol[0]
    elif options.plotPercentChange:
        vol = (vol - vol[0])*100/vol[0]
    axVol.plot(yr, vol, label=label, color=color, linestyle=linestyle, linewidth=1.5)

    VAF = f.variables['volumeAboveFloatation'][:] / scaleVol       
    if options.plotChange:
        VAF = VAF - VAF[0]
    elif options.plotPercentChange:
        VAF = (VAF - VAF[0])*100/VAF[0]
    axVAF.plot(yr, VAF, label=label, color=color, linestyle=linestyle, linewidth=1.5)
    
    volGround = f.variables['groundedIceVolume'][:] / scaleVol
    if options.plotChange:
        volGround = volGround - volGround[0]
    elif options.plotPercentChange:
        volGround = (volGround - volGround[0])*100/volGround[0]
    axVolGround.plot(yr, volGround, label=label, color=color, linestyle=linestyle, linewidth=1.5)

    volFloat = f.variables['floatingIceVolume'][:] / scaleVol
    if options.plotChange:
        volFloat = volFloat - volFloat[0]
    elif options.plotPercentChange:
        volFloat = (volFloat - volFloat[0])*100/volFloat[0]
    axVolFloat.plot(yr, volFloat, label=label, color=color, linestyle=linestyle, linewidth=1.5)

    areaGrd = f.variables['groundedIceArea'][:] / 1000.0**2
    if options.plotChange:
        areaGrd = areaGrd - areaGrd[0]
    elif options.plotPercentChange:
        areaGrd = (areaGrd - areaGrd[0])*100/areaGrd[0]
    axGrdArea.plot(yr, areaGrd, label=label, color=color, linestyle=linestyle, linewidth=1.5)

    areaFlt = f.variables['floatingIceArea'][:] / 1000.0**2
    if options.plotChange:
        areaFlt = areaFlt - areaFlt[0]
    elif options.plotPercentChange:
        areaFlt = (areaFlt - areaFlt[0])*100/areaFlt[0]
    axFltArea.plot(yr, areaFlt, label=label, color=color, linestyle=linestyle, linewidth=1.5)

    GLflux = f.variables['groundingLineFlux'][:]
    axGLflux.plot(yr, GLflux, label=label, color=color, linestyle=linestyle, linewidth=1.5)

    calvFlux = f.variables['totalCalvingFlux'][:]
    axCalvFlux.plot(yr, calvFlux, label=label, color=color, linestyle=linestyle, linewidth=1.5)

    totalBMB = f.variables['totalFloatingBasalMassBal'][:] / 1e12  # Convert kg/yr to Gt/yr
    axTotalBMB.plot(yr, totalBMB, label=label, color=color, linestyle=linestyle, linewidth=1.5)

    f.close()

# Plot each experiment with colors based on source and different line styles
for i, (exp_file, exp_name, source_dir) in enumerate(zip(experiment_files, found_experiments, experiment_sources)):
    color = source_color_map[source_dir]
    # Use different line styles for experiments from the same source
    source_exp_count = experiment_sources[:i+1].count(source_dir) - 1
    linestyle = line_styles[source_exp_count % len(line_styles)]
    
    plotStat(exp_file, exp_name, source_dir, color, linestyle)

# Add sea level axis only to VAF plot (once, after all experiments are plotted)
addSeaLevAx(axVAF)

# Add legend to the last subplot with smaller font
axTotalBMB.legend(loc='best', prop={'size': 5}, ncol=1)

# Create a summary of sources and experiments
sources_summary = {}
for exp, source in zip(found_experiments, experiment_sources):
    if source not in sources_summary:
        sources_summary[source] = []
    sources_summary[source].append(exp)

summary_text = "Sources: " + ", ".join([f"{source} ({len(exps)} exp.)" for source, exps in sources_summary.items()])

# Add title with experiment information
title_str = f"Global Statistics Comparison\n{summary_text}"
fig.suptitle(title_str, fontsize=11)

print("Generating plot.")
plt.tight_layout()

# Generate a meaningful save name
if options.plotSave:
    if options.plotSave.endswith('.png') or options.plotSave.endswith('.pdf') or options.plotSave.endswith('.jpg'):
        save_name = options.plotSave
    else:
        save_name = options.plotSave + '.png'
else:
    # Auto-generate filename based on sources and experiments
    source_names = "_".join(sorted(unique_sources))
    save_name = f'globalStats_{source_names}_{len(found_experiments)}exp.png'

if options.plotSave or not options.plotSave:  # Always save with auto-generated name if -s not specified
    fig.savefig(save_name, dpi=400, bbox_inches='tight')
    print(f"Figure saved as: {save_name}")

plt.show()

# Print summary
print(f"\n=== Summary ===")
print(f"Total experiments plotted: {num_experiments}")
for source in unique_sources:
    source_exps = [exp for exp, src in zip(found_experiments, experiment_sources) if src == source]
    print(f"{source}: {source_exps}")