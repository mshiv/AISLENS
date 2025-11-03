#!/usr/bin/env python
'''
Script to plot common time-series from one or more landice globalStats files.

Modified version that takes ensemble directory structure arguments instead of individual file paths.
Enhanced to support multiple ensembles and flexible experiment specification.

Original by Matt Hoffman, 8/23/2022
--
Shiva Muruganandham, 8/27/2024
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
from optparse import OptionParser
import matplotlib.pyplot as plt
import glob

rhoi = 910.0
rhosw = 1028.

print("** Gathering information.  (Invoke with --help for more details. All arguments are optional)")
parser = OptionParser(description=__doc__)
parser.add_option("-r", "--root", dest="rootDataDir", help="Root data directory path", metavar="PATH")
parser.add_option("-b", "--base", dest="ensembleBaseDir", help="Ensemble base directory/directories (comma-separated for multiple ensembles, e.g., 'CTRL,SSP585')", metavar="DIRNAME1,DIRNAME2")
parser.add_option("-e", "--experiments", dest="experimentList", help="Experiment specifications. Format options: 1) Simple list: 'EM1,EM2,EM4', 2) Ensemble-specific: 'CTRL:EM1,SSP585:EM2,SSP585:EM4', 3) Wildcard: 'EM*' to find all matching experiments", metavar="EXP_SPECS")
parser.add_option("-f", "--filename", dest="statsFilename", help="Statistics filename to look for in each experiment directory", default="globalStats.nc", metavar="FILENAME")
parser.add_option("-u", dest="units", help="units for mass/volume: m3, kg, Gt", default="Gt", metavar="UNITS")
parser.add_option("-c", dest="plotChange", help="plot time series as absolute change from initial.  (not applied to GL flux or calving flux)  Without this option, the full magnitude of time series is used", action='store_true', default=False)
parser.add_option("-p", dest="plotPercentChange", help="plot time series as percentage change from initial.  (not applied to GL flux or calving flux)  Without this option, the full magnitude of time series is used", action='store_true', default=False)
parser.add_option("-s", dest="plotSave", help="save figure")
parser.add_option("-x", "--xlim", dest="xlimits", help="X-axis limits as comma-separated values (e.g., '0,25' for years 0 to 25)", metavar="MIN,MAX")
parser.add_option("--search-all", dest="searchAll", help="Search all ensemble directories for experiments (ignores -b)", action='store_true', default=False)
parser.add_option("--list-available", dest="listAvailable", help="List all available experiments and exit", action='store_true', default=False)
options, args = parser.parse_args()


# Check for conflicting options
if options.plotChange and options.plotPercentChange:
    sys.exit("ERROR: Cannot use both -c (absolute change) and -p (percentage change) options simultaneously")

print("Using ice density of {} kg/m3 if required for unit conversions".format(rhoi))

def find_all_experiments(root_dir, ensemble_dirs, stats_filename):
    """Find all available experiments across ensemble directories."""
    available_experiments = {}
    
    for ensemble_dir in ensemble_dirs:
        if root_dir:
            full_ensemble_path = os.path.join(root_dir, ensemble_dir)
        else:
            full_ensemble_path = ensemble_dir
            
        if not os.path.exists(full_ensemble_path):
            print(f"Warning: Ensemble directory not found: {full_ensemble_path}")
            continue
            
        # Find all subdirectories that contain the stats file
        for item in os.listdir(full_ensemble_path):
            exp_path = os.path.join(full_ensemble_path, item)
            if os.path.isdir(exp_path):
                stats_file = os.path.join(exp_path, stats_filename)
                if os.path.exists(stats_file):
                    if ensemble_dir not in available_experiments:
                        available_experiments[ensemble_dir] = []
                    available_experiments[ensemble_dir].append(item)
    
    return available_experiments

def parse_experiment_specifications(experiment_list, ensemble_dirs, root_dir, stats_filename):
    """
    Parse experiment specifications and return list of (ensemble, experiment, file_path, display_name) tuples.
    
    Supports multiple formats:
    1. Simple list: 'EM1,EM2,EM4' - searches all ensembles
    2. Ensemble-specific: 'CTRL:EM1,SSP585:EM2,SSP585:EM4'
    3. Wildcard: 'EM*' - finds all matching experiments
    """
    experiment_specs = []
    
    if not experiment_list:
        sys.exit("ERROR: Must specify experiment list with -e/--experiments option")
    
    # Split by commas
    exp_parts = [exp.strip() for exp in experiment_list.split(',')]
    
    for exp_spec in exp_parts:
        if ':' in exp_spec:
            # Format: ENSEMBLE:EXPERIMENT
            ensemble_name, exp_name = exp_spec.split(':', 1)
            ensemble_name = ensemble_name.strip()
            exp_name = exp_name.strip()
            
            if ensemble_name not in ensemble_dirs:
                print(f"Warning: Specified ensemble '{ensemble_name}' not in ensemble directory list")
                continue
                
            # Handle wildcards in experiment name
            if '*' in exp_name or '?' in exp_name:
                # Find matching experiments in specific ensemble
                if root_dir:
                    search_path = os.path.join(root_dir, ensemble_name)
                else:
                    search_path = ensemble_name
                    
                if os.path.exists(search_path):
                    matching_dirs = glob.glob(os.path.join(search_path, exp_name))
                    for match_path in matching_dirs:
                        if os.path.isdir(match_path):
                            match_exp = os.path.basename(match_path)
                            stats_file = os.path.join(match_path, stats_filename)
                            if os.path.exists(stats_file):
                                display_name = f"{ensemble_name}:{match_exp}"
                                experiment_specs.append((ensemble_name, match_exp, stats_file, display_name))
            else:
                # Specific experiment in specific ensemble
                if root_dir:
                    exp_path = os.path.join(root_dir, ensemble_name, exp_name)
                else:
                    exp_path = os.path.join(ensemble_name, exp_name)
                    
                stats_file = os.path.join(exp_path, stats_filename)
                if os.path.exists(stats_file):
                    display_name = f"{ensemble_name}:{exp_name}"
                    experiment_specs.append((ensemble_name, exp_name, stats_file, display_name))
                else:
                    print(f"Warning: Stats file not found for {ensemble_name}:{exp_name} at {stats_file}")
                    
        else:
            # Format: EXPERIMENT (search all ensembles)
            exp_name = exp_spec.strip()
            found_in_ensembles = []
            
            # Handle wildcards
            if '*' in exp_name or '?' in exp_name:
                # Search all ensembles for matching experiments
                for ensemble_dir in ensemble_dirs:
                    if root_dir:
                        search_path = os.path.join(root_dir, ensemble_dir)
                    else:
                        search_path = ensemble_dir
                        
                    if os.path.exists(search_path):
                        matching_dirs = glob.glob(os.path.join(search_path, exp_name))
                        for match_path in matching_dirs:
                            if os.path.isdir(match_path):
                                match_exp = os.path.basename(match_path)
                                stats_file = os.path.join(match_path, stats_filename)
                                if os.path.exists(stats_file):
                                    display_name = f"{ensemble_dir}:{match_exp}"
                                    experiment_specs.append((ensemble_dir, match_exp, stats_file, display_name))
                                    found_in_ensembles.append(ensemble_dir)
            else:
                # Search for specific experiment in all ensembles
                for ensemble_dir in ensemble_dirs:
                    if root_dir:
                        exp_path = os.path.join(root_dir, ensemble_dir, exp_name)
                    else:
                        exp_path = os.path.join(ensemble_dir, exp_name)
                        
                    stats_file = os.path.join(exp_path, stats_filename)
                    if os.path.exists(stats_file):
                        display_name = f"{ensemble_dir}:{exp_name}"
                        experiment_specs.append((ensemble_dir, exp_name, stats_file, display_name))
                        found_in_ensembles.append(ensemble_dir)
                
                if not found_in_ensembles:
                    print(f"Warning: Experiment '{exp_name}' not found in any ensemble directory")
    
    return experiment_specs

# Parse ensemble directories
ensemble_dirs = []
if options.searchAll:
    # Search all directories in root for ensembles
    if not options.rootDataDir:
        sys.exit("ERROR: --search-all requires --root to be specified")
    
    for item in os.listdir(options.rootDataDir):
        item_path = os.path.join(options.rootDataDir, item)
        if os.path.isdir(item_path):
            ensemble_dirs.append(item)
    print(f"Auto-detected ensemble directories: {ensemble_dirs}")
    
elif options.ensembleBaseDir:
    ensemble_dirs = [ens.strip() for ens in options.ensembleBaseDir.split(',')]
    print(f"Using specified ensemble directories: {ensemble_dirs}")
else:
    sys.exit("ERROR: Must specify ensemble directories with -b/--base or use --search-all")

# List available experiments if requested
if options.listAvailable:
    print("\nAvailable experiments:")
    available = find_all_experiments(options.rootDataDir, ensemble_dirs, options.statsFilename)
    
    total_experiments = 0
    for ensemble, experiments in available.items():
        print(f"\n{ensemble}: ({len(experiments)} experiments)")
        for exp in sorted(experiments):
            print(f"  {exp}")
        total_experiments += len(experiments)
    
    print(f"\nTotal: {total_experiments} experiments across {len(available)} ensembles")
    print("\nUsage examples:")
    print("  # Plot specific experiments from specific ensembles:")
    print("  -e 'CTRL:EM1,SSP585:EM2,SSP585:EM4'")
    print("  # Plot same experiment from multiple ensembles:")
    print("  -e 'EM1,EM2' -b 'CTRL,SSP585'")
    print("  # Use wildcards:")
    print("  -e 'EM*' or -e 'SSP585:EM*'")
    sys.exit(0)

# Parse experiment specifications
experiment_specs = parse_experiment_specifications(
    options.experimentList, ensemble_dirs, options.rootDataDir, options.statsFilename
)

if not experiment_specs:
    sys.exit("ERROR: No valid experiments found")

print(f"\nFound {len(experiment_specs)} experiments to plot:")
for ensemble, exp, file_path, display_name in experiment_specs:
    print(f"  {display_name}: {file_path}")

# create axes to plot into
fig = plt.figure(1, figsize=(12, 12), facecolor='w')

nrow=3
ncol=3

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
plt.ylabel(f'volume{plotChangeStr} ({massUnit})')
plt.grid()
axX = axVol

axVAF = fig.add_subplot(nrow, ncol, 2, sharex=axX)
plt.xlabel('Year')
plt.ylabel(f'VAF{plotChangeStr} ({massUnit})')
plt.grid()

axVolGround = fig.add_subplot(nrow, ncol, 3, sharex=axX)
plt.xlabel('Year')
plt.ylabel(f'grounded volume{plotChangeStr} ({massUnit})')
plt.grid()

axVolFloat = fig.add_subplot(nrow, ncol, 4, sharex=axX)
plt.xlabel('Year')
plt.ylabel(f'floating volume{plotChangeStr} ({massUnit})')
plt.grid()

axGrdArea = fig.add_subplot(nrow, ncol, 5, sharex=axX)
plt.xlabel('Year')
plt.ylabel(f'grounded area{plotChangeStr} (km$^2$)')
plt.grid()

axFltArea = fig.add_subplot(nrow, ncol, 6, sharex=axX)
plt.xlabel('Year')
plt.ylabel(f'floating area{plotChangeStr} (km$^2$)')
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

# Define base colors for different ensembles and create color variations for experiments
ensemble_base_colors = plt.cm.Set1(np.linspace(0, 1, 9))  # Use Set1 colormap for distinct ensemble base colors

# Create mapping of ensembles to colors and experiments to color variations
ensemble_names_unique = list(set([ensemble for ensemble, _, _, _ in experiment_specs]))
ensemble_to_base_color = {}
for i, ensemble in enumerate(sorted(ensemble_names_unique)):
    ensemble_to_base_color[ensemble] = ensemble_base_colors[i % len(ensemble_base_colors)]

# Group experiments by ensemble to assign color variations within each ensemble
experiments_by_ensemble = {}
for ensemble, exp, file_path, display_name in experiment_specs:
    if ensemble not in experiments_by_ensemble:
        experiments_by_ensemble[ensemble] = []
    experiments_by_ensemble[ensemble].append((exp, file_path, display_name))

# Create mapping for color variations within each ensemble
def create_color_variations(base_color, n_variations):
    """Create n color variations from a base color by adjusting brightness and saturation."""
    import matplotlib.colors as mcolors
    
    # Convert to HSV for easier manipulation
    hsv = mcolors.rgb_to_hsv(base_color[:3])  # Only use RGB, ignore alpha
    
    variations = []
    if n_variations == 1:
        variations.append(base_color)
    else:
        # Create variations by adjusting value (brightness) and saturation
        for i in range(n_variations):
            # Adjust brightness: from 0.4 to 1.0
            brightness_factor = 0.4 + (0.6 * i / max(1, n_variations - 1))
            # Adjust saturation: from 0.6 to 1.0  
            saturation_factor = 0.6 + (0.4 * i / max(1, n_variations - 1))
            
            new_hsv = hsv.copy()
            new_hsv[1] = min(1.0, hsv[1] * saturation_factor)  # Saturation
            new_hsv[2] = min(1.0, hsv[2] * brightness_factor)  # Value/brightness
            
            new_rgb = mcolors.hsv_to_rgb(new_hsv)
            variations.append(new_rgb)
    
    return variations

experiment_to_color = {}
for ensemble, experiments in experiments_by_ensemble.items():
    base_color = ensemble_to_base_color[ensemble]
    n_experiments = len(experiments)
    color_variations = create_color_variations(base_color, n_experiments)
    
    for i, (exp, file_path, display_name) in enumerate(experiments):
        experiment_to_color[display_name] = color_variations[i]

def plotStat(fname, display_name, color):
    """Modified plotStat function to use only color variations (no line styles)."""
    print("Reading and plotting file: {} for experiment: {}".format(fname, display_name))

    f = Dataset(fname,'r')
    yr = f.variables['daysSinceStart'][:]/365.0
    yr = yr-yr[0]
    dt = f.variables['deltat'][:]/3.15e7
    print(f"{display_name} time span: {yr.max():.1f} years")

    vol = f.variables['totalIceVolume'][:] / scaleVol
    if options.plotChange:
        vol = vol - vol[0]
    elif options.plotPercentChange:
        vol = (vol - vol[0])*100/vol[0]
    axVol.plot(yr, vol, label=display_name, color=color, linewidth=1.5)

    VAF = f.variables['volumeAboveFloatation'][:] / scaleVol       
    if options.plotChange:
        VAF = VAF - VAF[0]
    elif options.plotPercentChange:
        VAF = (VAF - VAF[0])*100/VAF[0]
    axVAF.plot(yr, VAF, label=display_name, color=color, linewidth=1.5)
    
    volGround = f.variables['groundedIceVolume'][:] / scaleVol
    if options.plotChange:
        volGround = volGround - volGround[0]
    elif options.plotPercentChange:
        volGround = (volGround - volGround[0])*100/volGround[0]
    axVolGround.plot(yr, volGround, label=display_name, color=color, linewidth=1.5)

    volFloat = f.variables['floatingIceVolume'][:] / scaleVol
    if options.plotChange:
        volFloat = volFloat - volFloat[0]
    elif options.plotPercentChange:
        volFloat = (volFloat - volFloat[0])*100/volFloat[0]
    axVolFloat.plot(yr, volFloat, label=display_name, color=color, linewidth=1.5)

    areaGrd = f.variables['groundedIceArea'][:] / 1000.0**2
    if options.plotChange:
        areaGrd = areaGrd - areaGrd[0]
    elif options.plotPercentChange:
        areaGrd = (areaGrd - areaGrd[0])*100/areaGrd[0]
    axGrdArea.plot(yr, areaGrd, label=display_name, color=color, linewidth=1.5)

    areaFlt = f.variables['floatingIceArea'][:] / 1000.0**2
    if options.plotChange:
        areaFlt = areaFlt - areaFlt[0]
    elif options.plotPercentChange:
        areaFlt = (areaFlt - areaFlt[0])*100/areaFlt[0]
    axFltArea.plot(yr, areaFlt, label=display_name, color=color, linewidth=1.5)

    GLflux = f.variables['groundingLineFlux'][:]
    axGLflux.plot(yr, GLflux, label=display_name, color=color, linewidth=1.5)

    calvFlux = f.variables['totalCalvingFlux'][:]
    axCalvFlux.plot(yr, calvFlux, label=display_name, color=color, linewidth=1.5)

    totalBMB = f.variables['totalFloatingBasalMassBal'][:] / 1e12  # Convert kg/yr to Gt/yr
    axTotalBMB.plot(yr, totalBMB, label=display_name, color=color, linewidth=1.5)

    f.close()


# Parse x-axis limits if provided
xlim_range = None
if options.xlimits:
    try:
        xlim_values = [float(x.strip()) for x in options.xlimits.split(',')]
        if len(xlim_values) != 2:
            sys.exit("ERROR: X-axis limits must be exactly two comma-separated values (e.g., '0,25')")
        if xlim_values[0] >= xlim_values[1]:
            sys.exit("ERROR: X-axis minimum must be less than maximum")
        xlim_range = xlim_values
        print(f"Using X-axis limits: {xlim_range[0]} to {xlim_range[1]} years")
    except ValueError:
        sys.exit("ERROR: X-axis limits must be numeric values separated by comma (e.g., '0,25')")

# Plot each experiment with ensemble-specific color variations
for ensemble, exp, file_path, display_name in experiment_specs:
    color = experiment_to_color[display_name]
    plotStat(file_path, display_name, color)

# Add sea level axis only to VAF plot (once, after all experiments are plotted)
addSeaLevAx(axVAF)

# Apply x-axis limits to all subplots if specified
if xlim_range:
    print(f"Applying X-axis limits: {xlim_range}")
    for ax in [axVol, axVAF, axVolGround, axVolFloat, axGrdArea, axFltArea, axGLflux, axCalvFlux, axTotalBMB]:
        ax.set_xlim(xlim_range)

# Add legend to the last subplot
axTotalBMB.legend(loc='best', prop={'size': 6})

# Add title with experiment information
ensemble_names = list(set([ensemble for ensemble, _, _, _ in experiment_specs]))
exp_names = [display_name for _, _, _, display_name in experiment_specs]
title_str = f"Global Statistics Comparison\nEnsembles: {', '.join(sorted(ensemble_names))}\nExperiments: {', '.join(exp_names)}"
fig.suptitle(title_str, fontsize=10)

print("Generating plot.")
fig.tight_layout()

if options.plotSave:
    ensemble_str = "-".join(sorted(ensemble_names))
    exp_str = "-".join([name.replace(':', '_') for name in exp_names])
    save_name = f'globalStats_multi_ensemble_{ensemble_str}_{exp_str}.png'
    fig.savefig(save_name, dpi=400, bbox_inches='tight')
    print(f"Figure saved as: {save_name}")
    
plt.show()
