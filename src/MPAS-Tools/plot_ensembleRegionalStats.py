#!/usr/bin/env python
'''
Script to plot common time-series from one or more landice regionalStats files.
Currently only useful for whole-AIS simulations.

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
import itertools
from netCDF4 import Dataset
from optparse import OptionParser
import matplotlib.pyplot as plt
import glob

rhoi = 910.0

print("** Gathering information.  (Invoke with --help for more details. All arguments are optional)")
parser = OptionParser(description=__doc__)
parser.add_option("-r", "--root", dest="rootDataDir", help="Root data directory path", metavar="PATH")
parser.add_option("-b", "--base", dest="ensembleBaseDir", help="Ensemble base directory/directories (comma-separated for multiple ensembles, e.g., 'CTRL,SSP585')", metavar="DIRNAME1,DIRNAME2")
parser.add_option("-e", "--experiments", dest="experimentList", help="Experiment specifications. Format options: 1) Simple list: 'EM1,EM2,EM4', 2) Ensemble-specific: 'CTRL:EM1,SSP585:EM2,SSP585:EM4', 3) Wildcard: 'EM*' to find all matching experiments", metavar="EXP_SPECS")
parser.add_option("-f", "--filename", dest="statsFilename", help="Statistics filename to look for in each experiment directory", default="regionalStats.nc", metavar="FILENAME")
parser.add_option("-u", dest="units", help="units for mass/volume: m3, kg, Gt", default="Gt", metavar="UNITS")
parser.add_option("-n", dest="fileRegionNames", help="region name filename.  If not specified, will attempt to read region names from first experiment file.", metavar="FILENAME")
parser.add_option("-x", "--xlim", dest="xlimits", help="X-axis limits as comma-separated values (e.g., '0,25' for years 0 to 25)", metavar="MIN,MAX")
parser.add_option("--search-all", dest="searchAll", help="Search all ensemble directories for experiments (ignores -b)", action='store_true', default=False)
parser.add_option("--list-available", dest="listAvailable", help="List all available experiments and exit", action='store_true', default=False)
parser.add_option("-c", "--colors", dest="colors", help="Comma-separated list of Matplotlib colors to use for experiments (one per experiment). If fewer colors than experiments are supplied they will be cycled.", metavar="COL1,COL2,...", default=None)
parser.add_option("--legend-per-experiment", dest="legend_per_experiment", help="Include legend entries for each experiment (default: only first experiment label)", action='store_true', default=False)
parser.add_option("--colormap", dest="colormap", help="Name of Matplotlib colormap to sample distinct colors for experiments (e.g., 'tab10','viridis'). Overrides ensemble-based coloring when provided.", default=None)
options, args = parser.parse_args()

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

# Check experiment limit
if len(experiment_specs) > 8:
    print(f"Warning: {len(experiment_specs)} experiments specified. For readability, consider limiting to 8 or fewer.")

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

# Build string for titles about the runs in use
def create_color_variations(base_color, n_variations):
    """Create n color variations from a base color by adjusting brightness and saturation."""
    import matplotlib.colors as mcolors
    hsv = mcolors.rgb_to_hsv(base_color[:3])
    variations = []
    if n_variations == 1:
        variations.append(base_color)
    else:
        for i in range(n_variations):
            brightness_factor = 0.4 + (0.6 * i / max(1, n_variations - 1))
            saturation_factor = 0.6 + (0.4 * i / max(1, n_variations - 1))
            new_hsv = hsv.copy()
            new_hsv[1] = min(1.0, hsv[1] * saturation_factor)
            new_hsv[2] = min(1.0, hsv[2] * brightness_factor)
            new_rgb = mcolors.hsv_to_rgb(new_hsv)
            variations.append(new_rgb)
    return variations

# Determine colors for each experiment. If user supplied a colors list, use those (cycled if needed).
experiment_to_color = {}
if options.colors:
    user_colors = [c.strip() for c in options.colors.split(',') if c.strip()]
    if not user_colors:
        sys.exit("ERROR: --colors provided but no valid colors parsed")
    for i, (_, _, _, display_name) in enumerate(experiment_specs):
        experiment_to_color[display_name] = user_colors[i % len(user_colors)]
elif options.colormap:
    # Sample N colors from the requested colormap
    cmap = plt.cm.get_cmap(options.colormap)
    N = len(experiment_specs)
    sampled = [cmap(x) for x in np.linspace(0, 1, N)]
    for i, (_, _, _, display_name) in enumerate(experiment_specs):
        experiment_to_color[display_name] = sampled[i]
else:
    ensemble_base_colors = plt.cm.Set1(np.linspace(0, 1, 9))  # Use Set1 colormap for distinct ensemble base colors
    ensemble_names_unique = list(set([ensemble for ensemble, _, _, _ in experiment_specs]))
    ensemble_to_base_color = {}
    for i, ensemble in enumerate(sorted(ensemble_names_unique)):
        ensemble_to_base_color[ensemble] = ensemble_base_colors[i % len(ensemble_base_colors)]
    experiments_by_ensemble = {}
    for ensemble, exp, file_path, display_name in experiment_specs:
        if ensemble not in experiments_by_ensemble:
            experiments_by_ensemble[ensemble] = []
        experiments_by_ensemble[ensemble].append((exp, file_path, display_name))
    for ensemble, experiments in experiments_by_ensemble.items():
        base_color = ensemble_to_base_color[ensemble]
        n_experiments = len(experiments)
        color_variations = create_color_variations(base_color, n_experiments)
        for i, (exp, file_path, display_name) in enumerate(experiments):
            experiment_to_color[display_name] = color_variations[i]

runinfo = ""
for i, (_, _, _, display_name) in enumerate(experiment_specs):
    ensemble = display_name.split(':')[0]
    if i == 0:
        runinfo = f'{display_name}'
    else:
        runinfo = f'{runinfo}\n{display_name}'

if options.units == "m3":
   massUnit = "m$^3$"
elif options.units == "kg":
   massUnit = "kg"
elif options.units == "Gt":
   massUnit = "Gt"
else:
   sys.exit("Unknown mass/volume units")
print("Using volume/mass units of: ", massUnit)

# Get nRegions and yr from first file
f = Dataset(experiment_specs[0][2], 'r')  # Use first experiment file path
nRegions = len(f.dimensions['nRegions'])
yr = f.variables['daysSinceStart'][:]/365.0

# Get region names from file
if options.fileRegionNames:
   fn = Dataset(options.fileRegionNames, 'r')
   rNamesIn = fn.variables['regionNames'][:]
else:
   rNamesIn = f.variables['regionNames'][:]
# Process region names
rNamesOrig = list()
for r in range(nRegions):
    thisString = rNamesIn[r, :].tobytes().decode('utf-8').strip()  # convert from char array to string
    rNamesOrig.append(''.join(filter(str.isalnum, thisString)))  # this bit removes non-alphanumeric chars

# Antarctic data from:
# Rignot, E., Bamber, J., van den Broeke, M. et al. Recent Antarctic ice mass loss from radar interferometry
# and regional climate modelling. Nature Geosci 1, 106-110 (2008). https://doi.org/10.1038/ngeo102
# Table 1: Mass balance of Antarctica in gigatonnes (10^12 kg) per year by sector for the year 2000
# https://www.nature.com/articles/ngeo102/tables/1
# and
# Rignot, E., S. Jacobs, J. Mouginot, and B. Scheuchl. 2013. Ice-Shelf Melting Around Antarctica. Science 341 (6143): 266-70. https://doi.org/10.1126/science.1235798.
# Note: May want to switch to input+, net+
# Note: Some ISMIP6 basins combine multiple Rignot basins.  May want to separate if we update our regions.

ISMIP6basinInfo = {
        'ISMIP6BasinAAp': {'name': 'Dronning Maud Land', 'input': [60,9], 'outflow': [60,7], 'net': [0, 11], 'shelfMelt': [57.5]},
        'ISMIP6BasinApB': {'name': 'Enderby Land', 'input': [39,5], 'outflow': [40,2], 'net': [-1,5], 'shelfMelt': [24.6]},
        'ISMIP6BasinBC': {'name': 'Amery-Lambert', 'input': [73, 10], 'outflow': [77,4], 'net': [-4, 11], 'shelfMelt': [35.5]},
        'ISMIP6BasinCCp': {'name': 'Phillipi, Denman', 'input': [81, 13], 'outflow': [87,7], 'net':[-7,15], 'shelfMelt': [107.9]},
        'ISMIP6BasinCpD': {'name': 'Totten', 'input': [198,37], 'outflow': [207,13], 'net': [-8,39], 'shelfMelt': [102.3]},
        'ISMIP6BasinDDp': {'name': 'Mertz', 'input': [93,14], 'outflow': [94,6], 'net': [-2,16], 'shelfMelt': [22.8]},
        'ISMIP6BasinDpE': {'name': 'Victoria Land', 'input': [20,1], 'outflow': [22,3], 'net': [-2,4], 'shelfMelt': [22.9]},
        'ISMIP6BasinEF': {'name': 'Ross', 'input': [61+110,(10**2+7**2)**0.5], 'outflow': [49+80,(4**2+2^2)**0.5], 'net': [11+31,(11*2+7**2)**0.5], 'shelfMelt': [70.3]},
        'ISMIP6BasinFG': {'name': 'Getz', 'input': [108,28], 'outflow': [128,18], 'net': [-19,33], 'shelfMelt': [152.9]},
        'ISMIP6BasinGH': {'name': 'Thwaites/PIG', 'input': [177,25], 'outflow': [237,4], 'net': [-61,26], 'shelfMelt': [290.9]},
        'ISMIP6BasinHHp': {'name': 'Bellingshausen', 'input': [51,16], 'outflow': [86,10], 'net': [-35,19], 'shelfMelt': [76.3]},
        'ISMIP6BasinHpI': {'name': 'George VI', 'input': [71,21], 'outflow': [78,7], 'net': [-7,23], 'shelfMelt': [152.3]},
        'ISMIP6BasinIIpp': {'name': 'Larsen A-C', 'input': [15,5], 'outflow': [20,3], 'net': [-5,6], 'shelfMelt': [32.9]},
        'ISMIP6BasinIppJ': {'name': 'Larsen E', 'input': [8,4], 'outflow': [9,2], 'net': [-1,4], 'shelfMelt': [4.3]},
        'ISMIP6BasinJK': {'name': 'FRIS', 'input': [93+142, (8**2+11**2)**0.5], 'outflow': [75+145,(4**2+7**2)**0.5], 'net': [18-4,(9**2+13**2)**0.5], 'shelfMelt': [155.4]},
        'ISMIP6BasinKA': {'name': 'Brunt-Stancomb', 'input': [42+26,(8**2+7**2)**0.5], 'outflow': [45+28,(4**2+2**2)**0.5], 'net':[-3-1,(9**2+8**2)**0.5], 'shelfMelt': [10.4]}
        }

# Paolo 2023 net shelf melt values:
ISMIP6basinInfo = {
        'ISMIP6BasinAAp': {'name': 'Dronning Maud Land', 'input': [60,9], 'outflow': [60,7], 'net': [0, 11], 'shelfMelt': [37.49]},
        'ISMIP6BasinApB': {'name': 'Enderby Land', 'input': [39,5], 'outflow': [40,2], 'net': [-1,5], 'shelfMelt': [17.34]},
        'ISMIP6BasinBC': {'name': 'Amery-Lambert', 'input': [73, 10], 'outflow': [77,4], 'net': [-4, 11], 'shelfMelt': [21.03]},
        'ISMIP6BasinCCp': {'name': 'Phillipi, Denman', 'input': [81, 13], 'outflow': [87,7], 'net':[-7,15], 'shelfMelt': [40.27]},
        'ISMIP6BasinCpD': {'name': 'Totten', 'input': [198,37], 'outflow': [207,13], 'net': [-8,39], 'shelfMelt': [69.88]},
        'ISMIP6BasinDDp': {'name': 'Mertz', 'input': [93,14], 'outflow': [94,6], 'net': [-2,16], 'shelfMelt': [17.72]},
        'ISMIP6BasinDpE': {'name': 'Victoria Land', 'input': [20,1], 'outflow': [22,3], 'net': [-2,4], 'shelfMelt': [11.32]},
        'ISMIP6BasinEF': {'name': 'Ross', 'input': [61+110,(10**2+7**2)**0.5], 'outflow': [49+80,(4**2+2^2)**0.5], 'net': [11+31,(11*2+7**2)**0.5], 'shelfMelt': [40.1]},
        'ISMIP6BasinFG': {'name': 'Getz', 'input': [108,28], 'outflow': [128,18], 'net': [-19,33], 'shelfMelt': [119.3]},
        'ISMIP6BasinGH': {'name': 'Thwaites/PIG', 'input': [177,25], 'outflow': [237,4], 'net': [-61,26], 'shelfMelt': [191.04]},
        'ISMIP6BasinHHp': {'name': 'Bellingshausen', 'input': [51,16], 'outflow': [86,10], 'net': [-35,19], 'shelfMelt': [54.57]},
        'ISMIP6BasinHpI': {'name': 'George VI', 'input': [71,21], 'outflow': [78,7], 'net': [-7,23], 'shelfMelt': [85.53]},
        'ISMIP6BasinIIpp': {'name': 'Larsen A-C', 'input': [15,5], 'outflow': [20,3], 'net': [-5,6], 'shelfMelt': [23.09]},
        'ISMIP6BasinIppJ': {'name': 'Larsen E', 'input': [8,4], 'outflow': [9,2], 'net': [-1,4], 'shelfMelt': [16.51]},
        'ISMIP6BasinJK': {'name': 'FRIS', 'input': [93+142, (8**2+11**2)**0.5], 'outflow': [75+145,(4**2+7**2)**0.5], 'net': [18-4,(9**2+13**2)**0.5], 'shelfMelt': [54.21]},
        'ISMIP6BasinKA': {'name': 'Brunt-Stancomb', 'input': [42+26,(8**2+7**2)**0.5], 'outflow': [45+28,(4**2+2**2)**0.5], 'net':[-3-1,(9**2+8**2)**0.5], 'shelfMelt': [26.05]}
        }


# Parse region names to more usable names, if available
rNames = [None]*nRegions
for r in range(nRegions):
    if rNamesOrig[r] in ISMIP6basinInfo:
        rNames[r] = ISMIP6basinInfo[rNamesOrig[r]]['name']
    else:
        rNames[r] = rNamesOrig[r]

#print(rNames)

if nRegions <= 4:
    ncol = 2
elif nRegions <= 9:
    ncol = 3
elif nRegions <= 16:
    ncol = 4
elif nRegions <= 25:
    ncol = 5
else:
    sys.exit("ERROR: More than 25 regions found.  Attempting to plot this many regions is likely a bad idea.")
nrow = np.ceil(nRegions / ncol).astype('int') # Set nrow to have enough rows to plot number of regions based on ncol calculated above

# Set up Figure 1: volume stats overview
fig1, axs1 = plt.subplots(nrow, ncol, figsize=(13, 11), num=1)
fig1.suptitle(f'Mass change summary\n{runinfo}', fontsize=9)
for reg in range(nRegions):
   plt.sca(axs1.flatten()[reg])
   plt.xlabel('Year')
   plt.ylabel('volume change ({})'.format(massUnit))
   #plt.xticks(np.arange(22)*xtickSpacing)
   plt.grid()
   axs1.flatten()[reg].set_title(rNames[reg])
   if reg == 0:
      axX = axs1.flatten()[reg]
   else:
      axs1.flatten()[reg].sharex(axX)
   # plot obs if applicable
   if rNamesOrig[reg] in ISMIP6basinInfo:
       [mn, sig] = ISMIP6basinInfo[rNamesOrig[reg]]['net']
       axs1.flatten()[reg].fill_between(yr, yr*(mn-sig), yr*(mn+sig), color='b', alpha=0.2, label='grd obs')

# Set up Figure 2: grounded MB
fig2, axs2 = plt.subplots(nrow, ncol, figsize=(13, 11), num=2)
fig2.suptitle(f'Grounded mass change\n{runinfo}', fontsize=9)
for reg in range(nRegions):
   plt.sca(axs2.flatten()[reg])
   if reg // nrow == nrow-1:
      plt.xlabel('Year')
   if reg % ncol == 0:
      plt.ylabel('volume change ({})'.format(massUnit))
   #plt.xticks(np.arange(22)*xtickSpacing)
   plt.grid()
   axs2.flatten()[reg].set_title(rNames[reg])
   if reg == 0:
      axX = axs2.flatten()[reg]
   else:
      axs2.flatten()[reg].sharex(axX)
   # plot obs if applicable
   if rNamesOrig[reg] in ISMIP6basinInfo:
       [mn, sig] = ISMIP6basinInfo[rNamesOrig[reg]]['input']
       axs2.flatten()[reg].fill_between(yr, yr*(mn-sig), yr*(mn+sig), color='b', alpha=0.2, label='SMB obs')
       [mn, sig] = ISMIP6basinInfo[rNamesOrig[reg]]['outflow']
       axs2.flatten()[reg].fill_between(yr, -yr*(mn-sig), -yr*(mn+sig), color='g', alpha=0.2, label='outflow obs')
       [mn, sig] = ISMIP6basinInfo[rNamesOrig[reg]]['net']
       axs2.flatten()[reg].fill_between(yr, yr*(mn-sig), yr*(mn+sig), color='k', alpha=0.2, label='net obs')


# Set up Figure 3: floating MB
fig3, axs3 = plt.subplots(nrow, ncol, figsize=(13, 11), num=3)
fig3.suptitle(f'Floating mass change\n{runinfo}', fontsize=9)
for reg in range(nRegions):
   plt.sca(axs3.flatten()[reg])
   plt.xlabel('Year')
   plt.ylabel('volume change ({})'.format(massUnit))
   #plt.xticks(np.arange(22)*xtickSpacing)
   plt.grid()
   axs3.flatten()[reg].set_title(rNames[reg])
   if reg == 0:
      axX = axs3.flatten()[reg]
   else:
      axs3.flatten()[reg].sharex(axX)

# Set up Figure 4: area change
fig4, axs4 = plt.subplots(nrow, ncol, figsize=(13, 11), num=4)
fig4.suptitle(f'Area change\n{runinfo}', fontsize=9)
for reg in range(nRegions):
   plt.sca(axs4.flatten()[reg])
   plt.xlabel('Year')
   plt.ylabel('Area change (km^2)')
   #plt.xticks(np.arange(22)*xtickSpacing)
   plt.grid()
   axs4.flatten()[reg].set_title(rNames[reg])
   if reg == 0:
      axX = axs4.flatten()[reg]
   else:
      axs4.flatten()[reg].sharex(axX)


# Set up Figure 5
fig5, axs5 = plt.subplots(2,1, figsize=(13, 11), num=5)
fig5.suptitle(f'regional contributions\n{runinfo}', fontsize=9)
mnTot=0.0
sigTot = 0.0
for reg in range(nRegions):
   if rNamesOrig[reg] in ISMIP6basinInfo:
       [mn, sig] = ISMIP6basinInfo[rNamesOrig[reg]]['net']
       mnTot += mn
       sigTot += sig**2
sigTot = sigTot**0.5
axs5.flatten()[0].fill_between(yr, yr*(mnTot-sigTot), yr*(mnTot+sigTot), color='k', alpha=0.2, label='net obs')
plt.sca(axs5.flatten()[0])
plt.xlabel('Year')
plt.ylabel('Mass change (Gt)')
plt.grid()
axs5.flatten()[1].fill_between(yr, yr*(mnTot-sigTot), yr*(mnTot+sigTot), color='k', alpha=0.2, label='net obs')
plt.sca(axs5.flatten()[1])
plt.xlabel('Year')
plt.ylabel('VAF mass change (Gt)')
plt.grid()


# Set up Figure 6: melt rate vs obs
fig6, axs6 = plt.subplots(nrow, ncol, figsize=(13, 11), num=6)
fig6.suptitle(f'Ice-shelf melt rate\n{runinfo}', fontsize=9)
for reg in range(nRegions):
   plt.sca(axs6.flatten()[reg])
   plt.xlabel('Year')
   plt.ylabel('Ice-shelf melt rate (Gt/yr)')
   #plt.xticks(np.arange(22)*xtickSpacing)
   plt.grid()
   axs6.flatten()[reg].set_title(rNames[reg])
   if reg == 0:
      axX = axs6.flatten()[reg]
   else:
      axs6.flatten()[reg].sharex(axX)
   if rNamesOrig[reg] in ISMIP6basinInfo:
       mlt = ISMIP6basinInfo[rNamesOrig[reg]]['shelfMelt'][0]
       axs6.flatten()[reg].plot(yr, np.ones(yr.shape)*(mlt), color='k', label='melt obs')

# Set up unit conversion factors to be used when reading variables
if options.units == "m3":
    volUnitFactor = 1.0
    massUnitFactor = 1.0 / rhoi
elif options.units == "kg":
    volUnitFactor = rhoi
    massUnitFactor = 1.0
elif options.units == "Gt":
    volUnitFactor = rhoi / 1.0e12
    massUnitFactor = 1.0 / 1.0e12
else:
    sys.exit("ERROR: Unknown unit specified")

def plotStat(fname, display_name, color, addToLegend=False):
    """Modified plotStat function to use only color variations (no line styles)."""
    print("Reading and plotting file: {} for experiment: {}".format(fname, display_name))

    f = Dataset(fname,'r')
    yr = f.variables['daysSinceStart'][:]/365.0
    dt = f.variables['deltat'][:]/(3600.0*24.0*365.0) # in yr
    #yr = yr-yr[0]  # uncomment to align all start dates
    dtnR = np.tile(dt.reshape(len(dt),1), (1,nRegions))  # repeated per region with dim of nt,nRegions
    nRegionsLocal = len(f.dimensions['nRegions'])
    if nRegionsLocal != nRegions:
        sys.exit(f"ERROR: Number of regions in file {fname} does not match number of regions in first input file!")

    # Fig 1: summary plot
    vol = f.variables['regionalIceVolume'][:] * volUnitFactor
    lbl = f'{display_name} total' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs1.flatten()[r].plot(yr, vol[:,r] - vol[0,r], label=lbl, color=color, linewidth=1.5)

    VAF = f.variables['regionalVolumeAboveFloatation'][:] * volUnitFactor
    VAF = VAF[:,:] - VAF[0,:]
    lbl = f'{display_name} VAF' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs1.flatten()[r].plot(yr, VAF[:,r] - VAF[0,r], label=lbl, color=color, alpha=0.7, linewidth=1.5)

    volGround = f.variables['regionalGroundedIceVolume'][:] * volUnitFactor
    volGround = volGround[:,:] - volGround[0,:]
    lbl = f'{display_name} grd' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs1.flatten()[r].plot(yr, volGround[:,r] - volGround[0,r], label=lbl, color=color, alpha=0.8, linewidth=1.5)

    volFloat = f.variables['regionalFloatingIceVolume'][:] * volUnitFactor
    volFloat = volFloat[:,:] - volFloat[0,:]
    lbl = f'{display_name} flt' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs1.flatten()[r].plot(yr, volFloat[:,r] - volFloat[0,r], label=lbl, color=color, alpha=0.6, linewidth=1.5)

    # Fig 2: Grd MB ------------
    lbl = f'{display_name} vol chg' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs2.flatten()[r].plot(yr, volGround[:,r] - volGround[0,r], label=lbl, color=color, linewidth=1.5)

    grdSMB = f.variables['regionalSumGroundedSfcMassBal'][:] * massUnitFactor
    cumGrdSMB = np.cumsum(grdSMB*dtnR, axis=0)
    lbl = f'{display_name} SMB' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs2.flatten()[r].plot(yr, cumGrdSMB[:,r], label=lbl, color=color, alpha=0.7, linewidth=1.5)

    GLflux = f.variables['regionalSumGroundingLineFlux'][:] * massUnitFactor
    cumGLflux = np.cumsum(GLflux*dtnR, axis=0)
    lbl = f'{display_name} GL flux' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs2.flatten()[r].plot(yr, -1.0*cumGLflux[:,r], label=lbl, color=color, alpha=0.7, linewidth=1.5)

    GLMigflux = f.variables['regionalSumGroundingLineMigrationFlux'][:] * massUnitFactor
    cumGLMigflux = np.cumsum(GLMigflux*dtnR, axis=0)
    lbl = f'{display_name} GL mig flux' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs2.flatten()[r].plot(yr, -1.0*cumGLMigflux[:,r], label=lbl, color=color, alpha=0.6, linewidth=1.5)

    # sum of components
    grdSum = grdSMB - GLflux - GLMigflux # note negative sign on two GL terms - they are both positive grounded to floating
    cumGrdSum = np.cumsum(grdSum*dtnR, axis=0)
    lbl = f'{display_name} sum' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs2.flatten()[r].plot(yr, cumGrdSum[:,r], label=lbl, color=color, alpha=0.5, linewidth=0.8)
    grdSum2 = grdSMB - GLflux  # note negative sign on two GL terms - they are both positive grounded to floating
    cumGrdSum2 = np.cumsum(grdSum2*dtnR, axis=0)
    lbl = f'{display_name} sum, no GLmig' if addToLegend else '_nolegend_'
    for r in range(nRegions):
        axs2.flatten()[r].plot(yr, cumGrdSum2[:,r], label=lbl, linestyle=':', color=color, alpha=0.5, linewidth=0.8)

    # Fig 3: Flt MB ---------------
    lbl = f'{display_name} vol chg' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs3.flatten()[r].plot(yr, volFloat[:,r] - volFloat[0,r], label=lbl, color=color, linewidth=1.5)

    fltSMB = f.variables['regionalSumFloatingSfcMassBal'][:] * massUnitFactor
    cumFltSMB = np.cumsum(fltSMB*dtnR, axis=0)
    lbl = f'{display_name} SMB' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs3.flatten()[r].plot(yr, cumFltSMB[:,r], label=lbl, color=color, alpha=0.7, linewidth=1.5)

    lbl = f'{display_name} GL flux' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs3.flatten()[r].plot(yr, cumGLflux[:,r], label=lbl, color=color, alpha=0.7, linewidth=1.5)

    lbl = f'{display_name} GL mig flux' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs3.flatten()[r].plot(yr, cumGLMigflux[:,r], label=lbl, color=color, alpha=0.6, linewidth=1.5)

    clv = f.variables['regionalSumCalvingFlux'][:] * massUnitFactor
    cumClv = np.cumsum(clv*dtnR, axis=0)
    lbl = f'{display_name} calving' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs3.flatten()[r].plot(yr, -1.0*cumClv[:,r], label=lbl, color=color, alpha=0.7, linewidth=1.2)

    BMB = f.variables['regionalSumFloatingBasalMassBal'][:] * massUnitFactor
    cumBMB = np.cumsum(BMB*dtnR, axis=0)
    lbl = f'{display_name} BMB' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs3.flatten()[r].plot(yr, cumBMB[:,r], label=lbl, color=color, alpha=0.8, linewidth=1.2)

    # sum of components
    fltSum = fltSMB + GLflux + GLMigflux - clv + BMB
    cumFltSum = np.cumsum(fltSum*dtnR, axis=0)
    lbl = f'{display_name} sum' if addToLegend else '_nolegend_'
    for r in range(nRegions):
       axs3.flatten()[r].plot(yr, cumFltSum[:,r], label=lbl, color=color, alpha=0.5, linewidth=0.8)
    fltSum2 = fltSMB + GLflux - clv + BMB
    cumFltSum2 = np.cumsum(fltSum2*dtnR, axis=0)
    lbl = f'{display_name} sum, no GLmig' if addToLegend else '_nolegend_'
    for r in range(nRegions):
        axs3.flatten()[r].plot(yr, cumFltSum2[:,r], label=lbl, linestyle=':', color=color, alpha=0.5, linewidth=0.8)

    # Fig 4: area change  ---------------
    areaTot = f.variables['regionalIceArea'][:]/1000.0**2
    areaGrd = f.variables['regionalGroundedIceArea'][:]/1000.0**2
    areaFlt = f.variables['regionalFloatingIceArea'][:]/1000.0**2
    for r in range(nRegions):
        axs4.flatten()[r].plot(yr, areaTot[:,r] - areaTot[0,r], label=(f"{display_name} total area" if addToLegend else '_nolegend_'), color=color, linewidth=1.5)
        axs4.flatten()[r].plot(yr, areaGrd[:,r] - areaGrd[0,r], label=(f"{display_name} grd area" if addToLegend else '_nolegend_'), color=color, alpha=0.7, linewidth=1.5)
        axs4.flatten()[r].plot(yr, areaFlt[:,r] - areaFlt[0,r], label=(f"{display_name} flt area" if addToLegend else '_nolegend_'), color=color, alpha=0.6, linewidth=1.5)

    # Fig. 5:  select global stats ---------
    for r in range(nRegions):
        if rNamesOrig[r] == 'ISMIP6BasinGH':
           indTG = r
           break
    #print(f'TG index={indTG}')
    axs5.flatten()[0].plot(yr, volGround.sum(axis=1), label=f'{display_name} total', color=color, linewidth=1.5)
    volGroundnoTG = np.delete(volGround, indTG, 1)
    axs5.flatten()[0].plot(yr, volGroundnoTG.sum(axis=1), label=f'{display_name} no TG/PIG', color=color, alpha=0.7, linewidth=1.5)
    #for r in range(nRegions):
       #axs5.flatten()[0].plot(yr, VAF[:,r] - VAF[0,r], label=rNames[r], linestyle=sty)
    axs5.flatten()[1].plot(yr, VAF.sum(axis=1), label=f'{display_name} total', color=color, linewidth=1.5)
    VAFnoTG = np.delete(VAF, indTG, 1)
    axs5.flatten()[1].plot(yr, VAFnoTG.sum(axis=1), label=f'{display_name} no TG/PIG', color=color, alpha=0.7, linewidth=1.5)

    # Fig. 6:  melt rates ---------
    #BMB = np.where(BMB==0.0, BMB, np.nan*BMB)
    for r in range(nRegions):
        axs6.flatten()[r].plot(yr, -BMB[:,r], label=(f"{display_name} BMB" if addToLegend else '_nolegend_'), color=color, linewidth=1.5)

    f.close()

# Plot each experiment with assigned colors
for i, (ensemble, exp, file_path, display_name) in enumerate(experiment_specs):
    color = experiment_to_color[display_name]
    add_to_legend = True if options.legend_per_experiment else (i == 0)
    plotStat(file_path, display_name, color, addToLegend=add_to_legend)

# Apply x-axis limits to all subplots if specified
if xlim_range:
    print(f"Applying X-axis limits: {xlim_range}")
    all_axes = []
    all_axes.extend(axs1.flatten())
    all_axes.extend(axs2.flatten()) 
    all_axes.extend(axs3.flatten())
    all_axes.extend(axs4.flatten())
    all_axes.extend(axs5.flatten())
    all_axes.extend(axs6.flatten())
    
    for ax in all_axes:
        ax.set_xlim(xlim_range)

# Add legends and finalize plots
axs1.flatten()[-1].legend(loc='best', prop={'size': 5})
axs2.flatten()[-1].legend(loc='best', prop={'size': 5})
axs3.flatten()[-1].legend(loc='best', prop={'size': 5})
axs4.flatten()[-1].legend(loc='best', prop={'size': 6})
axs5.flatten()[0].legend(loc='best', prop={'size': 6})
axs6.flatten()[-1].legend(loc='best', prop={'size': 6})

print("Generating plot.")
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()
plt.show()
