#!/usr/bin/env python
'''
Plot common time-series from one or more landice regionalStats files -- SUBSET variant.

Same as plot_ensembleRegionalStats.py, but adds a --regions selector so you can plot only the region
indices you care about, instead of every region. This lifts the "<=25 regions" limit for region masks with
many regions (e.g. the 133-region draft-dependent masks): you pick a handful and it lays them out normally.

  --regions "33-37,40,42-45"    # 0-based indices into the nRegions dimension; ranges and singletons, mixed

Everything else (ensemble/experiment discovery, colors, units, x-limits) is unchanged.

Original by Matt Hoffman, 8/23/2022
Ensemble version + subset selector -- Shiva Muruganandham
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
parser.add_option("--regions", dest="regionSelection", help="Region indices to plot (0-based into the nRegions dimension). Comma-separated singletons and/or ranges, e.g. '33-37,40,42-45'. If omitted, plots ALL regions (still subject to the <=25 guard).", metavar="SPEC", default=None)
parser.add_option("--list-regions", dest="listRegions", help="Print the index -> region name table for the first file and exit.", action='store_true', default=False)
parser.add_option("--save", dest="savePrefix", help="Save the six figures to PREFIX_{summary,grounded,floating,area,contributions,meltrate}.png instead of opening interactive windows.", metavar="PREFIX", default=None)
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
        full_ensemble_path = os.path.join(root_dir, ensemble_dir) if root_dir else ensemble_dir
        if not os.path.exists(full_ensemble_path):
            print(f"Warning: Ensemble directory not found: {full_ensemble_path}")
            continue
        for item in os.listdir(full_ensemble_path):
            exp_path = os.path.join(full_ensemble_path, item)
            if os.path.isdir(exp_path):
                stats_file = os.path.join(exp_path, stats_filename)
                if os.path.exists(stats_file):
                    available_experiments.setdefault(ensemble_dir, []).append(item)
    return available_experiments

def parse_experiment_specifications(experiment_list, ensemble_dirs, root_dir, stats_filename):
    """Parse experiment specs -> list of (ensemble, experiment, file_path, display_name) tuples."""
    experiment_specs = []
    if not experiment_list:
        sys.exit("ERROR: Must specify experiment list with -e/--experiments option")
    exp_parts = [exp.strip() for exp in experiment_list.split(',')]
    for exp_spec in exp_parts:
        if ':' in exp_spec:
            ensemble_name, exp_name = (s.strip() for s in exp_spec.split(':', 1))
            if ensemble_name not in ensemble_dirs:
                print(f"Warning: Specified ensemble '{ensemble_name}' not in ensemble directory list")
                continue
            if '*' in exp_name or '?' in exp_name:
                search_path = os.path.join(root_dir, ensemble_name) if root_dir else ensemble_name
                if os.path.exists(search_path):
                    for match_path in glob.glob(os.path.join(search_path, exp_name)):
                        if os.path.isdir(match_path):
                            match_exp = os.path.basename(match_path)
                            stats_file = os.path.join(match_path, stats_filename)
                            if os.path.exists(stats_file):
                                experiment_specs.append((ensemble_name, match_exp, stats_file, f"{ensemble_name}:{match_exp}"))
            else:
                exp_path = os.path.join(root_dir, ensemble_name, exp_name) if root_dir else os.path.join(ensemble_name, exp_name)
                stats_file = os.path.join(exp_path, stats_filename)
                if os.path.exists(stats_file):
                    experiment_specs.append((ensemble_name, exp_name, stats_file, f"{ensemble_name}:{exp_name}"))
                else:
                    print(f"Warning: Stats file not found for {ensemble_name}:{exp_name} at {stats_file}")
        else:
            exp_name = exp_spec.strip()
            found_in_ensembles = []
            if '*' in exp_name or '?' in exp_name:
                for ensemble_dir in ensemble_dirs:
                    search_path = os.path.join(root_dir, ensemble_dir) if root_dir else ensemble_dir
                    if os.path.exists(search_path):
                        for match_path in glob.glob(os.path.join(search_path, exp_name)):
                            if os.path.isdir(match_path):
                                match_exp = os.path.basename(match_path)
                                stats_file = os.path.join(match_path, stats_filename)
                                if os.path.exists(stats_file):
                                    experiment_specs.append((ensemble_dir, match_exp, stats_file, f"{ensemble_dir}:{match_exp}"))
                                    found_in_ensembles.append(ensemble_dir)
            else:
                for ensemble_dir in ensemble_dirs:
                    exp_path = os.path.join(root_dir, ensemble_dir, exp_name) if root_dir else os.path.join(ensemble_dir, exp_name)
                    stats_file = os.path.join(exp_path, stats_filename)
                    if os.path.exists(stats_file):
                        experiment_specs.append((ensemble_dir, exp_name, stats_file, f"{ensemble_dir}:{exp_name}"))
                        found_in_ensembles.append(ensemble_dir)
                if not found_in_ensembles:
                    print(f"Warning: Experiment '{exp_name}' not found in any ensemble directory")
    return experiment_specs

def parse_region_selection(spec, nRegions):
    """Parse '33-37,40,42-45' -> ordered, de-duplicated list of 0-based region indices, validated."""
    sel = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo_s, hi_s = part.split('-', 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                sys.exit(f"ERROR: bad range '{part}' in --regions (expected like '33-37')")
            if lo > hi:
                lo, hi = hi, lo
            sel.extend(range(lo, hi + 1))
        else:
            try:
                sel.append(int(part))
            except ValueError:
                sys.exit(f"ERROR: bad region index '{part}' in --regions")
    seen, out = set(), []
    for i in sel:
        if i in seen:
            continue
        seen.add(i)
        if i < 0 or i >= nRegions:
            sys.exit(f"ERROR: region index {i} out of range 0..{nRegions - 1}")
        out.append(i)
    if not out:
        sys.exit("ERROR: no valid regions parsed from --regions")
    return out

# Parse ensemble directories
ensemble_dirs = []
if options.searchAll:
    if not options.rootDataDir:
        sys.exit("ERROR: --search-all requires --root to be specified")
    for item in os.listdir(options.rootDataDir):
        if os.path.isdir(os.path.join(options.rootDataDir, item)):
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
    total = 0
    for ensemble, experiments in available.items():
        print(f"\n{ensemble}: ({len(experiments)} experiments)")
        for exp in sorted(experiments):
            print(f"  {exp}")
        total += len(experiments)
    print(f"\nTotal: {total} experiments across {len(available)} ensembles")
    sys.exit(0)

# Parse experiment specifications (auto-discover if -e not provided)
if not options.experimentList:
    if not options.rootDataDir:
        sys.exit("ERROR: --root must be provided when auto-discovering experiments")
    available = find_all_experiments(options.rootDataDir, ensemble_dirs, options.statsFilename)
    experiment_specs = []
    for ensemble, exps in available.items():
        for exp in sorted(exps):
            stats_file = os.path.join(options.rootDataDir, ensemble, exp, options.statsFilename)
            experiment_specs.append((ensemble, exp, stats_file, f"{ensemble}:{exp}"))
    if not experiment_specs:
        sys.exit("ERROR: No experiments found under the provided root/base directories")
else:
    experiment_specs = parse_experiment_specifications(
        options.experimentList, ensemble_dirs, options.rootDataDir, options.statsFilename)
    if not experiment_specs:
        sys.exit("ERROR: No valid experiments found")

print(f"\nFound {len(experiment_specs)} experiments to plot:")
for ensemble, exp, file_path, display_name in experiment_specs:
    print(f"  {display_name}: {file_path}")
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
            variations.append(mcolors.hsv_to_rgb(new_hsv))
    return variations

# Determine colors for each experiment
experiment_to_color = {}
if options.colors:
    user_colors = [c.strip() for c in options.colors.split(',') if c.strip()]
    if not user_colors:
        sys.exit("ERROR: --colors provided but no valid colors parsed")
    for i, (_, _, _, display_name) in enumerate(experiment_specs):
        experiment_to_color[display_name] = user_colors[i % len(user_colors)]
elif options.colormap:
    cmap = plt.cm.get_cmap(options.colormap)
    sampled = [cmap(x) for x in np.linspace(0, 1, len(experiment_specs))]
    for i, (_, _, _, display_name) in enumerate(experiment_specs):
        experiment_to_color[display_name] = sampled[i]
else:
    ensemble_base_colors = plt.cm.Set1(np.linspace(0, 1, 9))
    ensemble_names_unique = list(set([ensemble for ensemble, _, _, _ in experiment_specs]))
    ensemble_to_base_color = {}
    for i, ensemble in enumerate(sorted(ensemble_names_unique)):
        ensemble_to_base_color[ensemble] = ensemble_base_colors[i % len(ensemble_base_colors)]
    experiments_by_ensemble = {}
    for ensemble, exp, file_path, display_name in experiment_specs:
        experiments_by_ensemble.setdefault(ensemble, []).append((exp, file_path, display_name))
    for ensemble, experiments in experiments_by_ensemble.items():
        color_variations = create_color_variations(ensemble_to_base_color[ensemble], len(experiments))
        for i, (exp, file_path, display_name) in enumerate(experiments):
            experiment_to_color[display_name] = color_variations[i]

runinfo = ""
for i, (_, _, _, display_name) in enumerate(experiment_specs):
    runinfo = f'{display_name}' if i == 0 else f'{runinfo}\n{display_name}'

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
f = Dataset(experiment_specs[0][2], 'r')
nRegions = len(f.dimensions['nRegions'])
yr = f.variables['daysSinceStart'][:] / 365.0

# Get region names
if options.fileRegionNames:
    fn = Dataset(options.fileRegionNames, 'r')
    rNamesIn = fn.variables['regionNames'][:]
else:
    rNamesIn = f.variables['regionNames'][:]
rNamesOrig = list()
for r in range(nRegions):
    thisString = rNamesIn[r, :].tobytes().decode('utf-8').strip()
    rNamesOrig.append(''.join(filter(str.isalnum, thisString)))

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

# Parse region names to friendlier names where available
rNames = [None] * nRegions
for r in range(nRegions):
    rNames[r] = ISMIP6basinInfo[rNamesOrig[r]]['name'] if rNamesOrig[r] in ISMIP6basinInfo else rNamesOrig[r]

# --list-regions: dump the index -> name table and exit (handy for building a --regions spec)
if options.listRegions:
    print(f"\n{nRegions} regions in {experiment_specs[0][2]}:")
    for r in range(nRegions):
        print(f"  {r:4d}  {rNames[r]}")
    sys.exit(0)

if options.regionSelection:
    regList = parse_region_selection(options.regionSelection, nRegions)
    print(f"\nPlotting {len(regList)} of {nRegions} regions (indices): {regList}")
    print("  " + ", ".join(f"{r}:{rNames[r]}" for r in regList))
else:
    regList = list(range(nRegions))
nPlot = len(regList)

# Grid layout is now driven by the NUMBER SELECTED, not the total region count
if nPlot <= 4:
    ncol = 2
elif nPlot <= 9:
    ncol = 3
elif nPlot <= 16:
    ncol = 4
elif nPlot <= 25:
    ncol = 5
else:
    sys.exit(f"ERROR: {nPlot} regions selected. Plotting more than 25 at once is a bad idea -- "
             f"narrow --regions (ranges/lists), or use --list-regions to choose.")
nrow = int(np.ceil(nPlot / ncol))

def _hide_unused(axs):
    """Turn off any trailing subplots beyond the number of selected regions."""
    flat = axs.flatten()
    for k in range(nPlot, len(flat)):
        flat[k].axis('off')

fig1, axs1 = plt.subplots(nrow, ncol, figsize=(13, 11), num=1, squeeze=False)
fig1.suptitle(f'Mass change summary\n{runinfo}', fontsize=9)
for p, reg in enumerate(regList):
    plt.sca(axs1.flatten()[p])
    plt.xlabel('Year'); plt.ylabel('volume change ({})'.format(massUnit)); plt.grid()
    axs1.flatten()[p].set_title(rNames[reg])
    if p == 0:
        axX = axs1.flatten()[p]
    else:
        axs1.flatten()[p].sharex(axX)
    if rNamesOrig[reg] in ISMIP6basinInfo:
        [mn, sig] = ISMIP6basinInfo[rNamesOrig[reg]]['net']
        axs1.flatten()[p].fill_between(yr, yr*(mn-sig), yr*(mn+sig), color='b', alpha=0.2, label='grd obs')
_hide_unused(axs1)

fig2, axs2 = plt.subplots(nrow, ncol, figsize=(13, 11), num=2, squeeze=False)
fig2.suptitle(f'Grounded mass change\n{runinfo}', fontsize=9)
for p, reg in enumerate(regList):
    plt.sca(axs2.flatten()[p])
    if p // nrow == nrow-1:
        plt.xlabel('Year')
    if p % ncol == 0:
        plt.ylabel('volume change ({})'.format(massUnit))
    plt.grid()
    axs2.flatten()[p].set_title(rNames[reg])
    if p == 0:
        axX = axs2.flatten()[p]
    else:
        axs2.flatten()[p].sharex(axX)
    if rNamesOrig[reg] in ISMIP6basinInfo:
        [mn, sig] = ISMIP6basinInfo[rNamesOrig[reg]]['input']
        axs2.flatten()[p].fill_between(yr, yr*(mn-sig), yr*(mn+sig), color='b', alpha=0.2, label='SMB obs')
        [mn, sig] = ISMIP6basinInfo[rNamesOrig[reg]]['outflow']
        axs2.flatten()[p].fill_between(yr, -yr*(mn-sig), -yr*(mn+sig), color='g', alpha=0.2, label='outflow obs')
        [mn, sig] = ISMIP6basinInfo[rNamesOrig[reg]]['net']
        axs2.flatten()[p].fill_between(yr, yr*(mn-sig), yr*(mn+sig), color='k', alpha=0.2, label='net obs')
_hide_unused(axs2)

fig3, axs3 = plt.subplots(nrow, ncol, figsize=(13, 11), num=3, squeeze=False)
fig3.suptitle(f'Floating mass change\n{runinfo}', fontsize=9)
for p, reg in enumerate(regList):
    plt.sca(axs3.flatten()[p])
    plt.xlabel('Year'); plt.ylabel('volume change ({})'.format(massUnit)); plt.grid()
    axs3.flatten()[p].set_title(rNames[reg])
    if p == 0:
        axX = axs3.flatten()[p]
    else:
        axs3.flatten()[p].sharex(axX)
_hide_unused(axs3)

fig4, axs4 = plt.subplots(nrow, ncol, figsize=(13, 11), num=4, squeeze=False)
fig4.suptitle(f'Area change\n{runinfo}', fontsize=9)
for p, reg in enumerate(regList):
    plt.sca(axs4.flatten()[p])
    plt.xlabel('Year'); plt.ylabel('Area change (km^2)'); plt.grid()
    axs4.flatten()[p].set_title(rNames[reg])
    if p == 0:
        axX = axs4.flatten()[p]
    else:
        axs4.flatten()[p].sharex(axX)
_hide_unused(axs4)

fig5, axs5 = plt.subplots(2, 1, figsize=(13, 11), num=5)
fig5.suptitle(f'regional contributions (selected regions)\n{runinfo}', fontsize=9)
mnTot = 0.0
sigTot = 0.0
for reg in regList:
    if rNamesOrig[reg] in ISMIP6basinInfo:
        [mn, sig] = ISMIP6basinInfo[rNamesOrig[reg]]['net']
        mnTot += mn
        sigTot += sig**2
sigTot = sigTot**0.5
axs5.flatten()[0].fill_between(yr, yr*(mnTot-sigTot), yr*(mnTot+sigTot), color='k', alpha=0.2, label='net obs')
plt.sca(axs5.flatten()[0]); plt.xlabel('Year'); plt.ylabel('Mass change (Gt)'); plt.grid()
axs5.flatten()[1].fill_between(yr, yr*(mnTot-sigTot), yr*(mnTot+sigTot), color='k', alpha=0.2, label='net obs')
plt.sca(axs5.flatten()[1]); plt.xlabel('Year'); plt.ylabel('VAF mass change (Gt)'); plt.grid()

fig6, axs6 = plt.subplots(nrow, ncol, figsize=(13, 11), num=6, squeeze=False)
fig6.suptitle(f'Ice-shelf melt rate\n{runinfo}', fontsize=9)
for p, reg in enumerate(regList):
    plt.sca(axs6.flatten()[p])
    plt.xlabel('Year'); plt.ylabel('Ice-shelf melt rate (Gt/yr)'); plt.grid()
    axs6.flatten()[p].set_title(rNames[reg])
    if p == 0:
        axX = axs6.flatten()[p]
    else:
        axs6.flatten()[p].sharex(axX)
    if rNamesOrig[reg] in ISMIP6basinInfo:
        mlt = ISMIP6basinInfo[rNamesOrig[reg]]['shelfMelt'][0]
        axs6.flatten()[p].plot(yr, np.ones(yr.shape)*(mlt), color='k', label='melt obs')
_hide_unused(axs6)

# Area-averaged sub-shelf melt rate (m/yr), comparable to Jourdain Fig 10
fig7, axs7 = plt.subplots(nrow, ncol, figsize=(13, 11), num=7, squeeze=False)
fig7.suptitle(f'Area-averaged sub-shelf melt rate\n{runinfo}', fontsize=9)
for p, reg in enumerate(regList):
    plt.sca(axs7.flatten()[p])
    plt.xlabel('Year'); plt.ylabel('Sub-shelf melt rate (m/yr)'); plt.grid()
    axs7.flatten()[p].set_title(rNames[reg])
    if p == 0:
        axX = axs7.flatten()[p]
    else:
        axs7.flatten()[p].sharex(axX)
_hide_unused(axs7)

# Unit conversion factors
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
    """Plot the selected regions (regList) from one regionalStats file. Subplot slot p <-> region index r."""
    print("Reading and plotting file: {} for experiment: {}".format(fname, display_name))

    f = Dataset(fname, 'r')
    yr = f.variables['daysSinceStart'][:] / 365.0
    dt = f.variables['deltat'][:] / (3600.0*24.0*365.0)  # in yr
    dtnR = np.tile(dt.reshape(len(dt), 1), (1, nRegions))
    nRegionsLocal = len(f.dimensions['nRegions'])
    if nRegionsLocal != nRegions:
        sys.exit(f"ERROR: Number of regions in file {fname} does not match number of regions in first input file!")

    # Fig 1: summary plot
    vol = f.variables['regionalIceVolume'][:] * volUnitFactor
    lbl = f'{display_name} total' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs1.flatten()[p].plot(yr, vol[:, r] - vol[0, r], label=lbl, color=color, linewidth=1.5)

    VAF = f.variables['regionalVolumeAboveFloatation'][:] * volUnitFactor
    VAF = VAF[:, :] - VAF[0, :]
    lbl = f'{display_name} VAF' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs1.flatten()[p].plot(yr, VAF[:, r] - VAF[0, r], label=lbl, color=color, alpha=0.7, linewidth=1.5)

    volGround = f.variables['regionalGroundedIceVolume'][:] * volUnitFactor
    volGround = volGround[:, :] - volGround[0, :]
    lbl = f'{display_name} grd' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs1.flatten()[p].plot(yr, volGround[:, r] - volGround[0, r], label=lbl, color=color, alpha=0.8, linewidth=1.5)

    volFloat = f.variables['regionalFloatingIceVolume'][:] * volUnitFactor
    volFloat = volFloat[:, :] - volFloat[0, :]
    lbl = f'{display_name} flt' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs1.flatten()[p].plot(yr, volFloat[:, r] - volFloat[0, r], label=lbl, color=color, alpha=0.6, linewidth=1.5)

    # Fig 2: Grd MB
    lbl = f'{display_name} vol chg' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs2.flatten()[p].plot(yr, volGround[:, r] - volGround[0, r], label=lbl, color=color, linewidth=1.5)

    grdSMB = f.variables['regionalSumGroundedSfcMassBal'][:] * massUnitFactor
    cumGrdSMB = np.cumsum(grdSMB*dtnR, axis=0)
    lbl = f'{display_name} SMB' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs2.flatten()[p].plot(yr, cumGrdSMB[:, r], label=lbl, color=color, alpha=0.7, linewidth=1.5)

    GLflux = f.variables['regionalSumGroundingLineFlux'][:] * massUnitFactor
    cumGLflux = np.cumsum(GLflux*dtnR, axis=0)
    lbl = f'{display_name} GL flux' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs2.flatten()[p].plot(yr, -1.0*cumGLflux[:, r], label=lbl, color=color, alpha=0.7, linewidth=1.5)

    GLMigflux = f.variables['regionalSumGroundingLineMigrationFlux'][:] * massUnitFactor
    cumGLMigflux = np.cumsum(GLMigflux*dtnR, axis=0)
    lbl = f'{display_name} GL mig flux' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs2.flatten()[p].plot(yr, -1.0*cumGLMigflux[:, r], label=lbl, color=color, alpha=0.6, linewidth=1.5)

    grdSum = grdSMB - GLflux - GLMigflux
    cumGrdSum = np.cumsum(grdSum*dtnR, axis=0)
    lbl = f'{display_name} sum' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs2.flatten()[p].plot(yr, cumGrdSum[:, r], label=lbl, color=color, alpha=0.5, linewidth=0.8)
    grdSum2 = grdSMB - GLflux
    cumGrdSum2 = np.cumsum(grdSum2*dtnR, axis=0)
    lbl = f'{display_name} sum, no GLmig' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs2.flatten()[p].plot(yr, cumGrdSum2[:, r], label=lbl, linestyle=':', color=color, alpha=0.5, linewidth=0.8)

    # Fig 3: Flt MB
    lbl = f'{display_name} vol chg' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs3.flatten()[p].plot(yr, volFloat[:, r] - volFloat[0, r], label=lbl, color=color, linewidth=1.5)

    fltSMB = f.variables['regionalSumFloatingSfcMassBal'][:] * massUnitFactor
    cumFltSMB = np.cumsum(fltSMB*dtnR, axis=0)
    lbl = f'{display_name} SMB' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs3.flatten()[p].plot(yr, cumFltSMB[:, r], label=lbl, color=color, alpha=0.7, linewidth=1.5)

    lbl = f'{display_name} GL flux' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs3.flatten()[p].plot(yr, cumGLflux[:, r], label=lbl, color=color, alpha=0.7, linewidth=1.5)

    lbl = f'{display_name} GL mig flux' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs3.flatten()[p].plot(yr, cumGLMigflux[:, r], label=lbl, color=color, alpha=0.6, linewidth=1.5)

    clv = f.variables['regionalSumCalvingFlux'][:] * massUnitFactor
    cumClv = np.cumsum(clv*dtnR, axis=0)
    lbl = f'{display_name} calving' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs3.flatten()[p].plot(yr, -1.0*cumClv[:, r], label=lbl, color=color, alpha=0.7, linewidth=1.2)

    BMB = f.variables['regionalSumFloatingBasalMassBal'][:] * massUnitFactor
    cumBMB = np.cumsum(BMB*dtnR, axis=0)
    lbl = f'{display_name} BMB' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs3.flatten()[p].plot(yr, cumBMB[:, r], label=lbl, color=color, alpha=0.8, linewidth=1.2)

    fltSum = fltSMB + GLflux + GLMigflux - clv + BMB
    cumFltSum = np.cumsum(fltSum*dtnR, axis=0)
    lbl = f'{display_name} sum' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs3.flatten()[p].plot(yr, cumFltSum[:, r], label=lbl, color=color, alpha=0.5, linewidth=0.8)
    fltSum2 = fltSMB + GLflux - clv + BMB
    cumFltSum2 = np.cumsum(fltSum2*dtnR, axis=0)
    lbl = f'{display_name} sum, no GLmig' if addToLegend else '_nolegend_'
    for p, r in enumerate(regList):
        axs3.flatten()[p].plot(yr, cumFltSum2[:, r], label=lbl, linestyle=':', color=color, alpha=0.5, linewidth=0.8)

    areaTot = f.variables['regionalIceArea'][:] / 1000.0**2
    areaGrd = f.variables['regionalGroundedIceArea'][:] / 1000.0**2
    areaFlt = f.variables['regionalFloatingIceArea'][:] / 1000.0**2
    for p, r in enumerate(regList):
        axs4.flatten()[p].plot(yr, areaTot[:, r] - areaTot[0, r], label=(f"{display_name} total area" if addToLegend else '_nolegend_'), color=color, linewidth=1.5)
        axs4.flatten()[p].plot(yr, areaGrd[:, r] - areaGrd[0, r], label=(f"{display_name} grd area" if addToLegend else '_nolegend_'), color=color, alpha=0.7, linewidth=1.5)
        axs4.flatten()[p].plot(yr, areaFlt[:, r] - areaFlt[0, r], label=(f"{display_name} flt area" if addToLegend else '_nolegend_'), color=color, alpha=0.6, linewidth=1.5)

    volGround_sel = volGround[:, regList]
    VAF_sel = VAF[:, regList]
    axs5.flatten()[0].plot(yr, volGround_sel.sum(axis=1), label=f'{display_name} total', color=color, linewidth=1.5)
    axs5.flatten()[1].plot(yr, VAF_sel.sum(axis=1), label=f'{display_name} total', color=color, linewidth=1.5)
    # only draw the "no TG/PIG" line if the Thwaites/PIG basin is actually among the selected regions
    tg_pos = next((p for p, r in enumerate(regList) if rNamesOrig[r] == 'ISMIP6BasinGH'), None)
    if tg_pos is not None:
        axs5.flatten()[0].plot(yr, np.delete(volGround_sel, tg_pos, 1).sum(axis=1), label=f'{display_name} no TG/PIG', color=color, alpha=0.7, linewidth=1.5)
        axs5.flatten()[1].plot(yr, np.delete(VAF_sel, tg_pos, 1).sum(axis=1), label=f'{display_name} no TG/PIG', color=color, alpha=0.7, linewidth=1.5)

    for p, r in enumerate(regList):
        axs6.flatten()[p].plot(yr, -BMB[:, r], label=(f"{display_name} BMB" if addToLegend else '_nolegend_'), color=color, linewidth=1.5)

    # Fig. 7:  area-averaged sub-shelf melt rate (m/yr) -- positive = melt (ice loss), no sign flip needed
    avgMelt = f.variables['regionalAvgSubshelfMelt'][:]   # m/yr, already an intensity
    for p, r in enumerate(regList):
        axs7.flatten()[p].plot(yr, avgMelt[:, r], label=(f"{display_name} melt" if addToLegend else '_nolegend_'), color=color, linewidth=1.5)

    f.close()


def _visible_axis_ylim(ax, xmin, xmax):
    """Return y-limits for the portion of plotted artists visible in [xmin, xmax]."""
    y_segments = []
    for line in ax.get_lines():
        xdata = np.asarray(line.get_xdata(orig=False), dtype=float)
        ydata = np.asarray(line.get_ydata(orig=False), dtype=float)
        if xdata.shape != ydata.shape:
            continue
        mask = (xdata >= xmin) & (xdata <= xmax) & np.isfinite(ydata)
        if np.any(mask):
            y_segments.append(ydata[mask])
    for collection in ax.collections:
        try:
            for path in collection.get_paths():
                vertices = np.asarray(path.vertices, dtype=float)
                if vertices.size == 0:
                    continue
                mask = (vertices[:, 0] >= xmin) & (vertices[:, 0] <= xmax) & np.isfinite(vertices[:, 1])
                if np.any(mask):
                    y_segments.append(vertices[:, 1][mask])
        except Exception:
            continue
    if not y_segments:
        return None
    ydata = np.concatenate(y_segments)
    ydata = ydata[np.isfinite(ydata)]
    if ydata.size == 0:
        return None
    ymin, ymax = np.min(ydata), np.max(ydata)
    pad = abs(ymin) * 0.01 + 1e-6 if ymin == ymax else 0.05 * (ymax - ymin)
    return ymin - pad, ymax + pad

# Plot each experiment
for i, (ensemble, exp, file_path, display_name) in enumerate(experiment_specs):
    color = experiment_to_color[display_name]
    add_to_legend = True if options.legend_per_experiment else (i == 0)
    plotStat(file_path, display_name, color, addToLegend=add_to_legend)

# Apply x-axis limits to all subplots if specified
if xlim_range:
    print(f"Applying X-axis limits: {xlim_range}")
    all_axes = []
    for axs in (axs1, axs2, axs3, axs4, axs5, axs6, axs7):
        all_axes.extend(axs.flatten())
    for ax in all_axes:
        ax.set_xlim(xlim_range)
        ylim_range = _visible_axis_ylim(ax, xlim_range[0], xlim_range[1])
        if ylim_range is not None:
            ax.set_ylim(ylim_range)

# Legends on the last PLOTTED subplot of each region figure (so they aren't on a hidden/empty axis)
axs1.flatten()[nPlot-1].legend(loc='best', prop={'size': 5})
axs2.flatten()[nPlot-1].legend(loc='best', prop={'size': 5})
axs3.flatten()[nPlot-1].legend(loc='best', prop={'size': 5})
axs4.flatten()[nPlot-1].legend(loc='best', prop={'size': 6})
axs5.flatten()[0].legend(loc='best', prop={'size': 6})
axs6.flatten()[nPlot-1].legend(loc='best', prop={'size': 6})
axs7.flatten()[nPlot-1].legend(loc='best', prop={'size': 6})

print("Generating plot.")
for fig in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
    fig.tight_layout()

if options.savePrefix:
    named = {'summary': fig1, 'grounded': fig2, 'floating': fig3,
             'area': fig4, 'contributions': fig5, 'meltrate': fig6, 'avgmelt': fig7}
    for name, fig in named.items():
        out = f"{options.savePrefix}_{name}.png"
        fig.savefig(out, dpi=130)
        print("wrote", out)
else:
    plt.show()
