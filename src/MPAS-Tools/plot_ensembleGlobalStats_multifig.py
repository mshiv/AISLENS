#!/usr/bin/env python
"""Plot global-mean statistics for ensemble experiments (lightweight).

This script creates one figure per variable for the requested experiments.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import glob
import numpy as np
from netCDF4 import Dataset
from optparse import OptionParser
import matplotlib.pyplot as plt

rhoi = 910.0
rhosw = 1028.0

parser = OptionParser(description=__doc__)
parser.add_option("-r", "--root", dest="rootDataDir", help="Root data directory path", metavar="PATH")
parser.add_option("-b", "--base", dest="ensembleBaseDir", help="Ensemble base directory/directories (comma-separated)", metavar="DIRNAME1,DIRNAME2")
parser.add_option("-e", "--experiments", dest="experimentList", help="Experiment specifications (see help text)", metavar="EXP_SPECS")
parser.add_option("-f", "--filename", dest="statsFilename", help="Statistics filename to look for in each experiment directory", default="globalStats.nc", metavar="FILENAME")
parser.add_option("-u", dest="units", help="units for mass/volume: m3, kg, Gt", default="Gt", metavar="UNITS")
parser.add_option("-c", dest="plotChange", help="plot time series as absolute change from initial", action='store_true', default=False)
parser.add_option("-p", dest="plotPercentChange", help="plot time series as percentage change from initial", action='store_true', default=False)
parser.add_option("-s", dest="plotSave", help="save figure (any value makes saving enabled)")
parser.add_option("-x", "--xlim", dest="xlimits", help="X-axis limits as comma-separated values (e.g., '0,25')", metavar="MIN,MAX")
parser.add_option("--search-all", dest="searchAll", help="Search all ensemble directories for experiments (ignores -b)", action='store_true', default=False)
parser.add_option("--list-available", dest="listAvailable", help="List all available experiments and exit", action='store_true', default=False)
parser.add_option("-v", "--variables", dest="varList", help="Comma-separated list of variables to plot (defaults to a common set)")

options, args = parser.parse_args()

if options.plotChange and options.plotPercentChange:
    sys.exit("ERROR: Cannot use both -c (absolute change) and -p (percentage change) options simultaneously")

if options.varList:
    vars_to_plot = [v.strip() for v in options.varList.split(',')]
else:
    vars_to_plot = [
        'totalIceVolume',
        'volumeAboveFloatation',
        'groundedIceVolume',
        'floatingIceVolume',
        'groundedIceArea',
        'floatingIceArea',
        'groundingLineFlux',
        'totalCalvingFlux',
        'totalFloatingBasalMassBal'
    ]

def find_all_experiments(root_dir, ensemble_dirs, stats_filename):
    available_experiments = {}
    for ensemble_dir in ensemble_dirs:
        full_ensemble_path = os.path.join(root_dir, ensemble_dir) if root_dir else ensemble_dir
        if not os.path.exists(full_ensemble_path):
            continue
        for item in os.listdir(full_ensemble_path):
            exp_path = os.path.join(full_ensemble_path, item)
            if os.path.isdir(exp_path):
                stats_file = os.path.join(exp_path, stats_filename)
                if os.path.exists(stats_file):
                    available_experiments.setdefault(ensemble_dir, []).append(item)
    return available_experiments

def parse_experiment_specifications(experiment_list, ensemble_dirs, root_dir, stats_filename):
    if not experiment_list:
        sys.exit("ERROR: Must specify experiment list with -e/--experiments option")
    experiment_specs = []
    exp_parts = [exp.strip() for exp in experiment_list.split(',')]
    for exp_spec in exp_parts:
        if ':' in exp_spec:
            ensemble_name, exp_name = [p.strip() for p in exp_spec.split(':', 1)]
            if ensemble_name not in ensemble_dirs:
                continue
            if '*' in exp_name or '?' in exp_name:
                search_path = os.path.join(root_dir, ensemble_name) if root_dir else ensemble_name
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
                exp_path = os.path.join(root_dir, ensemble_name, exp_name) if root_dir else os.path.join(ensemble_name, exp_name)
                stats_file = os.path.join(exp_path, stats_filename)
                if os.path.exists(stats_file):
                    display_name = f"{ensemble_name}:{exp_name}"
                    experiment_specs.append((ensemble_name, exp_name, stats_file, display_name))
        else:
            exp_name = exp_spec.strip()
            if '*' in exp_name or '?' in exp_name:
                for ensemble_dir in ensemble_dirs:
                    search_path = os.path.join(root_dir, ensemble_dir) if root_dir else ensemble_dir
                    if os.path.exists(search_path):
                        matching_dirs = glob.glob(os.path.join(search_path, exp_name))
                        for match_path in matching_dirs:
                            if os.path.isdir(match_path):
                                match_exp = os.path.basename(match_path)
                                stats_file = os.path.join(match_path, stats_filename)
                                if os.path.exists(stats_file):
                                    display_name = f"{ensemble_dir}:{match_exp}"
                                    experiment_specs.append((ensemble_dir, match_exp, stats_file, display_name))
            else:
                for ensemble_dir in ensemble_dirs:
                    exp_path = os.path.join(root_dir, ensemble_dir, exp_name) if root_dir else os.path.join(ensemble_dir, exp_name)
                    stats_file = os.path.join(exp_path, stats_filename)
                    if os.path.exists(stats_file):
                        display_name = f"{ensemble_dir}:{exp_name}"
                        experiment_specs.append((ensemble_dir, exp_name, stats_file, display_name))
    return experiment_specs

# Parse ensemble directories
ensemble_dirs = []
if options.searchAll:
    if not options.rootDataDir:
        sys.exit("ERROR: --search-all requires --root to be specified")
    for item in os.listdir(options.rootDataDir):
        if os.path.isdir(os.path.join(options.rootDataDir, item)):
            ensemble_dirs.append(item)
elif options.ensembleBaseDir:
    ensemble_dirs = [ens.strip() for ens in options.ensembleBaseDir.split(',')]
else:
    sys.exit("ERROR: Must specify ensemble directories with -b/--base or use --search-all")

if options.listAvailable:
    available = find_all_experiments(options.rootDataDir, ensemble_dirs, options.statsFilename)
    for ensemble, experiments in available.items():
        for exp in sorted(experiments):
            print(f"{ensemble}:{exp}")
    sys.exit(0)

if not options.experimentList:
    # No -e provided: auto-discover all experiments under each ensemble base
    if not options.rootDataDir:
        sys.exit("ERROR: --root must be provided when auto-discovering experiments")
    available = find_all_experiments(options.rootDataDir, ensemble_dirs, options.statsFilename)
    experiment_specs = []
    for ensemble, exps in available.items():
        for exp in sorted(exps):
            exp_path = os.path.join(options.rootDataDir, ensemble, exp)
            stats_file = os.path.join(exp_path, options.statsFilename)
            display_name = f"{ensemble}:{exp}"
            experiment_specs.append((ensemble, exp, stats_file, display_name))
    if not experiment_specs:
        sys.exit("ERROR: No experiments found under the provided root/base directories")
else:
    experiment_specs = parse_experiment_specifications(options.experimentList, ensemble_dirs, options.rootDataDir, options.statsFilename)
    if not experiment_specs:
        sys.exit("ERROR: No valid experiments found")

# Units scaling
if options.units == "m3":
    massUnit = "m$^3$"
    scaleVol = 1.0
elif options.units == "kg":
    massUnit = "kg"
    scaleVol = 1.0 / rhoi
elif options.units == "Gt":
    massUnit = "Gt"
    scaleVol = 1.0e12 / rhoi
else:
    sys.exit("Unknown mass/volume units")

def VAF2seaLevel(vol):
    return vol * scaleVol / 3.62e14 * rhoi / rhosw * 1000.0

def seaLevel2VAF(vol):
    return vol / scaleVol * 3.62e14 * rhosw / rhoi / 1000.0

def addSeaLevAx(ax):
    seaLevAx = ax.secondary_yaxis('right', functions=(VAF2seaLevel, seaLevel2VAF))
    seaLevAx.set_ylabel('Sea-level\nequivalent (mm)')

# Color mapping per ensemble: assign one base color per ensemble (in request order)
# and create hue/lightness variations for multiple experiments within the same ensemble.
# Build ordered list of unique ensemble names in the order they appear in experiment_specs.
ensemble_names_unique = []
for ensemble, _, _, _ in experiment_specs:
    if ensemble not in ensemble_names_unique:
        ensemble_names_unique.append(ensemble)
# Fallback to provided ensemble_dirs order if nothing found
if not ensemble_names_unique:
    ensemble_names_unique = list(ensemble_dirs)

# Use a larger categorical colormap for distinct base colors
base_cmap = plt.cm.get_cmap('tab20')
ensemble_base_colors = base_cmap(np.linspace(0, 1, 20))
ensemble_to_base_color = {}
for i, ensemble in enumerate(ensemble_names_unique):
    ensemble_to_base_color[ensemble] = ensemble_base_colors[i % len(ensemble_base_colors)]

experiments_by_ensemble = {}
for ensemble, exp, file_path, display_name in experiment_specs:
    experiments_by_ensemble.setdefault(ensemble, []).append((exp, file_path, display_name))

def create_color_variations(base_color, n_variations):
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

experiment_to_color = {}
for ensemble, experiments in experiments_by_ensemble.items():
    base_color = ensemble_to_base_color[ensemble]
    n_experiments = len(experiments)
    color_variations = create_color_variations(base_color, n_experiments)
    for i, (exp, file_path, display_name) in enumerate(experiments):
        experiment_to_color[display_name] = color_variations[i]

def read_time_and_var(fname, varname):
    with Dataset(fname, 'r') as f:
        yr = f.variables['daysSinceStart'][:] / 365.0
        yr = yr - yr[0]
        data = f.variables[varname][:]
        dt = f.variables.get('deltat')
        if dt is not None:
            _ = dt[:] / 3.15e7
    return yr, data

def plot_variable(varname, ax, display_name, fname, color):
    yr, data = read_time_and_var(fname, varname)

    plot_data = data
    # Apply unit conversions for specific variables
    if varname in ['totalIceVolume', 'volumeAboveFloatation', 'groundedIceVolume', 'floatingIceVolume']:
        plot_data = plot_data / scaleVol
    elif varname in ['groundedIceArea', 'floatingIceArea']:
        plot_data = plot_data / 1000.0**2
    elif varname == 'totalFloatingBasalMassBal':
        plot_data = plot_data / 1e12

    if options.plotChange:
        plot_data = plot_data - plot_data[0]
    elif options.plotPercentChange:
        plot_data = (plot_data - plot_data[0]) * 100.0 / plot_data[0]

    ax.plot(yr, plot_data, label=display_name, color=color, linewidth=1.5)
    return yr, plot_data

# Parse x-axis limits
xlim_range = None
if options.xlimits:
    try:
        xlim_values = [float(x.strip()) for x in options.xlimits.split(',')]
        if len(xlim_values) != 2:
            sys.exit("ERROR: X-axis limits must be exactly two comma-separated values (e.g., '0,25')")
        if xlim_values[0] >= xlim_values[1]:
            sys.exit("ERROR: X-axis minimum must be less than maximum")
        xlim_range = xlim_values
    except ValueError:
        sys.exit("ERROR: X-axis limits must be numeric values separated by comma (e.g., '0,25')")

# Precompute ensemble/experiment name lists used for filenames
ensemble_names = list(set([ensemble for ensemble, _, _, _ in experiment_specs]))
exp_names = [display_name for _, _, _, display_name in experiment_specs]
ensemble_str = "-".join(sorted(ensemble_names))
exp_str = "-".join([name.replace(':', '_') for name in exp_names])

# Loop over variables and create one figure per variable
for varname in vars_to_plot:
    fig = plt.figure(figsize=(9, 4), facecolor='w')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Year')
    # set ylabel according to variable
    if varname in ['totalIceVolume', 'volumeAboveFloatation', 'groundedIceVolume', 'floatingIceVolume']:
        ylabel = f"{varname}{' change' if options.plotChange or options.plotPercentChange else ''} ({massUnit})"
    elif varname in ['groundedIceArea', 'floatingIceArea']:
        ylabel = f"{varname}{' change' if options.plotChange or options.plotPercentChange else ''} (km$^2$)"
    elif varname == 'totalFloatingBasalMassBal':
        ylabel = f"{varname}{' change' if options.plotChange or options.plotPercentChange else ''} (Gt/yr)"
    else:
        ylabel = varname
    ax.set_ylabel(ylabel)
    ax.grid()

    plotted_series = []
    for ensemble, exp, file_path, display_name in experiment_specs:
        color = experiment_to_color[display_name]
        res = plot_variable(varname, ax, display_name, file_path, color)
        if res is not None:
            plotted_series.append(res)

    if varname == 'volumeAboveFloatation':
        addSeaLevAx(ax)

    if xlim_range:
        ax.set_xlim(xlim_range)
        # compute y-limits based only on data inside the x-range
        try:
            xmin, xmax = xlim_range
            y_vals = []
            for yr, pdata in plotted_series:
                try:
                    yr_arr = np.asarray(yr)
                    mask = (yr_arr >= xmin) & (yr_arr <= xmax)
                    if np.any(mask):
                        vals = np.asarray(pdata)[mask]
                        # filter NaNs
                        vals = vals[~np.isnan(vals)]
                        if vals.size:
                            y_vals.append(vals)
                except Exception:
                    continue
            if y_vals:
                concat = np.concatenate(y_vals)
                if concat.size:
                    ymin = float(concat.min())
                    ymax = float(concat.max())
                    if np.isfinite(ymin) and np.isfinite(ymax):
                        if ymax == ymin:
                            # expand a little when constant
                            pad = abs(ymin) * 0.01 if ymin != 0 else 1.0
                        else:
                            pad = 0.03 * (ymax - ymin)
                        ax.set_ylim(ymin - pad, ymax + pad)
        except Exception:
            # fallback: let matplotlib autoscale if anything goes wrong
            pass

    ax.legend(loc='best', prop={'size': 6})
    title_str = f"{varname} - Global Statistics\nEnsembles: {', '.join(sorted(ensemble_names))}\nExperiments: {', '.join(exp_names)}"
    fig.suptitle(title_str, fontsize=10)
    fig.tight_layout()

    if options.plotSave:
        safe_var = varname.replace('/', '_')
        save_name = f'globalStats_{safe_var}_{ensemble_str}_{exp_str}.png'
        fig.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")

plt.show()
