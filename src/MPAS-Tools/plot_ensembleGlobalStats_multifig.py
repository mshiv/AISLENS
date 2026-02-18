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
import fnmatch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
parser.add_option("-k", "--drop-restarts", dest="dropRestarts", help="Comma-separated experiment name patterns to drop earlier segments before the last restart (keep final monotonic tail aligned to restart time). Use '*' as wildcard. Example: -k SSP126-TRN:SSP12601,SSP585-*", metavar="LIST", default=None)
parser.add_option("--search-all", dest="searchAll", help="Search all ensemble directories for experiments (ignores -b)", action='store_true', default=False)
parser.add_option("--list-available", dest="listAvailable", help="List all available experiments and exit", action='store_true', default=False)
parser.add_option("-v", "--variables", dest="varList", help="Comma-separated list of variables to plot (defaults to a common set)")
parser.add_option("--dry-run", dest="dryRun", help="Print detected restart adjustments instead of plotting", action='store_true', default=False)
parser.add_option("--ensemble-colors", dest="ensemble_colors",
            help=("Comma-separated list of colors for ensemble bases. Two formats are supported: "
                "1) Positional colors: 'C1,C2,...' (order matches -b). "
                "2) Mapping entries: 'ENS1=#hex,ENS2=#hex' to set colors by ensemble name. "
                "You can mix mappings and positional entries; mappings are applied first. "
                "This option may be repeated: e.g. --ensemble-colors grey --ensemble-colors black"),
            metavar="C1,C2,...|ENS=COLOR,...", action='append', default=None)

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
base_cmap = plt.get_cmap('tab20')
ensemble_base_colors = base_cmap(np.linspace(0, 1, 20))
ensemble_to_base_color = {}
for i, ensemble in enumerate(ensemble_names_unique):
    ens_up = ensemble.upper()
    if ens_up.startswith('CTRL'):
        ensemble_to_base_color[ensemble] = '#1f77b4'
    elif 'SSP126' in ens_up:
        ensemble_to_base_color[ensemble] = '#ff7f0e'
    elif 'SSP585' in ens_up:
        ensemble_to_base_color[ensemble] = '#d62728'
    else:
        ensemble_to_base_color[ensemble] = ensemble_base_colors[i % len(ensemble_base_colors)]

# If user provided explicit ensemble colors on the CLI, use them.
# Supported formats:
#  - Positional colors: "#1f77b4,#ff7f0e,..." (order matches ensemble_names_unique)
#  - Mappings: "ENSEMBLE_NAME=#hex" (can be repeated and mixed with positional entries)
if getattr(options, 'ensemble_colors', None):
    try:
        raw = options.ensemble_colors
        entries = []
        # options.ensemble_colors may be a list (if flag repeated) or a single string
        if isinstance(raw, list):
            for item in raw:
                entries.extend([e.strip() for e in item.split(',') if e.strip()])
        else:
            entries = [e.strip() for e in raw.split(',') if e.strip()]

        # First apply any mapping entries of the form NAME=COLOR
        positional = []
        mapped = set()
        for ent in entries:
            if '=' in ent:
                name, col = [p.strip() for p in ent.split('=', 1)]
                if name in ensemble_names_unique:
                    ensemble_to_base_color[name] = col
                    mapped.add(name)
                else:
                    print(f"Warning: --ensemble-colors mapping refers to unknown ensemble '{name}'", file=sys.stderr)
            else:
                positional.append(ent)

        # Assign positional entries to ensembles not set via mappings, in order
        if positional:
            unset_ensembles = [ens for ens in ensemble_names_unique if ens not in mapped]
            if len(positional) > len(unset_ensembles):
                print(f"Warning: more positional colors ({len(positional)}) provided than available ensembles ({len(unset_ensembles)}); extra colors will be ignored", file=sys.stderr)
            for ens, col in zip(unset_ensembles, positional):
                ensemble_to_base_color[ens] = col
    except Exception as e:
        print(f"Warning: failed to parse --ensemble-colors: {e}; using default colors", file=sys.stderr)

experiments_by_ensemble = {}
for ensemble, exp, file_path, display_name in experiment_specs:
    experiments_by_ensemble.setdefault(ensemble, []).append((exp, file_path, display_name))

def create_color_variations(base_color, n_variations):
    import matplotlib.colors as mcolors
    # Ensure we have an RGB triple regardless of input type (hex string, RGB(A) tuple/array)
    try:
        base_rgb = mcolors.to_rgb(base_color)
    except Exception:
        # Fallback: if base_color is an array-like, try to slice first 3 components
        try:
            base_rgb = tuple(float(x) for x in base_color[:3])
        except Exception:
            raise ValueError(f"Unsupported base_color format: {base_color}")
    hsv = mcolors.rgb_to_hsv(base_rgb)
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

# Use the same base color for all experiments within an ensemble (no per-member shading)
experiment_to_color = {}
for ensemble, experiments in experiments_by_ensemble.items():
    base_color = ensemble_to_base_color[ensemble]
    for (exp, file_path, display_name) in experiments:
        experiment_to_color[display_name] = base_color

# Parse drop-restart patterns into a list for matching
drop_patterns = None
if options.dropRestarts:
    drop_patterns = [p.strip() for p in options.dropRestarts.split(',') if p.strip()]

def read_time_and_var(fname, varname, display_name=None, drop_patterns=None):
    with Dataset(fname, 'r') as f:
        yr = f.variables['daysSinceStart'][:] / 365.0
        yr = yr - yr[0]
        data = f.variables[varname][:]
        dt = f.variables.get('deltat')
        if dt is not None:
            _ = dt[:] / 3.15e7

    # If drop_patterns is provided, check whether this display_name matches any pattern
    apply_drop = False
    if drop_patterns and display_name:
        for pat in drop_patterns:
            pat = pat.strip()
            if not pat:
                continue
            # allow either exact match or fnmatch-style wildcard
            if fnmatch.fnmatchcase(display_name, pat) or display_name == pat:
                apply_drop = True
                break

            # Also allow matching ensemble or experiment components separately.
            # display_name is in the form 'ensemble:exp' — match either side.
            if ':' in display_name:
                ens_part, exp_part = display_name.split(':', 1)
                if fnmatch.fnmatchcase(ens_part, pat) or ens_part == pat or fnmatch.fnmatchcase(exp_part, pat) or exp_part == pat:
                    apply_drop = True
                    break

    if not apply_drop:
        return yr, data

    # Smart handling: find the last non-monotonic (non-increasing) jump and replace
    # the earlier segment between the matching timestamp and the restart with the final tail,
    # aligning the first tail timestamp to the restart timestamp (no rezeroing globally).
    try:
        diffs = np.diff(yr)
        non_increasing = np.where(diffs <= 0)[0]
        if non_increasing.size == 0:
            return yr, data

        tail_start = int(non_increasing[-1]) + 1
        # restart_time is the time at which the restarted tail begins
        restart_time = float(yr[tail_start])

        # earlier part before the tail
        earlier_yr = yr[:tail_start]
        earlier_data = data[:tail_start]

        # Remove any earlier timestamps that are >= restart_time (these belong to the
        # earlier, invalid forward run). Keep only earlier times strictly before restart_time.
        mask = earlier_yr < restart_time
        prefix_yr = earlier_yr[mask]
        prefix_data = earlier_data[mask]

        tail_yr = yr[tail_start:]
        tail_data = data[tail_start:]

        # shift tail so its first time equals the last value in prefix if prefix non-empty,
        # otherwise align to restart_time (no global rezeroing)
        if prefix_yr.size > 0:
            align_time = float(prefix_yr[-1])
        else:
            align_time = restart_time
        shift = align_time - float(tail_yr[0])
        tail_yr_shifted = tail_yr + shift

        new_yr = np.concatenate([prefix_yr, tail_yr_shifted]) if prefix_yr.size > 0 else tail_yr_shifted
        new_data = np.concatenate([prefix_data, tail_data]) if prefix_data.size > 0 else tail_data

        return new_yr, new_data
    except Exception:
        return yr, data

def plot_variable(varname, ax, display_name, fname, color):
    yr, data = read_time_and_var(fname, varname, display_name=display_name, drop_patterns=drop_patterns)

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

    # Replace per-experiment legend with ensemble-level legend (one entry per ensemble)
    ensemble_handles = []
    ensemble_labels = []
    for ens in ensemble_names_unique:
        base_color = ensemble_to_base_color.get(ens)
        if base_color is None:
            continue
        n = len(experiments_by_ensemble.get(ens, []))
        label = f"{ens} (n={n})" if n > 0 else ens
        ensemble_handles.append(Line2D([0], [0], color=base_color, lw=3))
        ensemble_labels.append(label)
    if ensemble_handles:
        ax.legend(ensemble_handles, ensemble_labels, loc='best', prop={'size': 6})
    title_str = f"{varname} - Global Statistics\nEnsembles: {', '.join(sorted(ensemble_names))}\nExperiments: {', '.join(exp_names)}"
    fig.suptitle(title_str, fontsize=10)
    fig.tight_layout()

    if options.plotSave:
        safe_var = varname.replace('/', '_')
        save_name = f'globalStats_{safe_var}_{ensemble_str}_{exp_str}.png'
        fig.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")

plt.show()
