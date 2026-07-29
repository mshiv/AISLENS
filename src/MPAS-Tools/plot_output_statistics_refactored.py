#!/usr/bin/env python
'''
Plot probability distribution functions of ice sheet model output variables 
across ensemble experiments at specified time steps.

Author: Shiva Muruganandham (refactored)
Date: December 2025
'''

import sys
import os
import numpy as np
import glob
from netCDF4 import Dataset
from optparse import OptionParser
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch

# Increase default font sizes for readability across figures
import matplotlib as mpl
mpl.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Constants
RHOI = 910.0  # Ice density (kg/m³)
RHOSW = 1028.0  # Seawater density (kg/m³)
MIN_EXPERIMENTS_THRESHOLD = 3  # Minimum experiments needed for valid PDF


def _vaf_to_sealevel_mm_factory(scale_vol):
    """Return pair of functions (vaf->mm, mm->vaf) for secondary axis.

    scale_vol: multiplier converting plotted units back to m^3.
    """
    def VAF2seaLevel(vol):
        # Convert plotted VAF to m^3, then to sea-level mm using 3.62e14 m^3/mm SLE
        try:
            vol_m3 = vol * scale_vol
            sle_mm = vol_m3 / 3.62e14 * (RHOI / RHOSW) * 1000.0
            return sle_mm
        except Exception:
            return np.nan

    def seaLevel2VAF(mm):
        try:
            vol_m3 = (mm / 1000.0) * 3.62e14 * (RHOSW / RHOI)
            return vol_m3 / scale_vol
        except Exception:
            return np.nan

    return VAF2seaLevel, seaLevel2VAF


def addSeaLevAx(ax, scale_vol):
    """Attach a secondary y-axis showing sea-level equivalent (mm)."""
    v2s, s2v = _vaf_to_sealevel_mm_factory(scale_vol)
    seaLevAx = ax.secondary_yaxis('right', functions=(v2s, s2v))
    seaLevAx.set_ylabel('Sea-level\nequivalent (mm)')



def parse_arguments():
    """Parse command-line arguments."""
    parser = OptionParser(description=__doc__)
    parser.add_option("-r", "--root", dest="rootDataDir", 
                      help="Root data directory path", metavar="PATH")
    parser.add_option("-b", "--base", dest="ensembleBaseDir", 
                      help="Ensemble base directory (relative to root)", metavar="DIRNAME")
    parser.add_option("-e", "--experiments", dest="experimentList", 
                      help="Comma-separated list of experiment run names", metavar="EXP1,EXP2")
    parser.add_option("-f", "--filename", dest="statsFilename", 
                      help="Statistics filename", default="globalStats.nc", metavar="FILENAME")
    parser.add_option("-u", dest="units", 
                      help="units for mass/volume: m3, kg, Gt", default="Gt", metavar="UNITS")
    parser.add_option("-v", "--variable", dest="variable", 
                      help="Variable to analyze", default="volumeAboveFloatation", metavar="VARNAME")
    parser.add_option("-t", "--timesteps", dest="timeSteps", 
                      help="Comma-separated time steps (years) for PDFs", 
                      default="0,50,100,200,300", metavar="T1,T2")
    parser.add_option("-c", dest="plotChange", 
                      help="Plot absolute change from initial", action='store_true', default=False)
    parser.add_option("-p", dest="plotPercentChange", 
                      help="Plot percentage change from initial", action='store_true', default=False)
    parser.add_option("-s", dest="plotSave", 
                      help="Save figure to file", metavar="FILENAME")
    # Output mode options: mutually exclusive; default will be mass-hist (probabilities)
    parser.add_option("--mass-hist", dest="mass_hist", help="Plot probability mass per bin evolution (default)", action='store_true', default=False)
    parser.add_option("--cdf", dest="cdf", help="Plot cumulative distribution function (CDF) evolution", action='store_true', default=False)
    parser.add_option("--density", dest="density", help="Plot probability density evolution (KDE) (density = default off)", action='store_true', default=False)
    parser.add_option("--heatmap", dest="heatmap", help="Plot distribution evolution as heatmap (x=time, y=variable bins)", action='store_true', default=False)
    parser.add_option("--percentiles", dest="percentiles", help="Plot percentiles over time (median and IQR by default)", action='store_true', default=False)
    parser.add_option("--no-bars", dest="no_bars", help="For mass-hist mode: do not draw bars, only show KDE-derived smooth curves", action='store_true', default=False)
    parser.add_option("--anomaly", dest="anomaly", help="Plot anomalies relative to the ensemble mean (subtract mean at each time)", action='store_true', default=False)
    parser.add_option("--anomaly-ref", dest="anomaly_ref", help="Compute anomalies relative to a specific experiment name (must be one of the names passed to -e)", metavar="EXP_NAME", default=None)
    parser.add_option("--time-series", dest="time_series", help="Produce time-series plot using all timestamps from the original files", action='store_true', default=False)
    parser.add_option("-x", "--xlim", dest="xlimits", help="X-axis limits as comma-separated values (e.g., '0,25')", metavar="MIN,MAX")
    parser.add_option("--std-normalize", dest="std_normalize",
                      help="Normalize standard-deviation panel: 'absolute' (default), 'cv' (σ/|mean|), or 'percent' (σ/|mean|*100)",
                      default='absolute', metavar="MODE")
    parser.add_option("--ensemble-colors", dest="ensemble_colors",
                      help="Comma-separated list of colors for ensemble bases (order matches -b)."
                      " Example: '#2ca02c,#1f77b4,#d62728'", metavar="C1,C2,...", default=None)
    
    options, args = parser.parse_args()
    
    # Validation
    if options.plotChange and options.plotPercentChange:
        sys.exit("ERROR: Cannot use both -c and -p options simultaneously")
    # Allow omitting -e/--experiments when --time-series is used (auto-discover experiments)
    if not options.experimentList and not options.time_series:
        sys.exit("ERROR: Must specify experiment list with -e/--experiments (or use --time-series to auto-discover)")
    if not options.ensembleBaseDir:
        sys.exit("ERROR: Must specify ensemble base directory with -b/--base")

    # Validate mutually-exclusive mode flags; default to mass-hist if none set
    modes = [options.mass_hist, options.cdf, options.density, options.heatmap, options.percentiles]
    if sum(bool(x) for x in modes) > 1:
        sys.exit("ERROR: Choose only one of --mass-hist, --cdf, --density, --heatmap, or --percentiles")
    if not any(modes):
        options.mass_hist = True
    
    return options



def get_scale_factor(units):
    """Get volume/mass scaling factor for specified units."""
    if units == "m3":
        return 1.0, "m$^3$"
    elif units == "kg":
        return 1.0/RHOI, "kg"
    elif units == "Gt":
        return 1.0e12 / RHOI, "Gt"
    else:
        sys.exit(f"ERROR: Unknown units '{units}'")

def get_unit_string(variable, base_unit, change_mode):
    var_lower = variable.lower()
    
    if 'volume' in var_lower or 'vaf' in var_lower:
        unit_base = base_unit
    elif 'area' in var_lower:
        unit_base = "km²"
    elif 'flux' in var_lower and base_unit == "Gt":
        unit_base = "Gt/yr"
    else:
        unit_base = ""

    if var_lower == 'dhdt' or 'dhdt' in var_lower:
        if change_mode == 'absolute':
            return " change (m/yr)"
        elif change_mode == 'percent':
            return " change (%)"
        else:
            return " (m/yr)"

    if change_mode == 'absolute':
        return f" change ({unit_base})" if unit_base else " change"
    elif change_mode == 'percent':
        return " change (%)"
    else:
        return f" ({unit_base})" if unit_base else ""



def extract_variable_data(fname, variable, scale_vol, change_mode=None):
    """Extract and scale variable time series from a single experiment file."""
    with Dataset(fname, 'r') as f:
        if 'daysSinceStart' not in f.variables:
            raise ValueError(f"Missing 'daysSinceStart' in {fname}; cannot compute times")
        yr = f.variables['daysSinceStart'][:] / 365.0
        yr = yr - yr[0]

        if variable in f.variables:
            var_data = f.variables[variable][:]
        else:
            # Special-case: allow computing dhdt from iceThicknessMean when
            # the requested 'dhdt' variable is not present in the stats file.
            if variable == 'dhdt' and 'iceThicknessMean' in f.variables:
                thickness = np.asarray(f.variables['iceThicknessMean'][:])
                if thickness.ndim == 1 and thickness.shape[0] == yr.shape[0]:
                    thickness_1d = thickness
                else:
                    if thickness.shape[0] == yr.shape[0]:
                        thickness_1d = np.nanmean(thickness.reshape(thickness.shape[0], -1), axis=1)
                    elif thickness.shape[-1] == yr.shape[0]:
                        newshape = (int(np.prod(thickness.shape[:-1])), thickness.shape[-1])
                        thickness_1d = np.nanmean(thickness.reshape(newshape), axis=0)
                    elif thickness.size == yr.shape[0]:
                        thickness_1d = thickness.ravel()
                    else:
                        raise ValueError(f"Cannot interpret shape {thickness.shape} of 'iceThicknessMean' in {fname}")

                delta = thickness_1d - thickness_1d[0]
                years = np.asarray(yr, dtype=float)
                years_safe = years.copy()
                if years_safe.size > 0:
                    years_safe[0] = np.nan
                dhdt = delta / years_safe
                if dhdt.size > 0:
                    dhdt = dhdt.astype(float)
                    dhdt[0] = 0.0
                var_data = dhdt
            else:
                raise ValueError(f"Variable '{variable}' not found in {fname}")
    
    var_lower = variable.lower()
    if 'volume' in var_lower or 'vaf' in var_lower:
        var_data = var_data / scale_vol
    elif 'area' in var_lower:
        var_data = var_data / 1e6
    elif 'flux' in var_lower and scale_vol > 1e10:
        var_data = var_data / 1e12

    if change_mode == 'absolute':
        var_data = var_data - var_data[0]
    elif change_mode == 'percent':
        var_data = (var_data - var_data[0]) * 100 / var_data[0]
    
    return yr, var_data

def interpolate_ensemble_data(experiment_files, experiment_names, variable, 
                               scale_vol, target_times, change_mode=None):
    """Extract and interpolate variable data from all experiments to target times."""
    all_var_data = {}
    min_experiments = max(MIN_EXPERIMENTS_THRESHOLD, int(0.5 * len(experiment_files)))
    
    for exp_file, exp_name in zip(experiment_files, experiment_names):
        try:
            years, var_data = extract_variable_data(exp_file, variable, scale_vol, change_mode)
        except Exception as e:
            print(f"Warning: Failed to extract data from {exp_name}: {e}", file=sys.stderr)
            continue
        
        valid_times = [t for t in target_times if years.min() <= t <= years.max()]
        if not valid_times:
            continue
        
        interp_func = interp1d(years, var_data, kind='linear', 
                               bounds_error=False, fill_value=np.nan)
        interp_var = interp_func(valid_times)
        
        for time_step, value in zip(valid_times, interp_var):
            if time_step not in all_var_data:
                all_var_data[time_step] = []
            if not np.isnan(value):
                all_var_data[time_step].append(value)
    
    valid_time_steps = [ts for ts in sorted(all_var_data.keys()) 
                        if len(all_var_data[ts]) >= min_experiments]
    
    return all_var_data, valid_time_steps



def calculate_distribution_stats(values):
    """Calculate comprehensive statistics for a distribution."""
    return {
        'n': len(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'median': np.median(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75),
        'skewness': stats.skew(values),
        'kurtosis': stats.kurtosis(values)
    }



def plot_pdf_grid(all_var_data, valid_time_steps, variable, unit_str, num_experiments):
    """Create grid of PDF plots for each time step."""
    n_plots = len(valid_time_steps)
    ncols = min(3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), facecolor='w')
    axes = np.array([axes]).flatten() if n_plots == 1 else axes.flatten()
    
    for i, time_step in enumerate(valid_time_steps):
        ax = axes[i]
        var_values = np.array(all_var_data[time_step])
        dist_stats = calculate_distribution_stats(var_values)
        
        ax.hist(var_values, bins='auto', alpha=0.6, density=True, 
                color='skyblue', edgecolor='black', label='Data')
        
        if dist_stats['std'] > 1e-10 and len(var_values) >= 3:
            kde = gaussian_kde(var_values)
            x_range = np.linspace(var_values.min(), var_values.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='KDE')
        
        ax.set_xlabel(f'{variable}{unit_str}')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Year {time_step:.0f} (N={dist_stats["n"]})\n'
                     f'μ={dist_stats["mean"]:.2f}, σ={dist_stats["std"]:.2f}, '
                     f'skew={dist_stats["skewness"]:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(f"{variable} Probability Distributions ({num_experiments} Experiments)", 
                 fontsize=14)
    plt.tight_layout()
    
    return fig

def plot_skewness_evolution(all_var_data, valid_time_steps, variable):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    time_points = []
    skewness_values = []
    
    for time_step in valid_time_steps:
        var_values = np.array(all_var_data[time_step])
        skewness_values.append(stats.skew(var_values))
        time_points.append(time_step)
    
    ax.plot(time_points, skewness_values, 'o-', linewidth=2, markersize=8, color='purple')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Symmetric')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Skewness')
    ax.set_title(f'{variable} Distribution Skewness Evolution\n'
                 'Positive: right-tailed, Negative: left-tailed')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    return fig

def plot_kde_evolution(all_var_data, valid_time_steps, variable, unit_str, num_experiments):
    """Plot smooth KDE-based PDFs for all time steps on one plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Determine overall data range
    all_values = np.concatenate([np.array(all_var_data[ts]) for ts in valid_time_steps])
    data_range = all_values.max() - all_values.min()
    x_range = np.linspace(all_values.min() - 0.1*data_range, 
                          all_values.max() + 0.1*data_range, 300)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_time_steps)))
    
    for i, time_step in enumerate(valid_time_steps):
        var_values = np.array(all_var_data[time_step])
        
        if len(var_values) >= 3 and np.std(var_values) > 1e-10:
            kde = gaussian_kde(var_values)
            kde_pdf = kde(x_range)
            ax.plot(x_range, kde_pdf, color=colors[i], linewidth=2.5, 
                    label=f'Year {time_step:.0f}')
            
            ax.axvline(np.mean(var_values), color=colors[i], 
                       linestyle='--', alpha=0.4, linewidth=1)
    
    ax.set_xlabel(f'{variable}{unit_str}')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{variable} PDF Evolution Over Time ({num_experiments} experiments)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_heatmap_evolution(all_var_data, valid_time_steps, variable, unit_str, num_experiments):
    """Plot distribution evolution as a heatmap (x=time, y=variable bins, color=probability mass)."""
    time_steps = sorted(valid_time_steps)

    all_values = np.concatenate([np.array(all_var_data[ts]) for ts in time_steps])
    bins = np.histogram_bin_edges(all_values, bins='auto')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    data_min, data_max = all_values.min(), all_values.max()
    data_range = data_max - data_min if data_max > data_min else 1.0
    x_grid = np.linspace(data_min - 0.1*data_range, data_max + 0.1*data_range, 2000)

    prob_matrix = np.zeros((len(bin_centers), len(time_steps)))

    for i, ts in enumerate(time_steps):
        vals = np.array(all_var_data[ts])
        if len(vals) == 0:
            continue
        if len(vals) >= 3 and np.std(vals) > 1e-12:
            kde = gaussian_kde(vals)
            kde_vals = kde(x_grid)
            for j in range(len(bins)-1):
                mask = (x_grid >= bins[j]) & (x_grid < bins[j+1])
                if mask.any():
                    prob_matrix[j, i] = np.trapz(kde_vals[mask], x_grid[mask])
        else:
            h, _ = np.histogram(vals, bins=bins)
            if h.sum() > 0:
                prob_matrix[:, i] = h / h.sum()

    col_sums = prob_matrix.sum(axis=0)
    nonzero = col_sums > 0
    prob_matrix[:, nonzero] = prob_matrix[:, nonzero] / col_sums[nonzero]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # imshow expects [rows, cols] with origin='lower' to have small bin at bottom
    extent = [min(time_steps) - 0.5*(time_steps[1]-time_steps[0]) if len(time_steps)>1 else min(time_steps)-0.5,
              max(time_steps) + 0.5*(time_steps[1]-time_steps[0]) if len(time_steps)>1 else max(time_steps)+0.5,
              bins[0], bins[-1]]

    im = ax.imshow(prob_matrix, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Probability mass')

    ax.set_xlabel('Time (years)')
    ax.set_ylabel(f'{variable}{unit_str}')
    ax.set_title(f'{variable} Distribution Heatmap Over Time ({num_experiments} experiments)')
    ax.set_xticks(time_steps)
    ax.grid(False)
    plt.tight_layout()
    return fig


def plot_percentiles_evolution(all_var_data, valid_time_steps, variable, unit_str, num_experiments):
    time_steps = sorted(valid_time_steps)
    medians = []
    p25 = []
    p75 = []

    for ts in time_steps:
        vals = np.array(all_var_data[ts])
        if len(vals) == 0:
            medians.append(np.nan)
            p25.append(np.nan)
            p75.append(np.nan)
            continue
        medians.append(np.median(vals))
        p25.append(np.percentile(vals, 25))
        p75.append(np.percentile(vals, 75))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_steps, medians, '-o', color='black', label='Median')
    ax.fill_between(time_steps, p25, p75, color='gray', alpha=0.3, label='IQR (25-75%)')

    ax.set_xlabel('Time (years)')
    ax.set_ylabel(f'{variable}{unit_str}')
    ax.set_title(f'{variable} Percentiles Over Time ({num_experiments} experiments)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def build_time_series_matrix(experiment_files, experiment_names, variable, scale_vol, change_mode):
    """Build a common time grid and interpolate all experiments to it."""
    series_list = []
    time_arrays = []

    for fpath, name in zip(experiment_files, experiment_names):
        try:
            years, var_data = extract_variable_data(fpath, variable, scale_vol, change_mode)
            years = np.asarray(years)
            if np.ma.isMaskedArray(years):
                years = np.ma.filled(years, np.nan)
            var_data = np.asarray(var_data)
            if np.ma.isMaskedArray(var_data):
                var_data = np.ma.filled(var_data, np.nan)

            time_arrays.append(years)
            series_list.append(var_data)
        except Exception as e:
            print(f"Warning: failed to read timeseries for {name}: {e}", file=sys.stderr)
            time_arrays.append(np.array([]))
            series_list.append(np.array([]))

    if len(time_arrays) == 0:
        return np.array([]), np.empty((0, 0))
    all_times_concat = np.concatenate([t for t in time_arrays if len(t) > 0]) if any(len(t)>0 for t in time_arrays) else np.array([])
    if all_times_concat.size == 0:
        return np.array([]), np.empty((len(series_list), 0))

    time_grid = np.unique(np.sort(all_times_concat))

    data_matrix = np.full((len(series_list), len(time_grid)), np.nan)
    for i, (t, s) in enumerate(zip(time_arrays, series_list)):
        if len(t) == 0 or len(s) == 0:
            continue
        if len(t) == 1:
            mask = np.isclose(time_grid, float(t[0]))
            if mask.any():
                data_matrix[i, mask] = float(s[0])
            continue

        try:
            interp = interp1d(t, s, kind='linear', bounds_error=False, fill_value=np.nan)
            data_matrix[i, :] = interp(time_grid)
        except Exception as e:
            print(f"Warning: interpolation failed for {experiment_names[i]}: {e}", file=sys.stderr)
            # leave row as NaNs in case of failure
            continue

    return time_grid, data_matrix


def plot_time_series(time_grid, data_matrix, experiment_names, variable, unit_str, mode='raw', ref_idx=None, scale_vol=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    n_exp = data_matrix.shape[0]
    ensemble_map = {}
    for idx, dname in enumerate(experiment_names):
        ens = dname.split(':', 1)[0]
        ensemble_map.setdefault(ens, []).append(idx)

    try:
        palette = globals().get('ensemble_to_base_color', {})
    except Exception:
        palette = {}

    legend_handles = []
    legend_labels = []

    for ens, indices in ensemble_map.items():
        subset = data_matrix[indices, :]
        with np.errstate(invalid='ignore'):
            if mode == 'anomaly':
                mean = np.nanmean(subset, axis=0)
                subset = subset - mean[np.newaxis, :]
            if subset.size == 0:
                continue
            p5 = np.nanpercentile(subset, 5, axis=0)
            p95 = np.nanpercentile(subset, 95, axis=0)
            p25 = np.nanpercentile(subset, 25, axis=0)
            p75 = np.nanpercentile(subset, 75, axis=0)
            mean_for_plot = np.nanmean(subset, axis=0)

        base_color = palette.get(ens, None)
        if base_color is None:
            base_color = plt.cm.tab10(len(legend_handles) % 10)


        pmin = np.nanmin(subset, axis=0)
        pmax = np.nanmax(subset, axis=0)
        ax.fill_between(time_grid, pmin, pmax, color=base_color, alpha=0.08)
        ax.fill_between(time_grid, p25, p75, color=base_color, alpha=0.2)

        ax.plot(time_grid, mean_for_plot, color=base_color, linewidth=1.5, linestyle='--')

        from matplotlib.lines import Line2D
        legend_handles.append(Line2D([0], [0], color=base_color, lw=3))
        legend_labels.append(f"{ens} (n={len(indices)})")

    if legend_handles:
        try:
            band_iqr = Patch(facecolor='gray', alpha=0.2, label='IQR (25%-75%)')
            band_range = Patch(facecolor='gray', alpha=0.08, label='5%-95% range')
            legend_handles_ext = list(legend_handles) + [band_iqr, band_range]
            legend_labels_ext = list(legend_labels) + [band_iqr.get_label(), band_range.get_label()]
            ax.legend(legend_handles_ext, legend_labels_ext, loc='best')
        except Exception:
            ax.legend(legend_handles, legend_labels, loc='best')

    if mode in ('anomaly', 'ref'):
        try:
            y_low = np.nanmin(p25)
            y_high = np.nanmax(p75)
            if np.isnan(y_low) or np.isnan(y_high) or (y_high - y_low) == 0:
                y_low = np.nanmin(data_matrix)
                y_high = np.nanmax(data_matrix)
            span = y_high - y_low
            if span == 0 or np.isclose(span, 0.0):
                pad = max(1.0, abs(y_high) * 0.01)
            else:
                pad = span * 0.1
            ax.set_ylim(y_low - pad, y_high + pad)
        except Exception:
            pass

    ax.set_xlabel('Time (years)')
    ax.set_ylabel(f'{variable}{unit_str}')
    try:
        if variable is not None and 'volumeabovefloatation' in variable.lower() and scale_vol is not None:
            addSeaLevAx(ax, scale_vol)
    except Exception:
        pass
    ax.set_title(f'{variable} Time Series ({n_exp} experiments)')
    ax.grid(True, alpha=0.3)
    try:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((3, 3))
        fmt.set_scientific(True)
        ax.yaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
    except Exception:
        pass
    plt.tight_layout()
    return fig


def plot_spread_ratio_time_series(time_grid, data_matrix, experiment_names, variable, unit_str):
    """Plot time series of (ensemble range) / (ensemble mean) per ensemble."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # Build ensemble -> member indices mapping from display names
    ensemble_map = {}
    for idx, dname in enumerate(experiment_names):
        ens = dname.split(':', 1)[0]
        ensemble_map.setdefault(ens, []).append(idx)

    # color mapping (caller sets global `ensemble_to_base_color`)
    palette = globals().get('ensemble_to_base_color', {})

    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []

    tiny = 1e-12
    for ens, indices in ensemble_map.items():
        subset = data_matrix[indices, :]
        if subset.size == 0:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.nanmean(subset, axis=0)
            pmin = np.nanmin(subset, axis=0)
            pmax = np.nanmax(subset, axis=0)
            spread = pmax - pmin
            mean_abs = np.abs(mean)
            mean_safe = mean_abs.copy()
            mean_safe[mean_safe < tiny] = np.nan
            ratio = spread / mean_safe

        base_color = palette.get(ens, None)
        if base_color is None:
            base_color = plt.cm.tab10(len(legend_handles) % 10)

        ratio_plot = ratio.copy()
        ratio_plot[~np.isfinite(ratio_plot)] = np.nan
        ratio_plot[ratio_plot <= 0.0] = np.nan

        ax.plot(time_grid, ratio_plot, '-', color=base_color, linewidth=1.5, label=f"{ens} (n={len(indices)})")

        legend_handles.append(Line2D([0], [0], color=base_color, lw=3))
        legend_labels.append(f"{ens} (n={len(indices)})")

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='best')

    ax.set_xlabel('Time (years)')
    ax.set_ylabel(f'Spread / |Mean| {unit_str} (log scale)')
    ax.set_yscale('log')
    ax.set_title(f'{variable} — Spread (range) / |Mean| (log y-axis)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_skew_kurt_time_series(time_grid, data_matrix, experiment_names, variable):
    from scipy import stats as _stats

    fig, (ax_ratio, ax_kurt) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ensemble_map = {}
    for idx, dname in enumerate(experiment_names):
        ens = dname.split(':', 1)[0]
        ensemble_map.setdefault(ens, []).append(idx)

    palette = globals().get('ensemble_to_base_color', {})

    ratio_handles = []
    ratio_labels = []
    kurt_handles = []
    kurt_labels = []

    tiny = 1e-12
    for ens, indices in ensemble_map.items():
        subset = data_matrix[indices, :]
        if subset.size == 0:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.nanmean(subset, axis=0)
            pmin = np.nanmin(subset, axis=0)
            pmax = np.nanmax(subset, axis=0)
            spread = pmax - pmin
            mean_abs = np.abs(mean)
            mean_safe = mean_abs.copy()
            mean_safe[mean_safe < tiny] = np.nan
            ratio = spread / mean_safe

        base_color = palette.get(ens, None)
        if base_color is None:
            base_color = plt.cm.tab10(len(ratio_handles) % 10)

        ratio_plot = ratio.copy()
        ratio_plot[~np.isfinite(ratio_plot)] = np.nan
        ratio_plot[ratio_plot <= 0.0] = np.nan

        lratio, = ax_ratio.plot(time_grid, ratio_plot, '-', color=base_color, linewidth=1.5, label=f"{ens} (n={len(indices)})")
        ratio_handles.append(lratio)
        ratio_labels.append(f"{ens} (n={len(indices)})")

            with np.errstate(invalid='ignore'):
                kurt_ts = _stats.kurtosis(subset, axis=0, nan_policy='omit')
        lk, = ax_kurt.plot(time_grid, kurt_ts, '-', color=base_color, linewidth=1.5, label=f"{ens} kurtosis")
        kurt_handles.append(lk)
        kurt_labels.append(f"{ens} (n={len(indices)})")

    if ratio_handles:
        ax_ratio.legend(ratio_handles, ratio_labels, loc='best')
    if kurt_handles:
        ax_kurt.legend(kurt_handles, kurt_labels, loc='best')

    ax_ratio.set_ylabel(f'Spread / |Mean| {unit_str} (log scale)')
    ax_ratio.set_yscale('log')
    ax_ratio.set_title(f'{variable} — Spread (range) / |Mean| (log y-axis)')
    ax_ratio.grid(True, alpha=0.3)

    ax_kurt.set_xlabel('Time (years)')
    ax_kurt.set_ylabel('Kurtosis')
    ax_kurt.set_title(f'{variable} — Kurtosis over time')
    ax_kurt.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_std_time_series(time_grid, data_matrix, experiment_names, variable, unit_str):
    """Plot ensemble standard deviation over time for each ensemble."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ensemble_map = {}
    for idx, dname in enumerate(experiment_names):
        ens = dname.split(':', 1)[0]
        ensemble_map.setdefault(ens, []).append(idx)

    palette = globals().get('ensemble_to_base_color', {})
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []

    for ens, indices in ensemble_map.items():
        subset = data_matrix[indices, :]
        if subset.size == 0:
            continue
        with np.errstate(invalid='ignore'):
            std_ts = np.nanstd(subset, axis=0)

        base_color = palette.get(ens, None)
        if base_color is None:
            base_color = plt.cm.tab10(len(legend_handles) % 10)

        ax.plot(time_grid, std_ts, '-', color=base_color, linewidth=1.5)

        legend_handles.append(Line2D([0], [0], color=base_color, lw=3))
        legend_labels.append(f"{ens} (n={len(indices)})")

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='best')

    ax.set_xlabel('Time (years)')
    ax.set_ylabel(f'Standard deviation {unit_str}')
    ax.set_title(f'{variable} — Ensemble standard deviation over time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_relative_std_time_series(time_grid, data_matrix, experiment_names, variable):
    """Plot ensemble σ relative to the absolute ensemble mean change (σ / |mean|)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ensemble_map = {}
    for idx, dname in enumerate(experiment_names):
        ens = dname.split(':', 1)[0]
        ensemble_map.setdefault(ens, []).append(idx)

    palette = globals().get('ensemble_to_base_color', {})
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []

    tiny = 1e-12
    for ens, indices in ensemble_map.items():
        subset = data_matrix[indices, :]
        if subset.size == 0:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.nanmean(subset, axis=0)
            std_ts = np.nanstd(subset, axis=0)
            mean_abs = np.abs(mean)
            mean_safe = mean_abs.copy()
            mean_safe[mean_safe < tiny] = np.nan
            rel = std_ts / mean_safe
            rel_plot = rel.copy()
            rel_plot[~np.isfinite(rel_plot)] = np.nan
            rel_plot[rel_plot <= 0.0] = np.nan

        base_color = palette.get(ens, None)
        if base_color is None:
            base_color = plt.cm.tab10(len(legend_handles) % 10)

        ax.plot(time_grid, rel_plot, '-', color=base_color, linewidth=1.5)

        legend_handles.append(Line2D([0], [0], color=base_color, lw=3))
        legend_labels.append(f"{ens} (n={len(indices)})")

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='best')

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Std / |Mean| (log scale)')
    ax.set_yscale('log')
    ax.set_title(f'{variable} — Ensemble σ relative to absolute ensemble mean (log y-axis)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_time_series_metrics_combined(time_grid, data_matrix, experiment_names, variable, unit_str, std_normalize='absolute'):
    """Create multi-panel figure: std dev, relative std, and skewness."""
    n_rows = 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5*n_rows), sharex=True)

    ensemble_map = {}
    for idx, dname in enumerate(experiment_names):
        ens = dname.split(':', 1)[0]
        ensemble_map.setdefault(ens, []).append(idx)

    palette = globals().get('ensemble_to_base_color', {})
    tiny = 1e-12

    ax0 = axes[0]
    tiny = 1e-12
    std_ylabel = None
    std_title = None
    for ens, indices in ensemble_map.items():
        subset = data_matrix[indices, :]
        if subset.size == 0:
            continue
        mean_ts = np.nanmean(subset, axis=0)
        std_ts = np.nanstd(subset, axis=0)
        cv_floor = max(tiny, 0.01 * np.nanmax(np.abs(mean_ts)))
        mean_safe = mean_ts.copy()
        mean_safe[np.abs(mean_safe) < cv_floor] = np.nan

        if std_normalize == 'absolute':
            plot_vals = std_ts
            std_ylabel = f'Standard deviation {unit_str}'
            std_title = f'{variable} — Ensemble standard deviation over time'
        elif std_normalize == 'cv':
            plot_vals = std_ts / np.abs(mean_safe)
            std_ylabel = 'Std / |Mean|'
            std_title = f'{variable} — Coefficient of variation (σ / |mean|)'
        elif std_normalize == 'percent':
            plot_vals = 100.0 * (std_ts / np.abs(mean_safe))
            std_ylabel = 'Std / |Mean| (%)'
            std_title = f'{variable} — Std as percent of mean (σ / |mean| * 100)'
        else:
            plot_vals = std_ts
            std_ylabel = f'Standard deviation {unit_str}'
            std_title = f'{variable} — Ensemble standard deviation over time'

        base_color = palette.get(ens, None) or plt.cm.tab10(len(ax0.lines) % 10)
        ax0.plot(time_grid, plot_vals, '-', color=base_color, linewidth=1.5, label=f"{ens} (n={len(indices)})")

    ax0.set_ylabel('Standard Deviation')
    ax0.set_title(f'Ensemble standard deviation (σ) for {variable}')
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc='best')

    ax1 = axes[1]
    for ens, indices in ensemble_map.items():
        subset = data_matrix[indices, :]
        if subset.size == 0:
            continue
        mean = np.nanmean(subset, axis=0)
        std_ts = np.nanstd(subset, axis=0)
        mean_abs = np.abs(mean)
        mean_safe = mean_abs.copy()
        mean_safe[mean_safe < tiny] = np.nan
        rel = std_ts / mean_safe
        rel_plot = rel.copy()
        rel_plot[~np.isfinite(rel_plot)] = np.nan
        rel_plot[rel_plot <= 0.0] = np.nan

        base_color = palette.get(ens, None) or plt.cm.tab10(len(ax1.lines) % 10)
        ax1.plot(time_grid, rel_plot, '-', color=base_color, linewidth=1.5, label=f"{ens} (n={len(indices)})")

    ax1.set_ylabel('Std / |Mean|')
    ax1.set_yscale('log')
    ax1.set_title(f'σ / Δ{variable} — Std deviation (σ) / mean change in {variable} (log y-axis)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    ax2 = axes[2]
    from scipy import stats as _stats
    kurt_dummy = []
    for ens, indices in ensemble_map.items():
        subset = data_matrix[indices, :]
        if subset.size == 0:
            continue
        with np.errstate(invalid='ignore'):
            skew_ts = _stats.skew(subset, axis=0, nan_policy='omit')

        base_color = palette.get(ens, None) or plt.cm.tab10(len(ax2.lines) % 10)
        ax2.plot(time_grid, skew_ts, '-', color=base_color, linewidth=1.5, label=f"{ens} (n={len(indices)})")

    ax2.set_ylabel('Skewness')
    ax2.set_xlabel('Time (years)')
    ax2.set_title('Skewness over time')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()
    return fig


def plot_probability_evolution(all_var_data, valid_time_steps, variable, unit_str, num_experiments, mode='mass', show_bars=True):
    """Plot probability evolution (mass, cdf, or density) for the ensemble."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    time_steps = sorted(valid_time_steps)

    all_values = np.concatenate([np.array(all_var_data[ts]) for ts in time_steps])
    data_min, data_max = all_values.min(), all_values.max()
    data_range = data_max - data_min if data_max > data_min else 1.0

    colors = plt.cm.viridis(np.linspace(0, 1, len(time_steps)))

    if mode == 'mass':
        bins = np.histogram_bin_edges(all_values, bins='auto')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        x_grid = np.linspace(data_min - 0.1*data_range, data_max + 0.1*data_range, 2000)

        global_max_prob = 0.0
        for i, ts in enumerate(time_steps):
            vals = np.array(all_var_data[ts])
            if len(vals) == 0:
                continue
            kde = gaussian_kde(vals)
            kde_vals = kde(x_grid)

            probs = np.empty(len(bins)-1)
            for j in range(len(bins)-1):
                mask = (x_grid >= bins[j]) & (x_grid < bins[j+1])
                if mask.any():
                    probs[j] = np.trapz(kde_vals[mask], x_grid[mask])
                else:
                    probs[j] = 0.0
            if probs.sum() > 0:
                probs /= probs.sum()

            global_max_prob = max(global_max_prob, probs.max() if len(probs) else 0.0)

            if show_bars:
                ax.bar(bin_centers, probs, width=np.diff(bins), align='center',
                       color=colors[i], alpha=0.25, label=f'Year {ts:.0f}')
            ax.plot(x_grid, kde_vals * (np.diff(bins).mean()), color=colors[i], linewidth=1.5, alpha=0.8)

        ax.set_ylabel('Probability mass (per bin)')
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(f'{variable}{unit_str}')
        ax.set_title(f'{variable} Probability Distribution Evolution (probability mass per bin, KDE-integrated)')
        if show_bars:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        ax.text(0.98, 0.95, f'Bins={len(bins)-1}\nMax bin prob={global_max_prob:.3g}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    elif mode == 'cdf':
        for i, ts in enumerate(time_steps):
            vals = np.sort(np.array(all_var_data[ts]))
            if len(vals) == 0:
                continue
            cdf = np.arange(1, len(vals)+1) / len(vals)
            ax.plot(vals, cdf, color=colors[i], linewidth=2.0, label=f'Year {ts:.0f}')

        ax.set_ylabel('Cumulative probability')
        ax.set_xlabel(f'{variable}{unit_str}')
        ax.set_title(f'{variable} CDF Evolution')
        ax.set_ylim(0.0, 1.0)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # diagnostic: median for last time step
        last_vals = np.sort(np.array(all_var_data[time_steps[-1]]))
        if len(last_vals) > 0:
            median = np.median(last_vals)
            ax.text(0.98, 0.95, f'Last median={median:.3g}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    else:  # density
        x_range = np.linspace(data_min - 0.1*data_range, data_max + 0.1*data_range, 300)
        for i, ts in enumerate(time_steps):
            vals = np.array(all_var_data[ts])
            if len(vals) >= 3 and np.std(vals) > 1e-10 and np.unique(vals).size > 1:
                try:
                    kde = gaussian_kde(vals)
                    pdf = kde(x_range)
                    ax.plot(x_range, pdf, color=colors[i], linewidth=2.0, label=f'Year {ts:.0f}')
                except Exception as e:
                    print(f"Warning: gaussian_kde failed for year {ts} (density plot fallback): {e}", file=sys.stderr)
                    h_bins = 50
                    h, bin_edges = np.histogram(vals, bins=h_bins, density=True)
                    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    ax.step(centers, h, where='mid', color=colors[i], linewidth=1.5, label=f'Year {ts:.0f} (hist)')
            else:
                if vals.size > 0:
                    y_vals = np.zeros_like(vals) + 0.0
                    ax.plot(vals, y_vals, '|', color=colors[i], markersize=10, label=f'Year {ts:.0f} (points)')
        ax.set_xlabel(f'{variable}{unit_str}')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{variable} PDF Evolution (KDE)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        try:
            last_vals = np.array(all_var_data[time_steps[-1]])
            kde = gaussian_kde(last_vals)
            pdf = kde(x_range)
            area = np.trapz(pdf, x_range)
            peak = pdf.max()
            ax.text(0.98, 0.95, f'Peak={peak:.3g}\nArea≈{area:.3f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        except Exception:
            pass

    plt.tight_layout()
    return fig

def print_statistics_summary(all_var_data, valid_time_steps):
    """Print comprehensive statistics table."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Year':<8} {'N':<5} {'Mean':<10} {'Std':<10} {'Median':<10} "
          f"{'Q25':<10} {'Q75':<10} {'Skew':<8} {'Kurt':<8}")
    print("-"*80)

    for time_step in valid_time_steps:
        var_values = np.array(all_var_data.get(time_step, []))
        if var_values.size == 0:
            print(f"{time_step:<8.0f} {'0':<5} {'nan':<10} {'nan':<10} {'nan':<10} {'nan':<10} {'nan':<10} {'nan':<8} {'nan':<8}")
            continue
        s = calculate_distribution_stats(var_values)
        print(f"{time_step:<8.0f} {s['n']:<5} {s['mean']:<10.2f} {s['std']:<10.2f} "
              f"{s['median']:<10.2f} {s['q25']:<10.2f} {s['q75']:<10.2f} "
              f"{s['skewness']:<8.2f} {s['kurtosis']:<8.2f}")
    
def main():
    # Parse command-line arguments
    options = parse_arguments()

    # Parse x-axis limits if provided
    xlim_range = None
    if getattr(options, 'xlimits', None):
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
    
    # Parse experiment list and time steps
    time_steps = [float(t.strip()) for t in options.timeSteps.split(',')]

    # Support multiple ensemble bases (comma-separated) and several experiment specification formats:
    #  - 'ENSEMBLE:EXP' -> explicit mapping
    #  - 'EXP' -> search EXP under all provided ensembles
    #  - wildcards supported in EXP (e.g. 'CTRL:EXP*' or 'EXP*')
    if not options.ensembleBaseDir:
        sys.exit("ERROR: Must specify ensemble base directory(s) with -b/--base")

    ensemble_dirs = [ens.strip() for ens in options.ensembleBaseDir.split(',')]

    experiment_specs = []

    if not options.experimentList:
        if not options.rootDataDir:
            sys.exit("ERROR: --root must be provided when auto-discovering experiments")
        for ens in ensemble_dirs:
            ensemble_path = os.path.join(options.rootDataDir, ens) if options.rootDataDir else ens
            if not os.path.exists(ensemble_path):
                print(f"Warning: ensemble directory not found: {ensemble_path}", file=sys.stderr)
                continue
            for item in os.listdir(ensemble_path):
                exp_path = os.path.join(ensemble_path, item)
                if os.path.isdir(exp_path):
                    stats_file = os.path.join(exp_path, options.statsFilename)
                    if os.path.exists(stats_file):
                        display_name = f"{ens}:{item}"
                        experiment_specs.append((ens, item, stats_file, display_name))
    else:
        exp_parts = [exp.strip() for exp in options.experimentList.split(',')]
        for exp_spec in exp_parts:
            if ':' in exp_spec:
                ens_name, exp_name = [p.strip() for p in exp_spec.split(':', 1)]
                if ens_name not in ensemble_dirs:
                    print(f"Warning: specified ensemble '{ens_name}' not in provided bases", file=sys.stderr)
                    continue
                if '*' in exp_name or '?' in exp_name:
                    search_path = os.path.join(options.rootDataDir, ens_name) if options.rootDataDir else ens_name
                    matches = glob.glob(os.path.join(search_path, exp_name))
                    for match in matches:
                        if os.path.isdir(match):
                            match_exp = os.path.basename(match)
                            stats_file = os.path.join(match, options.statsFilename)
                            if os.path.exists(stats_file):
                                display_name = f"{ens_name}:{match_exp}"
                                experiment_specs.append((ens_name, match_exp, stats_file, display_name))
                else:
                    exp_path = os.path.join(options.rootDataDir, ens_name, exp_name) if options.rootDataDir else os.path.join(ens_name, exp_name)
                    stats_file = os.path.join(exp_path, options.statsFilename)
                    if os.path.exists(stats_file):
                        display_name = f"{ens_name}:{exp_name}"
                        experiment_specs.append((ens_name, exp_name, stats_file, display_name))
                    else:
                        print(f"Warning: stats file not found for {ens_name}:{exp_name} -> {stats_file}", file=sys.stderr)
            else:
                exp_name = exp_spec
                if '*' in exp_name or '?' in exp_name:
                    for ens in ensemble_dirs:
                        search_path = os.path.join(options.rootDataDir, ens) if options.rootDataDir else ens
                        matches = glob.glob(os.path.join(search_path, exp_name))
                        for match in matches:
                            if os.path.isdir(match):
                                match_exp = os.path.basename(match)
                                stats_file = os.path.join(match, options.statsFilename)
                                if os.path.exists(stats_file):
                                    display_name = f"{ens}:{match_exp}"
                                    experiment_specs.append((ens, match_exp, stats_file, display_name))
                else:
                    found = False
                    for ens in ensemble_dirs:
                        exp_path = os.path.join(options.rootDataDir, ens, exp_name) if options.rootDataDir else os.path.join(ens, exp_name)
                        stats_file = os.path.join(exp_path, options.statsFilename)
                        if os.path.exists(stats_file):
                            display_name = f"{ens}:{exp_name}"
                            experiment_specs.append((ens, exp_name, stats_file, display_name))
                            found = True
                    if not found:
                        print(f"Warning: experiment '{exp_name}' not found under any provided ensemble base", file=sys.stderr)

    if not experiment_specs:
        sys.exit("ERROR: No valid experiments found with the provided -b/-e arguments")

    experiment_files = [spec[2] for spec in experiment_specs]
    experiment_names = [spec[3] for spec in experiment_specs]

    ensemble_names_unique = []
    for ens, _, _, _ in experiment_specs:
        if ens not in ensemble_names_unique:
            ensemble_names_unique.append(ens)
    base_cmap = plt.get_cmap('tab20')  # plt.cm.get_cmap removed in matplotlib >=3.9
    ensemble_to_base_color = {}
    for i, ens in enumerate(ensemble_names_unique):
        if ens.upper().startswith('CTRL'):
            ensemble_to_base_color[ens] = "#383E39"
        elif 'SSP126' in ens.upper():
            ensemble_to_base_color[ens] = "#236ddb"
        elif 'SSP585' in ens.upper():
            ensemble_to_base_color[ens] = "#dc2f2f"
        else:
            ensemble_to_base_color[ens] = base_cmap(i % base_cmap.N)

    # If user provided explicit ensemble colors on the CLI, use them (order matches the supplied -b bases)
    if getattr(options, 'ensemble_colors', None):
        try:
            provided = [c.strip() for c in options.ensemble_colors.split(',') if c.strip()]
            if len(provided) != len(ensemble_names_unique):
                print(f"Warning: --ensemble-colors provided {len(provided)} colors but found {len(ensemble_names_unique)} ensemble bases; ignoring override", file=sys.stderr)
            else:
                for ens_name, col in zip(ensemble_names_unique, provided):
                    ensemble_to_base_color[ens_name] = col
        except Exception as e:
            print(f"Warning: failed to parse --ensemble-colors: {e}; using default colors", file=sys.stderr)

    # Export the mapping to module globals so plotting helper functions (which lookup globals())
    # can access the ensemble base colors when they run.
    globals()['ensemble_to_base_color'] = ensemble_to_base_color

    # Diagnostic: print the resolved ensemble -> base color mapping
    print('Ensemble -> base color mapping:')
    for ens in ensemble_names_unique:
        print(f"  {ens} -> {ensemble_to_base_color.get(ens)}")

    # Map each display_name (ENSEMBLE:EXP) to a per-experiment color variation
    # Create small brightness variations so experiments within the same ensemble are distinct
    import matplotlib.colors as mcolors
    experiment_to_color = {}
    for ens in ensemble_names_unique:
        members = [d for e, _, _, d in experiment_specs if e == ens]
        n = len(members)
        base_color = ensemble_to_base_color[ens]
        hsv = mcolors.rgb_to_hsv(mcolors.to_rgb(base_color))
        variations = []
        if n == 1:
            variations = [base_color]
        else:
            for j in range(n):
                brightness = 0.5 + 0.5 * (j / max(1, n-1))
                new_hsv = hsv.copy()
                new_hsv[2] = min(1.0, hsv[2] * brightness)
                variations.append(mcolors.hsv_to_rgb(new_hsv))
        for member, col in zip(members, variations):
            experiment_to_color[member] = col
    
    # Get scaling
    scale_vol, mass_unit = get_scale_factor(options.units)
    
    # Determine change mode
    change_mode = None
    if options.plotChange:
        change_mode = 'absolute'
    elif options.plotPercentChange:
        change_mode = 'percent'
    
    # Build unit string once
    unit_str = get_unit_string(options.variable, mass_unit, change_mode)
    
    
    
    # Validate that all resolved files exist (experiment_files was built from experiment_specs)
    for file_path, display_name in zip(experiment_files, experiment_names):
        if not os.path.exists(file_path):
            sys.exit(f"ERROR: File not found: {file_path}")
    
    # Extract and interpolate data
    all_var_data, valid_time_steps = interpolate_ensemble_data(
        experiment_files, experiment_names, options.variable, 
        scale_vol, time_steps, change_mode
    )
    
    if not valid_time_steps:
        sys.exit("ERROR: No valid time steps with sufficient data")
    
    
    
    # Optionally convert to anomalies.
    data_for_plots = all_var_data
    display_variable = options.variable
    # resolved display-name for anomaly-ref (if provided and matched)
    resolved_ref_display_name = None

    # If a reference experiment is provided, compute anomalies relative to that run
    if options.anomaly_ref:
        user_ref = options.anomaly_ref

        # Try to resolve the user-provided reference name against the discovered
        # display names in `experiment_names`. Support several convenient forms:
        #  - exact display name: 'ENSEMBLE:EXP'
        #  - experiment suffix: 'EXP' (matches the part after the colon)
        #  - file basename prefix: matches start of the netCDF filename
        ref_name = None

        # 1) exact match
        if user_ref in experiment_names:
            ref_name = user_ref
        else:
            # 2) match by EXP part after colon
            matches = [nm for nm in experiment_names if nm.split(':', 1)[-1] == user_ref]
            if len(matches) == 1:
                ref_name = matches[0]
            elif len(matches) > 1:
                sys.exit(f"ERROR: ambiguous anomaly-ref '{user_ref}'; matches: {matches}")
            else:
                # 3) try matching by file basename (starts with provided string)
                matches = [nm for nm, fp in zip(experiment_names, experiment_files)
                           if os.path.basename(fp).startswith(user_ref)]
                if len(matches) == 1:
                    ref_name = matches[0]
                elif len(matches) > 1:
                    sys.exit(f"ERROR: ambiguous anomaly-ref '{user_ref}'; file matches: {matches}")

        if ref_name is None:
            # Provide helpful message listing available names for the user
            sample_names = ', '.join(experiment_names[:10]) + (', ...' if len(experiment_names) > 10 else '')
            sys.exit(f"ERROR: reference experiment '{user_ref}' not found among resolved experiments. "
                     f"Provide one of the display names (ENSEMBLE:EXP) or the EXP suffix. "
                     f"Examples: {sample_names}")

        # Find the corresponding file for the resolved reference and interpolate to target times
        ref_idx = experiment_names.index(ref_name)
        ref_file = experiment_files[ref_idx]
        try:
            ref_years, ref_var = extract_variable_data(ref_file, options.variable, scale_vol, change_mode)
        except Exception as e:
            sys.exit(f"ERROR: failed to extract reference data from {ref_name}: {e}")

        # Interpolate reference to the (already filtered) valid_time_steps
        interp_ref = interp1d(ref_years, ref_var, kind='linear', bounds_error=False, fill_value=np.nan)
        ref_vals = {ts: float(interp_ref(ts)) for ts in valid_time_steps}

        # Filter out time steps where reference is NaN (outside ref range)
        valid_time_steps_ref = [ts for ts in valid_time_steps if not np.isnan(ref_vals[ts])]
        if not valid_time_steps_ref:
            sys.exit(f"ERROR: reference '{ref_name}' has no overlap with valid time steps")

        # Build anomaly dataset: subtract reference value per time step from each ensemble member value
        anom_data = {}
        for ts in valid_time_steps_ref:
            vals = np.array(all_var_data.get(ts, []))
            if len(vals) == 0:
                anom_data[ts] = []
                continue
            anom_data[ts] = list(vals - ref_vals[ts])

        data_for_plots = anom_data
        valid_time_steps = valid_time_steps_ref
        display_variable = f"{options.variable} (anomaly vs {ref_name})"
        resolved_ref_display_name = ref_name

    elif options.anomaly:
        anom_data = {}
        for ts in valid_time_steps:
            vals = np.array(all_var_data.get(ts, []))
            if len(vals) == 0:
                anom_data[ts] = []
                continue
            mean_ts = np.mean(vals)
            anom_data[ts] = list(vals - mean_ts)
        data_for_plots = anom_data
        display_variable = f"{options.variable} (anomaly vs ensemble mean)"

    # Generate plots
    # Grid of per-timestep PDFs (kept for quick inspection)
    fig_grid = plot_pdf_grid(data_for_plots, valid_time_steps, display_variable, 
                              unit_str, len(experiment_names))

    # Skewness evolution
    fig_skew = plot_skewness_evolution(data_for_plots, valid_time_steps, display_variable)

    # Evolution plot: select heatmap/percentiles (time on x-axis) or traditional PDF modes
    if options.heatmap:
        fig_evolution = plot_heatmap_evolution(all_var_data, valid_time_steps, options.variable,
                                               unit_str, len(experiment_names))
        evo_suffix = '_heatmap'
    elif options.percentiles:
        fig_evolution = plot_percentiles_evolution(all_var_data, valid_time_steps, options.variable,
                                                   unit_str, len(experiment_names))
        evo_suffix = '_percentiles'
    else:
        # Unified probability evolution plot: default mode determined by CLI flags
        if options.mass_hist:
            mode = 'mass'
        elif options.cdf:
            mode = 'cdf'
        else:
            mode = 'density'

        # Generate the unified probability evolution plot; tolerate failures so
        # other figures still produce when KDE/data issues occur.
        try:
            fig_evolution = plot_probability_evolution(data_for_plots, valid_time_steps, display_variable,
                                                       unit_str, len(experiment_names), mode=mode,
                                                       show_bars=(not options.no_bars))
            if mode == 'mass':
                evo_suffix = '_prob_evolution'
            elif mode == 'cdf':
                evo_suffix = '_cdf_evolution'
            else:
                evo_suffix = '_pdf_evolution'
        except Exception as e:
            # Fail silently for the evolution plot per user request; print a short warning.
            print(f"Warning: failed to produce probability-evolution figure: {e}", file=sys.stderr)
            fig_evolution = None
            evo_suffix = '_prob_evolution_failed'
    
    # Save figures if requested
    if options.plotSave:
        # Save the grid and skewness as before
        fig_grid.savefig(options.plotSave, dpi=300, bbox_inches='tight')
        print(f"Saved: {options.plotSave}")

        skew_name = options.plotSave.replace('.png', '_skewness.png')
        fig_skew.savefig(skew_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {skew_name}")

        # Save the evolution figure with a name depending on the selected mode
        if fig_evolution is not None:
            evo_name = options.plotSave.replace('.png', f"{evo_suffix}.png")
            try:
                fig_evolution.savefig(evo_name, dpi=300, bbox_inches='tight')
                print(f"Saved: {evo_name}")
            except Exception as e:
                print(f"Warning: failed to save evolution figure: {e}", file=sys.stderr)
        else:
            print("Skipping saving probability-evolution figure (generation failed).")

    # Optional time series plot using all timestamps from original files
    if options.time_series:
        time_grid, data_matrix = build_time_series_matrix(experiment_files, experiment_names,
                                                         options.variable, scale_vol, change_mode)
        if time_grid.size == 0 or data_matrix.size == 0:
            print('Warning: no time-series data available; skipping time-series plot')
        else:
            # Determine mode for time series
            if resolved_ref_display_name is not None:
                ts_mode = 'ref'
                ref_idx = experiment_names.index(resolved_ref_display_name)
            elif options.anomaly_ref:
                # fallback: user passed a ref but resolution failed earlier (shouldn't happen)
                try:
                    ref_idx = experiment_names.index(options.anomaly_ref)
                    ts_mode = 'ref'
                except ValueError:
                    print(f"Warning: anomaly-ref '{options.anomaly_ref}' was not resolved; falling back to anomaly mode", file=sys.stderr)
                    ts_mode = 'anomaly'
                    ref_idx = None
            elif options.anomaly:
                ts_mode = 'anomaly'
                ref_idx = None
            else:
                ts_mode = 'raw'
                ref_idx = None

            fig_ts = plot_time_series(time_grid, data_matrix, experiment_names,
                                      display_variable, unit_str,
                                      mode=ts_mode, ref_idx=ref_idx, scale_vol=scale_vol)
            # Apply x-axis limits to time-series figure if requested
            if xlim_range is not None:
                try:
                    ax_ts = fig_ts.axes[0] if len(fig_ts.axes) > 0 else fig_ts.gca()
                    ax_ts.set_xlim(xlim_range)
                    print(f'Applied X-axis limits to time-series plot: {xlim_range}')
                except Exception as e:
                    print('Failed to apply x-axis limits to time-series plot:', e)

            # Save timeseries figure only if requested
            if options.plotSave:
                ts_name = options.plotSave.replace('.png', f"_timeseries")
                if resolved_ref_display_name:
                    ts_name = ts_name + f"_vs_{resolved_ref_display_name}.png"
                elif options.anomaly_ref:
                    ts_name = ts_name + f"_vs_{options.anomaly_ref}.png"
                elif options.anomaly:
                    ts_name = ts_name + '_anom.png'
                else:
                    ts_name = ts_name + '.png'

                fig_ts.savefig(ts_name, dpi=300, bbox_inches='tight')
                print(f"Saved: {ts_name}")
            # Create and optionally save the spread/mean ratio time series
            try:
                fig_metrics = plot_time_series_metrics_combined(time_grid, data_matrix, experiment_names, display_variable, unit_str, std_normalize=options.std_normalize)
                if xlim_range is not None:
                    try:
                        for ax in fig_metrics.axes:
                            ax.set_xlim(xlim_range)
                    except Exception:
                        pass
                if options.plotSave:
                    metrics_name = options.plotSave.replace('.png', f"_timeseries_metrics.png")
                    metrics_name = metrics_name + '.png' if not metrics_name.endswith('.png') else metrics_name
                    fig_metrics.savefig(metrics_name, dpi=300, bbox_inches='tight')
                    print(f"Saved: {metrics_name}")
            except Exception as e:
                print('Failed to produce combined metrics figure:', e)
            # Produce skewness/kurtosis as a separate figure
            try:
                fig_sk_kurt_ts = plot_skew_kurt_time_series(time_grid, data_matrix, experiment_names, display_variable)
                if xlim_range is not None:
                    try:
                        ax_sk = fig_sk_kurt_ts.axes[0] if len(fig_sk_kurt_ts.axes) > 0 else fig_sk_kurt_ts.gca()
                        ax_sk.set_xlim(xlim_range)
                    except Exception:
                        pass
                if options.plotSave:
                    sk_name_ts = options.plotSave.replace('.png', f"_skew_kurt_timeseries.png")
                    sk_name_ts = sk_name_ts + '.png' if not sk_name_ts.endswith('.png') else sk_name_ts
                    fig_sk_kurt_ts.savefig(sk_name_ts, dpi=300, bbox_inches='tight')
                    print(f"Saved: {sk_name_ts}")
            except Exception as e:
                print('Failed to produce skew/kurtosis timeseries figure:', e)
    
    # Print statistics
    print_statistics_summary(all_var_data, valid_time_steps)
    
    plt.show()

if __name__ == "__main__":
    main()
