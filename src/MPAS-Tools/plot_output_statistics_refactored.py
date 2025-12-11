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
from netCDF4 import Dataset
from optparse import OptionParser
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

# Constants
RHOI = 910.0  # Ice density (kg/m³)
RHOSW = 1028.0  # Seawater density (kg/m³)
MIN_EXPERIMENTS_THRESHOLD = 3  # Minimum experiments needed for valid PDF

# ============================================================================
# Configuration and Argument Parsing
# ============================================================================

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

# ============================================================================
# Unit Handling
# ============================================================================

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
    """
    Build appropriate unit string for axis labels.
    
    Args:
        variable: Variable name
        base_unit: Base unit string (e.g., "Gt", "m³")
        change_mode: 'absolute', 'percent', or None
    """
    var_lower = variable.lower()
    
    # Determine base unit based on variable type
    if 'volume' in var_lower or 'vaf' in var_lower:
        unit_base = base_unit
    elif 'area' in var_lower:
        unit_base = "km²"
    elif 'flux' in var_lower and base_unit == "Gt":
        unit_base = "Gt/yr"
    else:
        unit_base = ""
    
    # Add change modifier
    if change_mode == 'absolute':
        return f" change ({unit_base})" if unit_base else " change"
    elif change_mode == 'percent':
        return " change (%)"
    else:
        return f" ({unit_base})" if unit_base else ""

# ============================================================================
# Data Extraction and Processing
# ============================================================================

def extract_variable_data(fname, variable, scale_vol, change_mode=None):
    """
    Extract and scale variable time series from a single experiment file.
    
    Args:
        fname: Path to NetCDF file
        variable: Variable name to extract
        scale_vol: Volume/mass scaling factor
        change_mode: 'absolute', 'percent', or None
    
    Returns:
        years: Time array (years from start)
        var_data: Scaled and processed variable data
    """
    with Dataset(fname, 'r') as f:
        if variable not in f.variables:
            raise ValueError(f"Variable '{variable}' not found in {fname}")
        
        yr = f.variables['daysSinceStart'][:] / 365.0
        yr = yr - yr[0]  # Start from year 0
        
        var_data = f.variables[variable][:]
    
    # Apply scaling
    var_lower = variable.lower()
    if 'volume' in var_lower or 'vaf' in var_lower:
        var_data = var_data / scale_vol
    elif 'area' in var_lower:
        var_data = var_data / 1e6  # Convert to km²
    elif 'flux' in var_lower and scale_vol > 1e10:  # If units are Gt
        var_data = var_data / 1e12  # Convert to Gt/yr
    
    # Apply change calculations
    if change_mode == 'absolute':
        var_data = var_data - var_data[0]
    elif change_mode == 'percent':
        var_data = (var_data - var_data[0]) * 100 / var_data[0]
    
    return yr, var_data

def interpolate_ensemble_data(experiment_files, experiment_names, variable, 
                               scale_vol, target_times, change_mode=None):
    """
    Extract and interpolate variable data from all experiments to target times.
    
    Returns:
        all_var_data: Dict mapping time_step -> list of variable values
        valid_time_steps: List of time steps with sufficient data
    """
    all_var_data = {}
    min_experiments = max(MIN_EXPERIMENTS_THRESHOLD, int(0.5 * len(experiment_files)))
    
    for exp_file, exp_name in zip(experiment_files, experiment_names):
        try:
            years, var_data = extract_variable_data(exp_file, variable, scale_vol, change_mode)
        except Exception as e:
            print(f"Warning: Failed to extract data from {exp_name}: {e}", file=sys.stderr)
            continue
        
        # Interpolate to target times within available range
        valid_times = [t for t in target_times if years.min() <= t <= years.max()]
        if not valid_times:
            continue
        
        interp_func = interp1d(years, var_data, kind='linear', 
                               bounds_error=False, fill_value=np.nan)
        interp_var = interp_func(valid_times)
        
        # Store values
        for time_step, value in zip(valid_times, interp_var):
            if time_step not in all_var_data:
                all_var_data[time_step] = []
            if not np.isnan(value):
                all_var_data[time_step].append(value)
    
    # Filter time steps with sufficient data
    valid_time_steps = [ts for ts in sorted(all_var_data.keys()) 
                        if len(all_var_data[ts]) >= min_experiments]
    
    return all_var_data, valid_time_steps

# ============================================================================
# Statistics Calculation
# ============================================================================

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

# ============================================================================
# Plotting Functions
# ============================================================================

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
        
        # Histogram
        ax.hist(var_values, bins='auto', alpha=0.6, density=True, 
                color='skyblue', edgecolor='black', label='Data')
        
        # KDE overlay (only if sufficient variation)
        if dist_stats['std'] > 1e-10 and len(var_values) >= 3:
            kde = gaussian_kde(var_values)
            x_range = np.linspace(var_values.min(), var_values.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='KDE')
        
        # Labels and title
        ax.set_xlabel(f'{variable}{unit_str}')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Year {time_step:.0f} (N={dist_stats["n"]})\n'
                     f'μ={dist_stats["mean"]:.2f}, σ={dist_stats["std"]:.2f}, '
                     f'skew={dist_stats["skewness"]:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle(f"{variable} Probability Distributions ({num_experiments} Experiments)", 
                 fontsize=14)
    plt.tight_layout()
    
    return fig

def plot_skewness_evolution(all_var_data, valid_time_steps, variable):
    """Plot evolution of distribution skewness over time."""
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
        
        # Only plot KDE if sufficient variation exists
        if len(var_values) >= 3 and np.std(var_values) > 1e-10:
            kde = gaussian_kde(var_values)
            kde_pdf = kde(x_range)
            ax.plot(x_range, kde_pdf, color=colors[i], linewidth=2.5, 
                    label=f'Year {time_step:.0f}')
            
            # Mark mean with vertical line
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
    """Plot distribution evolution as a heatmap with time on the x-axis.

    Each column corresponds to a time step and each row to a variable bin.
    The color shows probability mass (bins sum to 1 per column/time).
    """
    time_steps = sorted(valid_time_steps)

    # Concatenate all values to determine binning
    all_values = np.concatenate([np.array(all_var_data[ts]) for ts in time_steps])
    bins = np.histogram_bin_edges(all_values, bins='auto')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # x grid for KDE integration
    data_min, data_max = all_values.min(), all_values.max()
    data_range = data_max - data_min if data_max > data_min else 1.0
    x_grid = np.linspace(data_min - 0.1*data_range, data_max + 0.1*data_range, 2000)

    prob_matrix = np.zeros((len(bin_centers), len(time_steps)))

    for i, ts in enumerate(time_steps):
        vals = np.array(all_var_data[ts])
        if len(vals) == 0:
            continue
        # KDE and integrate per bin to get probabilities
        if len(vals) >= 3 and np.std(vals) > 1e-12:
            kde = gaussian_kde(vals)
            kde_vals = kde(x_grid)
            for j in range(len(bins)-1):
                mask = (x_grid >= bins[j]) & (x_grid < bins[j+1])
                if mask.any():
                    prob_matrix[j, i] = np.trapz(kde_vals[mask], x_grid[mask])
        else:
            # Fallback to empirical histogram for tiny samples
            h, _ = np.histogram(vals, bins=bins)
            if h.sum() > 0:
                prob_matrix[:, i] = h / h.sum()

    # Normalize columns to sum to 1 (numerical stability)
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
    """Plot percentiles over time (median and IQR by default).

    Time is on the x-axis and variable values on the y-axis.
    """
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
    """Build a common time grid and interpolate all experiments to that grid.

    Returns:
      time_grid: 1D numpy array of sorted unique time points (years)
      data_matrix: 2D array shape (n_experiments, n_times) with NaNs where data missing
    """
    series_list = []
    time_arrays = []

    for fpath, name in zip(experiment_files, experiment_names):
        try:
            years, var_data = extract_variable_data(fpath, variable, scale_vol, change_mode)
            # Ensure years and var_data are plain numpy arrays (not masked arrays)
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

    # Build union time grid (sorted unique)
    if len(time_arrays) == 0:
        return np.array([]), np.empty((0, 0))
    all_times_concat = np.concatenate([t for t in time_arrays if len(t) > 0]) if any(len(t)>0 for t in time_arrays) else np.array([])
    if all_times_concat.size == 0:
        return np.array([]), np.empty((len(series_list), 0))

    time_grid = np.unique(np.sort(all_times_concat))

    # Interpolate each series to the grid
    data_matrix = np.full((len(series_list), len(time_grid)), np.nan)
    for i, (t, s) in enumerate(zip(time_arrays, series_list)):
        if len(t) == 0 or len(s) == 0:
            continue
        # If only a single time point is available, assign value at matching time(s)
        if len(t) == 1:
            # find grid indices that match this time (use isclose)
            mask = np.isclose(time_grid, float(t[0]))
            if mask.any():
                data_matrix[i, mask] = float(s[0])
            continue

        # For normal series, ensure no masked arrays are passed to interp1d
        try:
            interp = interp1d(t, s, kind='linear', bounds_error=False, fill_value=np.nan)
            data_matrix[i, :] = interp(time_grid)
        except Exception as e:
            print(f"Warning: interpolation failed for {experiment_names[i]}: {e}", file=sys.stderr)
            # leave row as NaNs in case of failure
            continue

    return time_grid, data_matrix


def plot_time_series(time_grid, data_matrix, experiment_names, variable, unit_str, mode='raw', ref_idx=None):
    """Plot ensemble time series.

    mode: 'raw' | 'anomaly' | 'ref'  (ref requires ref_idx)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    n_exp = data_matrix.shape[0]
    # Plot individual members (light lines). In 'ref' mode skip plotting the
    # reference run here so it does not obscure the ensemble mean/shading.
    for i in range(n_exp):
        if mode == 'ref' and ref_idx is not None and i == ref_idx:
            continue
        ax.plot(time_grid, data_matrix[i, :], color='gray', alpha=0.4, linewidth=0.8)

    # Compute central tendency and spread
    with np.errstate(invalid='ignore'):
        median = np.nanmedian(data_matrix, axis=0)
        mean = np.nanmean(data_matrix, axis=0)
        p25 = np.nanpercentile(data_matrix, 25, axis=0)
        p75 = np.nanpercentile(data_matrix, 75, axis=0)

    if mode == 'raw':
        central = mean
        label_central = 'Ensemble mean'
    elif mode == 'anomaly':
        central = mean
        # subtract ensemble mean from each member
        data_matrix = data_matrix - central[np.newaxis, :]
        median = np.nanmedian(data_matrix, axis=0)
        p25 = np.nanpercentile(data_matrix, 25, axis=0)
        p75 = np.nanpercentile(data_matrix, 75, axis=0)
        central = np.nanmean(data_matrix, axis=0)  # should be ~0
        label_central = 'Ensemble mean anomaly (≈0)'
    elif mode == 'ref':
        if ref_idx is None:
            raise ValueError('ref_idx required for ref mode')
        # reference series
        ref_series = data_matrix[ref_idx, :]
        # anomaly central: subtract reference then compute mean across other members
        data_minus_ref = data_matrix - ref_series[np.newaxis, :]
        data_excl_ref = np.delete(data_minus_ref, ref_idx, axis=0)
        median = np.nanmedian(data_excl_ref, axis=0)
        p25 = np.nanpercentile(data_excl_ref, 25, axis=0)
        p75 = np.nanpercentile(data_excl_ref, 75, axis=0)
        central = np.nanmean(data_excl_ref, axis=0)  # mean anomaly (excl ref)
        label_central = f'Mean anomaly vs {experiment_names[ref_idx]}'
        # also compute absolute ensemble mean excluding reference for bold plotting
        mean_excl_ref_abs = np.nanmean(np.delete(data_matrix, ref_idx, axis=0), axis=0)
    else:
        raise ValueError(f'Unknown mode {mode}')

    # Compute full range (min/max) for plotting the ensemble spread
    if mode == 'ref' and ref_idx is not None:
        pmin = np.nanmin(data_excl_ref, axis=0)
        pmax = np.nanmax(data_excl_ref, axis=0)
    else:
        pmin = np.nanmin(data_matrix, axis=0)
        pmax = np.nanmax(data_matrix, axis=0)

    # Plot fills (full range first - light, IQR next - slightly darker), then dashed mean,
    # then overlay bold lines (black absolute mean / red reference) so they remain visible.
    # Full ensemble range (light shade)
    ax.fill_between(time_grid, pmin, pmax, color='blue', alpha=0.08, label='Ensemble range')
    # IQR (darker shade)
    ax.fill_between(time_grid, p25, p75, color='blue', alpha=0.2, label='IQR (25-75%)')

    if mode == 'ref' and ref_idx is not None:
        # dashed anomaly central (mean anomaly excluding ref)
        # NOTE: do not plot the reference run itself here — values are anomalies
        # relative to that reference and would be a zero line.
        ax.plot(time_grid, central, color='blue', linewidth=1.8, linestyle='--', label=label_central)
    else:
        # default plotting for other modes: dashed ensemble mean and central on top
        try:
            mean_for_plot = np.nanmean(data_matrix, axis=0)
            mean_label = 'Ensemble mean'
        except Exception:
            mean_for_plot = np.nanmean(data_matrix, axis=0)
            mean_label = 'Ensemble mean'
        ax.plot(time_grid, mean_for_plot, color='blue', linewidth=1.5, linestyle='--', marker='o', markersize=3, markevery=max(1, len(time_grid)//50), label=mean_label)
        # central (solid) on top
        ax.plot(time_grid, central, color='black', linewidth=2.2, label=label_central)

    # If plotting anomalies (ref or ensemble), tighten y-limits to anomaly range so small values are visible
    if mode in ('anomaly', 'ref'):
        # compute range from percentiles to avoid extreme outliers
        try:
            y_low = np.nanmin(p25)
            y_high = np.nanmax(p75)
            if np.isnan(y_low) or np.isnan(y_high) or (y_high - y_low) == 0:
                # fallback to global min/max of data_matrix
                y_low = np.nanmin(data_matrix)
                y_high = np.nanmax(data_matrix)
            # expand slightly for breathing room
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
    ax.set_title(f'{variable} Time Series ({n_exp} experiments)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_probability_evolution(all_var_data, valid_time_steps, variable, unit_str, num_experiments, mode='mass', show_bars=True):
    """
    Plot probability evolution for the ensemble using one of three modes:
      - 'mass': probability mass per bin (default) -> y in [0,1]
      - 'cdf' : empirical cumulative distribution functions -> y in [0,1]
      - 'density': kernel density estimates (probability density)
    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    time_steps = sorted(valid_time_steps)

    # Concatenate all values to determine binning / x-range
    all_values = np.concatenate([np.array(all_var_data[ts]) for ts in time_steps])
    data_min, data_max = all_values.min(), all_values.max()
    data_range = data_max - data_min if data_max > data_min else 1.0

    colors = plt.cm.viridis(np.linspace(0, 1, len(time_steps)))

    if mode == 'mass':
        # Use shared bin edges across time steps so bars are comparable
        bins = np.histogram_bin_edges(all_values, bins='auto')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # x grid for KDE integration (fine resolution)
        x_grid = np.linspace(data_min - 0.1*data_range, data_max + 0.1*data_range, 2000)

        global_max_prob = 0.0
        for i, ts in enumerate(time_steps):
            vals = np.array(all_var_data[ts])
            if len(vals) == 0:
                continue
            # Build KDE for this time step and evaluate on x_grid
            kde = gaussian_kde(vals)
            kde_vals = kde(x_grid)

            # Integrate KDE over each bin to obtain probability mass per bin
            probs = np.empty(len(bins)-1)
            for j in range(len(bins)-1):
                mask = (x_grid >= bins[j]) & (x_grid < bins[j+1])
                if mask.any():
                    probs[j] = np.trapz(kde_vals[mask], x_grid[mask])
                else:
                    probs[j] = 0.0
            # numeric rounding may cause small drift; normalize
            if probs.sum() > 0:
                probs /= probs.sum()

            global_max_prob = max(global_max_prob, probs.max() if len(probs) else 0.0)

            # plot bars (semi-transparent) and overlay a smooth KDE line for reference
            if show_bars:
                ax.bar(bin_centers, probs, width=np.diff(bins), align='center',
                       color=colors[i], alpha=0.25, label=f'Year {ts:.0f}')
            # overlay KDE curve scaled for visibility (no legend label to avoid duplicates)
            ax.plot(x_grid, kde_vals * (np.diff(bins).mean()), color=colors[i], linewidth=1.5, alpha=0.8)

        ax.set_ylabel('Probability mass (per bin)')
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(f'{variable}{unit_str}')
        ax.set_title(f'{variable} Probability Distribution Evolution (probability mass per bin, KDE-integrated)')
        # show legend only if bars are drawn (legend per-year can be large)
        if show_bars:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # diagnostic: number of bins and peak probability
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
            if len(vals) >= 3 and np.std(vals) > 1e-10:
                kde = gaussian_kde(vals)
                pdf = kde(x_range)
                ax.plot(x_range, pdf, color=colors[i], linewidth=2.0, label=f'Year {ts:.0f}')
        ax.set_xlabel(f'{variable}{unit_str}')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{variable} PDF Evolution (KDE)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # diagnostic: show peak and area for the last plotted KDE (if any)
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
        var_values = np.array(all_var_data[time_step])
        s = calculate_distribution_stats(var_values)
        print(f"{time_step:<8.0f} {s['n']:<5} {s['mean']:<10.2f} {s['std']:<10.2f} "
              f"{s['median']:<10.2f} {s['q25']:<10.2f} {s['q75']:<10.2f} "
              f"{s['skewness']:<8.2f} {s['kurtosis']:<8.2f}")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    
    # Parse arguments
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
    experiment_names = [exp.strip() for exp in options.experimentList.split(',')]
    time_steps = [float(t.strip()) for t in options.timeSteps.split(',')]
    
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
    
    
    
    # Build and validate file paths
    experiment_files = []
    for exp_name in experiment_names:
        if options.rootDataDir:
            file_path = os.path.join(options.rootDataDir, options.ensembleBaseDir, 
                                     exp_name, options.statsFilename)
        else:
            file_path = os.path.join(options.ensembleBaseDir, exp_name, options.statsFilename)
        
        if not os.path.exists(file_path):
            sys.exit(f"ERROR: File not found: {file_path}")
        
        experiment_files.append(file_path)
    
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

    # If a reference experiment is provided, compute anomalies relative to that run
    if options.anomaly_ref:
        ref_name = options.anomaly_ref
        if ref_name not in experiment_names:
            sys.exit(f"ERROR: reference experiment '{ref_name}' not in supplied experiment list")

        # Find the corresponding file for the reference and interpolate to target times
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

        fig_evolution = plot_probability_evolution(data_for_plots, valid_time_steps, display_variable,
                                                   unit_str, len(experiment_names), mode=mode,
                                                   show_bars=(not options.no_bars))
        if mode == 'mass':
            evo_suffix = '_prob_evolution'
        elif mode == 'cdf':
            evo_suffix = '_cdf_evolution'
        else:
            evo_suffix = '_pdf_evolution'
    
    # Save figures if requested
    if options.plotSave:
        # Save the grid and skewness as before
        fig_grid.savefig(options.plotSave, dpi=300, bbox_inches='tight')
        print(f"Saved: {options.plotSave}")

        skew_name = options.plotSave.replace('.png', '_skewness.png')
        fig_skew.savefig(skew_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {skew_name}")

        # Save the evolution figure with a name depending on the selected mode
        evo_name = options.plotSave.replace('.png', f"{evo_suffix}.png")
        fig_evolution.savefig(evo_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {evo_name}")

    # Optional time series plot using all timestamps from original files
    if options.time_series:
        time_grid, data_matrix = build_time_series_matrix(experiment_files, experiment_names,
                                                         options.variable, scale_vol, change_mode)
        if time_grid.size == 0 or data_matrix.size == 0:
            print('Warning: no time-series data available; skipping time-series plot')
        else:
            # Determine mode for time series
            if options.anomaly_ref:
                ts_mode = 'ref'
                ref_idx = experiment_names.index(options.anomaly_ref)
            elif options.anomaly:
                ts_mode = 'anomaly'
                ref_idx = None
            else:
                ts_mode = 'raw'
                ref_idx = None

            fig_ts = plot_time_series(time_grid, data_matrix, experiment_names,
                                      display_variable, unit_str,
                                      mode=ts_mode, ref_idx=ref_idx)
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
                if options.anomaly_ref:
                    ts_name = ts_name + f"_vs_{options.anomaly_ref}.png"
                elif options.anomaly:
                    ts_name = ts_name + '_anom.png'
                else:
                    ts_name = ts_name + '.png'

                fig_ts.savefig(ts_name, dpi=300, bbox_inches='tight')
                print(f"Saved: {ts_name}")
    
    # Print statistics
    print_statistics_summary(all_var_data, valid_time_steps)
    
    plt.show()

if __name__ == "__main__":
    main()
