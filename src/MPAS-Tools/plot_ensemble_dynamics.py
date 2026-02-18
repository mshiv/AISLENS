#!/usr/bin/env python
"""
Plot ensemble dynamics diagnostics for ice-sheet ensemble statistics.

Produces a set of time-series and diagnostic plots for one or more ensembles
organized under ensemble base directories. Targets the variables
`totalIceVolume` and `volumeAboveFloatation` (VAF) but accepts any variable
present in the per-experiment `globalStats.nc` files.

Features implemented:
- Ensemble mean with 5-95% shaded band and standard deviation curve
- Spread/mean ratio (per ensemble)
- Time-evolving skewness and kurtosis (per ensemble)
- Member dV/dt and d2V/dt2 (per-member -> ensemble mean + spread)
- Event timing diagnostics (onset histograms/violins) using dV/dt and ΔV thresholds
- Fractional uncertainty (4σ / μ_vloss) at checkpoints
- Early-warning indicators: rolling variance and lag-1 autocorrelation on detrended VAF
- Member clustering on normalized ΔV trajectories (hierarchical clustering) and cluster medians
- Quantiles at selected horizons (box/violin plots)
- Normalized loss plots ΔV(t)/ΔV_end comparing ensembles

CLI design mirrors `plot_output_statistics_refactored.py` for ensemble discovery
and color mapping. No `-t` flag — all plots are along the natural time axis.

Usage (example):
python src/MPAS-Tools/plot_ensemble_dynamics.py \
    -r /path/to/data/MALI/diagnostics/ENSEMBLES \
    -b CTRL-SSN,SSP126,SSP585 \
    -e CTRL00,CTRL01,SSP12601,SSP58502 \
    -v totalIceVolume,volumeAboveFloatation \
    --save-prefix diagnostics_totalV

"""
import sys
import os
import glob
import numpy as np
from netCDF4 import Dataset
from optparse import OptionParser
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import detrend
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.colors as mcolors
import pandas as pd

# Default constants
MIN_EXPERIMENTS_THRESHOLD = 3
MIN_MEMBERS_PER_ENSEMBLE = 2  # require at least this many members to compute ensemble stats

# -----------------------------
# Argument parsing
# -----------------------------

def parse_arguments():
    p = OptionParser(description=__doc__)
    p.add_option("-r", "--root", dest="rootDataDir", help="Root data directory path", metavar="PATH")
    p.add_option("-b", "--base", dest="ensembleBaseDir", help="Ensemble base directories (comma-separated)", metavar="DIRS")
    p.add_option("-e", "--experiments", dest="experimentList", help="Comma-separated list of experiment run names (supports 'ENSEMBLE:EXP' or wildcards)")
    p.add_option("-f", "--filename", dest="statsFilename", default="globalStats.nc", help="Per-experiment stats filename")
    p.add_option("-v", "--variable", dest="variables", default="totalIceVolume,volumeAboveFloatation",
                 help="Comma-separated variable names to analyze (default: totalIceVolume,volumeAboveFloatation)")
    p.add_option("--save-prefix", dest="savePrefix", help="Save file prefix for output images (if omitted, show interactively)")
    p.add_option("--rolling-window", dest="rolling_window", type="int", default=25, help="Rolling window length in years for early-warning (default 25)")
    p.add_option("--onset-dvdt-thresh", dest="onset_dvdt", default=None, help="dV/dt threshold for retreat onset (Gt/yr). Can be comma-separated list")
    p.add_option("--onset-dV-drop", dest="onset_drop", default=None, help="ΔV drop thresholds (same units as variable) as comma list for onset detection")
    p.add_option("--checkpoints", dest="checkpoints", default="25,50,100,200", help="Years for fractional uncertainty checkpoints (comma-separated)")
    p.add_option("--clusters", dest="nclusters", type="int", default=3, help="Number of trajectory clusters to form (default 3)")
    opts, args = p.parse_args()

    if not opts.ensembleBaseDir:
        sys.exit("ERROR: Must specify ensemble base directory(s) with -b/--base")
    return opts

# -----------------------------
# Utilities: experiment discovery and color mapping
# -----------------------------

def discover_experiments(root, ensemble_bases, experiment_list, statsFilename):
    """Return tuples (ensemble, exp_name, stats_file, display_name)"""
    specs = []
    ensembles = [e.strip() for e in ensemble_bases.split(',')]
    if not experiment_list:
        # autodiscover
        if not root:
            sys.exit("ERROR: --root required for auto-discovery")
        for ens in ensembles:
            path = os.path.join(root, ens)
            if not os.path.exists(path):
                continue
            for item in os.listdir(path):
                exp_path = os.path.join(path, item)
                if os.path.isdir(exp_path):
                    statsf = os.path.join(exp_path, statsFilename)
                    if os.path.exists(statsf):
                        specs.append((ens, item, statsf, f"{ens}:{item}"))
    else:
        parts = [p.strip() for p in experiment_list.split(',')]
        for spec in parts:
            if ':' in spec:
                ens, exp = spec.split(':', 1)
                if '*' in exp or '?' in exp:
                    search = os.path.join(root, ens, exp) if root else os.path.join(ens, exp)
                    for match in glob.glob(search):
                        if os.path.isdir(match):
                            name = os.path.basename(match)
                            statsf = os.path.join(match, statsFilename)
                            if os.path.exists(statsf):
                                specs.append((ens, name, statsf, f"{ens}:{name}"))
                else:
                    path = os.path.join(root, ens, exp) if root else os.path.join(ens, exp)
                    statsf = os.path.join(path, statsFilename)
                    if os.path.exists(statsf):
                        specs.append((ens, exp, statsf, f"{ens}:{exp}"))
            else:
                exp = spec
                if '*' in exp or '?' in exp:
                    for ens in ensembles:
                        search = os.path.join(root, ens, exp) if root else os.path.join(ens, exp)
                        for match in glob.glob(search):
                            if os.path.isdir(match):
                                name = os.path.basename(match)
                                statsf = os.path.join(match, statsFilename)
                                if os.path.exists(statsf):
                                    specs.append((ens, name, statsf, f"{ens}:{name}"))
                else:
                    found = False
                    for ens in ensembles:
                        path = os.path.join(root, ens, exp) if root else os.path.join(ens, exp)
                        statsf = os.path.join(path, statsFilename)
                        if os.path.exists(statsf):
                            specs.append((ens, exp, statsf, f"{ens}:{exp}"))
                            found = True
                    if not found:
                        print(f"Warning: experiment '{exp}' not found under provided bases", file=sys.stderr)
    if not specs:
        sys.exit("ERROR: No experiments found")
    return specs


def build_ensemble_color_map(experiment_specs):
    ensemble_names = []
    for ens, _, _, _ in experiment_specs:
        if ens not in ensemble_names:
            ensemble_names.append(ens)
    # use public pyplot API (avoids deprecation warning in newer Matplotlib)
    base_cmap = plt.get_cmap('tab20')
    ensemble_to_base_color = {}
    for i, ens in enumerate(ensemble_names):
        if ens.upper().startswith('CTRL'):
            ensemble_to_base_color[ens] = '#1f77b4'
        elif 'SSP126' in ens.upper():
            ensemble_to_base_color[ens] = '#ff7f0e'
        elif 'SSP585' in ens.upper():
            ensemble_to_base_color[ens] = '#d62728'
        else:
            ensemble_to_base_color[ens] = base_cmap(i % base_cmap.N)
    # per-experiment variations
    experiment_to_color = {}
    for ens in ensemble_names:
        members = [d for e, _, _, d in experiment_specs if e == ens]
        n = len(members)
        base = ensemble_to_base_color[ens]
        hsv = mcolors.rgb_to_hsv(mcolors.to_rgb(base))
        variations = []
        if n == 1:
            variations = [base]
        else:
            for j in range(n):
                brightness = 0.5 + 0.5 * (j / max(1, n-1))
                new = hsv.copy()
                new[2] = min(1.0, hsv[2] * brightness)
                variations.append(mcolors.hsv_to_rgb(new))
        for mname, col in zip(members, variations):
            experiment_to_color[mname] = col
    return ensemble_to_base_color, experiment_to_color

# -----------------------------
# Data reading and time-grid
# -----------------------------

def extract_variable_series(fname, variable):
    with Dataset(fname, 'r') as f:
        if variable not in f.variables:
            raise KeyError(f"{variable} not in {fname}")
        yrs = f.variables['daysSinceStart'][:] / 365.0
        yrs = yrs - yrs[0]
        data = np.asarray(f.variables[variable][:])
    return yrs, data


def build_time_grid_matrix(experiment_files, experiment_names, variable):
    # similar approach to refactored script
    series = []
    times = []
    for fpath, dname in zip(experiment_files, experiment_names):
        try:
            yrs, vals = extract_variable_series(fpath, variable)
            times.append(np.asarray(yrs))
            series.append(np.asarray(vals))
        except Exception as e:
            print(f"Warning: failed to read {dname}: {e}", file=sys.stderr)
            times.append(np.array([]))
            series.append(np.array([]))
    if not any(len(t)>0 for t in times):
        return np.array([]), np.empty((0,0))
    all_times = np.unique(np.sort(np.concatenate([t for t in times if len(t)>0])))
    data_matrix = np.full((len(series), len(all_times)), np.nan)
    for i, (t, s) in enumerate(zip(times, series)):
        if len(t) == 0:
            continue
        if len(t) == 1:
            mask = np.isclose(all_times, float(t[0]))
            if mask.any():
                data_matrix[i, mask] = float(s[0])
            continue
        try:
            interp = interp1d(t, s, kind='linear', bounds_error=False, fill_value=np.nan)
            data_matrix[i, :] = interp(all_times)
        except Exception as e:
            print(f"Warning: interp failed for {experiment_names[i]}: {e}", file=sys.stderr)
    return all_times, data_matrix

# -----------------------------
# Diagnostics computations and plotting
# -----------------------------

def derivative_matrix(time_grid, data_matrix):
    # compute first derivative (dV/dt) per row (member) using central differences
    dt = np.gradient(time_grid)
    dvdt = np.gradient(data_matrix, axis=1) / dt[np.newaxis, :]
    # second derivative
    d2 = np.gradient(dvdt, axis=1) / dt[np.newaxis, :]
    return dvdt, d2


def rolling_var_and_ar1(series_1d, window_len_years, time_grid):
    # series_1d is 1D numpy array with NaNs allowed; convert to pandas Series indexed by time
    s = pd.Series(series_1d, index=time_grid)
    # use rolling window with 'window_len_years' centered; because time grid may be irregular, use window by count approximate
    window = max(3, int(window_len_years / np.median(np.diff(time_grid))))
    roll_var = s.rolling(window, center=True, min_periods=3).var()
    # lag-1 autocorrelation using rolling apply
    def lag1(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            return np.nan
        return pd.Series(arr).autocorr(lag=1)
    roll_ar1 = s.rolling(window, center=True, min_periods=3).apply(lag1, raw=False)
    return roll_var.values, roll_ar1.values


def detrend_safe(series_1d, time_grid):
    """Fill NaNs by linear interpolation (edge-filled) then detrend.

    Returns an array the same shape as input. If the entire series is NaN,
    returns an array of NaNs.
    """
    s = np.asarray(series_1d, dtype=float).copy()
    mask = np.isnan(s)
    if mask.all():
        return s
    if mask.any():
        good = ~mask
        try:
            f = interp1d(time_grid[good], s[good], bounds_error=False,
                         fill_value=(s[good][0], s[good][-1]))
            s[mask] = f(time_grid[mask])
        except Exception:
            # fallback: fill with nearest non-nan values
            s[mask] = np.nanmedian(s)
    # now detrend (safe because NaNs have been filled)
    try:
        return detrend(s)
    except Exception:
        return s


def cluster_normalized_trajectories(norm_matrix, nclusters=3):
    # norm_matrix shape (n_members, n_times)
    # replace NaNs with linear interpolation along time for clustering stability
    X = np.array(norm_matrix.copy())
    # Identify rows that are all-NaN and exclude them from clustering
    all_nan_rows = np.isnan(X).all(axis=1)
    valid_idx = np.where(~all_nan_rows)[0]
    if valid_idx.size == 0:
        # nothing to cluster; return zeros (label 0 = unclustered)
        return np.zeros(X.shape[0], dtype=int)

    X_valid = X[valid_idx].copy()
    # interpolate NaNs within valid rows
    for i in range(X_valid.shape[0]):
        row = X_valid[i]
        nans = np.isnan(row)
        if nans.any():
            goodx = np.where(~nans)[0]
            goodv = row[~nans]
            if goodx.size == 0:
                # Shouldn't happen because we filtered all-NaN rows, but guard anyway
                row[:] = np.nanmedian(row)
            else:
                f = interp1d(goodx, goodv, bounds_error=False, fill_value=(goodv[0], goodv[-1]))
                row[nans] = f(np.where(nans)[0])
            X_valid[i] = row

    # If there's only one valid member, assign it to cluster 1
    if X_valid.shape[0] == 1:
        labels_valid = np.array([1], dtype=int)
    else:
        # linkage on Euclidean of time series
        Z = linkage(X_valid, method='ward')
        # If requested clusters exceed members, fcluster will still assign up to nclusters
        labels_valid = fcluster(Z, t=nclusters, criterion='maxclust')

    # Build full label array with 0 for excluded members
    labels_full = np.zeros(X.shape[0], dtype=int)
    labels_full[valid_idx] = labels_valid
    return labels_full

# Plotting helpers

def save_or_show(fig, fname):
    if fname:
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        print(f"Saved {fname}")
        plt.close(fig)
    else:
        fig.show()

# -----------------------------
# Main plotting pipeline
# -----------------------------

def main():
    opts = parse_arguments()
    experiment_specs = discover_experiments(opts.rootDataDir, opts.ensembleBaseDir, opts.experimentList, opts.statsFilename)
    experiment_files = [s[2] for s in experiment_specs]
    experiment_names = [s[3] for s in experiment_specs]

    ensemble_to_base_color, experiment_to_color = build_ensemble_color_map(experiment_specs)

    variables = [v.strip() for v in opts.variables.split(',')]

    # parse onset thresholds and checkpoints
    onset_dvdt = None
    onset_drop = None
    if opts.onset_dvdt:
        onset_dvdt = [float(x) for x in opts.onset_dvdt.split(',')]
    if opts.onset_drop:
        onset_drop = [float(x) for x in opts.onset_drop.split(',')]
    checkpoints = [int(x) for x in opts.checkpoints.split(',')]

    # For each variable produce diagnostics
    for var in variables:
        time_grid, data_matrix = build_time_grid_matrix(experiment_files, experiment_names, var)
        if time_grid.size == 0 or data_matrix.size == 0:
            print(f"No data for {var}; skipping", file=sys.stderr)
            continue

        # Build ensemble map
        ensemble_map = {}
        for idx, dname in enumerate(experiment_names):
            ens = dname.split(':', 1)[0]
            ensemble_map.setdefault(ens, []).append(idx)

        # Basic per-ensemble stats
        for ens, indices in ensemble_map.items():
            if len(indices) < MIN_MEMBERS_PER_ENSEMBLE:
                print(f"Warning: ensemble '{ens}' has only {len(indices)} member(s); need >= {MIN_MEMBERS_PER_ENSEMBLE} to compute ensemble stats. Skipping.", file=sys.stderr)
                continue
            subset = data_matrix[indices, :]
            mean_ts = np.nanmean(subset, axis=0)
            p5 = np.nanpercentile(subset, 5, axis=0)
            p95 = np.nanpercentile(subset, 95, axis=0)
            std_ts = np.nanstd(subset, axis=0)
            # spread/mean
            with np.errstate(invalid='ignore', divide='ignore'):
                spread = np.nanmax(subset, axis=0) - np.nanmin(subset, axis=0)
                mean_safe = mean_ts.copy()
                tiny = 1e-12
                mean_safe[np.abs(mean_safe) < tiny] = np.nan
                spread_ratio = spread / mean_safe

            color = ensemble_to_base_color.get(ens, None)
            # Plot mean, 5-95 band, std, spread/mean ratio
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios':[3,1]}, sharex=True)
            ax0, ax1 = ax
            ax0.fill_between(time_grid, p5, p95, color=color, alpha=0.15)
            ax0.plot(time_grid, mean_ts, color=color, lw=1.8, label=f'{ens} mean')
            ax0.plot(time_grid, std_ts, color=color, lw=1.0, linestyle='--', label=f'{ens} std')
            ax0.set_ylabel(var)
            ax0.legend(loc='best')

            ax1.plot(time_grid, spread_ratio, color=color, lw=1.5)
            ax1.set_ylabel('spread/mean')
            ax1.set_xlabel('Time (years)')
            ax1.grid(True, alpha=0.3)

            title = f"{var} — {ens} mean, 5-95% band, std (lower) and spread/mean (bottom)"
            fig.suptitle(title)
            plt.tight_layout(rect=[0,0,1,0.96])
            outname = None
            if opts.savePrefix:
                outname = f"{opts.savePrefix}_{var}_{ens}_mean_spread.png"
            save_or_show(fig, outname)

            # Skewness / kurtosis
            with np.errstate(invalid='ignore'):
                skew_ts = stats.skew(subset, axis=0, nan_policy='omit')
                kurt_ts = stats.kurtosis(subset, axis=0, nan_policy='omit')
            fig2, ax2 = plt.subplots(1,1,figsize=(10,4))
            ax2.plot(time_grid, skew_ts, '-', color=color, lw=1.5, label='skew')
            ax2.plot(time_grid, kurt_ts, '--', color=color, lw=1.5, label='kurtosis')
            ax2.set_xlabel('Time (years)')
            ax2.set_ylabel('Moment')
            ax2.set_title(f'{var} skewness & kurtosis — {ens}')
            ax2.legend()
            plt.tight_layout()
            outname2 = f"{opts.savePrefix}_{var}_{ens}_sk_kurt.png" if opts.savePrefix else None
            save_or_show(fig2, outname2)

        # Rates and accelerations across members
        dvdt, d2 = derivative_matrix(time_grid, data_matrix)
        # per-ensemble plots
        for ens, indices in ensemble_map.items():
            sub_dv = dvdt[indices, :]
            sub_d2 = d2[indices, :]
            mean_dv = np.nanmean(sub_dv, axis=0)
            p25_dv = np.nanpercentile(sub_dv, 25, axis=0)
            p75_dv = np.nanpercentile(sub_dv, 75, axis=0)

            color = ensemble_to_base_color.get(ens, None)
            fig3, ax3 = plt.subplots(2,1,figsize=(12,8), sharex=True)
            ax3[0].fill_between(time_grid, p25_dv, p75_dv, color=color, alpha=0.2)
            ax3[0].plot(time_grid, mean_dv, color=color, lw=1.6, linestyle='--', label='mean dV/dt')
            ax3[0].set_ylabel('dV/dt')
            ax3[0].legend()

            mean_d2 = np.nanmean(sub_d2, axis=0)
            p25_d2 = np.nanpercentile(sub_d2, 25, axis=0)
            p75_d2 = np.nanpercentile(sub_d2, 75, axis=0)
            ax3[1].fill_between(time_grid, p25_d2, p75_d2, color=color, alpha=0.2)
            ax3[1].plot(time_grid, mean_d2, color=color, lw=1.6, linestyle='--', label='mean d2V/dt2')
            ax3[1].set_ylabel('d2V/dt2')
            ax3[1].set_xlabel('Time (years)')
            ax3[1].legend()

            plt.tight_layout()
            outname3 = f"{opts.savePrefix}_{var}_{ens}_rates.png" if opts.savePrefix else None
            save_or_show(fig3, outname3)

        # Event timing diagnostics (onset distributions)
        # For each member compute ΔV relative to start and dV/dt; find first passage times
        member_onset_times = { 'dvdt': {}, 'drop': {} }
        for i in range(data_matrix.shape[0]):
            traj = data_matrix[i, :]
            dv = np.gradient(traj, time_grid)
            # dvdt thresholds
            if onset_dvdt:
                for thr in onset_dvdt:
                    mask = dv <= -abs(thr)  # negative loss exceeding magnitude
                    inds = np.where(mask)[0]
                    t_on = time_grid[inds[0]] if inds.size else np.nan
                    member_onset_times['dvdt'].setdefault(thr, []).append(t_on)
            # drop thresholds
            if onset_drop:
                delta = traj[0] - traj  # positive when decreased
                for thr in onset_drop:
                    inds = np.where(delta >= abs(thr))[0]
                    t_on = time_grid[inds[0]] if inds.size else np.nan
                    member_onset_times['drop'].setdefault(thr, []).append(t_on)

        # Plot histograms/violins per ensemble for each threshold
        if onset_dvdt:
            for thr in onset_dvdt:
                fig4, ax4 = plt.subplots(1,1,figsize=(10,4))
                labels = []
                data = []
                for ens, indices in ensemble_map.items():
                    times = [member_onset_times['dvdt'][thr][i] for i in indices]
                    data.append([t for t in times if not np.isnan(t)])
                    labels.append(ens)
                # violin plot
                ax4.violinplot(data, showmeans=True)
                ax4.set_xticks(np.arange(1, len(labels)+1))
                ax4.set_xticklabels(labels)
                ax4.set_ylabel('Onset time (years)')
                ax4.set_title(f'Onset times (dV/dt <= -{thr})')
                plt.tight_layout()
                out4 = f"{opts.savePrefix}_{var}_onset_dvdt_{thr}.png" if opts.savePrefix else None
                save_or_show(fig4, out4)

        if onset_drop:
            for thr in onset_drop:
                fig5, ax5 = plt.subplots(1,1,figsize=(10,4))
                labels = []
                data = []
                for ens, indices in ensemble_map.items():
                    times = [member_onset_times['drop'][thr][i] for i in indices]
                    data.append([t for t in times if not np.isnan(t)])
                    labels.append(ens)
                ax5.violinplot(data, showmeans=True)
                ax5.set_xticks(np.arange(1, len(labels)+1))
                ax5.set_xticklabels(labels)
                ax5.set_ylabel('Onset time (years)')
                ax5.set_title(f'Onset times (ΔV >= {thr})')
                plt.tight_layout()
                out5 = f"{opts.savePrefix}_{var}_onset_drop_{thr}.png" if opts.savePrefix else None
                save_or_show(fig5, out5)

        # Fractional uncertainty (4σ/μ_vloss) at checkpoints
        # compute loss relative to start per member
        loss_matrix = data_matrix[ :, : ]
        loss_from_start = (loss_matrix[ :, 0][:, np.newaxis] - loss_matrix)
        # loss at checkpoints
        frac_unc = {}
        for cp in checkpoints:
            # find nearest index
            idx = np.argmin(np.abs(time_grid - cp))
            vals = loss_from_start[:, idx]
            mu = np.nanmean(vals)
            sigma = np.nanstd(vals)
            if np.isnan(mu) or mu == 0:
                frac_unc[cp] = np.nan
            else:
                frac_unc[cp] = 4.0 * sigma / mu
        # plot fractional uncertainty across checkpoints
        fig6, ax6 = plt.subplots(1,1,figsize=(8,4))
        cps = list(frac_unc.keys())
        vals = [frac_unc[c] for c in cps]
        ax6.plot(cps, vals, '-o')
        ax6.set_xlabel('Year')
        ax6.set_ylabel('4σ / μ_vloss')
        ax6.set_title(f'Fractional uncertainty for {var}')
        plt.tight_layout()
        out6 = f"{opts.savePrefix}_{var}_fractional_uncertainty.png" if opts.savePrefix else None
        save_or_show(fig6, out6)

        # Early-warning indicators on detrended VAF (rolling var and AR1)
        # Compute per-member rolling var and AR1 then average across ensemble.
        # Use detrend_safe to fill NaNs before detrending.
        roll_var_list = []
        roll_ar1_list = []
        for i in range(data_matrix.shape[0]):
            series_d = detrend_safe(data_matrix[i, :], time_grid)
            rv, ra = rolling_var_and_ar1(series_d, opts.rolling_window, time_grid)
            roll_var_list.append(rv)
            roll_ar1_list.append(ra)
        # stack and take nanmean along members
        if roll_var_list:
            roll_var_mean = np.nanmean(np.vstack(roll_var_list), axis=0)
        else:
            roll_var_mean = np.full_like(time_grid, np.nan, dtype=float)
        if roll_ar1_list:
            roll_ar1_mean = np.nanmean(np.vstack(roll_ar1_list), axis=0)
        else:
            roll_ar1_mean = np.full_like(time_grid, np.nan, dtype=float)
        fig7, ax7 = plt.subplots(2,1,figsize=(12,8), sharex=True)
        ax7[0].plot(time_grid, roll_var_mean, lw=1.5)
        ax7[0].set_ylabel('Rolling variance')
        ax7[1].plot(time_grid, roll_ar1_mean, lw=1.5)
        ax7[1].set_ylabel('Rolling AR(1)')
        ax7[1].set_xlabel('Time (years)')
        fig7.suptitle(f'Early-warning indicators (detrended {var})')
        plt.tight_layout(rect=[0,0,1,0.96])
        out7 = f"{opts.savePrefix}_{var}_early_warning.png" if opts.savePrefix else None
        save_or_show(fig7, out7)

        # Member clustering on normalized loss trajectories
        final_loss = loss_from_start[:, -1]
        # avoid divide-by-zero
        denom = np.where(np.abs(final_loss) < 1e-12, 1.0, final_loss)
        norm_trajs = loss_from_start / denom[:, np.newaxis]
        labels = cluster_normalized_trajectories(norm_trajs, nclusters=opts.nclusters)
        # plot cluster medians
        fig8, ax8 = plt.subplots(1,1,figsize=(10,5))
        for c in range(1, opts.nclusters+1):
            members = np.where(labels == c)[0]
            if members.size == 0:
                continue
            median = np.nanmedian(norm_trajs[members,:], axis=0)
            ax8.plot(time_grid, median, lw=2, label=f'cluster {c} (n={members.size})')
        ax8.set_xlabel('Time (years)')
        ax8.set_ylabel('Normalized loss ΔV/ΔV_end')
        ax8.legend()
        ax8.set_title(f'Cluster medians for {var}')
        plt.tight_layout()
        out8 = f"{opts.savePrefix}_{var}_clusters.png" if opts.savePrefix else None
        save_or_show(fig8, out8)

        # Quantiles at key horizons (boxplots)
        horizons = checkpoints
        fig9, ax9 = plt.subplots(1,1,figsize=(10,5))
        data_for_boxes = []
        labels = []
        for h in horizons:
            idx = np.argmin(np.abs(time_grid - h))
            vals = data_matrix[:, idx]
            data_for_boxes.append(vals[~np.isnan(vals)])
            labels.append(str(h))
        # Use tick labels explicitly to avoid deprecation of the `labels` kw in boxplot
        ax9.boxplot(data_for_boxes, showmeans=True)
        ax9.set_xticks(np.arange(1, len(labels) + 1))
        ax9.set_xticklabels(labels)
        ax9.set_xlabel('Year')
        ax9.set_ylabel(var)
        ax9.set_title(f'{var} quantiles at horizons')
        plt.tight_layout()
        out9 = f"{opts.savePrefix}_{var}_horizon_boxes.png" if opts.savePrefix else None
        save_or_show(fig9, out9)

        # Normalized loss ΔV(t)/ΔV_end per ensemble overlay
        fig10, ax10 = plt.subplots(1,1,figsize=(12,6))
        for ens, indices in ensemble_map.items():
            members = indices
            final = loss_from_start[members, -1]
            denom = np.where(np.abs(final) < 1e-12, 1.0, final)[:, np.newaxis]
            norm = loss_from_start[members, :] / denom
            mean_norm = np.nanmean(norm, axis=0)
            p25 = np.nanpercentile(norm, 25, axis=0)
            p75 = np.nanpercentile(norm, 75, axis=0)
            color = ensemble_to_base_color.get(ens, None)
            ax10.fill_between(time_grid, p25, p75, color=color, alpha=0.15)
            ax10.plot(time_grid, mean_norm, color=color, lw=1.8, label=f'{ens} mean norm')
        ax10.set_xlabel('Time (years)')
        ax10.set_ylabel('Normalized loss ΔV/ΔV_end')
        ax10.legend()
        ax10.set_title(f'Normalized loss for {var}')
        plt.tight_layout()
        out10 = f"{opts.savePrefix}_{var}_normalized_loss.png" if opts.savePrefix else None
        save_or_show(fig10, out10)

    print('Done')

if __name__ == '__main__':
    main()
