#!/usr/bin/env python
"""
plot_emergence_and_snr.py

Compute and plot signal-to-noise, time-of-emergence (ToE), windowed trends,
and spaghetti+fan charts comparing forced scenarios against a control ensemble.

This script intentionally preserves any physical (noise-induced) drift
present in the provided runs. It uses the same `globalStats`-style files
as `plot_output_statistics_refactored.py` and follows a similar CLI style.


"""
import os
import sys
from optparse import OptionParser
import numpy as np
from netCDF4 import Dataset
from scipy import stats
from scipy.signal import detrend
import matplotlib.pyplot as plt

# Utilities
def read_timeseries_from_stats(fname, varname):
    """Read years (from daysSinceStart) and a 1D variable time series from a globalStats file.

    Returns (years, vals)
    """
    with Dataset(fname, 'r') as f:
        if 'daysSinceStart' not in f.variables:
            raise ValueError(f"Missing daysSinceStart in {fname}")
        years = np.asarray(f.variables['daysSinceStart'][:]) / 365.0
        years = years - years[0]
        if varname not in f.variables:
            raise ValueError(f"Variable '{varname}' not found in {fname}")
        vals = np.asarray(f.variables[varname][:])
        # squeeze to 1D if needed
        vals = np.asarray(vals).squeeze()
        if vals.ndim != 1:
            # attempt to reduce multi-d arrays by spatial averaging if necessary
            vals = np.mean(vals.reshape((vals.shape[0], -1)), axis=1)
    return years, vals

def build_common_time_grid(series_years):
    """Build a sorted unique union time grid and return it.
    series_years: list of 1D arrays
    """
    concat = np.concatenate([s for s in series_years if s is not None and s.size > 0])
    if concat.size == 0:
        return np.array([])
    grid = np.unique(np.sort(concat))
    return grid

def interp_to_grid(years, vals, grid):
    from scipy.interpolate import interp1d
    if len(years) == 0:
        return np.full(grid.shape, np.nan)
    if years.size == 1:
        out = np.full(grid.shape, np.nan)
        idx = np.isclose(grid, float(years[0]))
        out[idx] = float(vals[0])
        return out
    interp = interp1d(years, vals, kind='linear', bounds_error=False, fill_value=np.nan)
    return interp(grid)

def compute_member_anomalies(matrix):
    """Subtract initial value from each member (row-wise).
    matrix: shape (n_members, n_times)
    returns anomalies same shape
    """
    with np.errstate(invalid='ignore'):
        init = matrix[:, 0:1]
        return matrix - init

def sliding_window_indices(grid, center, window_len):
    half = float(window_len) / 2.0
    mask = (grid >= (center - half)) & (grid <= (center + half))
    return mask

def windowed_control_stats(control_matrix, grid, window_len):
    """Compute per-time control std and lag-1 autocorrelation (averaged across members) using centered sliding windows.

    control_matrix shape: (n_control_members, n_times)
    returns: std_ts (n_times), rho_ts (n_times)
    """
    n_times = grid.size
    std_ts = np.full(n_times, np.nan)
    rho_ts = np.full(n_times, np.nan)
    for i, t in enumerate(grid):
        mask = sliding_window_indices(grid, t, window_len)
        if mask.sum() < 3:
            continue
        window_data = control_matrix[:, mask]  # shape (n_members, n_window)
        # flatten across members to estimate variability envelope
        pooled = window_data.flatten()
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size < 3:
            continue
        std_ts[i] = np.nanstd(pooled)
        # compute lag-1 autocorrelation per member, then median
        rhos = []
        for row in window_data:
            row = row[np.isfinite(row)]
            if row.size < 3:
                continue
            # lag-1 autocorr using Pearson on (x[:-1], x[1:])
            x1 = row[:-1]
            x2 = row[1:]
            if x1.size < 2:
                continue
            r = np.corrcoef(x1, x2)[0, 1]
            if np.isfinite(r):
                rhos.append(r)
        if rhos:
            rho_ts[i] = np.median(rhos)
    return std_ts, rho_ts

def compute_snr_ts(forced_matrix, control_matrix, grid, window_len):
    """Compute SNR(t) = |forced_mean(t) - control_mean(t)| / control_std_window(t).

    If multiple forced members are present, forced_mean is ensemble mean across members at each t.
    """
    forced_mean = np.nanmean(forced_matrix, axis=0)
    control_mean = np.nanmean(control_matrix, axis=0)
    control_std, _ = windowed_control_stats(control_matrix, grid, window_len)
    with np.errstate(invalid='ignore', divide='ignore'):
        snr = np.abs(forced_mean - control_mean) / control_std
        snr[~np.isfinite(snr)] = np.nan
    return snr, forced_mean, control_mean, control_std

def detect_toe(snr, grid, threshold=2.0, min_persistence_years=0.0):
    """Detect first time where snr >= threshold and (optionally) persists for min_persistence_years.
    Returns time of emergence or np.nan if none.
    """
    if np.all(np.isnan(snr)):
        return np.nan
    mask = snr >= threshold
    if not mask.any():
        return np.nan
    # If persistence requested, require consecutive times spanning min_persistence_years
    if min_persistence_years > 0 and mask.any():
        # find runs
        starts = np.where(np.diff(np.concatenate(([0], mask.astype(int)))) == 1)[0]
        ends = np.where(np.diff(np.concatenate((mask.astype(int), [0]))) == -1)[0]
        for s, e in zip(starts, ends):
            if grid[e-1] - grid[s] >= min_persistence_years:
                return grid[s]
        return np.nan
    else:
        # first True
        idx = np.argmax(mask)
        return grid[idx]

def moving_block_bootstrap_toe(forced_matrix, control_matrix, grid, window_len, threshold, block_len_years, n_boot=200):
    """Estimate ToE uncertainty by resampling control via moving-block bootstrap.

    For each bootstrap: resample blocks of control members/time to build surrogate control, compute SNR and ToE.
    Returns array of ToE values from bootstrap samples.
    """
    n_control, n_times = control_matrix.shape
    block_len = max(1, int(round(block_len_years / np.median(np.diff(grid)))))
    toe_samples = []
    rng = np.random.default_rng(12345)
    for b in range(n_boot):
        # generate bootstrap control by resampling rows (members) with replacement and circular-block resampling in time
        boot_control = np.full_like(control_matrix, np.nan)
        for i in range(n_control):
            # pick a member at random
            src = rng.integers(0, n_control)
            # pick a start index for block resampling
            out = []
            pos = 0
            while len(out) < n_times:
                start = rng.integers(0, n_times)
                end = min(n_times, start + block_len)
                out.extend(control_matrix[src, start:end].tolist())
            boot_control[i, :] = np.array(out[:n_times])
        snr_b, _, _, _ = compute_snr_ts(forced_matrix, boot_control, grid, window_len)
        toe_b = detect_toe(snr_b, grid, threshold)
        toe_samples.append(toe_b)
    return np.array(toe_samples)

def compute_windowed_trends(matrix, grid, trend_window):
    """Compute linear trends (slope per year) for each member using moving windows.
    Returns: windows_centers, trends_by_window (list of arrays shape (n_members,))
    For simplicity returns distribution of slopes per window (no AR1 se-adjustment here; see notes)
    """
    centers = grid
    n_members = matrix.shape[0]
    trends = []
    for i, t in enumerate(centers):
        mask = sliding_window_indices(grid, t, trend_window)
        if mask.sum() < 3:
            trends.append(np.full(n_members, np.nan))
            continue
        x = grid[mask]
        sls = np.full(n_members, np.nan)
        for m in range(n_members):
            y = matrix[m, mask]
            if np.sum(np.isfinite(y)) < 3:
                continue
            # fit linear slope using np.polyfit
            try:
                ok = np.isfinite(y)
                p = np.polyfit(x[ok], y[ok], 1)
                sls[m] = p[0]
            except Exception:
                sls[m] = np.nan
        trends.append(sls)
    return centers, np.array(trends)

def spaghetti_and_fan(grid, matrix, label, ax=None, color='C0', percentiles=(5,95), alpha_fill=0.25):
    """Plot member spaghetti and shaded percentile fan plus ensemble mean.
    matrix shape (n_members, n_times)
    returns fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,5))
    else:
        fig = ax.get_figure()
    n_members = matrix.shape[0]
    # plot members lightly
    for m in range(n_members):
        ax.plot(grid, matrix[m,:], color=color, alpha=0.25, linewidth=1)
    mean = np.nanmean(matrix, axis=0)
    p_low = np.nanpercentile(matrix, percentiles[0], axis=0)
    p_high = np.nanpercentile(matrix, percentiles[1], axis=0)
    ax.fill_between(grid, p_low, p_high, color=color, alpha=alpha_fill)
    ax.plot(grid, mean, color=color, linewidth=2.0, label=f"{label} mean")
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    return fig, ax

def variance_decomposition_time(forced_matrix, grid):
    """Compute within-member variance (across time per member then averaged) and between-member variance at each time.
    Returns time_series of within_var (mean across members of their variance in a local window) and between_var (var across members at time t).
    Simpler: compute between_var(t)=var across members at t; within_var(t)=mean of member-centered short-window variance (3-yr) around t.
    """
    n_members, n_times = forced_matrix.shape
    between = np.full(n_times, np.nan)
    within = np.full(n_times, np.nan)
    for i in range(n_times):
        col = forced_matrix[:, i]
        between[i] = np.nanvar(col)
        # small window for within variance estimate
        w = 3
        left = max(0, i - w)
        right = min(n_times, i + w + 1)
        member_vars = np.nanvar(forced_matrix[:, left:right], axis=1)
        within[i] = np.nanmean(member_vars)
    return within, between

def parse_args():
    p = OptionParser()
    p.add_option('-r','--root', dest='root', help='root data directory (contains ensemble base dirs)', default=None)
    p.add_option('-b','--base', dest='bases', help='comma-separated ensemble base dirs (relative to root or absolute)', metavar='BASE1,BASE2')
    p.add_option('-e','--experiments', dest='experiments', help='comma-separated list of experiment subdirs or names; if omitted auto-discovers under each base', default=None)
    p.add_option('-v','--variable', dest='variable', help='variable name in globalStats files (default volumeAboveFloatation)', default='volumeAboveFloatation')
    p.add_option('--control-name', dest='control_name', help='identifier for control ensemble base (default CTRL)', default='CTRL')
    p.add_option('--window', dest='window', help='window length in years for variability/SNR (default 20)', default=20.0, type='float')
    p.add_option('--trend-window', dest='trend_window', help='window length in years for windowed trends (default 40)', default=40.0, type='float')
    p.add_option('--snr-threshold', dest='snr_thresh', help='SNR threshold for ToE (default 2.0)', default=2.0, type='float')
    p.add_option('--block-len', dest='block_len', help='block length in years for moving-block bootstrap (default=window)', default=None, type='float')
    p.add_option('--bootstrap', dest='bootstrap', help='number of bootstrap draws for ToE uncertainty (default 200)', default=200, type='int')
    p.add_option('--plot-save', dest='plot_save', help='base filename to save plots (PNG will be appended)', default=None)
    p.add_option('--preserve-drift', dest='preserve_drift', action='store_true', default=True, help='Preserve any drift in the runs (default True)')
    p.add_option('--no-show', dest='no_show', action='store_true', default=False, help='Do not call plt.show()')
    opts, args = p.parse_args()
    return opts

def main():
    opts = parse_args()
    if not opts.bases:
        sys.exit('ERROR: must provide -b/--base with comma-separated ensemble base names')
    base_list = [b.strip() for b in opts.bases.split(',') if b.strip()]
    # discover experiments under each base
    experiment_specs = []  # tuples (ensemble_base, exp_dir, file_path, display_name)
    for base in base_list:
        base_path = os.path.join(opts.root, base) if opts.root else base
        if not os.path.isdir(base_path):
            print(f'Warning: base dir not found: {base_path}', file=sys.stderr)
            continue
        for item in sorted(os.listdir(base_path)):
            exp_dir = os.path.join(base_path, item)
            if not os.path.isdir(exp_dir):
                continue
            stats_file = os.path.join(exp_dir, 'globalStats.nc')
            if not os.path.exists(stats_file):
                continue
            display = f"{base}:{item}"
            experiment_specs.append((base, item, stats_file, display))

    if len(experiment_specs) == 0:
        sys.exit('ERROR: no experiments found under provided bases')

    # group by ensemble base
    ensembles = {}
    for ens, exp, fpath, disp in experiment_specs:
        ensembles.setdefault(ens, []).append((exp, fpath, disp))

    # extract timeseries for all experiments
    all_years = []
    disp_names = []
    file_paths = []
    series_list = []
    for ens, exp, fpath, disp in experiment_specs:
        try:
            yrs, vals = read_timeseries_from_stats(fpath, opts.variable)
        except Exception as e:
            print(f'Warning: failed to read {fpath}: {e}', file=sys.stderr)
            continue
        all_years.append(yrs)
        disp_names.append(disp)
        file_paths.append(fpath)
        series_list.append(vals)

    grid = build_common_time_grid(all_years)
    if grid.size == 0:
        sys.exit('ERROR: empty time grid')

    # interpolate all series to grid
    matrices = {}
    for ens in ensembles:
        members = ensembles[ens]
        arr = []
        for (exp, fpath, disp) in members:
            try:
                yrs, vals = read_timeseries_from_stats(fpath, opts.variable)
            except Exception:
                vals = np.full(grid.shape, np.nan)
            arr.append(interp_to_grid(yrs, vals, grid))
        matrices[ens] = np.vstack(arr)

    # identify control ensemble
    control_key = None
    for ens in matrices:
        if ens.upper().startswith(opts.control_name.upper()):
            control_key = ens
            break
    if control_key is None:
        # pick first ensemble as control if not found
        control_key = list(matrices.keys())[0]
        print(f'Note: control ensemble {opts.control_name} not found; using {control_key} as control', file=sys.stderr)

    # compute anomalies relative to each member's initial value
    for ens in list(matrices.keys()):
        matrices[ens] = compute_member_anomalies(matrices[ens])

    # Prepare output figures
    figs = []

    # Plot spaghetti + fan for each ensemble, overlay control envelope
    control_mat = matrices[control_key]
    fig_all, ax_all = plt.subplots(1,1,figsize=(12,6))
    for i, ens in enumerate(sorted(matrices.keys())):
        mat = matrices[ens]
        color = f'C{i%10}'
        label = ens
        # for control, use dashed mean
        for m in range(mat.shape[0]):
            ax_all.plot(grid, mat[m,:], color=color, alpha=0.15)
        mean = np.nanmean(mat, axis=0)
        p5 = np.nanpercentile(mat, 5, axis=0)
        p95 = np.nanpercentile(mat, 95, axis=0)
        ax_all.fill_between(grid, p5, p95, color=color, alpha=0.06)
        ax_all.plot(grid, mean, color=color, linewidth=2.0, label=f'{ens} mean')
    ax_all.set_title(f'{opts.variable} — member trajectories and ensemble fans (anomalies)')
    ax_all.set_xlabel('Time (years)')
    ax_all.set_ylabel(f'{opts.variable} anomaly')
    ax_all.legend(loc='best')
    ax_all.grid(True, alpha=0.3)
    figs.append((fig_all, 'spaghetti_fan.png'))

    # Compute SNR for each non-control ensemble vs control
    for i, ens in enumerate(sorted(matrices.keys())):
        if ens == control_key:
            continue
        forced_mat = matrices[ens]
        snr, forced_mean, control_mean, control_std = compute_snr_ts(forced_mat, control_mat, grid, opts.window)

        fig, ax = plt.subplots(1,1,figsize=(10,4))
        ax.plot(grid, snr, color='C1', linewidth=2.0, label=f'SNR (|{ens} - {control_key}| / control σ)')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('SNR')
        ax.axhline(opts.snr_thresh, color='k', linestyle='--', label=f'Threshold={opts.snr_thresh}')
        toe = detect_toe(snr, grid, opts.snr_thresh)
        if np.isfinite(toe):
            ax.axvline(toe, color='red', linestyle=':', label=f'ToE={toe:.1f}y')

        # bootstrap ToE uncertainty if requested
        if opts.bootstrap and opts.bootstrap > 0:
            block_len = opts.block_len if opts.block_len else opts.window
            toes = moving_block_bootstrap_toe(forced_mat, control_mat, grid, opts.window, opts.snr_thresh, block_len, n_boot=opts.bootstrap)
            toes = toes[np.isfinite(toes)]
            if toes.size > 0:
                ci_lo = np.nanpercentile(toes, 2.5)
                ci_hi = np.nanpercentile(toes, 97.5)
                ax.fill_betweenx([0, np.nanmax(snr[np.isfinite(snr)]) if np.any(np.isfinite(snr)) else 1.0], ci_lo, ci_hi, color='red', alpha=0.08, label=f'ToE 95% CI [{ci_lo:.1f},{ci_hi:.1f}]')

        ax.set_title(f'SNR over time: {ens} vs {control_key}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        figs.append((fig, f'snr_{ens}_vs_{control_key}.png'))

    # Windowed trend distributions: compare forced vs control
    for i, ens in enumerate(sorted(matrices.keys())):
        mat = matrices[ens]
        centers, trends = compute_windowed_trends(mat, grid, opts.trend_window)
        # plot distribution of slopes at selected windows (e.g., end, mid)
        # For brevity show three windows: early, mid, late
        idxs = [int(0.25*len(centers)), int(0.5*len(centers)), int(0.9*len(centers))]
        fig, axes = plt.subplots(1, len(idxs), figsize=(4*len(idxs),4))
        for j, idx in enumerate(idxs):
            ax = axes[j]
            data = trends[idx]
            data = data[np.isfinite(data)]
            if data.size == 0:
                ax.text(0.5,0.5,'No data', ha='center')
                continue
            ax.hist(data*100.0, bins=15, density=False, color=f'C{j}', alpha=0.7)
            ax.axvline(np.median(data*100.0), color='k', linestyle='--', label='median')
            ax.set_title(f'Window center {centers[idx]:.0f}y')
            ax.set_xlabel('Trend (units/year *100)')
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'Windowed trend distributions ({ens}) — window {opts.trend_window} years')
        figs.append((fig, f'trends_{ens}.png'))

    # Variance decomposition (within vs between) for a chosen ensemble (e.g., forced sum across members)
    # If multiple ensembles exist, show within/between for each
    figvar, axvar = plt.subplots(1,1,figsize=(10,4))
    for i, ens in enumerate(sorted(matrices.keys())):
        mat = matrices[ens]
        within, between = variance_decomposition_time(mat, grid)
        axvar.plot(grid, between, label=f'{ens} between var', color=f'C{i}')
        axvar.plot(grid, within, linestyle='--', color=f'C{i}', alpha=0.7, label=f'{ens} within var')
    axvar.set_yscale('log')
    axvar.set_xlabel('Time (years)')
    axvar.set_ylabel('Variance (log scale)')
    axvar.set_title('Variance decomposition: between vs within')
    axvar.legend(loc='best')
    axvar.grid(True, alpha=0.3)
    figs.append((figvar, 'variance_decomposition.png'))

    # Save plots if requested
    if opts.plot_save:
        for fig, fname in figs:
            out = opts.plot_save.replace('.png','') + '_' + fname
            try:
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'Saved: {out}')
            except Exception as e:
                print(f'Warning: failed to save {out}: {e}', file=sys.stderr)

    if not opts.no_show:
        plt.show()

if __name__ == '__main__':
    main()
