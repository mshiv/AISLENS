#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone ensemble map plotting for MALI with ratio output
This script reads ensemble-stats NetCDF files (one file per year) and, for a
requested variable (typically `thickness`) computes the ratio:

    ratio = <variable>_range / dhdt_mean

and writes a PNG and a compact NetCDF with the ratio per year. If the
stats files don't contain the required variables the script will warn and
skip the ratio for that year.

d assumes the ensembleStats_*.nc files have mesh
variables (`xCell`, `yCell`, `dcEdge`) and the stats variables
(e.g. `thickness_range`, `dhdt_mean`).

Usage example:

python3 src/MPAS-Tools/plot_ensemble_maps_ratio.py \
  --stats_files /path/to/ensembleStats_2025.nc \
  --years 2025 \
  --variables thickness \
  --run_dirs /path/to/run1/output,/path/to/run2/output \
  --run_names run1,run2 \
  --save_base /path/to/save/dir

"""

import os
import sys
import argparse
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm
import matplotlib.cm as cm

# grounding/mask bit values (match other plotting script)
groundingLineValue = 256
initialExtentValue = 1


def safe_flatten(var):
    """Return a 1-D numpy array for typical (1,nCells) or (nCells,) variables."""
    a = np.array(var)
    if a.ndim == 2 and a.shape[0] == 1:
        return a[0, :]
    if a.ndim == 1:
        return a
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    return a.ravel()


def _get_colormap_by_name(name):
    """Return a matplotlib colormap. Supports `cmocean:NAME` prefix.

    If `name` starts with `cmocean:`, try to import `cmocean.cm` and return
    the named colormap. Falls back to `plt.get_cmap` on any failure.
    """
    if not name:
        return plt.get_cmap(None)
    if isinstance(name, str) and name.startswith('cmocean:'):
        cmap_key = name.split(':', 1)[1]
        try:
            import cmocean
            cmap = getattr(cmocean.cm, cmap_key, None)
            if cmap is not None:
                return cmap
            else:
                print(f"  WARNING: cmocean colormap '{cmap_key}' not found; falling back to matplotlib cmap")
        except Exception as e:
            print(f"  WARNING: cmocean import failed ({e}); falling back to matplotlib cmap")
    try:
        return plt.get_cmap(name)
    except Exception:
        print(f"  WARNING: matplotlib cmap '{name}' not found; using default cmap")
        return plt.get_cmap(None)


def _make_gl_colors(user_colors, n):
    """Return a list of `n` colors for grounding-line overlays.

    Behavior:
    - If `user_colors` is provided (list of strings), assign those colors to the
      last grounding lines (so a single color will apply to the last GL plotted).
    - Pad the beginning with tab10-derived colors so the total length equals `n`.
    - If user supplied more colors than `n`, take the last `n` entries.
    """
    if user_colors is None:
        # simply use tab10 for all runs
        cols = cm.tab10(np.linspace(0, 1, n))
        return [tuple(c) for c in cols]

    user_list = list(user_colors)
    # number of padding colors needed at the front
    pad_count = max(0, n - len(user_list))
    pad = cm.tab10(np.linspace(0, 1, max(1, pad_count)))
    pad_colors = [tuple(c) for c in pad][:pad_count]
    cols = pad_colors + user_list
    # if user provided more colors than n, keep the last n (so user's last colors map to last runs)
    if len(cols) > n:
        cols = cols[-n:]
    # final safety: if still short, pad at end with tab10 cycle
    if len(cols) < n:
        extra = cm.tab10(np.linspace(0, 1, n - len(cols)))
        cols += [tuple(c) for c in extra]
    return cols


parser = argparse.ArgumentParser(description="Plot ensemble ratio: numerator / denominator (flexible)")
parser.add_argument("--stats_files", required=True,
                    help="Comma-separated ensemble stats NetCDF files (one per year).")
parser.add_argument("--years", required=True, help="Comma-separated years.")
parser.add_argument("--variables", required=False, help="(DEPRECATED) use --numerators; comma-separated variables (e.g. thickness)")
parser.add_argument("--numerators", required=False, help="Comma-separated numerator variables. Accepts exact variable names existing in the stats files (e.g. thickness_range, dhdt_std) or base names (e.g. thickness) which will try suffixes like '_range' first.")
parser.add_argument("--denominators", required=False, help="Comma-separated denominator variables (e.g. dhdt_mean). If a single name is given it will be used for all numerators. If omitted defaults to 'dhdt_mean'.")
parser.add_argument("--run_dirs", required=True, help="Comma-separated run output directories (for grounding line overlays)")
parser.add_argument("--run_names", required=True, help="Comma-separated run names (for legend)")
parser.add_argument("--save_base", required=False, default=None, help="Directory to save PNG and NetCDF outputs")
parser.add_argument("--gl_linewidth", required=False, default=0.7, type=float, help="Grounding line linewidth")
parser.add_argument("--cmap", required=False, default="viridis", help="Colormap for ratio plots")
parser.add_argument("--diverging", action="store_true", help="Use a diverging colormap centered at zero (red negative, blue positive).")
parser.add_argument("--diverging_cmap", required=False, default="RdBu", help="Diverging colormap name to use when --diverging is set (default: RdBu).")
parser.add_argument("--cbar_vmin", required=False, type=float, default=None, help="Hard-set colorbar minimum value (overrides automatic lower quantile).")
parser.add_argument("--cbar_vmax", required=False, type=float, default=None, help="Hard-set colorbar maximum value (overrides automatic upper quantile).")
parser.add_argument("--log_cbar", action="store_true", help="Use log scale for the colorbar (requires positive data).")
parser.add_argument("--gl_colors", required=False, default=None,
                    help="Optional comma-separated grounding-line colors for runs (overrides default colormap)."
                    )
parser.add_argument("--mask_ratio_below", required=False, type=float, default=None,
                    help="Mask (set to NaN) ratio values below this absolute threshold before plotting."
                    )
parser.add_argument("--mask_ratio_quantile", required=False, type=float, default=5.0,
                    help="Mask ratio values below this percentile (0-100). Default 5.0 masks values below the 5th percentile.")

# Optional initial thickness file used to compute absolute thickness change
parser.add_argument("--initial_thickness_file", required=False, default=None,
                    help="Path to initial output file containing 'thickness' (e.g. output_flux_all_timesteps_2000.nc). Required when using denominator 'abs_thickness_change'.")
parser.add_argument("--initial_thickness_time_index", required=False, type=int, default=0,
                    help="Time index to use from initial thickness file (default 0).")
args = parser.parse_args()
stats_files = args.stats_files.split(',')
years = [int(y) for y in args.years.split(',')]
# support legacy --variables argument
if args.numerators:
    variables = args.numerators.split(',')
elif args.variables:
    variables = args.variables.split(',')
else:
    raise SystemExit('ERROR: must provide --numerators (or legacy --variables)')
denominators = args.denominators.split(',') if args.denominators else None
run_dirs = args.run_dirs.split(',')
run_names = args.run_names.split(',')
save_base = args.save_base
gl_linewidth = args.gl_linewidth
cmap_name = args.cmap
use_diverging = args.diverging
diverging_cmap = args.diverging_cmap
cbar_vmin = args.cbar_vmin
cbar_vmax = args.cbar_vmax
use_log_cbar = args.log_cbar
initial_thickness_file = args.initial_thickness_file
initial_thickness_time_index = args.initial_thickness_time_index
gl_colors_arg = args.gl_colors.split(',') if args.gl_colors else None
mask_ratio_below = args.mask_ratio_below
mask_ratio_quantile = args.mask_ratio_quantile

if save_base:
    os.makedirs(save_base, exist_ok=True)

print(f"Plotting ratio for years={years} variables={variables}")

# simple grounding-line loader (only used for overlay aesthetic)

def load_grounding_for_year(run_dirs, run_names, year):
    entries = []
    # determine colors: prefer user-provided list, else fall back to tab10
    colors = _make_gl_colors(gl_colors_arg, len(run_dirs))
    for i, rd in enumerate(run_dirs):
        if not rd:
            continue
        mesh_file = os.path.join(rd, f"output_flux_all_timesteps_{year}_tAvg.nc")
        if not os.path.exists(mesh_file):
            continue
        try:
            m = Dataset(mesh_file)
            xCell = safe_flatten(m.variables['xCell'][:])
            yCell = safe_flatten(m.variables['yCell'][:])
            cellMask = None
            gl_mask = None
            extent_mask = None
            if 'cellMask' in m.variables:
                cmask = m.variables['cellMask'][:]
                if cmask.ndim > 1:
                    cmask = cmask[0]
                cellMask = cmask
                # extract grounding-line bit and initial extent bit
                try:
                    gl_mask = (cmask & groundingLineValue) // groundingLineValue
                    extent_mask = (cmask & initialExtentValue) // initialExtentValue
                except Exception:
                    gl_mask = None
                    extent_mask = None
            m.close()
            entries.append({'x': xCell, 'y': yCell, 'cellMask': cellMask, 'gl_mask': gl_mask, 'extent_mask': extent_mask, 'color': colors[i], 'name': run_names[i] if i < len(run_names) else f'run{i}'})
        except Exception as e:
            print(f"Warning: cannot load mesh {mesh_file}: {e}")
    return entries


for variable in variables:
    # variable here is a requested numerator token (could be 'thickness', 'thickness_range', 'dhdt_std', etc.)
    for stats_file, year in zip(stats_files, years):
        print(f"Processing {stats_file} (year {year}) for numerator={variable}")
        if not os.path.exists(stats_file):
            print(f"  WARNING: stats file not found: {stats_file}")
            continue
        try:
            f = Dataset(stats_file, 'r')
        except Exception as e:
            print(f"  ERROR opening {stats_file}: {e}")
            continue
        # Determine numerator variable name to read from file
        num_token = variable
        # If user provided an exact variable name present in file, use it
        if num_token in f.variables:
            num_name = num_token
        else:
            # try common derived names
            candidates = [f"{num_token}_range", f"{num_token}_std", f"{num_token}_mean"]
            found = None
            for c in candidates:
                if c in f.variables:
                    found = c
                    break
            if found is None:
                print(f"  WARNING: numerator '{num_token}' not found (tried { [num_token]+candidates }). Skipping.")
                f.close()
                continue
            num_name = found

        # Determine denominator variable (broadcasting rules)
        if denominators is None:
            den_token = 'dhdt_mean'
        else:
            if len(denominators) == 1:
                den_token = denominators[0]
            elif len(denominators) == len(variables):
                # match by position of variable in original variables list
                den_token = denominators[variables.index(variable)]
            else:
                print(f"  ERROR: number of denominators ({len(denominators)}) must be 1 or equal to numerators ({len(variables)}).")
                f.close()
                raise SystemExit(1)

        # Resolve or compute denominator. Support special token 'abs_thickness_change'
        if den_token == 'abs_thickness_change':
            # Need an initial thickness file path
            if not initial_thickness_file:
                print("  ERROR: denominator 'abs_thickness_change' requires --initial_thickness_file. Skipping.")
                f.close()
                continue
            # Read numerator and thickness_mean from stats file
            try:
                arr_num = safe_flatten(f.variables[num_name][:])
            except Exception as e:
                print(f"  ERROR reading numerator array: {e}")
                f.close()
                continue
            # thickness_mean should be present in ensemble stats
            if 'thickness_mean' not in f.variables:
                print("  WARNING: 'thickness_mean' not found in stats file; cannot compute abs_thickness_change. Skipping.")
                f.close()
                continue
            try:
                arr_thickness_mean = safe_flatten(f.variables['thickness_mean'][:])
            except Exception as e:
                print(f"  ERROR reading thickness_mean from stats: {e}")
                f.close()
                continue
            # Load initial thickness from provided file
            try:
                initf = Dataset(initial_thickness_file)
                init_thick = initf.variables['thickness'][:]
                initf.close()
            except Exception as e:
                print(f"  ERROR opening initial thickness file {initial_thickness_file}: {e}")
                f.close()
                continue
            # select time index if necessary
            try:
                if getattr(init_thick, 'ndim', 0) > 1:
                    if initial_thickness_time_index >= init_thick.shape[0]:
                        print(f"  ERROR: initial_thickness_time_index {initial_thickness_time_index} out of range for file {initial_thickness_file}. Skipping.")
                        f.close()
                        continue
                    init_slice = safe_flatten(init_thick[initial_thickness_time_index, :])
                else:
                    init_slice = safe_flatten(init_thick)
            except Exception as e:
                print(f"  ERROR processing initial thickness array: {e}")
                f.close()
                continue
            # compute absolute change per cell
            try:
                arr_den = np.abs(arr_thickness_mean - init_slice)
            except Exception as e:
                print(f"  ERROR computing abs_thickness_change: {e}")
                f.close()
                continue
            den_name = 'abs_thickness_change'
        else:
            # Resolve denom name similar to numerator
            if den_token in f.variables:
                den_name = den_token
            else:
                candidates = [f"{den_token}_mean", f"{den_token}_range", f"{den_token}_std"]
                found = None
                for c in candidates:
                    if c in f.variables:
                        found = c
                        break
                if found is None:
                    print(f"  WARNING: denominator '{den_token}' not found (tried { [den_token]+candidates }). Skipping.")
                    f.close()
                    continue
                den_name = found

            try:
                arr_num = safe_flatten(f.variables[num_name][:])
                arr_den = safe_flatten(f.variables[den_name][:])
            except Exception as e:
                print(f"  ERROR reading arrays: {e}")
                f.close()
                continue

        # mesh for plotting
        try:
            xCell = safe_flatten(f.variables['xCell'][:])
            yCell = safe_flatten(f.variables['yCell'][:])
            dcEdge = safe_flatten(f.variables['dcEdge'][:]) if 'dcEdge' in f.variables else None
        except Exception as e:
            print(f"  ERROR reading mesh from {stats_file}: {e}")
            f.close()
            continue

        f.close()

        # make sure arrays align
        if arr_num.shape != arr_den.shape:
            # attempt simple broadcast-friendly trimming to min length
            nmin = min(arr_num.size, arr_den.size)
            print(f"  WARNING: arr sizes differ (num={arr_num.size}, den={arr_den.size}). Trimming to {nmin}.")
            arr_num = arr_num[:nmin]
            arr_den = arr_den[:nmin]
            xCell = xCell[:nmin]
            yCell = yCell[:nmin]

        # compute ratio, avoid division by zero
        tiny = 1e-12
        den_safe = arr_den.astype(float)
        den_safe[np.abs(den_safe) < tiny] = np.nan
        ratio = arr_num / den_safe

        # Special handling when denominator is absolute thickness change:
        # Mask out small denominators (near-zero absolute change) so the
        # uncertainty-to-signal map highlights regions where variability
        # legitimately outpaces the local signal. We choose a conservative
        # threshold at the 5th positive-percentile of abs_thickness_change
        # (fall back to a tiny value if no positive denominators).
        if den_token == 'abs_thickness_change':
            try:
                pos_den = arr_den[np.isfinite(arr_den) & (arr_den > 0)]
                if pos_den.size > 0:
                    denom_thresh = float(np.nanquantile(pos_den, 0.05))
                else:
                    denom_thresh = 1e-6
                mask_small = (arr_den < denom_thresh) | (~np.isfinite(arr_den))
                n_masked = int(np.count_nonzero(mask_small))
                print(f"  INFO: masking {n_masked}/{arr_den.size} cells with abs_thickness_change < {denom_thresh:.3e}")
                # hide masked cells in ratio
                ratio = ratio.astype(float)
                ratio[mask_small] = np.nan
            except Exception as e:
                print(f"  WARNING: could not apply small-denominator masking for abs_thickness_change: {e}")
        # Quantile-based masking: mask values below the requested percentile
        if (mask_ratio_quantile is not None):
            try:
                q = float(mask_ratio_quantile)
                if (q < 0.0) or (q > 100.0):
                    print(f"  WARNING: --mask_ratio_quantile {q} out of range (0-100); skipping quantile masking")
                else:
                    qfrac = q / 100.0
                    if use_diverging:
                        vals = np.abs(ratio[np.isfinite(ratio)])
                    elif use_log:
                        vals = ratio[np.isfinite(ratio) & (ratio > 0)]
                    else:
                        vals = ratio[np.isfinite(ratio)]
                    if vals.size > 0:
                        thresh_q = float(np.nanquantile(vals, qfrac))
                    else:
                        thresh_q = 0.0
                    if use_diverging:
                        mask_q = (np.abs(ratio) < thresh_q) | (~np.isfinite(ratio))
                    else:
                        mask_q = (ratio < thresh_q) | (~np.isfinite(ratio))
                    n_masked_q = int(np.count_nonzero(mask_q))
                    print(f"  INFO: masking {n_masked_q}/{ratio.size} cells below {q}th percentile (thresh={thresh_q:.3e})")
                    ratio = ratio.astype(float)
                    ratio[mask_q] = np.nan
            except Exception as e:
                print(f"  WARNING: failed to apply --mask_ratio_quantile: {e}")

        # Additional user-requested masking: mask ratio values below absolute threshold
        if (mask_ratio_below is not None):
            try:
                thresh = float(mask_ratio_below)
                if use_diverging:
                    mask_low = (np.abs(ratio) < thresh) | (~np.isfinite(ratio))
                else:
                    # for non-diverging maps (and log maps) we mask values strictly less than thresh
                    mask_low = (ratio < thresh) | (~np.isfinite(ratio))
                n_masked2 = int(np.count_nonzero(mask_low))
                print(f"  INFO: masking {n_masked2}/{ratio.size} cells with ratio < {thresh}")
                ratio = ratio.astype(float)
                ratio[mask_low] = np.nan
            except Exception as e:
                print(f"  WARNING: failed to apply --mask_ratio_below: {e}")

        # prepare triangulation
        triang = tri.Triangulation(xCell, yCell)
        if dcEdge is not None:
            maxDist = np.max(dcEdge) * 2.0
            triMask = np.zeros(len(triang.triangles), dtype=bool)
            for t in range(len(triang.triangles)):
                triIdx = triang.triangles[t, :]
                # compute simple edge length checks
                if np.max([np.hypot(xCell[triIdx[0]]-xCell[triIdx[1]], yCell[triIdx[0]]-yCell[triIdx[1]]),
                           np.hypot(xCell[triIdx[1]]-xCell[triIdx[2]], yCell[triIdx[1]]-yCell[triIdx[2]]),
                           np.hypot(xCell[triIdx[0]]-xCell[triIdx[2]], yCell[triIdx[0]]-yCell[triIdx[2]])]) > maxDist:
                    triMask[t] = True
            triang.set_mask(triMask)

        # plotting
        fig, ax = plt.subplots(figsize=(7, 7))
        # choose colormap and normalization
        finite_mask = np.isfinite(ratio)
        if not np.any(finite_mask):
            print(f"  INFO: All ratio values are NaN for {stats_file}. Skipping.")
            plt.close(fig)
            continue

        # If user requested a log-scaled colorbar, validate and prepare positive-only values
        use_log = bool(use_log_cbar)
        if use_log and use_diverging:
            print("  WARNING: --log_cbar requested together with --diverging. Ignoring log scale and using diverging colormap.")
            use_log = False

        if use_log:
            # consider only positive finite values for log scaling
            pos_mask = np.isfinite(ratio) & (ratio > 0)
            if not np.any(pos_mask):
                print(f"  WARNING: --log_cbar requested but no positive ratio values present in {stats_file}; falling back to linear scale.")
                use_log = False

        # default quantile-based limits (use positive-only quantiles for log case)
        if use_log:
            qlow = np.nanquantile(ratio[pos_mask], 0.01)
            qhigh = np.nanquantile(ratio[pos_mask], 0.99)
        else:
            qlow = np.nanquantile(ratio, 0.01)
            qhigh = np.nanquantile(ratio, 0.99)

        # apply user overrides if provided
        vmin = qlow if cbar_vmin is None else cbar_vmin
        vmax = qhigh if cbar_vmax is None else cbar_vmax

        # sanity check for reversed bounds
        if (cbar_vmin is not None) and (cbar_vmax is not None) and (cbar_vmin >= cbar_vmax):
            print(f"  WARNING: --cbar_vmin ({cbar_vmin}) >= --cbar_vmax ({cbar_vmax}); falling back to quantile limits")
            vmin, vmax = qlow, qhigh

        if use_diverging:
            # If diverging requested, determine symmetric max_abs.
            if (cbar_vmin is None) and (cbar_vmax is not None):
                max_abs = abs(cbar_vmax)
            elif (cbar_vmax is None) and (cbar_vmin is not None):
                max_abs = abs(cbar_vmin)
            else:
                max_abs = max(abs(vmin), abs(vmax))
            norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
            cmap = _get_colormap_by_name(diverging_cmap)
        else:
            if use_log:
                # ensure vmin/vmax are positive for LogNorm
                if vmin <= 0 or vmax <= 0:
                    print(f"  WARNING: requested log colorbar but vmin/vmax include non-positive values; using positive quantile limits instead.")
                    vmin = qlow
                    vmax = qhigh
                # mask non-positive values to avoid LogNorm errors
                ratio = ratio.astype(float)
                ratio[~(np.isfinite(ratio) & (ratio > 0))] = np.nan
                norm = LogNorm(vmin=vmin, vmax=vmax)
                cmap = _get_colormap_by_name(cmap_name)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
                cmap = _get_colormap_by_name(cmap_name)
        tcol = ax.tripcolor(triang, ratio, cmap=cmap, shading='flat', norm=norm)
        ax.set_title(f"{num_name} / {den_name} — Year {year}")
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        # overlay grounding lines (best-effort)
        gls = load_grounding_for_year(run_dirs, run_names, year)
        for g in gls:
            # prefer the extracted grounding-line mask if present
            if g.get('gl_mask') is None:
                continue
            try:
                gl_tr = tri.Triangulation(g['x'], g['y'])
                # create triangle mask consistent with main triangulation
                gl_triMask = np.zeros(len(gl_tr.triangles), dtype=bool)
                for t in range(len(gl_tr.triangles)):
                    thisTri = gl_tr.triangles[t, :]
                    if (np.hypot(g['x'][thisTri[0]]-g['x'][thisTri[1]], g['y'][thisTri[0]]-g['y'][thisTri[1]]) > maxDist or
                        np.hypot(g['x'][thisTri[1]]-g['x'][thisTri[2]], g['y'][thisTri[1]]-g['y'][thisTri[2]]) > maxDist or
                        np.hypot(g['x'][thisTri[0]]-g['x'][thisTri[2]], g['y'][thisTri[0]]-g['y'][thisTri[2]]) > maxDist):
                        gl_triMask[t] = True
                gl_tr.set_mask(gl_triMask)
                ax.tricontour(gl_tr, g['gl_mask'], levels=[0.9999], colors=[g['color']], linewidths=gl_linewidth)
            except Exception:
                continue

        cbar = fig.colorbar(tcol, ax=ax, orientation='vertical', fraction=0.035, pad=0.03)
        cbar.set_label(f"{num_name} / {den_name} (units: derived)")

        if save_base:
            safe_num = num_name.replace('/', '_')
            safe_den = den_name.replace('/', '_')
            out_png = os.path.join(save_base, f"ensemble_ratio_{safe_num}_over_{safe_den}_{year}.png")
            fig.savefig(out_png, dpi=300, bbox_inches='tight')
            print(f"  Saved plot {out_png}")
            # write compact NetCDF
            out_nc = os.path.join(save_base, f"ensemble_ratio_{safe_num}_over_{safe_den}_{year}.nc")
            try:
                ncw = Dataset(out_nc, 'w')
                ncw.createDimension('nCells', xCell.size)
                xv = ncw.createVariable('xCell', 'f8', ('nCells',))
                yv = ncw.createVariable('yCell', 'f8', ('nCells',))
                rvname = f"{safe_num}_over_{safe_den}"
                rv = ncw.createVariable(rvname, 'f8', ('nCells',), zlib=True)
                xv[:] = xCell
                yv[:] = yCell
                rv[:] = ratio
                rv.units = 'derived (numerator / denominator)'
                ncw.close()
                print(f"  Saved ratio NetCDF {out_nc}")
            except Exception as e:
                print(f"  WARNING: failed to write ratio NetCDF: {e}")
        plt.close(fig)

print('Done')
