#!/usr/bin/env python3
"""Per-shelf diagnostic plots for saved parameter grids and satobs.

Generates: obs vs modeled scatter, residual histogram, draft bin counts, and parameter maps.
"""
import argparse
from pathlib import Path
import logging
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def load_param_grid(params_dir, name, suffix):
    p = Path(params_dir) / f"{name}_{suffix}.nc"
    if not p.exists():
        return None
    with xr.open_dataset(p) as ds:
        var = next(iter(ds.data_vars))
        # return a copy so the file can be closed safely
        return ds[var].values.copy()


def compute_modeled_flux(draft_mean, alpha0, alpha1, const_val, param_type):
    # param_type: 0 = linear (alpha0 + alpha1 * draft), 1 = constant (const_val)
    modeled = np.full_like(draft_mean, np.nan, dtype=float)
    mask_linear = (param_type == 0)
    mask_const = (param_type != 0)
    if np.any(mask_linear):
        modeled[mask_linear] = alpha0[mask_linear] + alpha1[mask_linear] * draft_mean[mask_linear]
    if np.any(mask_const):
        modeled[mask_const] = const_val[mask_const]
    return modeled


def main():
    p = argparse.ArgumentParser(description='Per-shelf diagnostic plots')
    p.add_argument('--shelf', required=True, help='Shelf name used in saved parameter filenames')
    p.add_argument('--params-dir', required=True, help='Directory where per-shelf parameter grid files live')
    p.add_argument('--satobs', required=True, help='Prepared satobs NetCDF (has flux and draft variables)')
    p.add_argument('--flux-var', default=None, help='Name of flux variable in satobs; default: first variable with two dims')
    p.add_argument('--draft-var', default=None, help='Name of draft variable in satobs')
    p.add_argument('--outdir', default='figures/per_shelf_diagnostics', help='Output directory for plots')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load parameter grids
    suffix_map = {
        'alpha1': 'draftDepenBasalMeltAlpha1',
        'alpha0': 'draftDepenBasalMeltAlpha0',
        'constant': 'draftDepenBasalMelt_constantMeltValue',
        'paramType': 'draftDepenBasalMelt_paramType',
           'minDraft': 'draftDepenBasalMelt_minDraft',
    }

    params = {}
    for k, pref in suffix_map.items():
        arr = load_param_grid(args.params_dir, pref, args.shelf)
        params[k] = arr

    # load satobs and choose variables
    sat = xr.open_dataset(args.satobs)
    # pick flux var
    if args.flux_var and args.flux_var in sat:
        flux_var = args.flux_var
    else:
        # pick first 2D variable
        flux_var = None
        for v in sat.data_vars:
            if sat[v].ndim >= 2:
                flux_var = v
                break
    if args.draft_var and args.draft_var in sat:
        draft_var = args.draft_var
    else:
        draft_var = None
        for v in sat.data_vars:
            if 'draft' in v.lower() or 'thickness' in v.lower():
                draft_var = v
                break

    if flux_var is None or draft_var is None:
        logger.error('Could not identify flux or draft variables in satobs; please provide --flux-var and --draft-var')
        return

    # compute per-cell mean over time (if time exists); otherwise use array as-is
    if 'time' in sat[flux_var].dims:
        flux_mean = np.asarray(sat[flux_var].mean(dim='time', skipna=True))
    else:
        flux_mean = np.asarray(sat[flux_var])

    if 'time' in sat[draft_var].dims:
        draft_mean = np.asarray(sat[draft_var].mean(dim='time', skipna=True))
    else:
        draft_mean = np.asarray(sat[draft_var])

    # Mask cells that have any parameter defined
    param_mask = None
    for k in ['alpha1', 'alpha0', 'constant', 'paramType']:
        if params[k] is not None:
            mask = ~np.isnan(params[k])
            param_mask = mask if param_mask is None else (param_mask | mask)
        # include minDraft presence in mask if available
        if params.get('minDraft') is not None:
            mask = ~np.isnan(params['minDraft'])
            param_mask = mask if param_mask is None else (param_mask | mask)

    if param_mask is None:
        logger.error('No parameter grids found in params-dir; cannot run diagnostics')
        return

    # Reduce to 1D arrays over masked cells
    mask_idx = np.where(param_mask)
    obs = flux_mean[mask_idx]
    draft_vals = draft_mean[mask_idx]

    # Extract parameter values for masked cells (use zeros if missing)
    alpha1 = params['alpha1'][mask_idx] if params['alpha1'] is not None else np.zeros_like(obs)
    alpha0 = params['alpha0'][mask_idx] if params['alpha0'] is not None else np.zeros_like(obs)
    const = params['constant'][mask_idx] if params['constant'] is not None else np.zeros_like(obs)
    ptype = params['paramType'][mask_idx] if params['paramType'] is not None else np.zeros_like(obs)
    # optional minDraft per-cell: use to filter out cells shallower than minDraft
    min_draft_vals = None
    if params.get('minDraft') is not None:
        try:
            min_draft_vals = params['minDraft'][mask_idx]
        except Exception:
            # If slicing fails (scalar), broadcast
            min_draft_vals = np.full_like(obs, params['minDraft'])

    modeled = compute_modeled_flux(draft_vals, alpha0, alpha1, const, ptype)

    # If minDraft is provided, exclude cells where draft < minDraft (not applicable)
    if min_draft_vals is not None:
        valid_cells = ~(np.isnan(min_draft_vals)) & (draft_vals >= min_draft_vals)
        if not np.any(valid_cells):
            logger.warning('After applying minDraft filter no valid cells remain for %s', args.shelf)
            return
        obs = obs[valid_cells]
        draft_vals = draft_vals[valid_cells]
        modeled = modeled[valid_cells]
        alpha1 = alpha1[valid_cells]
        alpha0 = alpha0[valid_cells]
        const = const[valid_cells]
        ptype = ptype[valid_cells]

    # Scatter plot obs vs modeled
    plt.figure(figsize=(6, 6))
    plt.scatter(modeled, obs, s=6, alpha=0.6)
    mn = np.nanmin(np.concatenate([modeled[~np.isnan(modeled)], obs[~np.isnan(obs)]]))
    mx = np.nanmax(np.concatenate([modeled[~np.isnan(modeled)], obs[~np.isnan(obs)]]))
    plt.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
    plt.xlabel('Modeled mean flux')
    plt.ylabel('Observed mean flux')
    plt.title(f'{args.shelf} obs vs modeled (means)')
    plt.tight_layout()
    plt.savefig(Path(outdir) / f'{args.shelf}_obs_vs_modeled.png')
    plt.close()

    # Residual histogram
    resid = obs - modeled
    plt.figure(figsize=(6, 4))
    plt.hist(resid[~np.isnan(resid)], bins=60)
    plt.xlabel('Residual (obs - model)')
    plt.title(f'{args.shelf} residual histogram (means)')
    plt.tight_layout()
    plt.savefig(Path(outdir) / f'{args.shelf}_residual_hist.png')
    plt.close()

    # Draft bin count histogram
    plt.figure(figsize=(6, 4))
    plt.hist(draft_vals[~np.isnan(draft_vals)], bins=30)
    plt.xlabel('Draft (m)')
    plt.ylabel('Cell count')
    plt.title(f'{args.shelf} draft bin counts')
    plt.tight_layout()
    plt.savefig(Path(outdir) / f'{args.shelf}_draft_bin_counts.png')
    plt.close()

    # Parameter maps (show as images with mask applied) — use available arrays
    def save_map(arr, name, cmap='RdBu_r'):
        if arr is None:
            return
        a = np.asarray(arr)
        if np.all(np.isnan(a)):
            return
        vmin = np.nanpercentile(a, 2)
        vmax = np.nanpercentile(a, 98)
        plt.figure(figsize=(6, 4))
        plt.imshow(a, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'{args.shelf} {name}')
        plt.tight_layout()
        plt.savefig(Path(outdir) / f'{args.shelf}_{name}.png')
        plt.close()

    # Save maps; original arrays are full-grid
    save_map(params['alpha1'], 'alpha1')
    save_map(params['alpha0'], 'alpha0')
    save_map(params['constant'], 'constantValue')

    logger.info('Wrote per-shelf diagnostics to %s', outdir)


if __name__ == '__main__':
    main()
