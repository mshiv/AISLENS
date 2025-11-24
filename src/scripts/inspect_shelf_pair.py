#!/usr/bin/env python3
"""Inspect parameter differences for a single shelf between fast and parallel runs.

Checks scalar parameter files first (draftDepenBasalMelt_params_{shelf}.nc) then
falls back to comprehensive grid files. Produces a short report and optional JSON
output.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)


def extract_scalar_from_ds(ds, varnames):
    out = {}
    for vn in varnames:
        if vn in ds:
            arr = np.asarray(ds[vn].values)
            if arr.size == 0:
                out[vn] = np.nan
            elif arr.size == 1:
                out[vn] = float(arr.item())
            else:
                valid = arr.flatten()[~np.isnan(arr.flatten())]
                out[vn] = float(valid.mean()) if valid.size > 0 else np.nan
        else:
            out[vn] = np.nan
    return out


def load_params(dirpath: Path, param_set: str, shelf_name: str):
    """Return a dict of parameters for the shelf from scalar file or grid fallback."""
    base = Path(dirpath) / param_set / 'comprehensive'
    scalar_file = base / f'draftDepenBasalMelt_params_{shelf_name}.nc'
    grid_file = base / f'draftDepenBasalMelt_comprehensive_{shelf_name}.nc'
    summary_file = base / 'comprehensive_summary.csv'

    vars_of_interest = [
        'draftDepenBasalMelt_minDraft',
        'draftDepenBasalMelt_constantMeltValue',
        'draftDepenBasalMeltAlpha0',
        'draftDepenBasalMeltAlpha1',
        'draftDepenBasalMelt_paramType'
    ]

    result = {
        'source': None,
        'scalars': {},
        'grid_stats': {},
        'summary_row': {}
    }

    if scalar_file.exists():
        try:
            ds = xr.open_dataset(scalar_file)
            result['scalars'] = extract_scalar_from_ds(ds, vars_of_interest)
            result['source'] = str(scalar_file)
            ds.close()
        except Exception as e:
            logger.warning(f"Failed to read scalar file {scalar_file}: {e}")

    # If scalars empty or NaN, try grid file
    if not result['scalars'] or all(np.isnan(list(result['scalars'].values()))):
        if grid_file.exists():
            try:
                ds = xr.open_dataset(grid_file)
                # compute simple stats for each var if present
                for vn in vars_of_interest:
                    if vn in ds:
                        arr = ds[vn].values.flatten()
                        valid = arr[~np.isnan(arr)]
                        result['grid_stats'][vn] = {
                            'count_valid': int(valid.size),
                            'mean': float(valid.mean()) if valid.size>0 else np.nan,
                            'median': float(np.median(valid)) if valid.size>0 else np.nan,
                            'min': float(valid.min()) if valid.size>0 else np.nan,
                            'max': float(valid.max()) if valid.size>0 else np.nan
                        }
                    else:
                        result['grid_stats'][vn] = {'count_valid': 0}
                result['source'] = str(grid_file)
                ds.close()
            except Exception as e:
                logger.warning(f"Failed to read grid file {grid_file}: {e}")

    # Try summary CSV row
    if summary_file.exists():
        try:
            df = pd.read_csv(summary_file)
            row = df[df['shelf_name'] == shelf_name]
            if not row.empty:
                result['summary_row'] = row.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"Failed to read summary CSV {summary_file}: {e}")

    return result


def compare_pairs(fast_dir, parallel_dir, param_set, shelf_name):
    fast = load_params(Path(fast_dir), param_set, shelf_name)
    par = load_params(Path(parallel_dir), param_set, shelf_name)

    report = {
        'shelf': shelf_name,
        'param_set': param_set,
        'fast_source': fast.get('source'),
        'parallel_source': par.get('source'),
        'differences': {}
    }

    keys = ['draftDepenBasalMelt_minDraft', 'draftDepenBasalMelt_constantMeltValue',
            'draftDepenBasalMeltAlpha0', 'draftDepenBasalMeltAlpha1', 'draftDepenBasalMelt_paramType']

    for k in keys:
        fv = fast.get('scalars', {}).get(k, np.nan)
        pv = par.get('scalars', {}).get(k, np.nan)
        report['differences'][k] = {
            'fast': None if np.isnan(fv) else float(fv),
            'parallel': None if np.isnan(pv) else float(pv),
            'delta': None if np.isnan(fv) or np.isnan(pv) else float(pv - fv)
        }

    # include grid stats if available
    report['fast_grid_stats'] = fast.get('grid_stats', {})
    report['parallel_grid_stats'] = par.get('grid_stats', {})

    # include summary CSV rows
    report['fast_summary'] = fast.get('summary_row', {})
    report['parallel_summary'] = par.get('summary_row', {})

    return report


def main():
    p = argparse.ArgumentParser(description='Inspect parameter differences between fast and parallel outputs')
    p.add_argument('--shelf', required=True, help='Shelf name to inspect')
    p.add_argument('--param-set', default='standard', help='Parameter set name')
    p.add_argument('--fast-dir', required=True, help='Base directory for fast script outputs (parent of param_set/comprehensive)')
    p.add_argument('--parallel-dir', required=True, help='Base directory for parallel script outputs')
    p.add_argument('--out', default=None, help='Optional JSON file to write the report')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    report = compare_pairs(args.fast_dir, args.parallel_dir, args.param_set, args.shelf)

    # Pretty-print
    print('\nParameter comparison report for shelf:', args.shelf)
    print('Parameter set:', args.param_set)
    print('\nScalar differences:')
    for k, v in report['differences'].items():
        print(f"  {k}: fast={v['fast']}  parallel={v['parallel']}  delta={v['delta']}")

    print('\nFast grid stats (counts, mean):')
    for k, s in report['fast_grid_stats'].items():
        if 'count_valid' in s:
            print(f"  {k}: {s.get('count_valid')} valid, mean={s.get('mean')}")

    print('\nParallel grid stats (counts, mean):')
    for k, s in report['parallel_grid_stats'].items():
        if 'count_valid' in s:
            print(f"  {k}: {s.get('count_valid')} valid, mean={s.get('mean')}")

    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(report, fh, indent=2)
        print('\nWrote JSON report to', args.out)


if __name__ == '__main__':
    main()
