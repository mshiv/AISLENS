#!/usr/bin/env python3
"""Add trend dataset variable into variability (vargen) files using xarray.

This script assumes both input files share the same Time coordinate length
(or have overlapping Time values). It will align on Time conservatively and
compute elementwise sum for the variable (default
`floatingBasalMassBalAdjustment`). The output file will contain the summed
variable (same name) and preserve other variables/coords from the var file.

Usage:
  python src/scripts/add_trend_to_vargen_xarray.py --trend TREND.nc --var VAR.nc --out OUT.nc
"""
from pathlib import Path
import argparse
import logging
import xarray as xr
import numpy as np

logger = logging.getLogger(__name__)


def add_trend_to_var(trend_path, var_path, out_path, varname='floatingBasalMassBalAdjustment', compress=True):
    trend_path = Path(trend_path)
    var_path = Path(var_path)
    out_path = Path(out_path)

    logger.info(f"Opening trend: {trend_path}")
    trend_ds = xr.open_dataset(trend_path)
    logger.info(f"Opening var:   {var_path}")
    var_ds = xr.open_dataset(var_path)

    if varname not in trend_ds and varname + '_var' in trend_ds:
        # handle alternate naming
        logger.info(f"Renaming {varname + '_var'} -> {varname} in trend file")
        trend_ds = trend_ds.rename({varname + '_var': varname})

    if varname not in trend_ds:
        raise KeyError(f"{varname} not found in trend file: {trend_path}")
    if varname not in var_ds:
        raise KeyError(f"{varname} not found in var file: {var_path}")

    # Align on Time coordinate values where possible
    try:
        t_times = trend_ds['Time'].values
        v_times = var_ds['Time'].values
        common = np.intersect1d(t_times, v_times)
    except Exception:
        common = np.array([])

    if common.size > 0:
        logger.info(f"Using intersection of Time coordinates: {common.size} steps")
        trend_sel = trend_ds.sel(Time=common)
        var_sel = var_ds.sel(Time=common)
    else:
        # fallback: require equal length or trim to shorter
        tlen = int(trend_ds.sizes.get('Time', 0))
        vlen = int(var_ds.sizes.get('Time', 0))
        if tlen == vlen:
            logger.info(f"Time lengths equal (n={tlen}); using full series")
            trend_sel = trend_ds
            var_sel = var_ds
        else:
            minlen = min(tlen, vlen)
            logger.warning(f"No exact Time overlap; trimming to min length={minlen} (trend={tlen}, var={vlen})")
            trend_sel = trend_ds.isel(Time=slice(0, minlen))
            var_sel = var_ds.isel(Time=slice(0, minlen))

    # Align arrays (ensures dims are compatible)
    T, V = xr.align(trend_sel[varname], var_sel[varname], join='inner')

    logger.info(f"Computing sum: var + trend for variable '{varname}'")
    summed = V + T

    # Build output dataset: copy var_ds and replace variable
    out_ds = var_sel.copy()
    out_ds[varname] = summed

    # Preserve variable attributes for the summed variable (if present in original)
    try:
        if varname in var_ds.data_vars:
            out_ds[varname].attrs = var_ds[varname].attrs.copy()
    except Exception:
        logger.debug('Could not copy variable attrs for %s', varname)

    # Preserve global attributes from the var file
    try:
        out_ds.attrs.update(getattr(var_ds, 'attrs', {}))
    except Exception:
        logger.debug('Could not copy global attrs')

    # Preserve xtime coordinate (subset or copy as appropriate)
    try:
        if 'xtime' in var_ds.coords or 'xtime' in var_ds:
            if 'Time' in out_ds.coords and 'Time' in var_ds.coords:
                # select xtime entries matching the output Time coordinate
                try:
                    out_ds['xtime'] = var_ds['xtime'].sel(Time=out_ds['Time'])
                except Exception:
                    # fallback to positional subset if sel fails
                    out_ds['xtime'] = var_ds['xtime'].isel(Time=slice(0, out_ds.sizes['Time']))
            else:
                out_ds['xtime'] = var_ds['xtime']
    except Exception:
        logger.debug('Could not preserve xtime coordinate')

    # Build encoding: prefer original encodings when available, otherwise use defaults
    encoding = {}
    for k in out_ds.data_vars:
        if compress:
            # try to reuse original encoding where possible
            try:
                if k in var_ds.data_vars and getattr(var_ds[k], 'encoding', None):
                    # copy known encoding entries (avoid engine-specific objects)
                    enc = {kk: vv for kk, vv in var_ds[k].encoding.items() if kk in ('zlib', 'complevel', 'chunksizes', 'shuffle')}
                    if not enc:
                        enc = {'zlib': True, 'complevel': 4}
                else:
                    enc = {'zlib': True, 'complevel': 4}
            except Exception:
                enc = {'zlib': True, 'complevel': 4}
        else:
            enc = {}
        encoding[k] = enc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing output to: {out_path}")
    out_ds.to_netcdf(out_path, format='NETCDF4', engine='netcdf4', encoding=encoding if encoding else None)
    logger.info("Write complete")


def main():
    p = argparse.ArgumentParser(description='Add trend variable into variability NetCDF using xarray')
    p.add_argument('--trend', required=True, help='Trend NetCDF file path')
    p.add_argument('--var', required=True, help='Variability NetCDF file path')
    p.add_argument('--out', required=True, help='Output NetCDF file path')
    p.add_argument('--varname', default='floatingBasalMassBalAdjustment', help='Variable name to add')
    p.add_argument('--no-compress', dest='compress', action='store_false', help='Disable zlib compression on output')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    add_trend_to_var(args.trend, args.var, args.out, varname=args.varname, compress=args.compress)


if __name__ == '__main__':
    main()
