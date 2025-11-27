#!/usr/bin/env python3
# Convert draft-dependence parameter NetCDF units from m/yr to kg m-2 s-1
import argparse
from pathlib import Path
import xarray as xr
import re
import sys


def detect_myr_units(units_str: str) -> bool:
    if units_str is None:
        return False
    s = units_str.lower()
    return bool(re.search(r"\bm\b.*(yr|year|/yr|yr-1|yr\^-1|per year)", s) or 'm of ice' in s)


def convert_dataset(ds: xr.Dataset, factor: float, inplace: bool = False) -> xr.Dataset:
    ds_out = ds if inplace else ds.copy(deep=True)
    for var in ds_out.data_vars:
        units = ds_out[var].attrs.get('units', None)
        if detect_myr_units(units):
            ds_out[var] = ds_out[var] * factor
            ds_out[var].attrs['units'] = 'kg m-2 s-1'
    return ds_out


def main():
    p = argparse.ArgumentParser(description="Convert draft-dependence parameter units between m/yr and kg m-2 s-1")
    p.add_argument('--dir', '-d', required=True, help='Directory containing NetCDF files to scan')
    p.add_argument('--inplace', action='store_true', help='Overwrite files in place (default: write new files)')
    p.add_argument('--out-suffix', default='_si', help='Suffix to append to output files when not using --inplace')
    p.add_argument('--rho', type=float, default=910.0, help='Density of ice in kg m-3 (default: 910)')
    p.add_argument('--seconds-per-year', type=float, default=365.0*24*3600, help='Seconds per year (default: 365*24*3600)')
    args = p.parse_args()

    base = Path(args.dir)
    factor = args.rho / args.seconds_per_year
    files = sorted([p for p in base.iterdir() if p.is_file() and p.suffix in ('.nc', '.nc4')])
    for f in files:
        with xr.open_dataset(f) as ds:
            ds_converted = convert_dataset(ds, factor, inplace=args.inplace)
            outpath = f if args.inplace else f.with_name(f.stem + args.out_suffix + f.suffix)
            ds_converted.to_netcdf(outpath)


if __name__ == '__main__':
    main()
