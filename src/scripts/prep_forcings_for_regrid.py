#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from aislens.utils import rename_dims_and_fillna, process_directory
from aislens.config import config


def regrid_to_mali(input_dir: Path, dims_to_rename: dict, fill_value=0, apply_og=True):
    """Apply rename/fill to the canonical OG forcing file and all files in input_dir.

    Parameters
    ----------
    input_dir : Path
        Directory containing forcing NetCDF files to process.
    dims_to_rename : dict
        Mapping of old dim names to new names (passed to rename_dims_and_fillna/process_directory).
    fill_value : scalar
        Value to fill NaNs with.
    apply_og : bool
        If True, also process the original canonical forcing file `config.FILE_FORCING_OG`.
    """
    # Always attempt to process canonical OG forcing file unless user requested otherwise
    if apply_og:
        try:
            rename_dims_and_fillna(config.FILE_FORCING_OG, dims_to_rename={k: v for k, v in dims_to_rename.items() if k in ("x", "y")}, fill_value=fill_value)
        except Exception as e:
            print(f"Failed to process canonical forcing file {config.FILE_FORCING_OG}: {e}")

    # Process all .nc files in provided input directory
    if input_dir and Path(input_dir).exists():
        process_directory(input_dir, dims_to_rename=dims_to_rename, fill_value=fill_value)
    else:
        print(f"Input directory {input_dir} does not exist; skipping directory processing.")


def _parse_dims(s: str):
    try:
        return json.loads(s)
    except Exception:
        # support simple comma-separated key=val pairs
        out = {}
        for part in s.split(','):
            if '=' in part:
                k, v = part.split('=', 1)
                out[k.strip()] = v.strip()
        return out


def main():
    parser = argparse.ArgumentParser(description='Prepare forcing files for regridding to MALI: rename dims and fill NaNs')
    parser.add_argument('--dir', '-d', dest='input_dir', type=str, default=str(config.DIR_FORCINGS),
                        help='Directory containing forcing NetCDF files to process (default: config.DIR_FORCINGS)')
    parser.add_argument('--fill-value', '-f', dest='fill_value', type=float, default=0,
                        help='Value to fill NaNs with (default: 0)')
    parser.add_argument('--dims', dest='dims', type=str, default='{"x":"x1","y":"y1","time":"Time"}',
                        help='JSON string or comma-separated mapping of dims to rename, e.g. "{\"x\":\"x1\",\"y\":\"y1\"}" or "x=x1,y=y1"')
    parser.add_argument('--no-og', dest='no_og', action='store_true', help='Do not process the canonical OG forcing file')
    args = parser.parse_args()

    dims_map = _parse_dims(args.dims)
    input_dir = Path(args.input_dir)

    regrid_to_mali(input_dir=input_dir, dims_to_rename=dims_map, fill_value=args.fill_value, apply_og=not args.no_og)


if __name__ == '__main__':
    main()