#!/usr/bin/env python3
"""
Generate MALI forcing realizations using EOF decomposition and phase randomization.

This script generates ensemble forcing datasets for the MALI ice sheet model by:
1. Loading extrapolated seasonality and variability datasets
2. Normalizing the variability data
3. Performing EOF decomposition on variability
4. Generating ensemble members through phase randomization
5. Combining randomized variability with seasonality
6. Saving forcing realizations with metadata

Prerequisites:
- Run prepare_data.py to generate extrapolated seasonality and variability datasets
- Extrapolation fills NaN values using nearest neighbor (see aislens.geospatial module)

Usage:
    python generate_forcings.py [--n-realizations N] [--load-existing-eof]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import xarray as xr
import numpy as np

from aislens.generator import eof_decomposition, phase_randomization, generate_data
from aislens.utils import setup_logging
from aislens.config import config

logger = logging.getLogger(__name__)


def load_and_prepare_data(seasonality_path=None, variability_path=None):
    """Load and prepare seasonality and variability datasets.

    If explicit paths are provided they are used, otherwise fallback to config paths.
    Returns (seasonality, variability, data_norm, data_tmean, data_tstd).
    """
    seasonality_file = Path(seasonality_path) if seasonality_path else Path(config.FILE_SEASONALITY_EXTRAPL)
    variability_file = Path(variability_path) if variability_path else Path(config.FILE_VARIABILITY_EXTRAPL)

    logger.info("Loading extrapolated seasonality and variability datasets...")
    for name, path in [("Seasonality", seasonality_file), ("Variability", variability_file)]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

    seasonality = xr.open_dataset(seasonality_file, chunks={config.TIME_DIM: 36})
    variability = xr.open_dataset(variability_file, chunks={config.TIME_DIM: 36})

    # Rename Time -> time on the loaded datasets (assign back to ensure change)
    if 'Time' in variability.dims:
        variability = variability.rename({"Time": "time"})
        logger.debug("Renamed 'Time' to 'time' in variability dataset")
    if 'Time' in seasonality.dims:
        seasonality = seasonality.rename({"Time": "time"})
        logger.debug("Renamed 'Time' to 'time' in seasonality dataset")
    
    if config.SORRM_FLUX_VAR not in variability:
        raise ValueError(f"Variable '{config.SORRM_FLUX_VAR}' not found in variability dataset")
    
    data = variability[config.SORRM_FLUX_VAR]
    logger.info(f"Normalizing data (variable: {config.SORRM_FLUX_VAR})...")
    data_tmean = data.mean('time')
    data_tstd = data.std('time').where(lambda x: x > 0, 1e-10)
    data_norm = (data - data_tmean) / data_tstd
    logger.info("Data normalization complete")
    
    return seasonality, variability, data_norm, data_tmean, data_tstd


def perform_eof_analysis(data_norm, load_existing=False):
    """Perform EOF decomposition or load existing EOF model."""
    nc_path = Path(config.FILE_EOF_MODEL)

    # Minimal wrapper that exposes the methods the rest of the pipeline expects
    class SimpleEOFModel:
        def __init__(self, eofs_da: xr.DataArray, pcs_da: xr.DataArray):
            self.eofs = eofs_da
            self.pcs = pcs_da

        # eofs and pcs are exposed as attributes on the instance

        @property
        def neofs(self):
            return int(self.eofs.sizes.get('mode', self.eofs.shape[0]))

        def reconstruct_randomized_X(self, new_pcs_arr: np.ndarray, mode_slice: slice):
            # Interpret caller slice as 1-based (match xeofs behaviour) and
            # convert to Python 0-based indices
            start = mode_slice.start or 1
            stop = mode_slice.stop or self.neofs
            step = mode_slice.step or 1
            py_start = max(0, int(start) - 1)
            py_stop = min(int(stop), self.neofs)
            indices = list(range(py_start, py_stop, int(step)))

            eofs_vals = self.eofs.values[indices, ...]  # (k_sel, y, x)
            pcs_sel = new_pcs_arr[:, indices]  # (ntime, k_sel)
            arr = np.tensordot(pcs_sel, eofs_vals, axes=([1], [0]))

            time_dim = self.pcs.dims[0]
            y_dim = self.eofs.dims[1] if len(self.eofs.dims) > 1 else 'y'
            x_dim = self.eofs.dims[2] if len(self.eofs.dims) > 2 else 'x'
            da = xr.DataArray(
                arr,
                dims=(time_dim, y_dim, x_dim),
                coords={time_dim: self.pcs.coords[self.pcs.dims[0]],
                        y_dim: self.eofs.coords[y_dim],
                        x_dim: self.eofs.coords[x_dim]},
            )
            return da

    # If user requested loading an existing model, only use NetCDF (pickle removed)
    if load_existing:
        if not nc_path.exists():
            raise FileNotFoundError(f"EOF NetCDF model not found: {nc_path}. Re-run without --load-existing-eof to recompute.")
        logger.info("Loading EOF model from NetCDF %s", nc_path)
        ds = xr.open_dataset(nc_path)
        eofs_da = ds['eofs']
        pcs_da = ds['pcs']
        nmodes = int(eofs_da.sizes.get('mode', eofs_da.shape[0]))
        model = SimpleEOFModel(eofs_da, pcs_da)
        logger.info("Loaded EOF model (NetCDF) with %d modes", nmodes)
        return model, pcs_da, nmodes

    # Compute EOF decomposition
    logger.info("Performing EOF decomposition...")
    model, eofs, pcs, nmodes, varexpl = eof_decomposition(data_norm)
    logger.info(f"EOF decomposition complete ({nmodes} modes retained)")

    # Persist a version-neutral NetCDF containing EOFs and PCs
    try:
        nc_path.parent.mkdir(parents=True, exist_ok=True)
        ds_out = xr.Dataset()
        if not isinstance(eofs, xr.DataArray):
            eofs = xr.DataArray(eofs)
        if not isinstance(pcs, xr.DataArray):
            pcs = xr.DataArray(pcs)
        ds_out['eofs'] = eofs
        ds_out['pcs'] = pcs
        ds_out.attrs['nmodes'] = int(nmodes)
        ds_out.attrs['explained_variance_ratio'] = np.asarray(varexpl)
        ds_out.to_netcdf(nc_path)
        logger.debug(f"EOF model (eofs+pcs) saved to NetCDF {nc_path}")
    except Exception as e:
        logger.warning("Failed to save EOF model NetCDF to %s: %s", nc_path, e)

    # Return the xeofs model (existing pipeline expects model object)
    return model, pcs, nmodes


def generate_and_save_forcings(model, pcs, nmodes, data_tmean, data_tstd,
                               seasonality, data, n_realizations,
                               include_seasonality=True):
    """Generate ensemble forcing realizations and save to disk."""
    logger.info(f"Generating {n_realizations} ensemble realizations...")
    new_pcs = phase_randomization(pcs.values, n_realizations)
    logger.info("Phase randomization complete")
    
    output_dir = Path(config.DIR_FORCINGS)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful_realizations = 0
    for i in range(n_realizations):
        try:
            logger.info(f"Generating realization {i+1}/{n_realizations}...")
            
            new_data = generate_data(model, new_pcs, i, nmodes, 1)
            new_data = (new_data * data_tstd) + data_tmean

            new_data = xr.DataArray(new_data, dims=data.dims, coords=data.coords,
                                   attrs=data.attrs.copy())
            new_data.name = data.name

            # Optionally add seasonality. If include_seasonality is False, save
            # the variability-only realization (useful for experiments).
            if include_seasonality:
                forcing = seasonality + new_data
            else:
                forcing = new_data

            forcing.attrs.update({
                'creation_date': datetime.now().isoformat(),
                'source': 'AISLENS forcing generator',
                'realization_number': i,
                'n_eof_modes': nmodes,
                # store as integer (0/1) to avoid netCDF attribute type issues
                'seasonality_included': int(bool(include_seasonality)),
                'description': f'MALI forcing realization {i} generated using EOF '
                              f'decomposition and phase randomization'
            })
            
            # filename reflects whether seasonality was included
            suffix = "" if forcing.attrs.get('seasonality_included', True) else "_no_ssn"
            fname = output_dir / f"forcing_realization_{i}{suffix}.nc"
            forcing.to_netcdf(fname)
            logger.debug(f"Saved realization {i} -> {fname}")
            successful_realizations += 1
        except Exception as e:
            logger.error(f"Failed to generate/save realization {i}: {e}")
    
    logger.info(f"Successfully generated {successful_realizations}/{n_realizations} realizations")
    if successful_realizations == 0:
        raise RuntimeError("Failed to generate any realizations")


def generate_forcings(n_realizations=None, load_existing_eof=False, include_seasonality=True,
                      seasonality_file=None, variability_file=None):
    """Main function to generate forcing realizations."""
    logger.info("AISLENS Forcing Generator")
    
    n_realizations = n_realizations or config.N_REALIZATIONS
    logger.info(f"Configuration: {n_realizations} realizations, load_existing_eof={load_existing_eof}")

    seasonality, variability, data_norm, data_tmean, data_tstd = load_and_prepare_data(
        seasonality_path=seasonality_file, variability_path=variability_file
    )
    model, pcs, nmodes = perform_eof_analysis(data_norm, load_existing=load_existing_eof)
    # Default: include seasonality. The CLI can toggle this.
    generate_and_save_forcings(model, pcs, nmodes, data_tmean, data_tstd,
                              seasonality, variability[config.SORRM_FLUX_VAR], n_realizations,
                              include_seasonality=include_seasonality)
    
    logger.info("Forcing generation complete!")
    logger.info(f"Output directory: {config.DIR_FORCINGS}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate MALI forcing realizations using EOF decomposition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python generate_forcings.py\n'
               '  python generate_forcings.py --n-realizations 50\n'
               '  python generate_forcings.py -n 100 --load-existing-eof'
    )
    parser.add_argument('--n-realizations', '-n', type=int, default=None,
                       help=f'Number of realizations (default: {config.N_REALIZATIONS})')
    parser.add_argument('--load-existing-eof', action='store_true',
                       help='Load existing EOF model instead of recomputing')
    parser.add_argument('--no-seasonality', action='store_true',
                        help='Do not add seasonality to the generated forcings (produce variability-only forcings)')
    parser.add_argument('--seasonality-file', type=str, default=None,
                        help='Path to explicit seasonality NetCDF (overrides config.FILE_SEASONALITY_EXTRAPL)')
    parser.add_argument('--variability-file', type=str, default=None,
                        help='Path to explicit variability NetCDF (overrides config.FILE_VARIABILITY_EXTRAPL)')
    args = parser.parse_args()
    
    output_dir = Path(config.DIR_PROCESSED)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "generate_forcings")
    
    include_seasonality = not args.no_seasonality
    generate_forcings(n_realizations=args.n_realizations,
                     load_existing_eof=args.load_existing_eof,
                     include_seasonality=include_seasonality,
                     seasonality_file=args.seasonality_file,
                     variability_file=args.variability_file)

