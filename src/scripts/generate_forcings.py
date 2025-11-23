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
import pickle
import xarray as xr
import numpy as np

from aislens.generator import eof_decomposition, phase_randomization, generate_data
from aislens.utils import setup_logging
from aislens.config import config

logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """Load and prepare seasonality and variability datasets."""
    logger.info("Loading extrapolated seasonality and variability datasets...")
    
    for name, path in [("Seasonality", config.FILE_SEASONALITY_EXTRAPL), 
                       ("Variability", config.FILE_VARIABILITY_EXTRAPL)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
    
    seasonality = xr.open_dataset(config.FILE_SEASONALITY_EXTRAPL, chunks={config.TIME_DIM: 36})
    variability = xr.open_dataset(config.FILE_VARIABILITY_EXTRAPL, chunks={config.TIME_DIM: 36})
    
    # Standardize time dimension name
    for ds, name in [(variability, "variability"), (seasonality, "seasonality")]:
        if 'Time' in ds.dims:
            ds.rename({"Time": "time"}, inplace=True)
            logger.debug(f"Renamed 'Time' to 'time' in {name} dataset")
    
    if config.SORRM_FLUX_VAR not in variability:
        raise ValueError(f"Variable '{config.SORRM_FLUX_VAR}' not found in variability dataset")
    
    # Normalize data
    data = variability[config.SORRM_FLUX_VAR]
    logger.info(f"Normalizing data (variable: {config.SORRM_FLUX_VAR})...")
    data_tmean = data.mean('time')
    data_tstd = data.std('time').where(lambda x: x > 0, 1e-10)  # Avoid division by zero
    data_norm = (data - data_tmean) / data_tstd
    logger.info("Data normalization complete")
    
    return seasonality, variability, data_norm, data_tmean, data_tstd


def perform_eof_analysis(data_norm, load_existing=False):
    """Perform EOF decomposition or load existing EOF model."""
    pickle_path = Path(config.FILE_EOF_MODEL)
    
    if load_existing:
        logger.info(f"Loading existing EOF model from {pickle_path}...")
        if not pickle_path.exists():
            raise FileNotFoundError(f"EOF model file not found: {pickle_path}")
        with open(pickle_path, "rb") as f:
            model = pickle.load(f)
        pcs, nmodes = model.pcs(), model.neofs
        logger.info(f"Loaded EOF model with {nmodes} modes")
    else:
        logger.info("Performing EOF decomposition...")
        model, _, pcs, nmodes, _ = eof_decomposition(data_norm)
        logger.info(f"EOF decomposition complete ({nmodes} modes retained)")
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"EOF model saved to {pickle_path}")
    
    return model, pcs, nmodes


def generate_and_save_forcings(model, pcs, nmodes, data_tmean, data_tstd, 
                               seasonality, data, n_realizations):
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
            
            # Generate synthetic variability and unnormalize
            new_data = generate_data(model, new_pcs, i, nmodes, 1)
            new_data = (new_data * data_tstd) + data_tmean
            
            # Convert to DataArray and combine with seasonality
            new_data = xr.DataArray(new_data, dims=data.dims, coords=data.coords, 
                                   attrs=data.attrs.copy())
            new_data.name = data.name
            forcing = seasonality + new_data
            
            # Add metadata
            forcing.attrs.update({
                'creation_date': datetime.now().isoformat(),
                'source': 'AISLENS forcing generator',
                'realization_number': i,
                'n_eof_modes': nmodes,
                'description': f'MALI forcing realization {i} generated using EOF '
                              f'decomposition and phase randomization'
            })
            
            # Save to netCDF
            forcing.to_netcdf(output_dir / f"forcing_realization_{i}.nc")
            logger.debug(f"Saved realization {i}")
            successful_realizations += 1
        except Exception as e:
            logger.error(f"Failed to generate/save realization {i}: {e}")
    
    logger.info(f"Successfully generated {successful_realizations}/{n_realizations} realizations")
    if successful_realizations == 0:
        raise RuntimeError("Failed to generate any realizations")


def generate_forcings(n_realizations=None, load_existing_eof=False):
    """Main function to generate forcing realizations."""
    logger.info("="*80)
    logger.info("AISLENS Forcing Generator")
    logger.info("="*80)
    
    n_realizations = n_realizations or config.N_REALIZATIONS
    logger.info(f"Configuration: {n_realizations} realizations, load_existing_eof={load_existing_eof}")
    
    seasonality, variability, data_norm, data_tmean, data_tstd = load_and_prepare_data()
    model, pcs, nmodes = perform_eof_analysis(data_norm, load_existing=load_existing_eof)
    generate_and_save_forcings(model, pcs, nmodes, data_tmean, data_tstd,
                              seasonality, variability[config.SORRM_FLUX_VAR], n_realizations)
    
    logger.info("="*80)
    logger.info("Forcing generation complete!")
    logger.info(f"Output directory: {config.DIR_FORCINGS}")
    logger.info("="*80)

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
    args = parser.parse_args()
    
    output_dir = Path(config.DIR_PROCESSED)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "generate_forcings")
    
    generate_forcings(n_realizations=args.n_realizations, 
                     load_existing_eof=args.load_existing_eof)

