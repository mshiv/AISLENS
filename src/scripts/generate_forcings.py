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
from aislens.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(config.DIR_PROCESSED) / 'generate_forcings.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """
    Load and prepare seasonality and variability datasets.
    
    Returns:
        tuple: (seasonality, variability, data_norm, data_tmean, data_tstd)
        
    Raises:
        FileNotFoundError: If required input files don't exist
        ValueError: If data validation fails
    """
    logger.info("Loading extrapolated seasonality and variability datasets...")
    
    # Check if input files exist
    if not Path(config.FILE_SEASONALITY_EXTRAPL).exists():
        raise FileNotFoundError(f"Seasonality file not found: {config.FILE_SEASONALITY_EXTRAPL}")
    if not Path(config.FILE_VARIABILITY_EXTRAPL).exists():
        raise FileNotFoundError(f"Variability file not found: {config.FILE_VARIABILITY_EXTRAPL}")
    
    try:
        seasonality = xr.open_dataset(config.FILE_SEASONALITY_EXTRAPL, chunks={config.TIME_DIM: 36})
        variability = xr.open_dataset(config.FILE_VARIABILITY_EXTRAPL, chunks={config.TIME_DIM: 36})
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise
    
    # Standardize time dimension name
    if 'Time' in variability.dims:
        variability = variability.rename({"Time": "time"})
        logger.info("Renamed 'Time' to 'time' in variability dataset")
    if 'Time' in seasonality.dims:
        seasonality = seasonality.rename({"Time": "time"})
        logger.info("Renamed 'Time' to 'time' in seasonality dataset")
    
    # Validate variable exists
    if config.SORRM_FLUX_VAR not in variability:
        raise ValueError(f"Variable '{config.SORRM_FLUX_VAR}' not found in variability dataset")
    
    # Extract and normalize data
    data = variability[config.SORRM_FLUX_VAR]
    logger.info(f"Normalizing data (variable: {config.SORRM_FLUX_VAR})...")
    data_tmean = data.mean('time')
    data_tstd = data.std('time')
    
    # Avoid division by zero
    data_tstd = data_tstd.where(data_tstd > 0, 1e-10)
    
    data_norm = (data - data_tmean) / data_tstd
    logger.info("Data normalization complete.")
    
    return seasonality, variability, data_norm, data_tmean, data_tstd


def perform_eof_analysis(data_norm, load_existing=False):
    """
    Perform EOF decomposition or load existing EOF model.
    
    Args:
        data_norm: Normalized data array
        load_existing: If True, load existing EOF model from pickle
        
    Returns:
        tuple: (model, pcs, nmodes)
        
    Raises:
        FileNotFoundError: If load_existing=True but model file doesn't exist
    """
    pickle_path = Path(config.FILE_EOF_MODEL)
    
    if load_existing:
        logger.info(f"Loading existing EOF model from {pickle_path}...")
        if not pickle_path.exists():
            raise FileNotFoundError(f"EOF model file not found: {pickle_path}")
        try:
            with open(pickle_path, "rb") as f:
                model = pickle.load(f)
            # Reconstruct PCs from loaded model
            pcs = model.pcs()
            nmodes = model.neofs
            logger.info(f"Loaded EOF model with {nmodes} modes")
            return model, pcs, nmodes
        except Exception as e:
            logger.error(f"Failed to load EOF model: {e}")
            raise
    else:
        logger.info("Performing EOF decomposition...")
        try:
            model, _, pcs, nmodes, _ = eof_decomposition(data_norm)
            logger.info(f"EOF decomposition complete ({nmodes} modes retained)")
            
            # Save model
            logger.info(f"Saving EOF model to {pickle_path}...")
            pickle_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info("EOF model saved successfully")
            
            return model, pcs, nmodes
        except Exception as e:
            logger.error(f"EOF decomposition failed: {e}")
            raise


def generate_and_save_forcings(model, pcs, nmodes, data_tmean, data_tstd, 
                               seasonality, data, n_realizations):
    """
    Generate ensemble forcing realizations and save to disk.
    
    Args:
        model: EOF model object
        pcs: Principal components
        nmodes: Number of EOF modes
        data_tmean: Temporal mean of data
        data_tstd: Temporal standard deviation of data
        seasonality: Seasonality dataset
        data: Original data array (for coords/attrs)
        n_realizations: Number of ensemble members to generate
    """
    logger.info(f"Generating {n_realizations} ensemble realizations...")
    
    # Perform phase randomization
    try:
        new_pcs = phase_randomization(pcs.values, n_realizations)
        logger.info("Phase randomization complete")
    except Exception as e:
        logger.error(f"Phase randomization failed: {e}")
        raise
    
    # Ensure output directory exists
    output_dir = Path(config.DIR_FORCINGS)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save each realization
    successful_realizations = 0
    for i in range(n_realizations):
        try:
            logger.info(f"Generating realization {i+1}/{n_realizations}...")
            
            # Generate synthetic variability
            new_data = generate_data(model, new_pcs, i, nmodes, 1)
            
            # Unnormalize
            new_data = (new_data * data_tstd) + data_tmean
            
            # Convert to DataArray with proper coordinates and attributes
            new_data = xr.DataArray(new_data, dims=data.dims, coords=data.coords)
            new_data.attrs = data.attrs.copy()
            new_data.name = data.name
            
            # Combine with seasonality to create final forcing
            forcing = seasonality + new_data
            
            # Add metadata
            forcing.attrs['creation_date'] = datetime.now().isoformat()
            forcing.attrs['source'] = 'AISLENS forcing generator'
            forcing.attrs['realization_number'] = i
            forcing.attrs['n_eof_modes'] = nmodes
            forcing.attrs['description'] = (
                f'MALI forcing realization {i} generated using EOF decomposition '
                f'and phase randomization of variability, combined with seasonality'
            )
            
            # Save to netCDF
            output_path = output_dir / f"forcing_realization_{i}.nc"
            forcing.to_netcdf(output_path)
            logger.info(f"Saved realization {i} to {output_path}")
            successful_realizations += 1
            
        except Exception as e:
            logger.error(f"Failed to generate/save realization {i}: {e}")
            continue
    
    logger.info(f"Successfully generated {successful_realizations}/{n_realizations} realizations")
    
    if successful_realizations == 0:
        raise RuntimeError("Failed to generate any realizations")


def generate_forcings(n_realizations=None, load_existing_eof=False):
    """
    Main function to generate forcing realizations.
    
    Args:
        n_realizations: Number of ensemble members (defaults to config.N_REALIZATIONS)
        load_existing_eof: If True, load existing EOF model instead of recomputing
    """
    logger.info("="*80)
    logger.info("AISLENS Forcing Generator")
    logger.info("="*80)
    
    try:
        # Use config value if not specified
        if n_realizations is None:
            n_realizations = config.N_REALIZATIONS
        logger.info(f"Configuration: {n_realizations} realizations, "
                   f"load_existing_eof={load_existing_eof}")
        
        # Load and prepare data
        seasonality, variability, data_norm, data_tmean, data_tstd = load_and_prepare_data()
        data = variability[config.SORRM_FLUX_VAR]
        
        # Perform EOF analysis
        model, pcs, nmodes = perform_eof_analysis(data_norm, load_existing=load_existing_eof)
        
        # Generate and save forcing realizations
        generate_and_save_forcings(
            model, pcs, nmodes, data_tmean, data_tstd,
            seasonality, data, n_realizations
        )
        
        logger.info("="*80)
        logger.info("Forcing generation complete!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("="*80)
        logger.error(f"FATAL ERROR: {e}")
        logger.error("="*80)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate MALI forcing realizations using EOF decomposition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default number of realizations (from config)
  python generate_forcings.py
  
  # Generate 50 realizations
  python generate_forcings.py --n-realizations 50
  
  # Load existing EOF model and generate 100 realizations
  python generate_forcings.py --n-realizations 100 --load-existing-eof
        """
    )
    parser.add_argument(
        '--n-realizations', '-n',
        type=int,
        default=None,
        help=f'Number of ensemble realizations to generate (default: {config.N_REALIZATIONS})'
    )
    parser.add_argument(
        '--load-existing-eof',
        action='store_true',
        help='Load existing EOF model from pickle file instead of recomputing'
    )
    
    args = parser.parse_args()
    
    try:
        generate_forcings(
            n_realizations=args.n_realizations,
            load_existing_eof=args.load_existing_eof
        )
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
