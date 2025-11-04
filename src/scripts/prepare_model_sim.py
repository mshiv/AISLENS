#!/usr/bin/env python3
"""
Prepare MPAS-Ocean model simulation data for MALI forcing generation.

This script processes ocean model output to generate seasonality and variability
components that will be used in EOF decomposition and ensemble generation:

1. Load MPAS-Ocean model simulation dataset
2. Subset data to desired time range
3. Detrend along time dimension (linear trend removal)
4. Deseasonalize to extract seasonal cycle
5. Remove draft dependence using pre-computed parameterizations
6. Calculate variability and seasonality components
7. Extrapolate components across entire ice sheet grid
8. Save processed datasets for downstream forcing generation

Prerequisites:
- Draft dependence parameterizations must be pre-computed for ice shelf regions
  (see calculate_draft_dependence_comprehensive.py)
- Ice shelf mask GeoJSON file must be available

Usage:
    python prepare_model_sim.py [--start-year YYYY] [--end-year YYYY] [--skip-extrapolation]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import geopandas as gpd

from aislens.dataprep import detrend_dim, deseasonalize, extrapolate_catchment_over_time
from aislens.utils import merge_catchment_files, subset_dataset_by_time, collect_directories, initialize_directories, write_crs
from aislens.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(config.DIR_PROCESSED) / 'prepare_model_sim.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def load_and_subset_model_data(start_year=None, end_year=None):
    """
    Load MPAS-Ocean model dataset and subset by time range.
    
    Args:
        start_year: Start year for subsetting (defaults to config.SORRM_START_YEAR)
        end_year: End year for subsetting (defaults to config.SORRM_END_YEAR)
        
    Returns:
        xr.Dataset: Time-subsetted model dataset with CRS
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If subsetting fails
    """
    logger.info("="*80)
    logger.info("Loading MPAS-Ocean Model Data")
    logger.info("="*80)
    
    # Use config defaults if not specified
    if start_year is None:
        start_year = config.SORRM_START_YEAR
    if end_year is None:
        end_year = config.SORRM_END_YEAR
    
    logger.info(f"Time range: {start_year}-{end_year}")
    
    # Check if model file exists
    if not Path(config.FILE_MPASO_MODEL).exists():
        raise FileNotFoundError(f"Model file not found: {config.FILE_MPASO_MODEL}")
    
    try:
        logger.info(f"Loading model from: {config.FILE_MPASO_MODEL}")
        model = xr.open_dataset(config.FILE_MPASO_MODEL, chunks={config.TIME_DIM: 36})
        model = write_crs(model, config.CRS_TARGET)
        logger.info(f"Model loaded successfully (shape: {model[config.SORRM_FLUX_VAR].shape})")
    except Exception as e:
        logger.error(f"Failed to load model dataset: {e}")
        raise
    
    # Subset by time
    try:
        logger.info(f"Subsetting data: {start_year} to {end_year}...")
        model_subset = subset_dataset_by_time(
            model,
            time_dim=config.TIME_DIM,
            start_year=start_year,
            end_year=end_year,
        )
        logger.info(f"Subsetting complete (shape: {model_subset[config.SORRM_FLUX_VAR].shape})")
        return model_subset
    except Exception as e:
        logger.error(f"Failed to subset model data: {e}")
        raise


def detrend_and_deseasonalize(model_subset):
    """
    Remove linear trend and seasonal cycle from model data.
    
    Args:
        model_subset: Time-subsetted model dataset
        
    Returns:
        tuple: (model_detrended, model_deseasonalized)
        
    Raises:
        ValueError: If required variable not found
    """
    logger.info("="*80)
    logger.info("Detrending and Deseasonalizing")
    logger.info("="*80)
    
    # Validate variable exists
    if config.SORRM_FLUX_VAR not in model_subset:
        raise ValueError(f"Variable '{config.SORRM_FLUX_VAR}' not found in model dataset")
    
    try:
        # Detrend (following prepare_data.py architecture)
        logger.info("Removing linear trend...")
        model_detrended = model_subset.copy()
        model_detrended[config.SORRM_FLUX_VAR] = detrend_dim(
            model_subset[config.SORRM_FLUX_VAR], 
            dim=config.TIME_DIM, 
            deg=1
        )
        logger.info("Detrending complete")
        
        # Deseasonalize
        logger.info("Removing seasonal cycle...")
        model_deseasonalized = deseasonalize(model_detrended)
        logger.info("Deseasonalization complete")
        
        return model_detrended, model_deseasonalized
        
    except Exception as e:
        logger.error(f"Failed during detrending/deseasonalization: {e}")
        raise


def load_ice_shelf_masks():
    """
    Load ice shelf mask GeoJSON and reproject to target CRS.
    
    Returns:
        gpd.GeoDataFrame: Ice shelf masks in target projection
        
    Raises:
        FileNotFoundError: If mask file doesn't exist
    """
    logger.info("Loading ice shelf masks...")
    
    if not Path(config.FILE_ICESHELFMASKS).exists():
        raise FileNotFoundError(f"Ice shelf masks not found: {config.FILE_ICESHELFMASKS}")
    
    try:
        icems = gpd.read_file(config.FILE_ICESHELFMASKS)
        icems = icems.to_crs({'init': config.CRS_TARGET})
        logger.info(f"Loaded {len(icems)} ice shelf masks")
        return icems
    except Exception as e:
        logger.error(f"Failed to load ice shelf masks: {e}")
        raise


def merge_draft_dependence_predictions(icems):
    """
    Merge pre-computed draft dependence predictions for all ice shelf regions.
    
    Args:
        icems: GeoDataFrame of ice shelf masks
        
    Returns:
        xr.Dataset: Merged draft dependence predictions
        
    Raises:
        FileNotFoundError: If any required prediction file is missing
    """
    logger.info("="*80)
    logger.info("Merging Draft Dependence Predictions")
    logger.info("="*80)
    
    # Build list of expected files
    pred_files = [
        config.DIR_ICESHELF_DEDRAFT_MODEL / f'draftDepenModelPred_{icems.name.values[i]}.nc'
        for i in config.ICE_SHELF_REGIONS
    ]
    
    # Check if all files exist
    missing_files = [f for f in pred_files if not f.exists()]
    if missing_files:
        logger.error(f"Missing {len(missing_files)} draft dependence prediction files:")
        for f in missing_files[:5]:  # Show first 5
            logger.error(f"  - {f}")
        raise FileNotFoundError(
            f"Missing draft dependence predictions. "
            f"Run calculate_draft_dependence_comprehensive.py first."
        )
    
    try:
        logger.info(f"Merging {len(pred_files)} prediction files...")
        draft_dependence_pred = merge_catchment_files(pred_files)
        logger.info("Draft dependence predictions merged successfully")
        return draft_dependence_pred
    except Exception as e:
        logger.error(f"Failed to merge draft dependence predictions: {e}")
        raise


def calculate_components(model_detrended, model_deseasonalized, draft_dependence_pred):
    """
    Calculate variability and seasonality components.
    
    Variability = Deseasonalized data - Draft dependence
    Seasonality = Detrended data - Deseasonalized data
    
    Args:
        model_detrended: Detrended model data
        model_deseasonalized: Deseasonalized model data
        draft_dependence_pred: Draft dependence predictions
        
    Returns:
        tuple: (model_variability, model_seasonality)
    """
    logger.info("="*80)
    logger.info("Calculating Variability and Seasonality Components")
    logger.info("="*80)
    
    try:
        # Reindex draft dependence to match deseasonalized data
        logger.info("Reindexing draft dependence predictions...")
        draft_dependence_pred = draft_dependence_pred.reindex_like(model_deseasonalized)
        logger.info("Reindexing complete")
        
        # Calculate components
        logger.info("Computing variability (deseasonalized - draft_dependence)...")
        model_variability = model_deseasonalized - draft_dependence_pred
        
        logger.info("Computing seasonality (detrended - deseasonalized)...")
        model_seasonality = model_detrended - model_deseasonalized
        
        logger.info("Component calculation complete")
        return model_variability, model_seasonality
        
    except Exception as e:
        logger.error(f"Failed to calculate components: {e}")
        raise


def save_intermediate_components(model_seasonality, model_variability):
    """
    Save seasonality and variability components before extrapolation.
    
    Args:
        model_seasonality: Seasonality dataset
        model_variability: Variability dataset
    """
    logger.info("="*80)
    logger.info("Saving Intermediate Components")
    logger.info("="*80)
    
    try:
        # Add metadata
        model_seasonality.attrs['creation_date'] = datetime.now().isoformat()
        model_seasonality.attrs['source'] = 'AISLENS model simulation preprocessor'
        model_seasonality.attrs['description'] = 'Seasonal cycle component (detrended - deseasonalized)'
        
        model_variability.attrs['creation_date'] = datetime.now().isoformat()
        model_variability.attrs['source'] = 'AISLENS model simulation preprocessor'
        model_variability.attrs['description'] = 'Variability component (deseasonalized - draft_dependence)'
        
        # Save
        logger.info(f"Saving seasonality to: {config.FILE_SEASONALITY}")
        model_seasonality.to_netcdf(config.FILE_SEASONALITY)
        
        logger.info(f"Saving variability to: {config.FILE_VARIABILITY}")
        model_variability.to_netcdf(config.FILE_VARIABILITY)
        
        logger.info("Intermediate components saved successfully")
        
    except Exception as e:
        logger.error(f"Failed to save intermediate components: {e}")
        raise


def extrapolate_and_save_components(model_variability, model_seasonality, icems):
    """
    Extrapolate components across entire ice sheet grid and save.
    
    Uses nearest neighbor extrapolation to fill values across full grid.
    
    Args:
        model_variability: Variability dataset
        model_seasonality: Seasonality dataset
        icems: Ice shelf masks GeoDataFrame
    """
    logger.info("="*80)
    logger.info("Extrapolating Components Across Ice Sheet Grid")
    logger.info("="*80)
    
    try:
        # Extrapolate variability
        logger.info("Extrapolating variability component...")
        logger.info("(This may take several minutes for large grids)")
        model_variability_extrapl = extrapolate_catchment_over_time(
            model_variability, 
            icems, 
            config, 
            config.SORRM_FLUX_VAR
        )
        
        # Fill remaining NaN values with 0 (following prepare_data.py pattern)
        logger.info("Filling remaining NaN values with 0...")
        model_variability_extrapl = model_variability_extrapl.fillna(0)
        logger.info("Variability extrapolation complete")
        
        # Extrapolate seasonality
        logger.info("Extrapolating seasonality component...")
        model_seasonality_extrapl = extrapolate_catchment_over_time(
            model_seasonality, 
            icems, 
            config, 
            config.SORRM_FLUX_VAR
        )
        
        logger.info("Filling remaining NaN values with 0...")
        model_seasonality_extrapl = model_seasonality_extrapl.fillna(0)
        logger.info("Seasonality extrapolation complete")
        
        # Add metadata
        model_variability_extrapl.attrs['creation_date'] = datetime.now().isoformat()
        model_variability_extrapl.attrs['source'] = 'AISLENS model simulation preprocessor'
        model_variability_extrapl.attrs['description'] = 'Extrapolated variability component across ice sheet grid'
        model_variability_extrapl.attrs['extrapolation_method'] = 'Nearest neighbor (see aislens.geospatial)'
        
        model_seasonality_extrapl.attrs['creation_date'] = datetime.now().isoformat()
        model_seasonality_extrapl.attrs['source'] = 'AISLENS model simulation preprocessor'
        model_seasonality_extrapl.attrs['description'] = 'Extrapolated seasonality component across ice sheet grid'
        model_seasonality_extrapl.attrs['extrapolation_method'] = 'Nearest neighbor (see aislens.geospatial)'
        
        # Save extrapolated components
        logger.info(f"Saving extrapolated variability to: {config.FILE_VARIABILITY_EXTRAPL}")
        model_variability_extrapl.to_netcdf(config.FILE_VARIABILITY_EXTRAPL)
        
        logger.info(f"Saving extrapolated seasonality to: {config.FILE_SEASONALITY_EXTRAPL}")
        model_seasonality_extrapl.to_netcdf(config.FILE_SEASONALITY_EXTRAPL)
        
        logger.info("Extrapolated components saved successfully")
        
    except Exception as e:
        logger.error(f"Failed during extrapolation: {e}")
        raise


def prepare_model_simulation(start_year=None, end_year=None, skip_extrapolation=False):
    """
    Main function to prepare model simulation data for forcing generation.
    
    Args:
        start_year: Start year for time subsetting (defaults to config)
        end_year: End year for time subsetting (defaults to config)
        skip_extrapolation: If True, skip extrapolation step (useful for testing)
    """
    logger.info("="*80)
    logger.info("AISLENS Model Simulation Preprocessor")
    logger.info("="*80)
    
    try:
        # Step 1: Load and subset model data
        model_subset = load_and_subset_model_data(start_year, end_year)
        
        # Step 2: Detrend and deseasonalize
        model_detrended, model_deseasonalized = detrend_and_deseasonalize(model_subset)
        
        # Step 3: Load ice shelf masks
        icems = load_ice_shelf_masks()
        
        # Step 4: Merge draft dependence predictions
        draft_dependence_pred = merge_draft_dependence_predictions(icems)
        
        # Step 5: Calculate variability and seasonality components
        model_variability, model_seasonality = calculate_components(
            model_detrended, model_deseasonalized, draft_dependence_pred
        )
        
        # Step 6: Save intermediate components
        save_intermediate_components(model_seasonality, model_variability)
        
        # Step 7: Extrapolate and save final components
        if skip_extrapolation:
            logger.info("Skipping extrapolation (--skip-extrapolation flag set)")
        else:
            extrapolate_and_save_components(model_variability, model_seasonality, icems)
        
        logger.info("="*80)
        logger.info("Model simulation preprocessing complete!")
        logger.info("="*80)
        logger.info(f"Outputs:")
        logger.info(f"  - Seasonality: {config.FILE_SEASONALITY}")
        logger.info(f"  - Variability: {config.FILE_VARIABILITY}")
        if not skip_extrapolation:
            logger.info(f"  - Seasonality (extrapolated): {config.FILE_SEASONALITY_EXTRAPL}")
            logger.info(f"  - Variability (extrapolated): {config.FILE_VARIABILITY_EXTRAPL}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("="*80)
        logger.error(f"FATAL ERROR: {e}")
        logger.error("="*80)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare MPAS-Ocean model simulation data for MALI forcing generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default time range from config
  python prepare_model_sim.py
  
  # Process custom time range
  python prepare_model_sim.py --start-year 2000 --end-year 2020
  
  # Skip extrapolation (for testing)
  python prepare_model_sim.py --skip-extrapolation
  
Prerequisites:
  - Draft dependence parameterizations must exist in:
    {config.DIR_ICESHELF_DEDRAFT_MODEL}
  - Run calculate_draft_dependence_comprehensive.py first
        """
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=None,
        help=f'Start year for time subsetting (default: {config.SORRM_START_YEAR})'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=None,
        help=f'End year for time subsetting (default: {config.SORRM_END_YEAR})'
    )
    parser.add_argument(
        '--skip-extrapolation',
        action='store_true',
        help='Skip extrapolation step (for testing intermediate outputs)'
    )
    parser.add_argument(
        '--init-dirs',
        action='store_true',
        help='Initialize required directories before processing'
    )
    
    args = parser.parse_args()
    
    # Initialize directories if requested
    if args.init_dirs:
        logger.info("Initializing required directories...")
        dirs_to_create = collect_directories(config)
        initialize_directories(dirs_to_create)
        logger.info("Directory initialization complete")
    
    try:
        prepare_model_simulation(
            start_year=args.start_year,
            end_year=args.end_year,
            skip_extrapolation=args.skip_extrapolation
        )
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
 