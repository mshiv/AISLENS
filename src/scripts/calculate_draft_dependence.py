#!/usr/bin/env python3
"""
Calculate draft dependence parameters from satellite observations.

Processes prepared satellite data to compute draftDepenBasalMeltAlpha parameters
for each ice shelf region using weighted regression (weight_power=0.25).

Prerequisites: Run prepare_satobs.py or prepare_data.py first

Usage: python calculate_draft_dependence.py
"""

import logging
from pathlib import Path
from time import time
import xarray as xr
import geopandas as gpd

from aislens.dataprep import dedraft_catchment
from aislens.utils import write_crs, setup_logging
from aislens.config import config

logger = logging.getLogger(__name__)


def calculate_draft_dependence():
    """Calculate draft dependence parameters for all ice shelf regions."""
    start_time = time()
    
    logger.info("Loading satellite observations and ice shelf masks...")
    satobs = write_crs(xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED), config.CRS_TARGET)
    icems = gpd.read_file(config.FILE_ICESHELFMASKS).to_crs({'init': config.CRS_TARGET})
    logger.debug(f"Processing {len(config.ICE_SHELF_REGIONS)} ice shelves")
    
    logger.info("Calculating draft dependence parameters...")
    for idx, i in enumerate(config.ICE_SHELF_REGIONS, 1):
        logger.debug(f"  [{idx}/{len(config.ICE_SHELF_REGIONS)}] {icems.name.values[i]}")
        dedraft_catchment(i, icems, satobs, config, 
                         save_dir=config.DIR_ICESHELF_DEDRAFT_SATOBS,
                         weights=True, weight_power=0.25,
                         save_pred=True, save_coefs=True)
    
    logger.info("Merging and saving draft dependence parameters...")
    param_files = [config.DIR_ICESHELF_DEDRAFT_SATOBS / f'draftDepenBasalMeltAlpha_{icems.name.values[i]}.nc'
                   for i in config.ICE_SHELF_REGIONS]
    draft_params = xr.merge([xr.open_dataset(f) for f in param_files]).fillna(0)
    draft_params.to_netcdf(config.FILE_DRAFT_DEPENDENCE)
    
    logger.info(f"Complete ({time() - start_time:.1f}s) → {config.FILE_DRAFT_DEPENDENCE}")


output_dir = Path(config.DIR_PROCESSED)
output_dir.mkdir(parents=True, exist_ok=True)
setup_logging(output_dir, "calculate_draft_dependence")

logger.info("DRAFT DEPENDENCE PARAMETER CALCULATION")
calculate_draft_dependence()
