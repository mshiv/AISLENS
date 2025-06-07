# Prepare satellite observations and model simulation data as required for 
# subsequent steps.
# This is the FIRST script to be run in the workflow.
# Satellite observations:
#   1. Load the satellite observation dataset.
#   2. Detrend the data along the time dimension (retain the mean value).
#   3. Deseasonalize the data.
#   4. Take the time-mean of the dataset.
#   5. Ensure that melt is converted to flux, if not, and is in SI units.
#   6. Save the prepared data fields of meltflux and draft.
# Model simulation data:
#   1. Load the model simulation dataset.
#   2. Subset the dataset to the desired time range.
#   3. Detrend the data along the time dimension.
#   4. Deseasonalize the data.
#   5. Dedraft the data.
#   6. Save the seasonality and variability components thus obtained.

from aislens.dataprep import detrend_dim, deseasonalize, dedraft_catchment, extrapolate_catchment_over_time
from aislens.utils import merge_catchment_files, subset_dataset_by_time, collect_directories, initialize_directories, write_crs
from aislens.config import config
import numpy as np
import xarray as xr
import geopandas as gpd

# initialize_directories(collect_directories(config))

def prepare_satellite_observations():
    # Load satellite observation dataset
    print("Preparing satellite observaticons...")
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS)
    print("Satellite observations loaded successfully.")
    # Detrend the data along the time dimension
    print("Detrending satellite observations...")
    satobs_deseasonalized = satobs.copy()
    satobs_detrended = detrend_dim(satobs_deseasonalized[config.SATOBS_FLUX_VAR], dim=config.TIME_DIM, deg=1)
    print("Satellite observations detrended successfully.")
    print("Deseasonalizing satellite observations...")
    # Deseasonalize the data
    satobs_deseasonalized[config.SATOBS_FLUX_VAR] = deseasonalize(satobs_detrended)
    print("Satellite observations deseasonalized successfully.")
    # TODO: Take the time-mean of the dataset
    # TODO: Ensure that melt is converted to flux, if not, and is in SI units

    # Save the prepared data
    satobs_deseasonalized.to_netcdf(config.FILE_PAOLO23_SATOBS_PREPARED)

if __name__ == "__main__":
    dirs_to_create = collect_directories(config)
    initialize_directories(dirs_to_create)
    prepare_satellite_observations()