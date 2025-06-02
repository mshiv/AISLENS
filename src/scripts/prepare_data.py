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

from aislens.dataprep import detrend_dim, deseasonalize, dedraft
from aislens.config import config
import xarray as xr

def prepare_satellite_observations():
    # Load satellite observation dataset
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS)
    
    # Detrend the data along the time dimension
    satobs_detrended = detrend_dim(satobs[config.SATOBS_FLUX_VAR], dim=config.TIME_DIM, deg=1)
    
    # Deseasonalize the data
    satobs_deseasonalized = deseasonalize(satobs_detrended)
    
    # Save the prepared data
    satobs_deseasonalized.to_netcdf(config.FILE_PAOLO23_SATOBS_PREPARED)

def prepare_model_simulation():
    # Load model simulation dataset
    model = xr.open_dataset(config.FILE_MPASO_MODEL)
    
    # Subset the dataset to the desired time range
    model_subset = model.sel({config.TIME_DIM: slice(config.START_YEAR, config.END_YEAR)})
    
    # Detrend, deseasonalize, and dedraft the data
    model_detrended = detrend_dim(model_subset[config.SORRM_FLUX_VAR], dim=config.TIME_DIM, deg=1)
    model_deseasonalized = deseasonalize(model_detrended)
    model_dedrafted = dedraft(model_deseasonalized, model_subset[config.SORRM_DRAFT_VAR])
    
    model_seasonality = model_detrended - model_deseasonalized
    
    # Save the processed components
    model_seasonality.to_netcdf(config.FILE_SEASONALITY)
    model_dedrafted.to_netcdf(config.FILE_VARIABILITY)

if __name__ == "__main__":
    prepare_satellite_observations()
    prepare_model_simulation()