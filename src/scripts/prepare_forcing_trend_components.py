from aislens.config import config
from aislens.dataprep import detrend_with_breakpoints_vectorized
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import ruptures as rpt


def detrend_forcing_trend_components(forcing_file_path):
    """
    Detrend the forcing trend components using a polynomial fit.
    
    Parameters:
    ds (xarray.Dataset): The dataset containing the forcing trend components.
    time_dim (str): The name of the time dimension in the dataset.
    deg (int): The degree of the polynomial to fit for detrending.
    
    Returns:
    xarray.Dataset: The dataset with detrended forcing trend components.
    """
    # Detrend each variable in the dataset
    ds = xr.open_dataset(forcing_file_path, chunks={config.TIME_DIM: 36})
    ds[config.MALI_FLOATINGBMB_VAR] = (ds[config.MALI_FLOATINGBMB_VAR].isel(Time=0) - ds[config.MALI_FLOATINGBMB_VAR])
    detrended_data = detrend_with_breakpoints_vectorized(ds[config.MALI_FLOATINGBMB_VAR],
                                                         dim="Time",        # Specify the dimension to detrend
                                                         deg=1,             # Degree of polynomial (e.g., 1 for linear detrending)
                                                         model="rbf",        # Cost model for ruptures
                                                         penalty=10         # Penalty value for change point detection
                                                         )
    detrended_data = detrended_data.to_dataset(name=config.AISLENS_FLOATINGBMB_VAR)  # Convert back to Dataset
    trend_with_breakpoints = ds - detrended_data
    trend_with_breakpoints.to_netcdf(Path(config.DIR_MALI_ISMIP6_FORCINGS) / "ISMIP6_SSP585_UKESM_FLOATINGBMB_TREND.nc")
    print("Detrending complete. Detrended data saved to 'ISMIP6_SSP585_UKESM_FLOATINGBMB_TREND.nc'.")


if __name__ == "__main__":
    # Define the path to the forcing file
    forcing_file_path = config.FILE_ISMIP6_SSP585_FORCING
    # Call the detrend function
    detrend_forcing_trend_components(forcing_file_path)