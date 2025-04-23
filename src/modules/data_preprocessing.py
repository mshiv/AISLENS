import xarray as xr
import numpy as np
from shapely.geometry import mapping

def detrend_dim(data, dim, deg):
    """
    Detrend data along a specified dimension using a polynomial fit.

    Args:
        data (xarray.DataArray): Input data.
        dim (str): Dimension along which to detrend.
        deg (int): Degree of the polynomial fit.

    Returns:
        xarray.DataArray: Detrended data.
    """
    p = data.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(data[dim], p.polyfit_coefficients)
    detrended = data - fit
    return detrended

def deseasonalize(data):
    """
    Remove the seasonal cycle from the data.

    Args:
        data (xarray.DataArray): Input data.

    Returns:
        xarray.DataArray: Deseasonalized data.
    """
    data_month = data.groupby("Time.month")
    data_clm = data_month.mean("Time")
    data_anm = data_month - data_clm
    original_mean = data.mean("Time")
    data_anm += original_mean
    return data_anm

def clip_data(total_data, basin, icems):
    """
    Clip the map to a specific domain.

    Args:
        total_data (xarray.DataArray): Input data.
        basin (str): Basin name.
        icems (GeoDataFrame): Ice shelf geometry.

    Returns:
        xarray.DataArray: Clipped data.
    """
    clipped_data = total_data.rio.clip(icems.loc[[basin], 'geometry'].apply(mapping), icems.crs)
    return clipped_data