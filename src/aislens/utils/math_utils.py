import numpy as np
import xarray as xr

def normalize(data):
    """
    Normalize the data by subtracting the mean and dividing by the standard deviation.

    Args:
        data (xarray.DataArray or xarray.Dataset): Input data.

    Returns:
        tuple: (normalized_data, mean, std)
    """
    mean = data.mean(dim="Time")
    std = data.std(dim="Time")
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def unnormalize(data, mean, std):
    """
    Unnormalize the data using the provided mean and standard deviation.

    Args:
        data (xarray.DataArray or xarray.Dataset): Normalized data.
        mean (xarray.DataArray or xarray.Dataset): Mean used for normalization.
        std (xarray.DataArray or xarray.Dataset): Standard deviation used for normalization.

    Returns:
        xarray.DataArray or xarray.Dataset: Unnormalized data.
    """
    return (data * std) + mean

def calculate_time_mean(data, dim="Time"):
    """
    Calculate the time mean of a dataset.

    Args:
        data (xarray.DataArray or xarray.Dataset): Input dataset.
        dim (str): Time dimension. Defaults to "Time".

    Returns:
        xarray.DataArray or xarray.Dataset: Time-averaged dataset.
    """
    return data.mean(dim=dim)

def calculate_spatial_mean(data, dims=("x", "y")):
    """
    Calculate the spatial mean of a dataset.

    Args:
        data (xarray.DataArray or xarray.Dataset): Input dataset.
        dims (tuple): Spatial dimensions. Defaults to ("x", "y").

    Returns:
        xarray.DataArray or xarray.Dataset: Spatially averaged dataset.
    """
    return data.mean(dim=dims)