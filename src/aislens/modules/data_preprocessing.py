import xarray as xr
import numpy as np
from shapely.geometry import mapping
from scipy import spatial
from pathlib import Path
from modules.extrapolation import fill_nan_with_nearest_neighbor_vectorized

def subset_dataset(file_path, dim, start, end, output_path=None, chunk_size=10):
    """
    Extract a subset of a NetCDF dataset based on a specified dimension and range.

    Args:
        file_path (str or Path): Path to the input NetCDF file.
        dim (str): Dimension to subset (e.g., "Time", "x", "y").
        start (int, float, or str): Start value for the subset range.
        end (int, float, or str): End value for the subset range.
        output_path (str or Path, optional): Path to save the subsetted dataset. If None, the dataset is not saved.
        chunk_size (int, optional): Chunk size for reading the dataset. Default is 10.

    Returns:
        xarray.Dataset: Subsetted dataset.
    """
    # Load the dataset with chunking
    dataset = xr.open_dataset(file_path, chunks={dim: chunk_size})

    # Ensure the dimension exists in the dataset
    if dim not in dataset.dims:
        raise ValueError(f"Dimension '{dim}' not found in the dataset.")

    # Subset the dataset
    subset = dataset.sel({dim: slice(start, end)})

    # Save the subsetted dataset if an output path is provided
    if output_path:
        subset.to_netcdf(output_path)
        print(f"Subsetted dataset saved to {output_path}")

    return subset


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
    clipped_data = clipped_data.drop("month", errors="ignore")
    return clipped_data


def process_ice_shelf(ds_data, iceShelfNum, icems):
    """
    Process data for a specific ice shelf.

    Args:
        ds_data (xarray.Dataset): Input dataset for a specific time step.
        iceShelfNum (int): Ice shelf index.
        icems (GeoDataFrame): Ice shelf geometries.

    Returns:
        xarray.Dataset: Processed dataset for the ice shelf.
    """
    ice_shelf_mask = icems.loc[[iceShelfNum], 'geometry'].apply(mapping)
    ds = clip_data(ds_data, iceShelfNum, icems)
    
    # Vectorized filling of NaN values
    ds = ds.map(fill_nan_with_nearest_neighbor_vectorized, keep_attrs=True)
    
    ds = ds.rio.clip(ice_shelf_mask, icems.crs)
    return ds

def merge_datasets(results):
    """
    Merge datasets from multiple ice shelves.

    Args:
        results (list): List of xarray.Dataset objects.

    Returns:
        xarray.Dataset: Merged dataset.
    """
    return xr.merge(results)

def copy_subset_data(ds_data, merged_ds):
    """
    Copy data from merged datasets into the original dataset.

    Args:
        ds_data (xarray.Dataset): Original dataset.
        merged_ds (xarray.Dataset): Merged dataset.

    Returns:
        xarray.Dataset: Updated dataset with merged data.
    """
    x_indices = np.searchsorted(ds_data.x, merged_ds.x)
    y_indices = np.searchsorted(ds_data.y, merged_ds.y)

    mask = np.zeros((ds_data.sizes['y'], ds_data.sizes['x']), dtype=bool)
    mask[np.ix_(y_indices, x_indices)] = True

    ds_result = ds_data.copy(deep=True)

    for var in merged_ds.data_vars:
        if var in ds_result:
            full_sized_data = np.full(ds_result[var].shape, np.nan)
            full_sized_data[np.ix_(y_indices, x_indices)] = merged_ds[var].values
            ds_result[var] = xr.where(np.isnan(full_sized_data), ds_result[var], full_sized_data)

    return ds_result

def normalize(data):
    """
    Normalize the input data.

    Args:
        data (xarray.DataArray): Input data.

    Returns:
        tuple: (normalized data, mean, std)
    """
    data_mean = data.mean("time")
    data_std = data.std("time")
    data_demeaned = data - data_mean
    data_normalized = data_demeaned / data_std
    return data_normalized, data_mean, data_std

def unnormalize(data, mean, std):
    """
    Unnormalize the data.

    Args:
        data (xarray.DataArray): Normalized data.
        mean (xarray.DataArray): Mean used for normalization.
        std (xarray.DataArray): Standard deviation used for normalization.

    Returns:
        xarray.DataArray: Unnormalized data.
    """
    return (data * std) + mean


def subset_dataset(file_path, dim, start, end, output_path=None, chunk_size=10):
    """
    Extract a subset of a NetCDF dataset based on a specified dimension and range.

    Args:
        file_path (str or Path): Path to the input NetCDF file.
        dim (str): Dimension to subset (e.g., "Time", "x", "y").
        start (int, float, or str): Start value for the subset range.
        end (int, float, or str): End value for the subset range.
        output_path (str or Path, optional): Path to save the subsetted dataset. If None, the dataset is not saved.
        chunk_size (int, optional): Chunk size for reading the dataset. Default is 10.

    Returns:
        xarray.Dataset: Subsetted dataset.
    """
    # Load the dataset with chunking
    dataset = xr.open_dataset(file_path, chunks={dim: chunk_size})

    # Ensure the dimension exists in the dataset
    if dim not in dataset.dims:
        raise ValueError(f"Dimension '{dim}' not found in the dataset.")

    # Subset the dataset
    subset = dataset.sel({dim: slice(start, end)})

    # Save the subsetted dataset if an output path is provided
    if output_path:
        subset.to_netcdf(output_path)
        print(f"Subsetted dataset saved to {output_path}")

    return subset

