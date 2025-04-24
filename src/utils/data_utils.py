import xarray as xr
import os
import geopandas as gpd
from shapely.geometry import mapping

def load_dataset(file_path, chunks=None):
    """
    Load a dataset from a NetCDF file with optional chunking.

    Args:
        file_path (str): Path to the NetCDF file.
        chunks (dict, optional): Chunk sizes for lazy loading. Defaults to None.

    Returns:
        xarray.Dataset: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return xr.open_dataset(file_path, chunks=chunks)

def subset_dataset(data, dim, start, end):
    """
    Subset a dataset along a specified dimension.

    Args:
        data (xarray.Dataset or xarray.DataArray): Input dataset.
        dim (str): Dimension to subset (e.g., "Time", "x", "y").
        start (int, float, or str): Start value for the subset range.
        end (int, float, or str): End value for the subset range.

    Returns:
        xarray.Dataset or xarray.DataArray: Subsetted dataset.
    """
    if dim not in data.dims:
        raise ValueError(f"Dimension '{dim}' not found in the dataset.")
    return data.sel({dim: slice(start, end)})

def mask_dataset(data, mask, crs):
    """
    Mask a dataset using a spatial mask.

    Args:
        data (xarray.DataArray or xarray.Dataset): Input dataset.
        mask (GeoDataFrame): Geospatial mask.
        crs (str): Coordinate reference system of the mask.

    Returns:
        xarray.DataArray or xarray.Dataset: Masked dataset.
    """
    if not hasattr(data, "rio"):
        raise AttributeError("Dataset must be a GeoDataset with rioxarray enabled.")
    return data.rio.clip(mask.geometry.apply(mapping), crs)

def save_dataset(data, file_path):
    """
    Save a dataset to a NetCDF file.

    Args:
        data (xarray.Dataset or xarray.DataArray): Dataset to save.
        file_path (str): Path to save the NetCDF file.
    """
    data.to_netcdf(file_path)
    print(f"Dataset saved to {file_path}")

def ensure_directory_exists(directory):
    """
    Ensure that a directory exists. Create it if it doesn't.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")