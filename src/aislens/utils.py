import numpy as np
import xarray as xr
import os
from pathlib import Path
import geopandas as gpd
from shapely.geometry import mapping


import scipy
import numpy as np
import xarray as xr
from scipy import spatial
import urllib
import gc
from aislens.config import config

##################################################################
# Utilities to create the data directory structure and 
# download all necessary data files.
# The data directory structure is as follows:
# data/
# ├── external
# │   ├── meansatobsPaolo23.nc
# │   └── sorrmv21.nc
# ├── interim
# ├── processed
# ├── raw
# └── tmp

##################################################################

def create_data_directories(base_dir="data"):
    """
    Create the data directory structure as specified.
    """
    subdirs = ["external", "interim", "processed", "raw", "tmp"]
    for sub in subdirs:
        Path(base_dir, sub).mkdir(parents=True, exist_ok=True)

def fetch_file(target_dir, filename, source):
    """
    Download or symlink a file to the target directory.
    If source is an https link, download the file.
    If source is a local path, create a symlink.
    """
    target_path = Path(target_dir) / filename
    if str(source).startswith("https://") or str(source).startswith("http://"):
        print(f"Downloading {filename} from {source}...")
        urllib.request.urlretrieve(source, target_path)
    else:
        print(f"Creating symlink for {filename} from {source}...")
        if target_path.exists():
            target_path.unlink()
        os.symlink(os.path.abspath(source), target_path)

def prepare_external_data(sorrmv21_src, meansatobsPaolo23_src, base_dir="data"):
    """
    Prepare the external data directory and fetch the required files.
    """
    create_data_directories(base_dir)
    external_dir = Path(base_dir) / "external"
    fetch_file(external_dir, "sorrmv21.nc", sorrmv21_src)
    fetch_file(external_dir, "meansatobsPaolo23.nc", meansatobsPaolo23_src)


##################################################################
# Data utilities for xarray datasets
# These modules provides helper functions for data manipulation
# mainly for data preprocessing
##################################################################

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

def subset_dataset_by_time(ds, time_dim, start_year, end_year):
    """
    Subset an xarray Dataset or DataArray along a cftime-based time dimension by year.
    
    Parameters:
      ds: xarray.Dataset or xarray.DataArray
      time_dim: str, name of the time dimension (e.g., "Time")
      start_year: int, start year (exclusive)
      end_year: int, end year (inclusive)
      
    Returns:
      Subsetted xarray object
    """
    years = ds[time_dim].dt.year
    # If .dt.year fails (e.g., non-standard calendars), fallback to list comprehension
    # if isinstance(years, xr.DataArray):
    #    years = years.values
    #    years = np.array([t.year for t in model[config.TIME_DIM].values])
    mask = (years > start_year) & (years <= end_year)
    return ds.isel({time_dim: mask})

def copy_subset_data(ds_data, merged_ds):
    """
    Copy data from merged datasets into the original dataset. 
    This utility function is used to update the original dataset with data from merged datasets,
    and is useful in the extrapolation workflows where we merge data from multiple ice shelf catchments
    and extrapolate data to the entire ice sheet grid.

    Args:
        ds_data (xarray.Dataset): Original dataset.
        merged_ds (xarray.Dataset): Merged dataset.

    Returns:
        xarray.Dataset: Updated dataset with merged data.
    """
    # Find the indices in ds_data that correspond to merged_ds coordinates
    x_indices = np.searchsorted(ds_data.x, merged_ds.x)
    y_indices = np.searchsorted(ds_data.y, merged_ds.y)
    # Create a boolean mask for the subset area in ds_data based on sizes of x and y dimensions.
    mask = np.zeros((ds_data.sizes['y'], ds_data.sizes['x']), dtype=bool)
    mask[np.ix_(y_indices, x_indices)] = True
    # Create a new dataset with the same structure as ds_data
    ds_result = ds_data.copy(deep=True)
    # Update the values in ds_result where the mask is True
    for var in merged_ds.data_vars:
        if var in ds_result:
            # Create a full-sized array with NaNs
            full_sized_data = np.full(ds_result[var].shape, np.nan)
            # Fill in the data from merged_ds
            full_sized_data[np.ix_(y_indices, x_indices)] = merged_ds[var].values
            # Update ds_result, preserving the original values where merged_ds doesn't have data
            ds_result[var] = xr.where(np.isnan(full_sized_data), ds_result[var], full_sized_data)
    return ds_result

def merge_catchment_data(results):
    """
    Merge in-memory, loaded datasets from multiple ice shelf catchments. Used in the extrapolation workflows.

    Args:
        results (list): List of xarray.Dataset objects.

    Returns:
        xarray.Dataset: Merged dataset.
    """
    return xr.merge(results)

def merge_catchment_files(filepaths):
    """
    Merge a list of NetCDF files into a single xarray.Dataset.

    Args:
        filepaths (list): List of paths to NetCDF files.

    Returns:
        xarray.Dataset: Merged dataset.
    """
    datasets = [xr.open_dataset(fp) for fp in filepaths]
    merged = xr.merge(datasets)
    for ds in datasets:
        ds.close()
    return merged

##################################################################
# Geospatial utilities for xarray and geopandas datasets
# These modules provides helper functions for manipulating the
# data using geospatial masks and projections
# The functions are designed to work with xarray and geopandas
##################################################################

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

def write_crs(ds, crs='epsg:3031'):
    """
    Write CRS information to an xarray dataset.

    Args:
        ds (xarray.Dataset): Dataset to update.
        crs (str): Coordinate reference system.

    Returns:
        xarray.Dataset: Updated dataset.
    """
    ds.rio.write_crs(crs, inplace=True)
    return ds


##################################################################
# Data utilities for xarray datasets
# These modules provides helper functions to extrapolate the dataset
# as required to create final forcing fields required by MPAS-LI.
##################################################################

def fill_nan_with_nearest_neighbor(da):
    """
    Fill NaN values in the data using nearest-neighbor interpolation.

    Args:
        da (xarray.DataArray): Input data.

    Returns:
        xarray.DataArray: Data with NaN values filled.
    """
    data = da.values
    nan_indices = np.argwhere(np.isnan(data))
    non_nan_indices = np.argwhere(~np.isnan(data))
    non_nan_values = data[~np.isnan(data)]

    # Create a KDTree for fast nearest-neighbor lookup
    tree = spatial.KDTree(non_nan_indices)

    # For each NaN value, find the nearest non-NaN value
    for nan_index in nan_indices:
        _, nearest_index = tree.query(nan_index)
        data[tuple(nan_index)] = non_nan_values[nearest_index]

    return xr.DataArray(data, dims=da.dims, coords=da.coords, attrs=da.attrs)

def fill_nan_with_nearest_neighbor_vectorized(da):
    """
    Fill NaN values in the data using vectorized nearest-neighbor interpolation.

    Args:
        da (xarray.DataArray): Input data.

    Returns:
        xarray.DataArray: Data with NaN values filled.
    """
    data = da.values
    mask = np.isnan(data)

    # Get coordinates of all points and non-NaN points
    coords = np.array(np.nonzero(np.ones_like(data))).T
    valid_coords = coords[~mask.ravel()]
    valid_values = data[~mask]

    # Use KDTree for efficient nearest neighbor search
    tree = spatial.cKDTree(valid_coords)
    _, indices = tree.query(coords[mask.ravel()])

    # Fill NaN values
    data_filled = data.copy()
    data_filled[mask] = valid_values[indices]

    return xr.DataArray(data_filled, dims=da.dims, coords=da.coords, attrs=da.attrs)

def delaunay_interp_weights(xy, uv, d=2):
    """
    Compute Delaunay interpolation weights.
    Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    
    Reference: Trevor Hillebrand
    Args:
        xy (array): Input x, y coordinates.
        uv (array): Output (MPAS-LI) x, y coordinates.
        d (int): Dimensionality (default is 2).

    Returns:
        tuple: (vertices, weights, outside_indices, tree)
    """
    tri = scipy.spatial.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    weights = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    weights = np.hstack((weights, 1 - weights.sum(axis=1, keepdims=True)))
    tree = scipy.spatial.cKDTree(xy)
    return vertices, weights, tree

def nn_interp_weights(xy, uv, d=2):
    """
    Compute nearest-neighbor interpolation weights.

    Args:
        xy (array): Input x, y coordinates.
        uv (array): Output (MPAS-LI) x, y coordinates.
        d (int): Dimensionality (default is 2).

    Returns:
        array: Indices of nearest neighbors.
    """
    tree = scipy.spatial.cKDTree(xy)
    _, idx = tree.query(uv, k=1)
    return idx

##################################################################
# Functions to create final forcing fields required by MALI.
##################################################################


def rename_dims_and_fillna(file_path, dims_to_rename=None, fill_value=0):
    """
    Rename dimensions and fill NaN values in a NetCDF file.

    Args:
        file_path (str or Path): Path to the NetCDF file.
        dims_to_rename (dict, optional): Dictionary mapping old dimension names to new names (e.g., {'x': 'x1', 'y': 'y1'}).
        fill_value (int or float, optional): Value to replace NaN values with. Default is 0.

    Returns:
        xarray.Dataset: Modified dataset.
    """
    # Open the dataset
    ds = xr.open_dataset(file_path)

    # Rename dimensions if specified
    if dims_to_rename:
        ds = ds.rename(dims_to_rename)

    # Fill NaN values with the specified fill value
    ds = ds.fillna(fill_value)

    # Save the modified dataset, overwriting the original file
    ds.to_netcdf(file_path)
    print(f"Updated {file_path.name}: renamed dimensions and filled NaNs with {fill_value}.")
    return ds


def process_directory(directory, dims_to_rename=None, fill_value=0):
    """
    Process all NetCDF files in a directory: rename dimensions and fill NaN values.

    Args:
        directory (str or Path): Path to the directory containing NetCDF files.
        dims_to_rename (dict, optional): Dictionary mapping old dimension names to new names (e.g., {'x': 'x1', 'y': 'y1'}).
        fill_value (int or float, optional): Value to replace NaN values with. Default is 0.

    Returns:
        None
    """
    directory = Path(directory)

    # Loop through all .nc files in the directory
    for file_path in directory.glob("*.nc"):
        rename_dims_and_fillna(file_path, dims_to_rename=dims_to_rename, fill_value=fill_value)


##################################################################
# IO utilities for file handling
# These modules provides helper functions for file handling
# such as creating directories, checking file existence, etc.
##################################################################

def ensure_dir_exists(file_path):
    """
    Ensure that the directory for the given file path exists.
    If it doesn't exist, create it.

    Args:
        file_path (str or Path): Path to the file or directory.
    """
    directory = Path(file_path).parent
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def is_directory_path(attr_name, path_obj):
    """
    Check if the given attribute name and path object represent a directory path. 
    Directory paths are defined to start with 'DIR_' or end with '_DIR' and are Path objects.
    Args:
        attr_name (str): Name of the attribute.
        path_obj (Path): Path object to check.
    Returns:
        bool: True if the attribute name indicates 
              a directory path and the path object is a Path object, False otherwise.
    """
    
    return (
        isinstance(path_obj, Path)
        and (
            attr_name.startswith("DIR_") or attr_name.endswith("_DIR")
            or "DIR_" in attr_name
        )
    )

def collect_directories(config_obj):
    directories = []
    for attr_name in dir(config_obj):
        # Skip dunder and private/protected attributes
        if attr_name.startswith("_"):
            continue
        value = getattr(config_obj, attr_name)
        if is_directory_path(attr_name, value):
            directories.append(value)
    return directories

def initialize_directories(directories=collect_directories(config)):
    """
    Ensure that all directories in the given list exist.

    Args:
        directories (list of str or Path): List of directory paths to check/create.
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def cleanup(*variables):
    """
    Delete interim variables and collect garbage.
    """
    for var in variables:
        del var
    print('Deleted interim variables')
    gc.collect()