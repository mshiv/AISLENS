import numpy as np
import xarray as xr
import os
from pathlib import Path
import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union


import scipy
from scipy import ndimage
import numpy as np
import xarray as xr
from scipy import spatial
from sklearn.neighbors import BallTree
import urllib
import gc
from aislens.config import config
import cftime
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter, medfilt
import logging
from datetime import datetime


##################################################################
# Logging utilities
##################################################################

def setup_logging(output_dir, script_name="script"):
    """
    Setup logging to file and console.
    
    Args:
        output_dir: Directory to save log file
        script_name: Name prefix for log file (default: "script")
    
    Returns:
        Path to the log file
    """
    log_file = Path(output_dir) / f'{script_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    
    return log_file


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
    # Conservative merge: prefer non-NaN values from any input dataset.
    # This avoids placeholder NaNs overwriting real values and keeps merging
    # deterministic irrespective of input ordering.
    if not results:
        return xr.Dataset()

    # Use a deep copy of the first dataset as a template for coords/dims
    acc = results[0].copy(deep=True)
    # Initialize accumulator variables to NaN so we can fill from any file
    for var in acc.data_vars:
        acc[var] = acc[var] * np.nan

    for ds in results:
        # Try aligning datasets to the accumulator grid if necessary
        try:
            ds_al = ds.reindex_like(acc, method=None)
        except Exception:
            ds_al = ds

        for var in ds_al.data_vars:
            if var not in acc.data_vars:
                acc[var] = ds_al[var]
            else:
                # Where accumulator is NaN and ds has a value, take ds value
                acc[var] = xr.where(~np.isnan(acc[var]), acc[var], ds_al[var])

    return acc

def merge_catchment_files(filepaths):
    """
    Merge a list of NetCDF files into a single xarray.Dataset.

    Args:
        filepaths (list): List of paths to NetCDF files.

    Returns:
        xarray.Dataset: Merged dataset.
    """
    datasets = [xr.open_dataset(fp) for fp in filepaths]
    try:
        merged = merge_catchment_data(datasets)
    finally:
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

def fill_nan_with_nearest_neighbor_vectorized_balltree(da):
    data = da.values  # extract values from DataArray
    mask = np.isnan(data)  # create a mask of NaN values
    
    # Get coordinates of all points and non-NaN points
    coords = np.array(np.nonzero(np.ones_like(data))).T
    valid_coords = coords[~mask.ravel()]
    valid_values = data[~mask]
    
    # Use BallTree for efficient nearest neighbor search
    tree = BallTree(valid_coords)
    _, indices = tree.query(coords[mask.ravel()], k=1)
    
    # Fill NaN values
    data_filled = data.copy()
    data_filled[mask] = valid_values[indices.ravel()]
    
    return data_filled

def fill_nan_with_nearest_neighbor_ndimage(da):
    """
    Fill NaN values in a 2D xarray.DataArray with the value of the nearest non-NaN neighbor.
    Uses scipy.ndimage.distance_transform_edt for maximum speed.
    
    Parameters:
        da: xarray.DataArray (2D)
    Returns:
        xarray.DataArray with NaNs filled
    """
    # Dask-aware: compute if needed, preserve coords/dims/attrs
    is_xr = hasattr(da, 'values')
    if is_xr:
        data = da.compute().values if hasattr(da.data, 'compute') else da.values
        dims = da.dims
        coords = {dim: da.coords[dim] for dim in da.dims}
        attrs = da.attrs
    else:
        data = np.asarray(da)
        dims = None
        coords = None
        attrs = None

    mask = np.isnan(data)
    if not np.any(mask):
        return da.copy() if is_xr else data.copy()

    # Use ndimage distance transform to get nearest-non-NaN indices
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    data_filled = data[tuple(idx)]

    if is_xr:
        return xr.DataArray(data_filled, dims=dims, coords=coords, attrs=attrs)
    return data_filled


def align_mask_to_template(mask_da, template_da):
    """Align a boolean mask DataArray to a template DataArray.
    Tries to reindex by nearest coords and transpose dims when needed.
    Returns a boolean xarray.DataArray with the same dims/coords as template_da.
    """
    # If mask is numpy array, convert directly
    if not hasattr(mask_da, 'values'):
        arr = np.asarray(mask_da).astype(bool)
        return xr.DataArray(arr, dims=template_da.dims, coords={d: template_da.coords[d] for d in template_da.dims})

    # Ensure mask is computed if dask-backed
    if hasattr(mask_da.data, 'compute'):
        try:
            mask_da = mask_da.compute()
        except Exception:
            pass

    # If shapes and dims already match, ensure ordering matches
    try:
        if tuple(mask_da.shape) == tuple(template_da.shape):
            if mask_da.dims != template_da.dims:
                # transpose to template dims ordering
                mask_da = mask_da.transpose(*template_da.dims)
            return mask_da.astype(bool)
    except Exception:
        pass

    # Try reindexing by nearest neighbor on both dims
    try:
        mapping = {template_da.dims[0]: template_da.coords[template_da.dims[0]],
                   template_da.dims[1]: template_da.coords[template_da.dims[1]]}
        mask_al = mask_da.reindex(mapping, method='nearest', fill_value=False)
        # ensure dims order
        if mask_al.dims != template_da.dims:
            mask_al = mask_al.transpose(*template_da.dims)
        return mask_al.astype(bool)
    except Exception:
        # Fall back to rasterization option upstream
        raise


def rasterize_ice_mask(icems, template_da):
    """Rasterize geopandas GeoDataFrame `icems` onto the `template_da` grid.
    Returns a boolean xarray.DataArray aligned to template_da dims/coords.
    """
    try:
        from rasterio.features import rasterize
    except Exception as e:
        raise RuntimeError('rasterio is required for rasterize_ice_mask') from e

    # Ensure template has rioxarray transform
    try:
        transform = template_da.rio.transform()
    except Exception as e:
        raise RuntimeError('template_da must have rioxarray spatial metadata (rio.transform)') from e

    out_shape = tuple(template_da.shape)
    geoms = [(mapping(g), 1) for g in icems.geometry]
    mask_arr = rasterize(geoms, out_shape=out_shape, transform=transform, fill=0, dtype='uint8')
    return xr.DataArray(mask_arr.astype(bool), dims=template_da.dims, coords={d: template_da.coords[d] for d in template_da.dims})


def compute_nearest_index_map(mask, cache_path=None, overwrite=False):
    """Compute (and optionally cache) nearest-index map for filling NaNs.

    Parameters
    ----------
    mask : xarray.DataArray or numpy.ndarray
        Boolean mask where True indicates NaN/missing cells that should be filled.
        (This matches existing code where mask = np.isnan(data)).
    cache_path : str or Path, optional
        If provided, save (or load) the index map to/from this .npz file. When loading,
        the function returns the cached map unless overwrite=True.
    overwrite : bool
        If True and cache_path exists, recompute and overwrite the cached file.

    Returns
    -------
    index_map : numpy.ndarray
        Integer array of shape (ndim, y, x) containing indices of the nearest
        non-NaN cell for each grid cell. Use as index_map to fill arrays repeatedly.
    """
    # Accept xarray or numpy-like
    if hasattr(mask, 'values'):
        # If Dask-backed, compute to memory
        mask_arr = mask.compute().values if hasattr(mask.data, 'compute') else mask.values
    else:
        mask_arr = np.asarray(mask)

    # cache handling
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not overwrite:
            try:
                npz = np.load(str(cache_path))
                idx = npz['indices']
                return idx
            except Exception:
                # fall through and recompute
                pass

    # Use ndimage distance transform to compute nearest non-NaN indices
    # mask_arr is True where NaN (holes). This matches existing helpers using mask.
    idx = ndimage.distance_transform_edt(mask_arr, return_distances=False, return_indices=True)
    # Ensure integer indices for safe indexing
    idx = idx.astype(np.intp)

    if cache_path is not None:
        # Save as .npz with key 'indices'
        np.savez_compressed(str(cache_path), indices=idx)

    return idx


def apply_nearest_index_map(arr, index_map):
    """Apply a precomputed nearest-index map to fill NaNs in a 2D array.

    Parameters
    ----------
    arr : xarray.DataArray or numpy.ndarray
        2D array to fill. If xarray and Dask-backed, compute the slice before calling.
    index_map : numpy.ndarray
        Output from compute_nearest_index_map (shape (2, y, x)).

    Returns
    -------
    filled : same type as input (numpy array or xarray.DataArray)
    """
    is_xr = False
    if hasattr(arr, 'values'):
        is_xr = True
        da = arr
        data = da.compute().values if hasattr(da.data, 'compute') else da.values
    else:
        data = np.asarray(arr)

    mask = np.isnan(data)
    if not np.any(mask):
        return arr.copy() if is_xr else data.copy()

    nearest_y = index_map[0]
    nearest_x = index_map[1]
    filled = data.copy()
    filled[mask] = data[nearest_y[mask], nearest_x[mask]]

    if is_xr:
        return xr.DataArray(filled, dims=da.dims, coords=da.coords, attrs=da.attrs)
    return filled


def fill_with_index_map(da, index_map):
    """Convenience wrapper: fill an xarray.DataArray using a precomputed index_map.

    This is Dask-aware for per-slice (2D) fills: pass a 2D DataArray (one time slice).
    """
    return apply_nearest_index_map(da, index_map)

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
    ds = xr.open_dataset(file_path, chunks={config.TIME_DIM: 36}) #, decode_times=False)

    # Rename dimensions if specified
    if dims_to_rename:
        ds = ds.rename(dims_to_rename)

    # Fill NaN values with the specified fill value
    ds = ds.fillna(fill_value)
    tmp_path = str(file_path) + ".tmp"

    # Save the modified dataset, overwriting the original file
    ds.to_netcdf(tmp_path)
    ds.close()
    os.replace(tmp_path, file_path)  # Atomically replace original file
    print(f"Updated {file_path.name}: renamed dimensions and filled NaNs with {fill_value}.")


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

################################
# --- Helper Functions ---
################################

def get_distance_to_gl(mlt, icems, insar_gl_shape):
    gl_union = unary_union(insar_gl_shape.to_crs(icems.crs).geometry)
    gl_points = []
    for geom in gl_union.geoms if hasattr(gl_union, 'geoms') else [gl_union]:
        if geom.geom_type == 'LineString':
            gl_points.extend(np.array(geom.coords))
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                gl_points.extend(np.array(line.coords))
    gl_points = np.array(gl_points)
    tree = cKDTree(gl_points)
    x_coords, y_coords = np.meshgrid(mlt['x'].values, mlt['y'].values, indexing='xy')
    points_flat = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    distances, _ = tree.query(points_flat)
    return distances.reshape(mlt.shape)

def find_draft_threshold_knee(h, mlt, window=15, poly=2):
    hvals = h.values.flatten()
    mvals = mlt.values.flatten()
    mask = ~np.isnan(hvals) & ~np.isnan(mvals)
    hvals = hvals[mask]
    mvals = mvals[mask]
    if len(hvals) < 10:
        return 1000
    sort_idx = np.argsort(hvals)
    h_sorted = hvals[sort_idx]
    m_sorted = mvals[sort_idx]
    bins = np.linspace(h_sorted.min(), h_sorted.max(), 100)
    digitized = np.digitize(h_sorted, bins)
    bin_medians = np.array([np.median(m_sorted[digitized == i]) for i in range(1, len(bins))])
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    win = min(window, len(bin_medians)//2*2+1)
    if win < 3: win = 3
    smoothed = savgol_filter(bin_medians, window_length=win, polyorder=poly)
    d2 = np.gradient(np.gradient(smoothed))
    knee_idx = np.argmax(d2)
    threshold = bin_centers[knee_idx]
    return threshold

def find_draft_threshold_knee_smooth(h, mlt, window=15, poly=2, min_draft=None, max_draft=None, quantile_clip=(0.01, 0.99)):
    hvals = h.values.flatten()
    mvals = mlt.values.flatten()
    mask = ~np.isnan(hvals) & ~np.isnan(mvals)
    hvals = hvals[mask]
    mvals = mvals[mask]
    # Clip outliers using quantiles
    if len(hvals) < 10:
        return 1000
    qlo, qhi = np.quantile(hvals, quantile_clip)
    hmask = (hvals >= qlo) & (hvals <= qhi)
    hvals = hvals[hmask]
    mvals = mvals[hmask]
    sort_idx = np.argsort(hvals)
    h_sorted = hvals[sort_idx]
    m_sorted = mvals[sort_idx]
    bins = np.linspace(h_sorted.min(), h_sorted.max(), 100)
    digitized = np.digitize(h_sorted, bins)
    # Use median for robustness
    bin_medians = np.array([np.median(m_sorted[digitized == i]) if np.sum(digitized == i) > 3 else np.nan for i in range(1, len(bins))])
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # Remove bins with too few points
    valid = ~np.isnan(bin_medians)
    bin_medians = bin_medians[valid]
    bin_centers = bin_centers[valid]
    if len(bin_medians) < 5:
        return 1000
    # Robust smoothing
    win = min(window, len(bin_medians)//2*2+1)
    if win < 3: win = 3
    smoothed = medfilt(bin_medians, kernel_size=win)
    if len(smoothed) >= win:
        smoothed = savgol_filter(smoothed, window_length=win, polyorder=min(poly, win-1))
    d2 = np.gradient(np.gradient(smoothed))
    knee_idx = np.argmax(d2)
    threshold = bin_centers[knee_idx]
    # Enforce physical bounds
    if min_draft is not None:
        threshold = max(threshold, min_draft)
    if max_draft is not None:
        threshold = min(threshold, max_draft)
    return threshold


################################################################
# Weighting Methods for draft dependence calculation
################################################################

def draft_threshold_weight(h, threshold=1000):
    w = np.where(h.values > threshold, 1.0, 0.0)
    w[np.isnan(h.values)] = np.nan
    return w

def draft_threshold_weight_dynamic(h, mlt):
    threshold = find_draft_threshold_knee(h, mlt)
    print(f"Dynamic threshold for shelf: {threshold}")
    w = np.where(h.values > threshold, 1.0, 0.0)
    w[np.isnan(h.values)] = np.nan
    return w

def draft_threshold_weight_dynamic_smooth(h, mlt):
    threshold = find_draft_threshold_knee_smooth(h, mlt)
    print(f"Dynamic threshold for shelf (smooth): {threshold}")
    w = np.where(h.values > threshold, 1.0, 0.0)
    w[np.isnan(h.values)] = np.nan
    return w

def inv_GLdistance_weight(mlt, distances_2d, a=1):
    epsilon = 1e-6
    w = (1.0/(distances_2d + epsilon))**a
    w[np.isnan(mlt.values)] = np.nan
    return w

def draft_x_inv_GLdistance_weight(mlt, h, distances_2d, a=1, b=1):
    epsilon = 1e-6
    w = (np.abs(h.values)**a) * (1.0/(distances_2d + epsilon))**b
    w[np.isnan(mlt.values)] = np.nan
    return w

def melt_x_inv_GLdistance_weight(mlt, distances_2d, a=1, b=1):
    epsilon = 1e-6
    w = (np.abs(mlt.values)**a) * (1.0/(distances_2d + epsilon))**b
    w[np.isnan(mlt.values)] = np.nan
    return w

def melt_x_draft_weight(mlt, h, a=1, b=1):
    w = (np.abs(mlt.values)**a) * (np.abs(h.values)**b)
    w[np.isnan(mlt.values)] = np.nan
    return w

def gaussian_draft_weight(h, mu=1200, sigma=300):
    w = np.exp(-0.5 * ((h.values-mu)/sigma)**2)
    w[np.isnan(h.values)] = np.nan
    return w

def gaussian_draft_weight_dynamic(h):
    # Compute mean and std only for non-NaN values
    valid = ~np.isnan(h.values)
    mu = np.nanmean(h.values)
    sigma = np.nanstd(h.values)
    w = np.exp(-0.5 * ((h.values - mu) / sigma) ** 2)
    w[~valid] = np.nan
    return w

def sigmoid_weight(h, mu=None, k=0.005, a=1):
    """
    Sigmoid weighting function with optional power law.
    Args:
        h: draft array
        mu: center of sigmoid (default: mean of h)
        k: steepness
        a: power to raise the sigmoid (default 1)
    Returns:
        Weight array, same shape as h
    """
    if mu is None:
        mu = np.nanmean(h.values)
    w = 1 / (1 + np.exp(-k * (h.values - mu)))
    w = w ** a
    w[np.isnan(h.values)] = np.nan
    return w

def sigmoid_x_inv_GLdistance_weight(mlt, h, distances_2d, a=1, b=1, k=0.005, mu=None):
    """
    Combined weight: (sigmoid_draft^a) * (inv_GLdistance^b)
    - a: power for sigmoid_draft
    - b: power for inv_GLdistance
    - k: steepness for sigmoid
    - mu: center for sigmoid (default: mean draft)
    """
    if mu is None:
        mu = np.nanmean(h.values)
    # Sigmoid draft weight
    sigmoid = 1 / (1 + np.exp(-k * (h.values - mu)))
    # Inverse GL distance weight
    epsilon = 1e-6
    inv_dist = 1.0 / (distances_2d + epsilon)
    # Combine with powers
    w = (sigmoid ** a) * (inv_dist ** b)
    w[np.isnan(mlt.values)] = np.nan
    return w

def uniform_weight(mlt, h):
    w = np.ones_like(mlt.values)
    w[np.isnan(mlt.values)] = np.nan
    return w

def abs_melt_weight(mlt, h):
    w = np.abs(mlt.values)
    w[np.isnan(mlt.values)] = np.nan
    return w

def draft_weight(mlt, h, a=1):
    w = np.abs(h.values)**a
    w[np.isnan(mlt.values)] = np.nan
    return w

def combined_weight(mlt, h, distances_2d, a=1, b=1, c=1, d=1, e=1, f=1):
    # a: melt, b: draft, c: inv_distance, d: gaussian, e: static threshold, f: dynamic threshold
    w = (np.abs(mlt.values)**a) * (np.abs(h.values)**b)
    if distances_2d is not None:
        w *= (1.0/(distances_2d + 1e-6))**c
    w *= gaussian_draft_weight(h, mu=1200, sigma=300)**d
    w *= draft_threshold_weight(h, threshold=1000)**e
    w *= draft_threshold_weight_dynamic(h, mlt)**f
    w[np.isnan(mlt.values)] = np.nan
    return w