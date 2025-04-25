import numpy as np
import xarray as xr
import os
import geopandas as gpd
from shapely.geometry import mapping
import scipy

######################################################
# Data utilities for xarray and geopandas
######################################################


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


def find_ice_shelf_index(ice_shelf_name, icems):
    """
    Find the index of an ice shelf by name.

    Args:
        ice_shelf_name (str): Name of the ice shelf.
        icems (GeoDataFrame): Ice shelf geometries.

    Returns:
        int: Index of the ice shelf.
    """
    return icems[icems['name'] == ice_shelf_name].index[0]

def read_ice_shelves_mask(file_path, target_crs="EPSG:3031"):
    """
    Read the ice shelves mask from a GeoJSON or shapefile and reproject it to the target CRS.

    Args:
        file_path (str): Path to the GeoJSON or shapefile containing the ice shelves mask.
        target_crs (str): Target coordinate reference system (CRS). Defaults to "EPSG:3031".

    Returns:
        geopandas.GeoDataFrame: Ice shelves mask reprojected to the target CRS.
    """
    # Read the mask file
    ice_shelves_mask = gpd.read_file(file_path)

    # Reproject to the target CRS
    ice_shelves_mask = ice_shelves_mask.to_crs(target_crs)

    return ice_shelves_mask

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