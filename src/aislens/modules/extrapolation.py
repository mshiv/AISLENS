import numpy as np
import xarray as xr
from scipy import spatial

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