import xarray as xr
import numpy as np
from shapely.geometry import mapping
from scipy import spatial

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

def fill_nan_with_nearest_neighbor_vectorized(da):
    """
    Fill NaN values in a DataArray using nearest neighbors.

    Args:
        da (xarray.DataArray): Input data with NaNs.

    Returns:
        xarray.DataArray: Data with NaNs filled.
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