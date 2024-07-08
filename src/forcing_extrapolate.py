import sys
import os
os.environ['USE_PYGEOS'] = '0'
import gc
import collections
from pathlib import Path

import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams, cycler
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
from statsmodels.tsa.seasonal import seasonal_decompose


import numpy as np
import xarray as xr
from xeofs.xarray import EOF
import rioxarray

import dask
import distributed

import scipy
from scipy import signal
import cftime
from shapely.geometry import mapping
from xarrayutils.utils import linear_trend, xr_linregress
import pandas as pd
import cmocean

from scipy import spatial


# File path directories

# inDirName = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Get full path of the aislens_emulation directory. All file IO is relative to this path.
main_dir = Path.cwd().parent
#dir_ext_data = 'data/external/'
#dir_interim_data = 'data/interim/'
DIR_external = 'data/external/'
DIR_processed = 'data/processed/'
DIR_interim = 'data/interim/'
FILE_MeltDraftObs = 'ANT_G1920V01_IceShelfMeltDraft.nc'
FILE_basalMeltObs_deSeasonalized = 'obs23_melt_anm.nc'
FILE_iceShelvesShape = 'iceShelves.geojson'
FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'
FILE_SORRMv21_ICV_300y = "SORRMv21_variability_300y.nc"

# Load data and ice shelf masks
varSORRM = xr.open_dataset(main_dir / DIR_processed / FILE_SORRMv21_ICV_300y)
# varSORRM = varSORRM.timeMonthly_avg_landIceFreshwaterFlux

# Load ice shelf masks
iceShelves = gpd.read_file(main_dir / DIR_external / FILE_iceShelvesShape)
icems = iceShelves.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();

def clip_data(total_data, basin):
    clipped_data = total_data.rio.clip(icems.loc[[basin],'geometry'].apply(mapping),icems.crs)
    #clipped_data = clipped_data.dropna('time',how='all')
    #clipped_data = clipped_data.dropna('y',how='all')
    #clipped_data = clipped_data.dropna('x',how='all')
    clipped_data = clipped_data.drop("month")
    return clipped_data

def find_ice_shelf_index(ice_shelf_name):
    return icems[icems['name']==ice_shelf_name].index[0]

def fill_nan_with_nearest_neighbor(da):
    # Convert to numpy array
    data = da.values
    
    # Get the indices of NaN and non-NaN values
    nan_indices = np.argwhere(np.isnan(data))
    non_nan_indices = np.argwhere(~np.isnan(data))
    non_nan_values = data[~np.isnan(data)]
    
    # Create a KDTree for fast nearest-neighbor lookup
    tree = spatial.KDTree(non_nan_indices)
    
    # For each NaN value, find the nearest non-NaN value
    for nan_index in nan_indices:
        _, nearest_index = tree.query(nan_index)
        data[tuple(nan_index)] = non_nan_values[nearest_index]
    
    # Create a new DataArray with filled values
    filled_da = xr.DataArray(data, dims=da.dims, coords=da.coords, attrs=da.attrs)
    return filled_da

import xarray as xr
import numpy as np
from scipy import spatial
import rioxarray
import geopandas as gpd
from pathlib import Path

def fill_nan_with_nearest_neighbor_vectorized(da):
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
    ice_shelf_mask = icems.loc[[iceShelfNum], 'geometry'].apply(mapping)
    ds = clip_data(ds_data, iceShelfNum)
    
    # Vectorized filling of NaN values
    ds = ds.map(fill_nan_with_nearest_neighbor_vectorized, keep_attrs=True)
    
    ds = ds.rio.clip(ice_shelf_mask, icems.crs)
    return ds

def merge_datasets(results):
    merged = xr.merge(results)
    return merged

# Assuming ds_data and merged_ds are already loaded
# ds_data = xr.open_dataset('path_to_ds_data.nc')
# merged_ds = xr.open_dataset('path_to_merged_ds.nc')
def copy_subset_data(ds_data, merged_ds):
    # Find the indices in ds_data that correspond to merged_ds coordinates
    x_indices = np.searchsorted(ds_data.x, merged_ds.x)
    y_indices = np.searchsorted(ds_data.y, merged_ds.y)

    # Get the sizes of the x and y dimensions
    x_size = ds_data.sizes['x']
    y_size = ds_data.sizes['y']

    # Create a boolean mask for the subset area in ds_data
    mask = np.zeros((y_size, x_size), dtype=bool)
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

# Create a dummy xarray dataArray in the shape of varSORRM.timeMonthly_avg_landIceFreshwaterFlux

varSORRM_extrapl_array = np.empty(varSORRM.timeMonthly_avg_landIceFreshwaterFlux.shape)
varSORRM_extrapl_array[:] = np.nan

varSORRM_extrapl = xr.DataArray(varSORRM_extrapl_array, coords=varSORRM.timeMonthly_avg_landIceFreshwaterFlux.coords, dims = varSORRM.timeMonthly_avg_landIceFreshwaterFlux.dims, attrs=varSORRM.timeMonthly_avg_landIceFreshwaterFlux.attrs)
varSORRM_extrapl = xr.Dataset(data_vars=dict(timeMonthly_avg_landIceFreshwaterFlux=(varSORRM_extrapl)))

for t in range(len(varSORRM.time)):  # Change to range(len(varSORRM.time)) for full processing
    ds_data = varSORRM.isel(time=t).rename({'x1': 'x', 'y1': 'y'})
    
    # Process all ice shelves in parallel
    ice_shelf_range = range(33, 133)
    results = [process_ice_shelf(ds_data, iceShelfNum, icems) for iceShelfNum in ice_shelf_range]

    merged_ds = merge_datasets(results)
    result_ds = copy_subset_data(ds_data, merged_ds)

    varSORRM_extrapl.timeMonthly_avg_landIceFreshwaterFlux[t] = result_ds.timeMonthly_avg_landIceFreshwaterFlux
    print(f"Completed time step {t}")

# Save the updated varSORRM
varSORRM_extrapl.to_netcdf(main_dir / DIR_processed / 'SORRMv21_variability_300y_NNextrapl.nc')
print("Processing complete. Updated dataset saved.")