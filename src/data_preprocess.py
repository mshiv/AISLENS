# Script to preprocess the data
# Input is a raw 3D xarray dataset, presumably with dimensions (time, x, y)
# Output is a preprocessed 3D xarray dataset, with the same dimensions
# Steps:
# 1. Detrend the data along the time dimension using a linear fit - This is done to remove any long-term trends in the data, physical or not
# 2. Remove the seasonal cycle from the data - This is done to remove any seasonal variability in the data, so that we can focus on the interannual variability
# 3. Clip the data to a specific domain - This is done to focus on a specific region of interest
# 4. Remove the draft dependence from the data - This is done to remove the dependence of the melt rate on the ice shelf draft, and a function is defined for each ice shelf

import sys
import os
os.environ['USE_PYGEOS'] = '0'
import gc
from pathlib import Path

import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
import geopandas as gpd

import numpy as np
import xarray as xr
from xeofs.xarray import EOF
import rioxarray

from shapely.geometry import mapping
from sklearn.linear_model import LinearRegression

# File path directories
# Load dataset

main_dir = Path.cwd().parent
DIR_external = 'data/external/'
DIR_interim = 'data/interim/'
DIR_processed = 'data/processed/'
DIR_external = 'data/external/'


FILE_MeltDraftObs = 'ANT_G1920V01_IceShelfMeltDraft.nc'
FILE_basalMeltObs_deSeasonalized = 'obs23_melt_anm.nc'
FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'
FILE_iceShelvesShape = 'iceShelves.geojson'

# Ocean model output
# Load ocean model data for plotting as well
yr1 = 300
yr2 = 900
SORRMv21 = xr.open_dataset(main_dir / DIR_external / 'SORRMv2.1.ISMF/regridded_output/' / FILE_SORRMv21, chunks={"Time":36})
SORRMv21_flux = SORRMv21.timeMonthly_avg_landIceFreshwaterFlux[yr1*12:yr2*12]
SORRMv21_draft = SORRMv21.timeMonthly_avg_ssh

ICESHELVES_MASK = gpd.read_file(main_dir / DIR_external / FILE_iceShelvesShape)
icems = ICESHELVES_MASK.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();


# Helper functions

def detrend_dim(data, dim, deg):
    # Store the original mean
    #original_mean = data.mean(dim=dim)
    # detrend along a single dimension
    p = data.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(data[dim], p.polyfit_coefficients)
    detrended = data - fit
    # Add back the original mean
    #detrended += original_mean
    return detrended

def clip_data(total_data, basin):
    """
    Clip the map to a specific domain
    data: input data (xarray DataArray)
    domain: domain name (string), as defined in the ice shelf geometry file (icems)
    """
    clipped_data = total_data.rio.clip(icems.loc[[basin],'geometry'].apply(mapping),icems.crs)
    #clipped_data = clipped_data.dropna('time',how='all')
    #clipped_data = clipped_data.dropna('y',how='all')
    #clipped_data = clipped_data.dropna('x',how='all')
    #clipped_data = clipped_data.drop("month")
    return clipped_data

def find_ice_shelf_index(ice_shelf_name):
    return icems[icems['name']==ice_shelf_name].index[0]

def deseasonalize(data):
    # Group data by month
    data_month = data.groupby("Time.month")
    # Calculate climatological mean for each month
    data_clm = data_month.mean("Time")
    # Calculate deseasonalized anomalies
    data_anm = data_month - data_clm
    # Add back the original mean
    original_mean = data.mean("Time")
    data_anm += original_mean
    return data_anm

def dedraft(data, draft):
    data_tm = data.mean(dim='Time')
    draft_tm = draft.mean(dim='Time')
    data_stack = data_tm.stack(z=('x', 'y'))
    draft_stack = draft_tm.stack(z=('x', 'y'))
    data_stack_noNaN = data_stack.fillna(0)
    draft_stack_noNaN = draft_stack.fillna(0)
    data_stack_noNaN_vals = data_stack_noNaN.values.reshape(-1,1)
    draft_stack_noNaN_vals = draft_stack_noNaN.values.reshape(-1,1)
    reg = LinearRegression().fit(draft_stack_noNaN_vals, data_stack_noNaN_vals)
    data_pred_stack_noNaN_vals = reg.predict(draft_stack_noNaN_vals).reshape(-1)
    data_pred_stack_noNaN = data_stack_noNaN.copy(data=data_pred_stack_noNaN_vals)
    data_pred_stack = data_pred_stack_noNaN.where(~data_stack.isnull(), np.nan)
    data_pred = data_pred_stack.unstack('z').transpose()
    #data_dedraft = data - data_pred
    return data_pred #reg.coef_, reg.intercept_, data_pred, data_dedraft

# Define a function to write_crs for the xarray dataset, with the crs input parameter defaulting to a string "epsg:3031"
def write_crs(ds, crs='epsg:3031'):
    ds.rio.write_crs(crs, inplace=True)
    return ds

# Detrend the data

# Method 1: Detrend the time series of spatial mean melt rate using a linear trend that is unique at each spatial point
SORRMv21_flux_detrend_perpixel = detrend_dim(SORRMv21_flux, 'Time', 1).compute()
SORRMv21_flux_detrend_perpixel_ts = SORRMv21_flux_detrend_perpixel.mean(dim=['x', 'y']).compute()
print("Data detrended")

# Remove the seasonal cycle
# Deseasonalize
SORRMv21_flux_detrend_perpixel_deseasonalize = deseasonalize(SORRMv21_flux_detrend_perpixel).compute()
SORRMv21_flux_detrend_perpixel_deseasonalize_ts = SORRMv21_flux_detrend_perpixel_deseasonalize.mean(dim=['x', 'y']).compute()
print("Data deseasonalized")

# Remove the draft dependence

print('Removing draft dependence...')
iceShelfRegions = range(33,133)

# write_crs for the data to be clipped
SORRMv21_flux_detrend_perpixel_deseasonalize = write_crs(SORRMv21_flux_detrend_perpixel_deseasonalize)
SORRMv21_draft = write_crs(SORRMv21_draft)

for i in iceShelfRegions:
    print('extracting data for catchment {}'.format(icems.name.values[i]))
    mlt = clip_data(SORRMv21_flux_detrend_perpixel_deseasonalize, i)
    h = clip_data(SORRMv21_draft, i)
    mlt_tm = mlt.mean(dim='Time')
    h_tm = h.mean(dim='Time')
    print('calculating linear regression for catchment {}'.format(icems.name.values[i]))
    mlt_pred = dedraft(mlt, h)

    mlt_pred.name = 'draftDepenBasalMeltPred'
    mlt_pred.attrs['long_name'] = 'Predicted flux of mass through the ocean surface based on draft dependence coefficients. Positive into ocean.'
    mlt_pred.attrs['units'] = 'kg m^-2 s^-1'

    mlt_pred.to_netcdf(main_dir / DIR_interim / 'draft_dependence/sorrm/{}_draftPred.nc'.format(icems.name.values[i]))
    print('{} file saved'.format(icems.name.values[i]))

    del mlt, h, mlt_tm, h_tm, mlt_pred
    print('deleted interim variables')
    gc.collect()
print('draft dependence removed, predicted flux files saved for individual ice shelves')

# Merge draft dependence parameters for all ice shelves into a single xarray dataset

iceShelfRegions = range(33,133)
ds = xr.Dataset()
for i in iceShelfRegions:
    ds = xr.merge([ds, xr.open_dataset(main_dir / DIR_interim / 'draft_dependence/sorrm/{}_draftPred.nc'.format(icems.name.values[i]))])
ds.to_netcdf(main_dir / DIR_interim / 'draft_dependence/sorrm/SORRMv21_draftDependencePred.nc')

print('merged draft dependence parameters for all ice shelves into a single xarray dataset')

# Remove draft dependence from the data
SORRMv21_flux_detrend_perpixel_deseasonalize_dedraft = SORRMv21_flux_detrend_perpixel_deseasonalize - ds#['draftDepenBasalMeltPred']
#SORRMv21_flux_detrend_perpixel_deseasonalize_dedraft_ts = SORRMv21_flux_detrend_perpixel_deseasonalize_dedraft.mean(dim=['x', 'y']).compute()

# Save the preprocessed data
SORRMv21_variability = SORRMv21_flux_detrend_perpixel_deseasonalize_dedraft

# Rename name attribute for the variable
SORRMv21_variability.name = 'landIceFreshwaterFluxVariability'
SORRMv21_variability.to_netcdf(main_dir / DIR_processed / 'draft_dependence/sorrm/SORRMv21_variability.nc')
print('Preprocessed data saved')