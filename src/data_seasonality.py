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

# Detrend the data

# Method 1: Detrend the time series of spatial mean melt rate using a linear trend that is unique at each spatial point
SORRMv21_flux_detrend_perpixel = detrend_dim(SORRMv21_flux, 'Time', 1).compute()
#SORRMv21_flux_detrend_perpixel_ts = SORRMv21_flux_detrend_perpixel.mean(dim=['x', 'y']).compute()
print("Data detrended")

# Remove the seasonal cycle
# Deseasonalize
SORRMv21_flux_detrend_perpixel_deseasonalize = deseasonalize(SORRMv21_flux_detrend_perpixel).compute()
#SORRMv21_flux_detrend_perpixel_deseasonalize_ts = SORRMv21_flux_detrend_perpixel_deseasonalize.mean(dim=['x', 'y']).compute()
print("Data deseasonalized")

# Save only the seasonality cycle from the above data, not the deseasonalized data.
# The seasonality cycle here is given by the difference between SORRMv21_flux_detrend_perpixel and SORRMv21_flux_detrend_perpixel_deseasonalize

SORRMv21__seasonality = SORRMv21_flux_detrend_perpixel - SORRMv21_flux_detrend_perpixel_deseasonalize

# Rename name attribute for the variable
#SORRMv21_variability.attrs['name'] = 'landIceFreshwaterFluxVariability'
SORRMv21__seasonality.to_netcdf(main_dir / DIR_processed / 'draft_dependence/sorrm/SORRMv21_variability.nc')
print('Preprocessed data saved')