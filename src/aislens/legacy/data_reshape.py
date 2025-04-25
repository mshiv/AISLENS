# Script to preprocess the data
# The data_preprocess.py script did not reshape the final dataset back to the original grid size. This script adds that step. 
# Refer to the AIS-sat-obs.ipynb code for the implementation in the iceShelfRegions loop.

# Import necessary libraries

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

# Load the SORRM_variability.nc dataset as well as the original SORRMv_21 dataset

yr1 = 300
yr2 = 900
SORRMv21 = xr.open_dataset(main_dir / DIR_external / 'SORRMv2.1.ISMF/regridded_output/' / FILE_SORRMv21, chunks={"Time":36})
SORRMv21_flux = SORRMv21.timeMonthly_avg_landIceFreshwaterFlux[yr1*12:yr2*12]
SORRMv21_draft = SORRMv21.timeMonthly_avg_ssh

ICESHELVES_MASK = gpd.read_file(main_dir / DIR_external / FILE_iceShelvesShape)
icems = ICESHELVES_MASK.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();

sorrmv21_variability = xr.open_dataset(main_dir / DIR_processed / 'SORRMv21_variability.nc')
sorrmv21_variability = sorrmv21_variability.__xarray_dataarray_variable__


# Helper functions

# Define a function to write_crs for the xarray dataset, with the crs input parameter defaulting to a string "epsg:3031"
def write_crs(ds, crs='epsg:3031'):
    ds.rio.write_crs(crs, inplace=True)
    return ds

mlt_orig = SORRMv21_flux.mean(dim='Time')
mlt_var = np.full((mlt_orig.shape[0], mlt_orig.shape[1], mlt_orig.shape[1]), 0)

for i in sorrmv21_variability.Time:
    mlt_var[i,:,:] = xr.DataArray(sorrmv21_variability[i,:,:], dims=['y', 'x'], coords={'x': mlt_orig.x, 'y': mlt_orig.y})

mlt_var.to_netcdf(main_dir / DIR_processed / 'draft_dependence/sorrm/SORRMv21_variability_resized.nc')