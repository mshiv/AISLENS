# Dedraft both datasets (observations and model output)
# Refer dedraft
# TODO : Refactor to take input/output filepaths and type of regions as cli arguments
# TODO : Add Dask chunking for i/o when large files

import sys
import os
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
from xarrayutils.utils import linear_trend, xr_linregress
import gc

main_dir = Path.cwd().parent # Main directory path of project repository - all filepaths are relative to this

# File path directories
DIR_external = 'data/external/'
DIR_interim = 'data/interim/'
DIR_processed = 'data/processed/'
DIR_external = 'data/external/'

# DATA FILENAMES
FILE_MeltDraftObs = 'ANT_G1920V01_IceShelfMeltDraft.nc'
FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'


yr1 = 300
yr2 = 900

FILE_SORRMv21_DETREND_DESEASONALIZE =  'SORRMv21_yr1-yr2_DETREND_DESEASONALIZE.nc'.format(yr1,yr2)
FILE_iceShelvesShape = 'iceShelves.geojson'

SORRMv21 = xr.open_dataset(main_dir.parent / 'aislens_emulation/' / DIR_external / 'SORRMv2.1.ISMF/regridded_output/' / FILE_SORRMv21, chunks={"Time":36})
SORRMv21_DRAFT = SORRMv21.timeMonthly_avg_ssh

SORRMv21_DETREND_DESEASONALIZE = xr.open_dataset(main_dir / DIR_interim / FILE_SORRMv21_DETREND_DESEASONALIZE, chunks={"Time":36})
SORRMv21_DETREND_DESEASONALIZE_FLUX = SORRMv21_DETREND_DESEASONALIZE.__xarray_dataarray_variable__

ICESHELVES_MASK = gpd.read_file(main_dir / DIR_external / FILE_iceShelvesShape)
icems = ICESHELVES_MASK.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();

SORRMv21_DETREND_DESEASONALIZE_FLUX.rio.write_crs("epsg:3031",inplace=True);
SORRMv21_DRAFT.rio.write_crs("epsg:3031",inplace=True);

if 'time' in SORRMv21_DRAFT.dims:
    tdim = 'time'
elif 'Time' in SORRMv21_DRAFT.dims:
    tdim = 'Time'

SORRMv21_DRAFT_TMEAN = SORRMv21_DRAFT.mean(tdim)

IMBIEregions = range(6,33)
iceShelfRegions = range(33,133)

for i in iceShelfRegions:
    print('extracting data for catchment {}'.format(icems.name.values[i]))
    mlt = SORRMv21_DETREND_DESEASONALIZE_FLUX.__xarray_dataarray_variable__.rio.clip(icems.loc[[i],'geometry'].apply(mapping),icems.crs,drop=False)
    # mlt = MELTDRAFT_OBS.melt.rio.clip(icems.loc[[i],'geometry'].apply(mapping),icems.crs,drop=False)
    mlt_mean = mlt.mean(tdim)
    # Dedraft: Linear Regression with SSH over chosen basin
    print('calculating linear regression for catchment {}'.format(icems.name.values[i]))
    mlt_rgrs = xr_linregress(SORRMv21_DRAFT, mlt_mean, dim=tdim) # h = independent variable
    mlt_rgrs.to_netcdf(main_dir / DIR_interim / 'dedraft/iceShelfRegions/{}_DEDRAFT_PARAMS.nc'.format(icems.name.values[i]))
    mlt_prd = mlt_rgrs.slope*SORRMv21_DRAFT_TMEAN + mlt_rgrs.intercept
    # flx_ddrft = flx - flx_prd
    mlt_prd.to_netcdf(main_dir / DIR_interim / 'dedraft/iceShelfRegions/{}_DEDRAFT_REGRESS.nc'.format(icems.name.values[i]))
    print('{} file saved'.format(icems.name.values[i]))
    del mlt, mlt_mean, mlt_rgrs, mlt_prd
    print('deleted interim variables')
    gc.collect()

# Dedraft for each ice shelf region

# SORRMv21_DETREND_DESEASONALIZE_DEDRAFT.to_netcdf(main_dir / DIR_interim / 'SORRMv21_{}-{}_DETREND_DESEASONALIZE.nc'.format(yr1,yr2))
