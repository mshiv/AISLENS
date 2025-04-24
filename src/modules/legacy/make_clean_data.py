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

FILE_SORRMv21_DETREND_DESEASONALIZE =  'SORRMv21_{}-{}_DETREND_DESEASONALIZE.nc'.format(yr1,yr2)
FILE_iceShelvesShape = 'iceShelves.geojson'

# This was also for Yr 300 - 900
FILE_SORRMv21_DRAFT_DEPENDENCE_FIT_iceShelfRegions = 'SORRMv21_DRAFT_DEPENDENCE_FIT_iceShelfRegions.nc'

SORRMv21_DETREND_DESEASONALIZE = xr.open_dataset(main_dir / DIR_interim / FILE_SORRMv21_DETREND_DESEASONALIZE, chunks={"Time":36})
SORRMv21_DRAFT_DEPENDENCE_FIT_iceShelfRegions = xr.open_dataset(main_dir / DIR_interim / 'dedraft/' / FILE_SORRMv21_DRAFT_DEPENDENCE_FIT_iceShelfRegions, chunks={"Time":36})

SORRMv21_DETREND_DESEASONALIZE_DEDRAFT = SORRMv21_DETREND_DESEASONALIZE.__xarray_dataarray_variable__ - SORRMv21_DRAFT_DEPENDENCE_FIT_iceShelfRegions.__xarray_dataarray_variable__
SORRMv21_DETREND_DESEASONALIZE_DEDRAFT.to_netcdf(main_dir / DIR_interim / 'SORRMv21_{}-{}_DETREND_DESEASONALIZE_DEDRAFT.nc'.format(yr1,yr2))