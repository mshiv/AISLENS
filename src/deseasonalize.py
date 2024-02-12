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

import numpy as np
import xarray as xr
import rioxarray

import dask
import distributed

import scipy
from scipy import signal
import cftime
from shapely.geometry import mapping
from xarrayutils.utils import linear_trend, xr_linregress
import pandas as pd

# Main directory path of project repository - all filepaths are relative to this
main_dir = Path.cwd().parent
DIR_external = 'data/external/'
DIR_interim = 'data/interim/'

FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'

yr1 = 300
yr2 = 900

FILE_SORRMv21_DETREND = 'SORRMv21_{}-{}_DETREND.nc'.format(yr1,yr2)

SORRMv21_DETREND = xr.open_dataset(main_dir / DIR_interim / FILE_SORRMv21_DETREND, chunks={"Time":36})
SORRMv21_DETREND_FLUX = SORRMv21_DETREND.__xarray_dataarray_variable__

# Deseasonalize: Remove climatologies to isolate anomalies / deseasonalize
SORRMv21_DETREND_FLUX_MONTH = SORRMv21_DETREND_FLUX.groupby("Time.month")
SORRMv21_DETREND_FLUX_CLM = SORRMv21_DETREND_FLUX_MONTH.mean("Time") # Climatologies
SORRMv21_DETREND_FLUX_ANM = SORRMv21_DETREND_FLUX_MONTH - SORRMv21_DETREND_FLUX_CLM # Deseasonalized anomalies

SORRMv21_DETREND_FLUX_ANM.to_netcdf(main_dir / DIR_interim / 'SORRMv21_{}-{}_DETREND_DESEASONALIZE.nc'.format(yr1,yr2))

