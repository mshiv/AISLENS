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

main_dir = Path.cwd().parent
DIR_basalMeltObs = 'data/external/Paolo2023/'
FILE_MeltDraftObs = 'ANT_G1920V01_IceShelfMeltDraft.nc'

MELTDRAFT_OBS = xr.open_dataset(main_dir / DIR_basalMeltObs / FILE_MeltDraftObs, chunks={"x":729, "y":729})
obs23_melt = MELTDRAFT_OBS.melt
obs23_draft = MELTDRAFT_OBS.draft

# Time series of spatial mean melt
obs23_melt_ts = obs23_melt.mean(dim=['x', 'y'])

# Deseasonalize: Remove climatologies to isolate anomalies / deseasonalize
obs23_melt_month = obs23_melt.groupby("time.month")
obs23_melt_clm = obs23_melt_month.mean("time") # Climatologies
obs23_melt_anm = obs23_melt_month - obs23_melt_clm # Deseasonalized anomalies

obs23_melt_anm.to_netcdf(main_dir / 'data/processed/obs23_melt_anm.nc')

