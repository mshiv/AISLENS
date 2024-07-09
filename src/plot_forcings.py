
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

import dask
import distributed

import cftime
from shapely.geometry import mapping
import pandas as pd
import cmocean


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


# Load ice shelf masks
iceShelves = gpd.read_file(main_dir / DIR_external / FILE_iceShelvesShape)
icems = iceShelves.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();

varSORRM_extrapl = xr.open_dataset(main_dir / DIR_processed / 'SORRMv21_variability_300y_NNextrapl.nc')

# Note the colorbar extent is set to half of the max of the 1% and 99% quantiles in the raw data variable for better visualization.
icv_vmin = np.nanquantile(varSORRM_extrapl.timeMonthly_avg_landIceFreshwaterFlux.values, 0.01)
icv_vmax = np.nanquantile(varSORRM_extrapl.timeMonthly_avg_landIceFreshwaterFlux.values, 0.99)

sns.set_theme(style="whitegrid")

for t in range(len(varSORRM.time)):
    plt.clf()
    plt.figure(figsize=(15,15))
    ax1 = plt.subplot(111,projection=ccrs.SouthPolarStereo())

    icems[33:133].plot(ax=ax1,color='antiquewhite', linewidth=0,zorder=1)
    icems[33:133].boundary.plot(ax=ax1,color='k', linewidth=0.2,zorder=7)

    ax1.patch.set_facecolor(color='lightsteelblue')
    ax1.add_feature(cartopy.feature.LAND, color='ghostwhite', zorder=1)
    ax1.coastlines(lw=0.2)
    varSORRM_extrapl.timeMonthly_avg_landIceFreshwaterFlux[t].plot(ax=ax1, cmap="RdBu_r", vmin=icv_vmin, vmax=icv_vmax, add_colorbar=True)
    plt.savefig('ICV_300y_NNextrapl_t-{:04d}.png'.format(t), dpi=300)
    plt.close()