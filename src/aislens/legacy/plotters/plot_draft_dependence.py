# Import necessary libraries
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

def clip_data(total_data, basin):
    """
    Clip the map to a specific domain
    data: input data (xarray DataArray)
    domain: domain name (string), as defined in the ice shelf geometry file (icems)
    """
     # TODO: Include a step here to convert from domain name string to the domain index number used in the ice shelf geometry file
    clipped_data = total_data.rio.clip(icems.loc[[basin],'geometry'].apply(mapping),icems.crs)
    #clipped_data = clipped_data.dropna('time',how='all')
    #clipped_data = clipped_data.dropna('y',how='all')
    #clipped_data = clipped_data.dropna('x',how='all')
    #clipped_data = clipped_data.drop("month")
    return clipped_data

# Load the dataset
main_dir = Path.cwd().parent
DIR_external = 'data/external/'
DIR_processed = 'data/processed/'
FILE_MeltDraftObs = 'ANT_G1920V01_IceShelfMeltDraft.nc'
FILE_basalMeltObs_deSeasonalized = 'obs23_melt_anm.nc'
FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'
FILE_iceShelvesShape = 'iceShelves.geojson'


# Satellite observations
MELTDRAFT_OBS = xr.open_dataset(main_dir / DIR_external / FILE_MeltDraftObs, chunks={"x":729, "y":729})
obs23_melt = MELTDRAFT_OBS.melt
obs23_draft = MELTDRAFT_OBS.draft

obs23_melt_anm = xr.open_dataset(main_dir / DIR_processed / FILE_basalMeltObs_deSeasonalized, chunks={"x":729, "y":729})
obs23_melt_anm = obs23_melt_anm.melt




# Ocean model output
# Load ocean model data for plotting as well
yr1 = 300
yr2 = 900
SORRMv21 = xr.open_dataset(main_dir.parent / 'aislens_emulation/' / DIR_external / 'SORRMv2.1.ISMF/regridded_output/' / FILE_SORRMv21, chunks={"Time":36})
SORRMv21_flux = SORRMv21.timeMonthly_avg_landIceFreshwaterFlux[yr1*12:yr2*12]
SORRMv21_draft = SORRMv21.timeMonthly_avg_ssh

#SORRMv21_flux_tm = SORRMv21_flux.mean(dim='Time').compute()
#SORRMv21_draft_tm = SORRMv21_draft.mean(dim='Time').compute()

#SORRMv21_flux_tm.rio.write_crs("epsg:3031",inplace=True);
#SORRMv21_draft_tm.rio.write_crs("epsg:3031",inplace=True);


ICESHELVES_MASK = gpd.read_file(main_dir / DIR_external / FILE_iceShelvesShape)
icems = ICESHELVES_MASK.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();


# Time mean of melt rate and draft
obs23_melt_tm = obs23_melt.mean(dim='time').compute()
obs23_draft_tm = obs23_draft.mean(dim='time').compute()
obs23_melt_anm_tm = obs23_melt_anm.mean(dim='time').compute()

obs23_melt_tm.rio.write_crs("epsg:3031",inplace=True);
obs23_melt_anm_tm.rio.write_crs("epsg:3031",inplace=True);
obs23_draft_tm.rio.write_crs("epsg:3031",inplace=True);

IMBIEregions = range(6,33)
iceShelfRegions = range(33,133)

sns.set_theme(style="whitegrid")

for i in iceShelfRegions:
    fig, ax = plt.subplots(1, 1, figsize=[12, 8])
    axins = ax.inset_axes([0.65, 0.6, 0.45, 0.35])
    print('extracting data for catchment {}'.format(icems.name.values[i]))
    melt_anm = clip_data(obs23_melt_anm_tm, i)
    melt = clip_data(obs23_melt_tm, i)
    z = clip_data(obs23_draft_tm, i)
    #sorrm_melt = clip_data(SORRMv21_flux_tm, i)
    #sorrm_z = clip_data(SORRMv21_draft_tm, i)
    ax.scatter(melt, z,color='r',s=1)
    ax.scatter(melt_anm, z,color='b',s=1)
    #ax.scatter(sorrm_melt, sorrm_z,color='b',s=2)
    ax.set_xlabel('Melt Rate (m/ yr)')
    #ax.set_xlabel('Freshwater Flux (kg/m2/s)')
    ax.set_ylabel('Draft (m)')
    ax.set_ylim(2000,0)
    #ax.set_xlim(0, 1e-3)
    #ax.set_xscale('log')
    ax.set_title(icems.name[i])
    icems[33:133].plot(ax=axins,linewidth=0.3)
    icems.loc[[i],'geometry'].plot(ax=axins,color='r')
    plt.savefig(main_dir / "reports/figures/interim/draft_dependence/obs23/{}_obs23_comp.png".format(icems.name[i]))
    print("saved fig {}.png".format(icems.name[i]))
    plt.close()
    del melt, z, melt_anm
    print('deleted interim variables')
    gc.collect()