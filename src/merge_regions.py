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


data = SORRMv21
# ds = data.melt
ds = data.timeMonthly_avg_landIceFreshwaterFlux

np_flux_array = np.empty(ds[0].shape)
np_flux_array[:] = np.nan

iceshelves_rgrs_array = xr.DataArray(np_flux_array, coords=ds[0].coords, dims = ds[0].dims, attrs=ds.attrs)
#iceshelves_rgrs = xr.Dataset(data_vars=dict(timeMonthly_avg_landIceFreshwaterFlux=(iceshelves_rgrs_array)), coords=data.coords, attrs=data.timeMonthly_avg_landIceFreshwaterFlux.attrs)
iceshelves_rgrs = xr.Dataset(data_vars=dict(melt=(iceshelves_rgrs_array)))

IMBIEregions = range(6,33)
iceShelfRegions = range(33,133)

for i in iceShelfRegions:
    iceshelves_rgrs_catchment = xr.open_dataset(main_dir / DIR_interim /'{}_DEDRAFT_REGRESS.nc'.format(icems.name.values[i]))
    # Commented out the below to remain consistent with prior interim data generation steps. 
    # Use default xarray variable name throughout for now.
    #iceshelves_rgrs_catchment['melt'] = iceshelves_rgrs_catchment['__xarray_dataarray_variable__']
    #iceshelves_rgrs_catchment = iceshelves_rgrs_catchment.drop(['__xarray_dataarray_variable__'])
    iceshelves_rgrs = xr.merge([iceshelves_rgrs, iceshelves_rgrs_catchment], compat='no_conflicts')


iceshelves_rgrs.to_netcdf(main_dir / DIR_interim / "dedraft/SORRMv21_DRAFT_DEPENDENCE_FIT.nc")