# Script to preprocess the data
# Input is a raw 3D xarray dataset, presumably with dimensions (time, x, y)
# Output is a preprocessed 3D xarray dataset, with the same dimensions
# Steps:
# 1. Detrend the data along the time dimension using a linear fit - This is done to remove any long-term trends in the data, physical or not
# 2. Remove the seasonal cycle from the data - This is done to remove any seasonal variability in the data, so that we can focus on the interannual variability
# 3. Clip the data to a specific domain - This is done to focus on a specific region of interest
# 4. Remove the draft dependence from the data - This is done to remove the dependence of the melt rate on the ice shelf draft, and a function is defined for each ice shelf

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
from xarrayutils.utils import linear_trend, xr_linregress
import cftime
import dask
import distributed
import scipy
from scipy import signal

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
SORRMv21 = xr.open_dataset(main_dir.parent / 'aislens_emulation/' / DIR_external / 'SORRMv2.1.ISMF/regridded_output/' / FILE_SORRMv21, chunks={"Time":36})
SORRMv21_flux = SORRMv21.timeMonthly_avg_landIceFreshwaterFlux[yr1*12:yr2*12]
SORRMv21_draft = SORRMv21.timeMonthly_avg_ssh

# Detrend the data




# Remove the seasonal cycle

# Clip the data

# Remove the draft dependence

