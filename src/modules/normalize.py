#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 14, 2024

@author: Shivaprakash Muruganandham

Coarsen the input dataset from monthly to annual resolution
// TODO - test todo

"""
import sys
import os
os.environ['USE_PYGEOS'] = '0'
import gc
import collections
from pathlib import Path
from optparse import OptionParser

import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
# import seaborn as sns

import xarray as xr
import pandas as pd
import geopandas as gpd
import rioxarray
import scipy
from scipy import signal
import cftime
from shapely.geometry import mapping
from xarrayutils.utils import linear_trend, xr_linregress

import dask
import distributed


# Main directory path of project repository - all filepaths are relative to this
main_dir = Path.cwd().parent
DIR_external = 'data/external/'
DIR_interim = 'data/interim/'

parser = OptionParser(description=__doc__)
parser.add_option("-f", dest="fileInName", help="input filename", default="SORRMv21.ISMF.FULL.nc", metavar="FILENAME")
options, args = parser.parse_args()

p = Path(options.fileInName)

ds = xr.open_dataset(options.fileInName, chunks={"Time":36}) # Dataset
da = ds.__xarray_dataarray_variable__ # Data Array

# Normalize 
def normalize(data):
    data_tmean = data.mean('Time').compute()
    data_tstd = data.std('Time').compute()
    data_demeaned = data - data_tmean
    data_normalized = data_demeaned/data_tstd
    return data_tmean, data_tstd, data_normalized

da_tmean, da_tstd, da_normalized = normalize(da)

da_normalized.to_netcdf(main_dir / DIR_interim / '{}_NORMALIZE.nc'.format(p.stem))
da_tmean.to_netcdf(main_dir / DIR_interim / 'normalize/{}_TMEAN.nc'.format(p.stem))
da_tstd.to_netcdf(main_dir / DIR_interim / 'normalize/{}_TSTD.nc'.format(p.stem))
