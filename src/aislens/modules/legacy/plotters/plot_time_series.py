#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 13, 2024

@author: Shivaprakash Muruganandham

Script to plot time series of freshwater fluxes for an arbitrary number of files, 
specified to a certain region/domain. These domains can be individual ice shelves,
IMBIE catchments, larger sectors, or the entire ice sheet. Regions are defined by 
.geojson files, taken from geometric_features.

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

parser = OptionParser(description=__doc__)
parser.add_option("-1", dest="file1inName", help="input filename", default="SORRMv21.ISMF.FULL.nc", metavar="FILENAME")
parser.add_option("-2", dest="file2inName", help="input filename", metavar="FILENAME")
parser.add_option("-3", dest="file3inName", help="input filename", metavar="FILENAME")
parser.add_option("-4", dest="file4inName", help="input filename", metavar="FILENAME")
parser.add_option("-r", dest="regionName", help="Name of region of interest (ice shelves, IMBIE, sectors, AIS)", action='store_true', default='Antarctica')
options, args = parser.parse_args()

main_dir = Path.cwd().parent
#dir_ext_data = 'data/external/'
#dir_interim_data = 'data/interim/'
DIR_external = 'data/external/'
DIR_interim = 'data/interim/'
FILE_iceShelvesShape = 'iceShelves.geojson'

ICESHELVES_MASK = gpd.read_file(main_dir / DIR_external / FILE_iceShelvesShape)
icems = ICESHELVES_MASK.to_crs({'init': 'epsg:3031'});
crs = ccrs.SouthPolarStereo();

def clip_data(total_data, basin):
    total_data.rio.write_crs("epsg:3031",inplace=True);
    clipped_data = total_data.rio.clip(icems.loc[[basin],'geometry'].apply(mapping),icems.crs,drop=False)
    #clipped_data = clipped_data.dropna('time',how='all')
    #clipped_data = clipped_data.dropna('y',how='all')
    #clipped_data = clipped_data.dropna('x',how='all')
    #clipped_data = clipped_data.drop("month")
    return clipped_data

def time_series(data, domain):
    # Calculate time series of freshwater fluxes - spatial mean over the domain defined
    data = clip_data(data, domain)
    data_ts = data.mean(['y','x'])
    return data_ts

def plot_data(fname, region):
    print("Reading and plotting file: {}".format(fname))
    ds = xr.open_dataset(fname, chunks={"Time":36}) # Dataset
    da = ds.__xarray_dataarray_variable__ # Data Array
    #da = ds.timeMonthly_avg_landIceFreshwaterFlux
    da_ts = time_series(da, region) 

    fig, axs = plt.subplots(1, 2, figsize=(25, 8), gridspec_kw={'width_ratios': [1, 3]})
    axs[0].psd(da_ts);
    axs[0].set_xlabel('Frequency (cycles/month)')
    axs[0].set_xscale('log')
    axs[0].set_title('PSD of Spatially Averaged Basal Freshwater Flux')

    da_ts.plot(ax=axs[1], label=fname)
    axs[1].set_xlabel('Time (years)')
    axs[1].set_title('Spatially Averaged Basal Freshwater Flux')

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.1)) 
    #TODO: Update legend placement to outside axes, remove filepath from legend
    fig.suptitle('Time Series and Power Spectral Density of Basal Freshwater Flux for {}'.format(icems.name.values[region]))

region = int(options.regionName)

plot_data(options.file1inName, region)

if(options.file2inName):
   plot_data(options.file2inName, region)

if(options.file3inName):
   plot_data(options.file3inName, region)

if(options.file4inName):
   plot_data(options.file4inName, region)

print("Generating plot.")
# fig.tight_layout()
plt.show()