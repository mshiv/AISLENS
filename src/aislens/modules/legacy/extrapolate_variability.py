# Given 3 data files:
# 1. Netcdf file of landIceFreshwaterFlux variability in three dimensions (time, x, y)
# 2. Netcdf file / geojson file of current ice shelf grounding line extent (mask of 1s at current GL position, 0s elsewhere)
# 3. Netcdf file / geojson file of future ice shelf grounding line extent (mask of 1s at future GL position, 0s elsewhere)

# This script will:
# Extrapolate the data point of landIceFreshwaterFlux variability to pixel locations between the two grounding line extents based on a nearest neighbor approach
# Save the extrapolated data to a new netcdf file

# Usage: python extrapolate_variability.py <input_data> <current_gl> <future_gl> <output_file>

import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.warp
import rasterio.transform
import rasterio.crs
import fiona
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import shape
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion

