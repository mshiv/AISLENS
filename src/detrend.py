from pathlib import Path
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt


# TODO : Dask implementation of the detrend function
# Refer: https://ncar.github.io/esds/posts/2022/dask-debug-detrend/

def detrend_dim(data, dim, deg):
    # detrend along a single dimension
    p = data.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(data[dim], p.polyfit_coefficients)
    return data - fit

# Main directory path of project repository - all filepaths are relative to this
main_dir = Path.cwd().parent
DIR_external = 'data/external/'
DIR_interim = 'data/interim/'

# DATASET FILEPATHS
# Ocean model output - E3SM (SORRMv2.1.ISMF), data received from Darin Comeau / Matt Hoffman at LANL
# This is a 1000 year long simulation of the ocean. 
# The data is regridded to a 10 km resolution and is available at a monthly output.
# We make use of years 300 to 900 for our analysis:
#   - removing the first 300 years as spin-up 
#   - removing the last 100 years to avoid any unnecessary drift

FILE_SORRMv21 = 'Regridded_SORRMv2.1.ISMF.FULL.nc'
FILE_iceShelvesShape = 'iceShelves.geojson'

# Ocean model output
# Should this be hard-coded here? Or should it be passed as an argument? (default values being as below)
yr1 = 300
yr2 = 900
SORRMv21 = xr.open_dataset(main_dir.parent / 'aislens_emulation/' / DIR_external / 'SORRMv2.1.ISMF/regridded_output/' / FILE_SORRMv21, chunks={"Time":36})
SORRMv21_flux = SORRMv21.timeMonthly_avg_landIceFreshwaterFlux[yr1*12:yr2*12]
SORRMv21_draft = SORRMv21.timeMonthly_avg_ssh

flux_detrend = detrend_dim(SORRMv21_flux,"Time",1)
flux_detrend = flux_detrend.compute()

flux_detrend.to_netcdf(main_dir / DIR_interim / "SORRMv21_{}-{}_DETREND.nc".format(yr1,yr2))