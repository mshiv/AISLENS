# This script defines the statistical generator
"""
Methods involved:
1. eof_decomposition: performs EOF decomposition on the input data
2. phase_randomization: performs phase randomization on the PCs of the EOF decomposition
3. generation: Generates synthetic data using the phase randomized PCs
4. unnormalize: Unnormalizes the generated data
5. save: Saves the generated data to a netcdf file
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
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import scipy
from scipy import signal
import cftime
from shapely.geometry import mapping
from xarrayutils.utils import linear_trend, xr_linregress
from xeofs.xarray import EOF

import dask
import distributed

parser = OptionParser(description=__doc__)
parser.add_option("-f", dest="fileInName", help="input normalized data filename", default="SORRMv21.ISMF.FULL.nc", metavar="FILENAME")
parser.add_option("-n", dest="nRealizations", help="number of ensemble members to be generated", default=5, metavar="N_REALIZATIONS")
options, args = parser.parse_args()

p = Path(options.fileInName)

main_dir = Path.cwd().parent
#dir_ext_data = 'data/external/'
#dir_interim_data = 'data/interim/'
DIR_external = 'data/external/'
DIR_interim = 'data/interim/'
DIR_processed = 'data/processed/'
FILE_iceShelvesShape = 'iceShelves.geojson'

def eof_decomposition(data):
    """
    Performs EOF decomposition on the input data
    """
    model = EOF(data)
    model.solve()
    eofs = model.eofs()
    pcs = model.pcs()
    nmodes = model.n_modes
    varexpl = model.explained_variance_ratio()
    return model, eofs, pcs, nmodes, varexpl

def phase_randomization(pcs, n_realizations):
    """
    Performs phase randomization on the PCs of the EOF decomposition
    """
    t_length = pcs.shape[0]
    new_pcs = np.empty((n_realizations,pcs.shape[0],pcs.shape[1]))

    for i in range(n_realizations):
        for m in range(nmodes):
            fl = pcs[:,m]
            fl_fourier = np.fft.rfft(fl)
            random_phases = np.exp(np.random.uniform(0,2*np.pi,int(len(fl)/2+1))*1.0j)
            fl_fourier_new = fl_fourier*random_phases
            new_pcs[i,:,m] = np.fft.irfft(fl_fourier_new)
        print('calculated ifft for realization {}, all modes'.format(i))
    return new_pcs

def generate_data(model,n_realization,mode,mode_skip):
    # mode can be any int in (1,nmodes), for cases 
    # when dimensionality reduction is preferred on the reconstructed dataset
    data_reconstr = model.reconstruct_randomized_X(new_fl[n_realization],slice(1,mode,mode_skip))
    #flux_reconstr = flux_reconstr.dropna('time',how='all')
    #flux_reconstr = flux_reconstr.dropna('y',how='all')
    #flux_reconstr = flux_reconstr.dropna('x',how='all')
    #flux_reconstr = flux_reconstr.drop("month")
    return data_reconstr


data = xr.open_dataset(options.fileInName)
data = data.__xarray_dataarray_variable__[:3600]

#TODO: change tdim from 'time' to 'Time', as the EOF method requires.
data = data.rename({"Time":"time"})

# Normalize 
data_tmean = data.mean('time')
data_tstd = data.std('time')
data_demeaned = data - data_tmean
data_norm = data_demeaned/data_tstd

# Perform EOF decomposition on normalized data
model, eofs, pcs, nmodes, varexpl = eof_decomposition(data_norm)

"""
# Save model pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

file_pi = open(str(main_dir / "data/interim/" / "norm_model.obj"), 'wb') 
pickle.dump(model, file_pi)
file_pi.close()
"""

n_realizations = options.nRealizations
new_pcs = phase_randomization(pcs, n_realizations)

# Generate dataset realizations

## Standard EOF/PCA implementation
# Can use the xeofs-rand package, or directly generate using sklearn PCA.


for i in range(n_realizations):
    data_reconstr = generate_data(model, i, 3600, 1)
    data_reconstr = (data_reconstr*data_tstd)+data_tmean
    #melt_reconstr = flux_reconstr*sec_per_year/rho_fw
    #melt_reconstr = melt_reconstr.rename('rec{}'.format(n_realizations))
    data_reconstr = data_reconstr.rename('sorrmvar_rec-{}'.format(n_realizations))
    data_reconstr.to_netcdf(main_dir / DIR_processed / '{}_sorrm-var-rec_{}.nc'.format(p.stem, i))
    print('reconstructed realization # {}'.format(i))