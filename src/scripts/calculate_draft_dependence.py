# This script preprocesses satellite observation data to calculate and extrapolate 
# draft dependence parameters.
# Run this script after running prepare_data.py
# Steps:
#   1. Load the satellite observation dataset.
#   2. Calculate draft dependence parameters (draftDepenBasalMeltAlpha0 and 
#      draftDepenBasalMeltAlpha1) using the dedraft function.
#       2.1. This dedrafting is done by first, splitting the dataset into 
#            different ice shelf regions and calculating draft params separately 
#            for each one. Do this by using the ice shelf masks.
#       2.2. Then, the draft params are merged across the entire ice sheet.
#       2.3. The draft params are then saved to a file.
#   3. Extrapolate the draft dependence parameters to the entire ice sheet grid.
#       3.1. This is done by filling NaN values with the nearest neighbor values 
#            using the fill_nan_with_nearest_neighbor_vectorized function.
#   4. Save the extrapolated parameters to a specified output path.

from aislens.dataprep import dedraft, setup_draft_depen_field
from aislens.geospatial import find_ice_shelf_index, clip_data, process_ice_shelf
from aislens.utils import fill_nan_with_nearest_neighbor_vectorized, initialize_directories, write_crs
from aislens.config import config
import xarray as xr
from shapely.geometry import mapping

# Load the prepared satellite observation dataset
satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
icems = xr.open_dataset(config.FILE_ICESHELFMASKS)

# 3 main functions to be run:
# 1. dedraft_ice_shelf_region: Dedraft the satellite observation data for each ice shelf region.
# 2. process_ice_shelf_region: Process a single ice shelf region: perform regression, create DataArrays, mask, write output.
# 3. calculate_draft_dependence: Calculate draft dependence parameters for all ice shelf regions.
# Dedraft the satellite observation data for each ice shelf region
# Loop through each ice shelf region defined in the configuration
# Merge the dedrafted data across the entire ice sheet

def process_catchment(i, icems, satobs, config):
    print(f'Extracting data for catchment {icems.name.values[i]}')
    ds = clip_data(satobs, i, icems)
    ds_tm = ds.mean(dim='time')
    
    print(f'Calculating linear regression for catchment {icems.name.values[i]}')
    mlt_coef, mlt_intercept = dedraft(ds.melt, ds.draft)
    
    # Retrieve attribute keys explicitly for clarity
    alpha0_key, alpha1_key = list(config.DATA_ATTRS.keys())
    mlt_coef_ds = setup_draft_depen_field(ds_tm.melt, mlt_coef, alpha1_key, i, icems)
    mlt_intercept_ds = setup_draft_depen_field(ds_tm.melt, mlt_intercept, alpha0_key, i, icems)
    
    # Combine and save
    mlt_coefs = xr.Dataset({
        mlt_coef_ds.name: mlt_coef_ds,
        mlt_intercept_ds.name: mlt_intercept_ds
        })
    mlt_coefs.to_netcdf(config.DIR_ICESHELF_DEDRAFT / f'draftDepenBasalMeltAlpha_{icems.name.values[i]}.nc')
    print(f'{icems.name.values[i]} file saved')

def calculate_draft_dependence(icems, satobs, config):
    for i in config.ICE_SHELF_REGIONS:
        process_catchment(i, icems, satobs, config)
    draft_dependence_params = xr.Dataset()
    for i in config.ICE_SHELF_REGIONS:
        draft_dependence_params = xr.merge([draft_dependence_params, xr.open_dataset(config.DIR_ICESHELF_DEDRAFT / 'draftDepenBasalMeltAlpha_{}.nc'.format(icems.name.values[i]))])
    # Extrapolate draft dependence parameters
    draft_dependence_params = draft_dependence_params.fillna(0)  # Fill NaN values with 0 before extrapolation
    # draft_dependence_extrapolated = fill_nan_with_nearest_neighbor_vectorized(draft_dependence_params)
    # Save the extrapolated parameters
    draft_dependence_params.to_netcdf(config.FILE_DRAFT_DEPENDENCE)

if __name__ == "__main__":
    calculate_draft_dependence(icems, satobs, config)