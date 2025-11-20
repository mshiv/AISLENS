# Prepare satellite observations and model simulation data as required for 
# subsequent steps.
# This is the FIRST script to be run in the workflow.
# Satellite observations:
#   1. Load the satellite observation dataset.
#   2. Detrend the data along the time dimension (retain the mean value).
#   3. Deseasonalize the data.
#   4. Take the time-mean of the dataset.
#   5. Ensure that melt is converted to flux, if not, and is in SI units.
#   6. Save the prepared data fields of meltflux and draft.
# Model simulation data:
#   1. Load the model simulation dataset.
#   2. Subset the dataset to the desired time range.
#   3. Detrend the data along the time dimension.
#   4. Deseasonalize the data.
#   5. Dedraft the data.
#   6. Save the seasonality and variability components thus obtained.

from aislens.dataprep import detrend_dim, deseasonalize, dedraft_catchment, extrapolate_catchment_over_time
from aislens.utils import merge_catchment_files, subset_dataset_by_time, collect_directories, initialize_directories, write_crs
from aislens.config import config
import numpy as np
import xarray as xr
import geopandas as gpd

# initialize_directories(collect_directories(config))

def prepare_satellite_observations():
    # Load satellite observation dataset
    print("Preparing satellite observaticons...")
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS)
    print("Satellite observations loaded successfully.")
    # Detrend the data along the time dimension
    print("Detrending satellite observations...")
    satobs_deseasonalized = satobs.copy()
    satobs_detrended = detrend_dim(satobs_deseasonalized[config.SATOBS_FLUX_VAR], dim=config.TIME_DIM, deg=1)
    print("Satellite observations detrended successfully.")
    print("Deseasonalizing satellite observations...")
    # Deseasonalize the data
    satobs_deseasonalized[config.SATOBS_FLUX_VAR] = deseasonalize(satobs_detrended)
    print("Satellite observations deseasonalized successfully.")
    # TODO: Take the time-mean of the dataset
    # TODO: Ensure that melt is converted to flux, if not, and is in SI units

    # Save the prepared data
    satobs_deseasonalized.to_netcdf(config.FILE_PAOLO23_SATOBS_PREPARED)

def prepare_model_simulation():
    # Load model simulation dataset
    print("Preparing model simulation data...")
    model = xr.open_dataset(config.FILE_MPASO_MODEL, chunks={config.TIME_DIM: 36})
    model = write_crs(model, config.CRS_TARGET)

    print("Model simulation data loaded successfully.")
    print("Subsetting model simulation data by time...")
    model_subset = subset_dataset_by_time(model,
                                          time_dim=config.TIME_DIM,
                                          start_year=config.SORRM_START_YEAR,
                                          end_year=config.SORRM_END_YEAR,
                                          )
    print("Model simulation data subsetted successfully.")
    print("Detrending model simulation data...")
    # Detrend, deseasonalize, and dedraft the data
    model_detrended = model_subset.copy()
    model_detrended[config.SORRM_FLUX_VAR] = detrend_dim(model_subset[config.SORRM_FLUX_VAR], dim=config.TIME_DIM, deg=1)
    print("Model simulation data detrended successfully.")
    print("Deseasonalizing model simulation data...")
    model_deseasonalized = deseasonalize(model_detrended)
    print("Model simulation data deseasonalized successfully.")
    print("Dedrafting model simulation data...")
    icems = gpd.read_file(config.FILE_ICESHELFMASKS);
    icems = icems.to_crs({'init': config.CRS_TARGET});

    for i in config.ICE_SHELF_REGIONS:
        dedraft_catchment(i, icems, model_deseasonalized, config, 
                          save_dir=config.DIR_ICESHELF_DEDRAFT_MODEL,
                          save_pred=True
                          )
    draft_dependence_pred = xr.Dataset()
    for i in config.ICE_SHELF_REGIONS:
        draft_dependence_pred = xr.merge([draft_dependence_pred, xr.open_dataset(config.DIR_ICESHELF_DEDRAFT_MODEL / 'draftDepenModelPred_{}.nc'.format(icems.name.values[i]))])
    print("Model simulation data dedrafted successfully.")
    print("Merging draft dependence predictions across all ice shelf regions...")
    draft_dependence_pred = merge_catchment_files([config.DIR_ICESHELF_DEDRAFT_MODEL / f'draftDepenModelPred_{icems.name.values[i]}.nc'
                                                   for i in config.ICE_SHELF_REGIONS
                                                   ])
    print("Draft dependence predictions merged successfully.")
    print("Calculating variability and seasonality components...")
    model_variability = model_deseasonalized - draft_dependence_pred
    model_seasonality = model_detrended - model_deseasonalized
    print("Saving model components...")
    model_seasonality.to_netcdf(config.FILE_SEASONALITY)
    model_variability.to_netcdf(config.FILE_VARIABILITY)
    print("Model components saved successfully.")
    print("Processing complete. Extrapolating components to the entire ice sheet grid...")

    # Extrapolate the seasonality and variability components to the entire ice sheet grid
    model_variability_extrapl = extrapolate_catchment_over_time(model_variability, 
                                                                icems, config, 
                                                                config.SORRM_FLUX_VAR
                                                                )
    model_seasonality_extrapl = extrapolate_catchment_over_time(model_seasonality, 
                                                                icems, config, 
                                                                config.SORRM_FLUX_VAR
                                                                )
    print("Components extrapolated successfully.")
    # Save the processed components
    model_seasonality_extrapl.to_netcdf(config.FILE_SEASONALITY_EXTRAPL)
    model_variability_extrapl.to_netcdf(config.FILE_VARIABILITY_EXTRAPL)
    print("Processing complete. Model components saved.")


if __name__ == "__main__":
    dirs_to_create = collect_directories(config)
    initialize_directories(dirs_to_create)
    #prepare_satellite_observations()
    prepare_model_simulation() 