# This file defines file paths and constants used throughout the codebase.

from pathlib import Path
from pyprojroot import here

# Define the root directory of the project
BASE_DIR =  here() # or use here(project_files=[".git"])

##############################################################################
### DIRECTORY PATHS
##############################################################################

# Define base data directories
DIR_EXTERNAL = BASE_DIR / "data/external"
DIR_INTERIM = BASE_DIR / "data/interim"
DIR_PROCESSED = BASE_DIR / "data/processed"
DIR_TMP = BASE_DIR / "data/tmp"
# Output paths for interim datasets
DIR_ICESHELF_DEDRAFT = DIR_INTERIM / "draft_dependence"
# Output paths for processed datasets
DIR_DRAFT_DEPENDENCE = DIR_PROCESSED / "draft_dependence"
DIR_VARGENS = DIR_PROCESSED / "vargen_realizations"
DIR_FORCINGS = DIR_PROCESSED / "forcings" # this is the sum of extrapolated seasonality and vargen_realizations

# Output paths for final MALI grid fields
DIR_MALI_FORCINGS = DIR_PROCESSED / "mali_grid/forcings"
DIR_MALI_DRAFT_DEPENDENCE = DIR_PROCESSED / "mali_grid/draft_dependence"

##############################################################################
# FILENAMES
##############################################################################
# Define file paths for input datasets
FILE_PAOLO23_SATOBS = DIR_EXTERNAL / "ANT_G1920V01_IceShelfMeltDraft.nc"
FILE_MPASO_MODEL = DIR_EXTERNAL /  "Regridded_SORRMv2.1.ISMF.FULL.nc"
FILE_ICESHELFMASKS = DIR_EXTERNAL / "iceShelves.geojson"
# Define file paths for intermediate and processed datasets
FILE_PAOLO23_SATOBS_PREPARED = DIR_PROCESSED / "satellite_observations_prepared.nc"
FILE_MPASO_MODEL_PREPARED = DIR_PROCESSED / "model_simulation_prepared.nc"
FILE_DRAFT_DEPENDENCE = DIR_PROCESSED / "draft_dependence_params.nc"
FILE_SEASONALITY = DIR_PROCESSED / "sorrm_seasonality.nc"
FILE_VARIABILITY = DIR_PROCESSED / "sorrm_variability.nc"
FILE_FORCING = DIR_PROCESSED / "sorrm_forcing.nc"

##############################################################################
# CONFIG object to store all file paths and constants
##############################################################################

CONFIG = {
    # Input paths
    "satobs_data_raw": FILE_PAOLO23_SATOBS,
    "model_data_raw": FILE_MPASO_MODEL,
    "iceshelf_masks": FILE_ICESHELFMASKS,

    # Output paths for prepared datasets
    "satobs_data_prepared": FILE_PAOLO23_SATOBS_PREPARED,
    "model_data_prepared": FILE_MPASO_MODEL_PREPARED,

    # Output paths for intermediate and processed datasets
    "sorrm_seasonality": FILE_SEASONALITY,
    "sorrm_variability": FILE_VARIABILITY,
    "sorrm_forcing": FILE_FORCING,
    "draft_dependence_params": FILE_DRAFT_DEPENDENCE,

    # Output directories for dedraft parameters and regressions
    "dedraft_params_dir": DIR_ICESHELF_DEDRAFT,
    "dedraft_regress_dir": DIR_ICESHELF_DEDRAFT,

    # Output paths for final MALI grid fields
    "mali_forcings_output_path": DIR_MALI_FORCINGS,
    "mali_draft_dependence_output_path": DIR_MALI_DRAFT_DEPENDENCE,

    # Constants for processing
    "grid_file": FILE_ICESHELFMASKS,  # Assuming this is the grid file
    "time_dim": "Time",  # Name of the time dimension
    "draft_var": "draft",  # Name of the draft variable
    "flux_var": "freshwater_flux",  # Name of the freshwater flux variable
    "start_year": 2000,  # Example start year for subsetting
    "end_year": 2020,  # Example end year for subsetting
    "n_realizations": 10,  # Number of realizations for forcing generation
    "crs_target": "EPSG:3031",  # Target CRS for geospatial data
}