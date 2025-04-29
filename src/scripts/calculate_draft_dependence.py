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

from aislens.dataprep import dedraft
from aislens.utils import fill_nan_with_nearest_neighbor_vectorized
from aislens.config import config
import xarray as xr

def calculate_draft_dependence():
    # Load the prepared satellite observation dataset
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
    
    # Calculate draft dependence parameters
    draft_dependence = dedraft(satobs[config.SATOBS_FLUX_VAR], satobs[config.SATOBS_DRAFT_VAR])
    
    # Extrapolate draft dependence parameters
    draft_dependence_extrapolated = fill_nan_with_nearest_neighbor_vectorized(draft_dependence)
    
    # Save the extrapolated parameters
    draft_dependence_extrapolated.to_netcdf(config.FILE_DRAFT_DEPENDENCE)

if __name__ == "__main__":
    calculate_draft_dependence()