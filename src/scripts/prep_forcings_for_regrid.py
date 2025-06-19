# This script regrids the draft dependence parameters and forcing anomalies to the 
# MPAS grid.
# It is the last script to be run in the workflow.
# STEPS:
#   1. Load the draft dependence parameters and forcing anomalies dataset.
#   2. Rename the variables and dimensions to match the MPAS grid specifications 
#      defined by the user in the config file. (These are based on the dictionaries 
#      defined in src/MPAS-Tools/interpolate_to_mpasli_grid_mod.py). 
#      This step and the next are together performed using 
#      aislens.utils.rename_dims_and_fillna and aislens.utils.process_directory
#       2.1. Dimensions: x1, y1, Time.
#       2.2. Draft dependence parameters: 
#                   draftDepenBasalMeltAlpha0, draftDepenBasalMeltAlpha1
#            Forcing anomalies: 
#                   floatingBasalMassBalAdjustment
#   3. Fill any remaining NaN values in these 3 fields with 0.
#   3. Regrid the dataset to the MPAS grid using 
#      src/MPAS-Tools/interpolate_to_mpasli_grid_mod.py (via CLI arguments).
#   4. Save the regridded datasets to a specified output path.
#   5. Print a message indicating the completion of the regridding process.

from aislens.utils import rename_dims_and_fillna, process_directory
from aislens.config import config
import cftime

def regrid_to_mali():
    # Regrid draft dependence parameters
    #rename_dims_and_fillna(config.FILE_DRAFT_DEPENDENCE, dims_to_rename={"x": "x1", "y": "y1"}, fill_value=0)
    # Regrid forcing realizations
    process_directory(config.DIR_FORCINGS, dims_to_rename={"x": "x1", "y": "y1", "time": "Time"}, fill_value=0)
    # process_directory(config.DIR_FORCINGS, fill_value=0)

if __name__ == "__main__":
    regrid_to_mali()