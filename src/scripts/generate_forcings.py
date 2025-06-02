# This script generates forcing realizations using EOF decomposition and 
# phase randomization.
# Run this script after running prepare_data.py.
# Steps:
#   1. Load the variability and seasonality datasets.
#   2. Use the variability dataset to generate realizations:
#       2.1. Normalize the dataset.
#       2.2. Perform EOF decomposition.
#       2.3. Perform phase randomization on the principal components based on 
#            n_realizations provided by user.
#       2.4. Generate synthetic data using the EOF model and randomized principal
#            components.
#       2.5. Unnormalize the generated data.
#   3. Extrapolate the original and generated variability datasets to the entire 
#      ice sheet grid.
#       3.1. This is done by filling NaN values with the nearest neighbor values
#            using the fill_nan_with_nearest_neighbor_vectorized function.
#       3.2. The extrapolated datasets are saved to a specified output path.
#   4. Extrapolate the seasonality dataset to the entire ice sheet grid similarly.
#       4.1. This is done by filling NaN values with the nearest neighbor values
#            using the fill_nan_with_nearest_neighbor_vectorized function.
#       4.2. The extrapolated dataset is saved to a specified output path.
#   5. Add the extrapolated seasonality to each of the variability datasets to 
#      create the final forcing dataset for each ensemble member.
#   6. Save the final forcing datasets to a specified output path.

from aislens.generator import generate_variability_realizations
from aislens.config import config
import xarray as xr

def generate_forcings():
    # Load extrapolated seasonality and variability datasets
    seasonality = xr.open_dataset(config.FILE_SEASONALITY_EXTRAPL)
    variability = xr.open_dataset(config.FILE_VARIABILITY_EXTRAPL)
    
    # Generate variability realizations
    realizations = generate_variability_realizations(variability, n_realizations=config.N_REALIZATIONS)
    
    # Add seasonality to each realization and save
    for i, realization in enumerate(realizations):
        forcing = realization + seasonality
        forcing.to_netcdf(config.DIR_FORCINGS / f"forcing_realization_{i}.nc")

if __name__ == "__main__":
    generate_forcings()