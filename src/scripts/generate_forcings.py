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

from aislens.generator import eof_decomposition, phase_randomization, generate_data
from aislens.config import config
import xarray as xr
import pickle
from pathlib import Path

def generate_forcings():
    # Load extrapolated seasonality and variability datasets
    seasonality = xr.open_dataset(config.FILE_SEASONALITY_EXTRAPL, chunks={config.TIME_DIM: 36})
    variability = xr.open_dataset(config.FILE_VARIABILITY_EXTRAPL, chunks={config.TIME_DIM: 36})
    # Verify that the time dimension in the dataset is named "time"
    if 'Time' in variability.dims:
        variability = variability.rename({"Time": "time"})
    if 'Time' in seasonality.dims:
        seasonality = seasonality.rename({"Time": "time"})
    data = variability[config.SORRM_FLUX_VAR]
    data_tmean = data.mean('time')
    data_tstd = data.std('time')
    data_norm = (data - data_tmean) / data_tstd
    print("Data normalization complete.")
    print("Performing EOF decomposition...")
    # Comment this out if you want to load the model from a pickle file instead of running EOF.
    model, _, pcs, nmodes, _ = eof_decomposition(data_norm)
    print("EOF DECOMP COMPLETE.")
    print("Save model to pickle file...")
    pickle_path = Path(config.DIR_PROCESSED) / "model.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    # Uncomment the following lines to load the model from a pickle file instead of running EOF above.
    # with open(pickle_path, "rb") as f:
    #     model = pickle.load(f)
    n_realizations = config.N_REALIZATIONS
    print(f"Phase randomization for {n_realizations} realizations...")
    new_pcs = phase_randomization(pcs.values, n_realizations)
    print("Phase randomization complete.")
    print("Generating synthetic data...")
    for i in range(n_realizations):
        new_data = generate_data(model, new_pcs, i, nmodes, 1)
        new_data = (new_data * data_tstd) + data_tmean
        new_data = xr.DataArray(new_data, dims=data.dims, coords=data.coords)
        new_data.attrs = data.attrs
        new_data.name = data.name
        forcing = seasonality + new_data
        forcing.to_netcdf(config.DIR_FORCINGS / f"forcing_realization_{i}.nc")
        print(f"Generated forcing realization {i} and saved to {config.DIR_FORCINGS / f'forcing_realization_{i}.nc'}")  

if __name__ == "__main__":
    generate_forcings()