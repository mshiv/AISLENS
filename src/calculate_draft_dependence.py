# Preprocess satellite observation data to calculate draft dependence parameterization

from modules.io import load_dataset, save_dataset
from modules.data_preprocessing import detrend, deseasonalize
from modules.draft_dependence import calculate_draft_dependence
from modules.masks import clip_to_ice_shelves

def preprocess_satellite_observations():
    # Load satellite observations
    satobs_file = "data/external/basalmelt_satobs_data.nc"
    satobs_data = load_dataset(satobs_file)

    # Calculate time mean of freshwater flux
    mean_flux = satobs_data["freshwater_flux"].mean(dim="Time")

    # Dedraft the dataset
    draft_dependence_params = calculate_draft_dependence(
        satobs_data["draft"], mean_flux
    )

    # Save draft dependence parameters
    save_dataset(
        draft_dependence_params, "data/processed/draft_dependence_params.nc"
    )