# Create and save final forcing files that include a trend (0 for control), seasonal and variability component.
from modules.io import load_dataset, save_dataset
from modules.generator import generate_realizations
from modules.normalize import normalize, unnormalize

def generate_variability_realizations(n_gen):
    # Load variability signal
    variability_file = "data/interim/basalmelt_variability_model_data.nc"
    variability_data = load_dataset(variability_file)

    # Normalize the data
    normalized_data, mean, std = normalize(variability_data)

    # Generate realizations
    realizations = generate_realizations(normalized_data, n_gen)

    # Unnormalize and save each realization
    for i, realization in enumerate(realizations):
        unnormalized_realization = unnormalize(realization, mean, std)
        save_dataset(
            unnormalized_realization,
            f"data/processed/forcings/basalmelt_variability_generated_{i}.nc",
        )

def create_forcing_files(n_gen):
    # Load seasonal signal
    seasonality_file = "data/interim/basalmelt_seasonality_model_data.nc"
    seasonality_data = load_dataset(seasonality_file)

    for i in range(n_gen):
        # Load variability realization
        variability_file = f"data/processed/forcings/basalmelt_variability_generated_{i}.nc"
        variability_data = load_dataset(variability_file)

        # Add seasonal signal to create forcing
        forcing_data = variability_data + seasonality_data

        # Save forcing file
        save_dataset(
            forcing_data,
            f"data/processed/forcings/basalmelt_forcing_anomaly_{i}.nc",
        )