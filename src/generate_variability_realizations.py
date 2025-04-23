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