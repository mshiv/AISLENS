from src.create_forcings import create_forcing_files
from src.derive_draft_dependence import preprocess_satellite_observations
from src.extract_variability import preprocess_model_data
from src.generate_variability_realizations import generate_variability_realizations

def main():
    # Step 1: Preprocess satellite observations
    preprocess_satellite_observations()

    # Step 2: Preprocess model data
    preprocess_model_data()

    # Step 3: Generate variability realizations
    n_gen = 10  # Number of realizations
    generate_variability_realizations(n_gen)

    # Step 4: Create forcing files
    create_forcing_files(n_gen)

if __name__ == "__main__":
    main()