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