# Prepare satellite observations and model simulation data as required for subsequent steps.

from modules.io import load_dataset, save_dataset
from modules.data_preprocess import detrend_dim

def prepare_satobs_data(input_path, output_path):
    """
    Prepare satellite observation data by calculating the time mean.

    Args:
        input_path (str): Path to the input satellite observation dataset.
        output_path (str): Path to save the prepared dataset.

    Returns:
        xarray.Dataset: Prepared satellite observation dataset.
    """
    # Load the dataset
    satobs_data = load_dataset(input_path)

    # Calculate the time mean
    time_mean = satobs_data.mean(dim="Time")

    # Save the prepared dataset
    save_dataset(time_mean, output_path)
    print(f"Satellite observation data prepared and saved to {output_path}")

    return time_mean

def prepare_model_data(input_path, output_path, start_year, end_year):
    """
    Prepare model simulation data by subsetting and detrending.

    Args:
        input_path (str): Path to the input model dataset.
        output_path (str): Path to save the prepared dataset.
        start_year (int): Start year for subsetting the dataset.
        end_year (int): End year for subsetting the dataset.

    Returns:
        xarray.Dataset: Prepared model dataset.
    """
    # Load the dataset
    model_data = load_dataset(input_path)

    # Subset the dataset to the desired time range
    subset_data = model_data.sel(Time=slice(start_year, end_year))

    # Perform detrending along the time dimension
    detrended_data = detrend_dim(subset_data, dim="Time", deg=1)

    # Save the prepared dataset
    save_dataset(detrended_data, output_path)
    print(f"Model data prepared and saved to {output_path}")

    return detrended_data

def run_prepare_data_workflow():
    """
    Run the data preparation workflow for satellite observations and model data.
    """
    print("Running data preparation workflow...")

    # Prepare satellite observation data
    prepare_satobs_data(
        input_path="data/external/basalmelt_satobs_data.nc",
        output_path="data/interim/satobs_time_mean.nc"
    )

    # Prepare model simulation data
    prepare_model_data(
        input_path="data/external/basalmelt_model_data.nc",
        output_path="data/interim/model_detrended.nc",
        start_year=300,
        end_year=900
    )

    print("Data preparation workflow complete.")