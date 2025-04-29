# Prepare satellite observations and model simulation data as required for subsequent steps.

from aislens.io import load_dataset, save_dataset
from aislens.dataprep import detrend_dim
from aislens.config import CONFIG

def prepare_satobs_data(input_path, output_path, time_dim):
    """
    Prepare satellite observation data by calculating the time mean.

    Args:
        input_path (str): Path to the input satellite observation dataset.
        output_path (str): Path to save the prepared dataset.
        time_dim (str): Name of the time dimension.

    Returns:
        xarray.Dataset: Prepared satellite observation dataset.
    """
    # Load the dataset
    satobs_data = load_dataset(input_path)

    # Calculate the time mean
    time_mean = satobs_data.mean(dim=time_dim)

    # Save the prepared dataset
    save_dataset(time_mean, output_path)
    print(f"Satellite observation data prepared and saved to {output_path}")

    return time_mean

def prepare_model_data(input_path, output_path, start_year, end_year, time_dim):
    """
    Prepare model simulation data by subsetting and detrending.

    Args:
        input_path (str): Path to the input model dataset.
        output_path (str): Path to save the prepared dataset.
        start_year (int): Start year for subsetting the dataset.
        end_year (int): End year for subsetting the dataset.
        time_dim (str): Name of the time dimension.

    Returns:
        xarray.Dataset: Prepared model dataset.
    """
    # Load the dataset
    model_data = load_dataset(input_path)

    # Subset the dataset to the desired time range
    subset_data = model_data.sel({time_dim: slice(start_year, end_year)})

    # Perform detrending along the time dimension
    detrended_data = detrend_dim(subset_data, dim=time_dim, deg=1)

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
        input_path=CONFIG["satobs_input_path"],
        output_path=CONFIG["satobs_output_path"],
        time_dim=CONFIG["time_dim"]
    )

    # Prepare model simulation data
    prepare_model_data(
        input_path=CONFIG["model_input_path"],
        output_path=CONFIG["model_output_path"],
        start_year=CONFIG["start_year"],
        end_year=CONFIG["end_year"],
        time_dim=CONFIG["time_dim"]
    )

    print("Data preparation workflow complete.")

if __name__ == "__main__":
    run_prepare_data_workflow()