from aislens.dataprep import detrend_dim, deseasonalize
from aislens.io import load_dataset, save_dataset
from aislens.config import CONFIG

def extract_variability(input_path, output_path, time_dim):
    """
    Preprocess model simulation data to extract variability and seasonality components.

    Args:
        input_path (str): Path to the input dataset.
        output_path (str): Path to save the processed dataset.
        time_dim (str): Name of the time dimension.
    """
    # Load the dataset
    dataset = load_dataset(input_path)

    # Detrend the dataset
    detrended_data = detrend_dim(dataset, dim=time_dim, deg=1)

    # Deseasonalize the dataset
    deseasonalized_data = deseasonalize(detrended_data)

    # Save the processed dataset
    save_dataset(deseasonalized_data, output_path)
    print(f"Variability data saved to {output_path}")

if __name__ == "__main__":
    extract_variability(
        input_path=CONFIG["model_input_path"],
        output_path=CONFIG["variability_output_path"],
        time_dim=CONFIG["time_dim"]
    )