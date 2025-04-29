from aislens.dataprep import dedraft, extrapolate
from aislens.io import load_dataset, save_dataset
from aislens.config import CONFIG

def preprocess_satellite_observations(input_path, output_path, draft_var, flux_var, grid_file):
    """
    Preprocess satellite observation data to calculate and extrapolate draft dependence.

    Args:
        input_path (str): Path to the input satellite observation dataset.
        output_path (str): Path to save the draft dependence parameters.
        draft_var (str): Name of the draft variable.
        flux_var (str): Name of the freshwater flux variable.
        grid_file (str): Path to the ice sheet grid file for extrapolation.
    """
    # Load satellite observations
    satobs_data = load_dataset(input_path)

    # Calculate draft dependence
    draft_dependence_params = dedraft(satobs_data[flux_var], satobs_data[draft_var])

    # Extrapolate draft dependence parameters to the entire ice sheet grid
    extrapolated_params = extrapolate(draft_dependence_params, grid_file)

    # Save the extrapolated parameters
    save_dataset(extrapolated_params, output_path)
    print(f"Draft dependence parameters saved to {output_path}")

if __name__ == "__main__":
    preprocess_satellite_observations(
        input_path=CONFIG["satobs_input_path"],
        output_path=CONFIG["draft_dependence_output_path"],
        draft_var=CONFIG["draft_var"],
        flux_var=CONFIG["flux_var"],
        grid_file=CONFIG["grid_file"]
    )