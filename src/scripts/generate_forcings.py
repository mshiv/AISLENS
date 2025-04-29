from aislens.generator import eof_decomposition, phase_randomization, generate_data
from aislens.dataprep import normalize, unnormalize, extrapolate
from aislens.io import load_dataset, save_dataset
from aislens.config import CONFIG

def generate_forcings(input_path, output_path, n_realizations, grid_file):
    """
    Generate forcing realizations using EOF decomposition and phase randomization.

    Args:
        input_path (str): Path to the input dataset.
        output_path (str): Path to save the generated forcing realizations.
        n_realizations (int): Number of realizations to generate.
        grid_file (str): Path to the ice sheet grid file for extrapolation.
    """
    # Load the dataset
    dataset = load_dataset(input_path)

    # Normalize the dataset
    normalized_data, norm_params = normalize(dataset)

    # Perform EOF decomposition
    model, eofs, pcs, nmodes, varexpl = eof_decomposition(normalized_data)

    # Perform phase randomization
    randomized_pcs = phase_randomization(pcs, n_realizations)

    # Generate synthetic data
    synthetic_data = generate_data(model, randomized_pcs, realization_idx=0, mode=nmodes, mode_skip=1)

    # Unnormalize the generated data
    unnormalized_data = unnormalize(synthetic_data, norm_params)

    # Extrapolate the generated data to the entire ice sheet grid
    extrapolated_data = extrapolate(unnormalized_data, grid_file)

    # Save the extrapolated data
    save_dataset(extrapolated_data, output_path)
    print(f"Forcing realizations saved to {output_path}")

if __name__ == "__main__":
    generate_forcings(
        input_path=CONFIG["variability_output_path"],
        output_path=CONFIG["forcing_output_path"],
        n_realizations=CONFIG["n_realizations"],
        grid_file=CONFIG["grid_file"]
    )