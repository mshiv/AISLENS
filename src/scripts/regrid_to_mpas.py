# Regrid the draft dependence parameters and forcing anomalies to the MPAS grid
from aislens.geospatial import regrid_to_mpas
from aislens.io import load_dataset, save_dataset
from aislens.config import CONFIG

def regrid_data(input_path, output_path, mpas_grid_file):
    """
    Regrid draft dependence parameters and forcing anomalies to the MPAS grid.

    Args:
        input_path (str): Path to the input dataset.
        output_path (str): Path to save the regridded dataset.
        mpas_grid_file (str): Path to the MPAS grid file.
    """
    # Load the dataset
    dataset = load_dataset(input_path)

    # Regrid the dataset
    regridded_data = regrid_to_mpas(dataset, mpas_grid_file)

    # Save the regridded dataset
    save_dataset(regridded_data, output_path)
    print(f"Regridded data saved to {output_path}")

if __name__ == "__main__":
    regrid_data(
        input_path=CONFIG["forcing_output_path"],
        output_path=CONFIG["regridded_output_path"],
        mpas_grid_file=CONFIG["mpas_grid_file"]
    )