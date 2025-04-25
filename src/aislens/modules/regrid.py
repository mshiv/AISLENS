import os
import xarray as xr
from pathlib import Path

def rename_dims_and_fillna(file_path, dims_to_rename=None, fill_value=0):
    """
    Rename dimensions and fill NaN values in a NetCDF file.

    Args:
        file_path (str or Path): Path to the NetCDF file.
        dims_to_rename (dict, optional): Dictionary mapping old dimension names to new names (e.g., {'x': 'x1', 'y': 'y1'}).
        fill_value (int or float, optional): Value to replace NaN values with. Default is 0.

    Returns:
        xarray.Dataset: Modified dataset.
    """
    # Open the dataset
    ds = xr.open_dataset(file_path)

    # Rename dimensions if specified
    if dims_to_rename:
        ds = ds.rename(dims_to_rename)

    # Fill NaN values with the specified fill value
    ds = ds.fillna(fill_value)

    # Save the modified dataset, overwriting the original file
    ds.to_netcdf(file_path)
    print(f"Updated {file_path.name}: renamed dimensions and filled NaNs with {fill_value}.")
    return ds


def process_directory(directory, dims_to_rename=None, fill_value=0):
    """
    Process all NetCDF files in a directory: rename dimensions and fill NaN values.

    Args:
        directory (str or Path): Path to the directory containing NetCDF files.
        dims_to_rename (dict, optional): Dictionary mapping old dimension names to new names (e.g., {'x': 'x1', 'y': 'y1'}).
        fill_value (int or float, optional): Value to replace NaN values with. Default is 0.

    Returns:
        None
    """
    directory = Path(directory)

    # Loop through all .nc files in the directory
    for file_path in directory.glob("*.nc"):
        rename_dims_and_fillna(file_path, dims_to_rename=dims_to_rename, fill_value=fill_value)