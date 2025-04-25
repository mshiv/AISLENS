import xarray as xr
from pathlib import Path

def load_dataset(file_path):
    """
    Load a dataset from a NetCDF file.

    Args:
        file_path (str): Path to the NetCDF file.

    Returns:
        xarray.Dataset: Loaded dataset.
    """
    return xr.open_dataset(file_path)

def save_dataset(data, file_path):
    """
    Save a dataset to a NetCDF file.

    Args:
        data (xarray.DataArray): Data to save.
        file_path (str): Path to save the file.

    Returns:
        None
    """
    data.to_netcdf(file_path)
    print(f"Saved dataset to {file_path}")