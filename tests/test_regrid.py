import unittest
import xarray as xr
import numpy as np
from pathlib import Path
from modules.regrid import rename_dims_and_fillna, process_directory

class TestRegrid(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for mock NetCDF files
        self.test_dir = Path("test_nc_files")
        self.test_dir.mkdir(exist_ok=True)

        # Create a mock NetCDF file
        self.file_path = self.test_dir / "mock_dataset.nc"
        self.dataset = xr.Dataset(
            {
                "var": (["x", "y"], np.random.rand(5, 5)),
            },
            coords={
                "x": np.arange(5),
                "y": np.arange(5),
            },
        )
        self.dataset.to_netcdf(self.file_path)

    def tearDown(self):
        # Clean up the temporary directory and files
        for file in self.test_dir.glob("*.nc"):
            file.unlink()
        self.test_dir.rmdir()

    def test_rename_dims_and_fillna(self):
        # Test renaming dimensions and filling NaNs
        dims_to_rename = {"x": "x1", "y": "y1"}
        modified_ds = rename_dims_and_fillna(self.file_path, dims_to_rename=dims_to_rename, fill_value=0)

        # Check that dimensions are renamed
        self.assertIn("x1", modified_ds.dims)
        self.assertIn("y1", modified_ds.dims)
        self.assertNotIn("x", modified_ds.dims)
        self.assertNotIn("y", modified_ds.dims)

        # Check that NaN values are replaced with 0
        self.assertTrue((modified_ds["var"].values == 0).sum() == 0)  # No NaNs should remain

    def test_process_directory(self):
        # Create another mock NetCDF file in the directory
        another_file_path = self.test_dir / "another_mock_dataset.nc"
        another_dataset = xr.Dataset(
            {
                "var": (["x", "y"], np.random.rand(5, 5)),
            },
            coords={
                "x": np.arange(5),
                "y": np.arange(5),
            },
        )
        another_dataset.to_netcdf(another_file_path)

        # Test processing the directory
        dims_to_rename = {"x": "x1", "y": "y1"}
        process_directory(self.test_dir, dims_to_rename=dims_to_rename, fill_value=0)

        # Verify that both files were processed
        for file_path in self.test_dir.glob("*.nc"):
            modified_ds = xr.open_dataset(file_path)