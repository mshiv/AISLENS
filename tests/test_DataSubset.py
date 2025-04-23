import unittest
import xarray as xr
import numpy as np
from modules.data_preprocessing import subset_dataset
from pathlib import Path

class TestDataSubset(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset
        self.file_path = "mock_dataset.nc"
        self.dataset = xr.Dataset(
            {
                "var": (["Time", "x", "y"], np.random.rand(10, 5, 5)),
            },
            coords={
                "Time": np.arange(10),
                "x": np.arange(5),
                "y": np.arange(5),
            },
        )
        self.dataset.to_netcdf(self.file_path)

    def tearDown(self):
        # Clean up the mock dataset
        Path(self.file_path).unlink()

    def test_subset_time(self):
        # Test subsetting by Time dimension
        subset = subset_dataset(self.file_path, dim="Time", start=2, end=5)
        self.assertEqual(len(subset["Time"]), 4)
        self.assertTrue((subset["Time"] == [2, 3, 4, 5]).all())

    def test_invalid_dimension(self):
        # Test invalid dimension
        with self.assertRaises(ValueError):
            subset_dataset(self.file_path, dim="invalid_dim", start=0, end=5)

if __name__ == "__main__":
    unittest.main()