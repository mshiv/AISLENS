# Extrapolation Index-Map Test

This test validates the index-map optimization for `extrapolate_catchment_over_time`:
- **Correctness**: results match the baseline method (within floating-point tolerance) and don't fill ocean pixels
- **Performance**: timing comparison between the two methods on repeated fills with the same template

## Test Design

Location: `tests/test_extrapolation_indexmap.py` (pytest)

Setup:
1. Create a 40×40 synthetic grid with rioxarray CRS and affine transform metadata
2. Add a small GeoDataFrame with 1–2 ice-shelf polygons
3. Create a synthetic dataset with a few time steps, with NaNs inside and outside the shelves
4. Run the fill without index-map optimization, capture result A and time tA
5. Run the fill with index-map optimization, capture result B and time tB
6. Verify:
   - A and B are numerically equal on the ice shelf (within tolerance using `np.allclose`)
   - Both A and B have exactly zero outside ice shelves
   - Report timing difference (no hard failure on marginal differences)

Commands to run locally
-----------------------
Run pytest for the single test (after test file creation):


```bash
# from repo root
pytest -q tests/test_extrapolation_indexmap.py

# or just run this one test by nameynthetic grid must have `rio` metadata (CRS and transform); the test will write that metadata using rioxarray APIs.
```

- If running with dask-enabled arrays in the CI environment, ensure dask is available. The test will keep arrays small and call `.compute()` as needed so it runs fine without a scheduler.


## Implementation

- Generate all synthetic data within the test; do not read external files
- Use `tempfile` module for cache files and remove after test completion
- Measure timing with `time.perf_counter()` and log results via `caplog` or print statements
- Use `pytest.mark.skipif` to gracefully skip if `rasterio`/`rioxarray` are unavailable

**Common issues:**
- Rasterization fails if CRS or transform is missing → verify both are set on the synthetic template
- Memory errors on large grids → start with small grid (40×40 is sufficient)
- Marginal timing differences (<5%) are acceptable; focus on correctness first
---------------
- If rasterization fails with a message about missing transform/CRS, ensure the synthetic template DataArray has `rio.write_crs()` called and a correct affine transform set (the test includes this).
- If index_map mode raises memory errors on unexpectedly large grids, lower the synthetic grid size.

Next steps
## Next Steps

Implement `tests/test_extrapolation_indexmap.py` following the design above. Verify locally with pytest before committing.