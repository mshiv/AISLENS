import time
import numpy as np
import xarray as xr
import pytest

import aislens.dataprep as dataprep
from aislens.config import config


def test_extrapolation_indexmap_equivalence(monkeypatch):
    """
    Lightweight integration test for extrapolate_catchment_over_time.

    Strategy:
    - Monkeypatch heavy geospatial and per-catchment internals so the function runs deterministically.
    - Create a small synthetic dataset with only NaNs.
    - Provide a synthetic rasterized ice mask (central square True).
    - Make the fake per-catchment extrapolation return fields filled with 1.0 (simulates ndimage per-catchment fill).
    - Make the fake fill_with_index_map also fill NaNs with 1.0 (simulates index-map behavior).
    - Verify outputs are identical inside the ice mask and zeros outside the mask.
    """

    var = config.SORRM_FLUX_VAR

    # Small synthetic grid
    nx, ny, nt = 20, 20, 3
    x = np.arange(nx)
    y = np.arange(ny)
    times = np.arange(nt)

    data = np.full((nt, ny, nx), np.nan, dtype=float)
    ds = xr.Dataset({var: (("Time", "y", "x"), data)}, coords={"Time": times, "x": x, "y": y})

    # Ensure only a small number of ice-shelf regions are iterated
    monkeypatch.setattr(dataprep.config, 'ICE_SHELF_REGIONS', [0], raising=False)

    # Fake extrapolate_catchment: fill NaNs with 1.0 (simulating ndimage per-catchment fill)
    def fake_extrapolate_catchment(ds_data, i, icems, precomputed_masks=None):
        da = ds_data[var]
        out = da.copy()
        out.values = np.where(np.isnan(out.values), 1.0, out.values)
        return out

    monkeypatch.setattr(dataprep, 'extrapolate_catchment', fake_extrapolate_catchment)

    # Fake merge_catchment_data: return a Dataset with the variable name mapped to the first result
    monkeypatch.setattr(
        dataprep,
        'merge_catchment_data',
        lambda results: xr.Dataset({var: results[0]})
    )

    # Fake copy_subset_data: just return merged result (no spatial copying needed here)
    monkeypatch.setattr(dataprep, 'copy_subset_data', lambda ds_data, merged: merged)

    # Create a synthetic rasterized mask: central 8x8 square True, else False
    mask_arr = np.zeros((ny, nx), dtype=bool)
    mask_arr[6:14, 6:14] = True
    mask = xr.DataArray(mask_arr, coords={"y": y, "x": x}, dims=("y", "x"))

    monkeypatch.setattr(dataprep, 'rasterize_ice_mask', lambda icems, template: mask)

    # Fake compute_nearest_index_map (returns a dummy object)
    monkeypatch.setattr(dataprep, 'compute_nearest_index_map', lambda mask_for_map, cache_path=None: object())

    # Fake fill_with_index_map: fill NaNs with 1.0 (same behavior as fake_extrapolate_catchment)
    monkeypatch.setattr(dataprep, 'fill_with_index_map', lambda arr, index_map: arr.where(~np.isnan(arr), other=1.0))

    # Run without index_map
    t0 = time.perf_counter()
    out_noidx = dataprep.extrapolate_catchment_over_time(ds, None, config, var, use_index_map=False)
    t_noidx = time.perf_counter() - t0

    # Run with index_map
    t0 = time.perf_counter()
    out_idx = dataprep.extrapolate_catchment_over_time(ds, None, config, var, use_index_map=True, index_map_cache_path=None)
    t_idx = time.perf_counter() - t0

    a = out_noidx[var].values
    b = out_idx[var].values

    # Sanity: shapes must match
    assert a.shape == b.shape

    # Inside mask -> should be 1.0 for both
    for t in range(nt):
        inside_no = a[t][mask_arr]
        inside_idx = b[t][mask_arr]
        assert np.allclose(inside_no, 1.0, atol=1e-12)
        assert np.allclose(inside_idx, 1.0, atol=1e-12)

    # Outside mask -> should be exactly 0.0 for both (rasterized mask applied)
    for t in range(nt):
        outside_no = a[t][~mask_arr]
        outside_idx = b[t][~mask_arr]
        assert np.allclose(outside_no, 0.0, atol=1e-12)
        assert np.allclose(outside_idx, 0.0, atol=1e-12)

    # Report timings (not a strict assertion)
    print(f"timings: no-index {t_noidx:.4f}s, index {t_idx:.4f}s")
