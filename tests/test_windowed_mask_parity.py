import time
import numpy as np
import xarray as xr
import pytest

import aislens.dataprep as dataprep
from aislens.config import config
import pandas as pd
from shapely.geometry import box


def test_windowed_mask_vs_clip_equivalence(monkeypatch):
    """
    Test that using precomputed per-shelf windows/masks (fast .isel+mask path)
    yields the same final outputs as the fallback clip-based path. We monkeypatch
    internals so the test is deterministic and lightweight.
    """

    var = config.SORRM_FLUX_VAR

    # Small synthetic grid
    nx, ny, nt = 20, 20, 3
    x = np.arange(nx)
    y = np.arange(ny)
    times = np.arange(nt)

    data = np.full((nt, ny, nx), np.nan, dtype=float)
    ds = xr.Dataset({var: (("Time", "y", "x"), data)}, coords={"Time": times, "x": x, "y": y})

    # Limit to a single shelf region to keep logic simple
    monkeypatch.setattr(dataprep.config, 'ICE_SHELF_REGIONS', [0], raising=False)

    # Prepare a central mask (8x8 square) used as the rasterized ice mask
    mask_arr = np.zeros((ny, nx), dtype=bool)
    mask_arr[6:14, 6:14] = True
    raster_mask = xr.DataArray(mask_arr, coords={"y": y, "x": x}, dims=("y", "x"))

    # Make rasterize_ice_mask return our synthetic mask
    monkeypatch.setattr(dataprep, 'rasterize_ice_mask', lambda icems, template: raster_mask)

    # Ensure merge/copy are identity-like for merged returned datasets.
    # The per-catchment result may be either a DataArray or a Dataset depending
    # on the code path; make the stub robust to both.
    def _merge_stub(results):
        r0 = results[0]
        if isinstance(r0, xr.Dataset):
            # Prefer the configured variable name if present, otherwise take the first var
            if var in r0.data_vars:
                da = r0[var]
            else:
                first_var = next(iter(r0.data_vars))
                da = r0[first_var]
        else:
            da = r0
        return xr.Dataset({var: da})

    monkeypatch.setattr(dataprep, 'merge_catchment_data', _merge_stub)

    # Fake ndimage fill to deterministically fill NaNs with 1.0
    def fake_ndimage_fill(da):
        out = da.copy()
        out.values = np.where(np.isnan(out.values), 1.0, out.values)
        return out

    monkeypatch.setattr(dataprep, 'fill_nan_with_nearest_neighbor_ndimage', fake_ndimage_fill)

    # Build a precomputed_masks dict that the fast path will consume
    pre_mask = {}
    # window covering 6:14 in y and 6:14 in x
    ys = y[6:14]
    xs = x[6:14]
    mask_small = mask_arr[6:14, 6:14]
    mask_da_small = xr.DataArray(mask_small, coords={"y": ys, "x": xs}, dims=("y", "x"))
    pre_mask[0] = {'window': (6, 14, 6, 14), 'mask': mask_da_small}

    # Provide a minimal icems-like DataFrame so dataprep.extrapolate_catchment can access
    # `.loc`, `.name.values` and `.geometry`. The geometry itself won't be used because
    # rasterize_ice_mask is monkeypatched above.
    icems = pd.DataFrame({'name': np.array(['SHELF0']), 'geometry': [box(-1, -1, 1, 1)]})
    # Attach a crs attribute (some callers expect it)
    icems.crs = None

    # Case 1: precomputed masks available -> expect fast path to be used
    monkeypatch.setattr(dataprep, 'compute_shelf_windows_and_masks', lambda template, icems_arg: pre_mask)

    # For the fallback path we will monkeypatch extrapolate_catchment to simulate
    # the old clip-based behavior: simply return a DataArray filled with 1.0
    def fake_extrapolate_catchment_old(ds_data, i, icems, precomputed_masks=None):
        da = ds_data[var]
        out = da.copy()
        out.values = np.where(np.isnan(out.values), 1.0, out.values)
        return out

    # When precomputed masks exist we want to run the normal extrapolate_catchment (fast path)
    # so do NOT monkeypatch dataprep.extrapolate_catchment here.

    out_pre = dataprep.extrapolate_catchment_over_time(ds, icems, config, var, use_index_map=False)

    # Case 2: no precomputed masks -> force old clip-based path by returning None
    monkeypatch.setattr(dataprep, 'compute_shelf_windows_and_masks', lambda template, icems_arg: None)
    # Monkeypatch extrapolate_catchment to the old behavior
    monkeypatch.setattr(dataprep, 'extrapolate_catchment', fake_extrapolate_catchment_old)

    out_fallback = dataprep.extrapolate_catchment_over_time(ds, icems, config, var, use_index_map=False)

    a = out_pre[var].values
    b = out_fallback[var].values

    # Shapes must match
    assert a.shape == b.shape

    # Inside mask -> should be 1.0 for both
    for t in range(nt):
        inside_pre = a[t][mask_arr]
        inside_fall = b[t][mask_arr]
        assert np.allclose(inside_pre, 1.0, atol=1e-12)
        assert np.allclose(inside_fall, 1.0, atol=1e-12)

    # Outside mask -> should be exactly 0.0 for both (rasterized mask applied in extrapolate_catchment_over_time)
    for t in range(nt):
        outside_pre = a[t][~mask_arr]
        outside_fall = b[t][~mask_arr]
        assert np.allclose(outside_pre, 0.0, atol=1e-12)
        assert np.allclose(outside_fall, 0.0, atol=1e-12)
