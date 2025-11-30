import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from aislens.dataprep import dedraft

def make_synthetic():
    # small grid 4x4, 3 time steps
    x = np.arange(4)
    y = np.arange(4)
    time = np.arange(3)
    # draft increases linearly across x (same each time)
    draft = np.zeros((3, 4, 4))
    for t in range(3):
        for j in range(4):
            draft[t, :, j] = np.linspace(10, 40, 4)
    # construct melt = 2.5 * draft + 5 + small noise
    melt = 2.5 * draft + 5.0 + np.random.normal(0, 0.01, draft.shape)
    # mask ~half the cells (set to NaN)
    mask = np.zeros((4,4), dtype=bool)
    mask[::2, ::2] = True  # mask alternating cells
    for t in range(3):
        melt[t][mask] = np.nan
        draft[t][mask] = np.nan

    da_melt = xr.DataArray(melt, dims=('Time', 'y', 'x'),
                           coords={'Time': time, 'y': y, 'x': x})
    da_draft = xr.DataArray(draft, dims=('Time', 'y', 'x'),
                            coords={'Time': time, 'y': y, 'x': x})
    return da_melt, da_draft

def test_masked_fit_recovers_slope():
    melt, draft = make_synthetic()
    coef, intercept, pred = dedraft(melt, draft, weights=None)
    # coef ~ 2.5, intercept ~ 5.0
    assert np.isclose(coef[0], 2.5, atol=0.1)
    assert np.isclose(intercept, 5.0, atol=0.2)
    # predictions inside masked cells should be NaN
    pred_tm = pred.mean(dim='Time') if 'Time' in pred.dims else pred
    assert np.all(np.isnan(pred_tm.values[::2, ::2]))