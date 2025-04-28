"""
This module contains functions for preprocessing and analyzing ice shelf data.
"""
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
from shapely.geometry import mapping
from scipy import spatial
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression

def detrend_dim(data, dim, deg):
    """
    Detrend data along a specified dimension using a polynomial fit.

    Args:
        data (xarray.DataArray): Input data.
        dim (str): Dimension along which to detrend.
        deg (int): Degree of the polynomial fit.

    Returns:
        xarray.DataArray: Detrended data.
    """
    p = data.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(data[dim], p.polyfit_coefficients)
    detrended = data - fit
    return detrended

def deseasonalize(data):
    """
    Remove the seasonal cycle from the data.

    Args:
        data (xarray.DataArray): Input data.

    Returns:
        xarray.DataArray: Deseasonalized data.
    """
    data_month = data.groupby("Time.month")
    data_clm = data_month.mean("Time")
    data_anm = data_month - data_clm
    original_mean = data.mean("Time")
    data_anm += original_mean
    return data_anm

def dedraft(data, draft):
    """
    Remove draft dependence from the data using linear regression.

    Args:
        data (xarray.DataArray): Input data.
        draft (xarray.DataArray): Draft data.

    Returns:
        xarray.DataArray: Predicted draft dependence.
    """
    data_tm = data.mean(dim='Time')
    draft_tm = draft.mean(dim='Time')
    data_stack = data_tm.stack(z=('x', 'y'))
    draft_stack = draft_tm.stack(z=('x', 'y'))
    data_stack_noNaN = data_stack.fillna(0)
    draft_stack_noNaN = draft_stack.fillna(0)
    reg = LinearRegression().fit(draft_stack_noNaN.values.reshape(-1, 1), data_stack_noNaN.values.reshape(-1, 1))
    data_pred_stack_noNaN_vals = reg.predict(draft_stack_noNaN.values.reshape(-1, 1)).reshape(-1)
    data_pred_stack_noNaN = data_stack_noNaN.copy(data=data_pred_stack_noNaN_vals)
    data_pred_stack = data_pred_stack_noNaN.where(~data_stack.isnull(), np.nan)
    data_pred = data_pred_stack.unstack('z').transpose()
    return data_pred