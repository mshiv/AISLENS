"""
This module contains functions for preprocessing and analyzing ice shelf data and ocean forcing datasets.
"""
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
from shapely.geometry import mapping
from scipy import spatial
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
import ruptures as rpt

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

def detect_breakpoints(arr, model="l2", penalty=10):
    """
    Detect breakpoints in a 1D array using ruptures.
    
    Parameters:
        arr (numpy.ndarray): 1D array of input data.
        model (str): The cost model for change point detection (default is "l2").
        penalty (int): The penalty value for the change point detection algorithm.

    Returns:
        list: List of breakpoints indices.
    """
    algo = rpt.Pelt(model=model).fit(arr)
    return algo.predict(pen=penalty)

def detrend_segmented(arr, time, breakpoints, deg=1):
    """
    Detrend a 1D array segment-by-segment based on breakpoints.
    
    Parameters:
        arr (numpy.ndarray): 1D array of input data.
        time (numpy.ndarray): 1D array of time values.
        breakpoints (list): List of breakpoint indices.
        deg (int): Degree of the polynomial fit (default is 1 for linear).

    Returns:
        numpy.ndarray: Detrended array.
    """
    detrended_values = np.zeros_like(arr)
    start = 0
    for end in breakpoints:
        segment = slice(start, end)
        time_segment = time[segment]
        values_segment = arr[segment]

        # Fit a polynomial to the segment
        coeffs = np.polyfit(time_segment, values_segment, deg)
        fit = np.polyval(coeffs, time_segment)

        # Detrend the segment
        detrended_values[segment] = values_segment - fit

        # Update the start index for the next segment
        start = end

    return detrended_values

def detrend_with_breakpoints_vectorized(data, dim, deg=1, model="l2", penalty=10):
    """
    Automatically detrend an xarray.DataArray with breakpoints along a specified dimension.

    Parameters:
        data (xarray.DataArray): The input data to detrend.
        dim (str): The dimension along which to detrend.
        deg (int): The degree of the polynomial fit (default is 1 for linear).
        model (str): The cost model for change point detection (default is "l2").
        penalty (int): The penalty value for the change point detection algorithm.

    Returns:
        xarray.DataArray: The detrended data with breakpoints applied.
    
    References:
        - "Ruptures: Change point detection in Python"
        - https://github.com/deepcharles/ruptures/discussions/139
    """
    def _process_single_slice(arr, coord_vals):
        # Detect breakpoints using ruptures
        breakpoints = detect_breakpoints(arr, model=model, penalty=penalty)
        # Detrend the data segment by segment
        return detrend_segmented(arr, coord_vals, breakpoints, deg=deg)

    detrended_data = xr.apply_ufunc(
        _process_single_slice,
        data,
        data[dim],
        input_core_dims=[[dim], [dim]],  # Specify core dimensions
        output_core_dims=[[dim]],        # Specify output dimension
        vectorize=True,                  # Enable vectorization
        #dask="parallelized",             # Allow Dask parallelization
        output_dtypes=[data.dtype]       # Ensure correct output dtype
    )

    return detrended_data

def detrend_with_breakpoints_ts(data, dim, deg=1, model="l2", penalty=10):
    """
    Automatically detrend data with breakpoints based on changes in the slope.
    This function is used only for prototyping and testing, or for quick analysis using spatially averaged time series data.
    For production, use detrend_with_breakpoints_vectorized.

    Parameters:
        data (xarray.DataArray): The input data to detrend.
        dim (str): The dimension along which to detrend.
        deg (int): The degree of the polynomial fit (default is 1 for linear).
        model (str): The cost model for change point detection (default is "l2").
                    1. "l1" for absolute error
                    2. "l2" for squared error
                    3. "rbf" for Gaussian error
                    4. "linear" for linear error
                    5. "normal" for normal error
        penalty (int): The penalty value for the change point detection algorithm.

    Returns:
        xarray.DataArray: The detrended data with breakpoints applied.
    """
    # Extract the data and dimension values as numpy arrays
    time = data[dim].values
    values = data.values

    # Ensure the data is 1D
    if values.ndim > 1:
        raise ValueError("Input data must be one-dimensional.")

    # Detect breakpoints using the ruptures library
    algo = rpt.Pelt(model=model).fit(values)
    breakpoints = algo.predict(pen=penalty)

    # Initialize an array to store the detrended data
    detrended_values = np.zeros_like(values)

    # Fit and detrend each segment
    start = 0
    for end in breakpoints:
        segment = slice(start, end)
        time_segment = time[segment]
        values_segment = values[segment]

        # Fit a polynomial to the segment
        coeffs = np.polyfit(time_segment, values_segment, deg)
        fit = np.polyval(coeffs, time_segment)

        # Detrend the segment
        detrended_values[segment] = values_segment - fit

        # Update the start index for the next segment
        start = end

    # Create a new xarray.DataArray with the detrended values
    detrended_data = xr.DataArray(
        detrended_values,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs
    )

    return detrended_data