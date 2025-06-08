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
from aislens.config import config
from aislens.geospatial import clip_data
from aislens.utils import fill_nan_with_nearest_neighbor_vectorized, fill_nan_with_nearest_neighbor_vectorized_balltree, merge_catchment_data, copy_subset_data, write_crs

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
    # Store the original mean
    original_mean = data.mean(dim=dim)
    # Detrend along a single dimension
    p = data.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(data[dim], p.polyfit_coefficients)
    detrended = data - fit
    # Add back the original mean
    detrended += original_mean
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
        reg.coef_: Coefficients of the regression.
        reg.intercept_: Intercept of the regression.
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
    return reg.coef_, reg.intercept_, data_pred

def setup_draft_depen_field(param_ref, param_data, param_name, i, icems):
    """
    Set up a DataArray for draft dependence parameter field.

    Args:
        param_ref (xarray.Dataset): Time-averaged dataset.
        param_data (xarray.Dataset): Parameter data calculated by dedraft.

    Returns:
        xarray.DataArray: Initialized DataArray for the parameter.
    """
    param_ds = xr.DataArray(
        np.full((param_ref.shape[0], param_ref.shape[1]), param_data),
        dims=['y', 'x'],
        coords={'x': param_ref.x, 'y': param_ref.y}
    )
    param_ds.name = param_name
    param_ds.attrs["long_name"] = config.DATA_ATTRS[param_name]['long_name']
    param_ds.attrs["units"] = config.DATA_ATTRS[param_name]['units']
    write_crs(param_ds)
    # Create a boolean mask for the ice shelf of interest using the icems geometry
    # # This mask will be used to filter the two data arrays above to define values only inside the ice shelf mask region
    # Filter the mlt_coef_ds data array using this mask
    param_ds = clip_data(param_ds, i, icems)
    return param_ds

def dedraft_catchment(
    i, icems, data, config, 
    save_dir, 
    save_pred=False, 
    save_coefs=False
    ):
    """
    This function processes individual ice shelf catchments to dedraft and:
    - save predicted melt (to isolate and calculate variability in model data), 
    - or save regression coefficients (to calculate draft dependence from observations).
    
    Args:
        i (int): Index for ice shelf catchment.
        icems: geoDataframe with ice shelf masks.
        data (xarray.DataArray): Input data containing melt and draft fields for model or observations.
        config: Configuration object containing paths and attributes.
        save_dir (Path): Directory to save output.
        save_pred (bool): Save predicted melt (model).
        save_coefs (bool): Save regression coefficients (obs).
    """
    catchment_name = icems.name.values[i]
    print(f'Extracting data for catchment {catchment_name}')
    ds = clip_data(data, i, icems)
    ds_tm = ds.mean(dim=config.TIME_DIM)
    # Choose the correct variable names based on the data type
    if config.SORRM_FLUX_VAR in ds.data_vars:
        # If the data is from the model, use SORRM variables
        flux_var = config.SORRM_FLUX_VAR
        draft_var = config.SORRM_DRAFT_VAR
    elif config.SATOBS_FLUX_VAR in ds.data_vars:
        # For satellite observations, use the SATOBS variables
        flux_var = config.SATOBS_FLUX_VAR
        draft_var = config.SATOBS_DRAFT_VAR

    print(f'Calculating draft dependent linear regression for catchment {catchment_name}')
    coef, intercept, pred = dedraft(ds[flux_var], ds[draft_var])

    if save_coefs:
        # Retrieve attribute keys explicitly for clarity
        alpha0_key, alpha1_key = list(config.DATA_ATTRS.keys())
        coef_ds = setup_draft_depen_field(ds_tm.melt, coef, alpha1_key, i, icems)
        intercept_ds = setup_draft_depen_field(ds_tm.melt, intercept, alpha0_key, i, icems)
        ds_out = xr.Dataset({coef_ds.name: coef_ds, intercept_ds.name: intercept_ds})
        filename = save_dir / f'draftDepenBasalMeltAlpha_{catchment_name}.nc'
        ds_out.to_netcdf(filename)
        print(f'{catchment_name} coefficients file saved: {filename}')
    if save_pred:
        filename = save_dir / f'draftDepenModelPred_{catchment_name}.nc'
        pred.to_netcdf(filename)
        print(f'{catchment_name} prediction file saved: {filename}')

def extrapolate_catchment(data, i, icems):
    """
    Extrapolate specified data field for a given ice shelf catchment into the interior of the ice sheet.
    Args:
        ds_data (xarray.DataArray): Input data containing melt and draft fields for model or observations.
        i (int): Index for ice shelf catchment.
        icems: geoDataframe with ice shelf masks.
    Returns:
        xarray.DataArray: Extrapolated data field for the given ice shelf catchment.
    """
    ice_shelf_mask = icems.loc[[i], 'geometry'].apply(mapping)
    ds = clip_data(data, i, icems)
    #ds = ds.map(fill_nan_with_nearest_neighbor_vectorized, keep_attrs=True)
    ds = ds.map(fill_nan_with_nearest_neighbor_vectorized_balltree, keep_attrs=True)
    ds = ds.rio.clip(ice_shelf_mask, icems.crs)
    return ds

def extrapolate_catchment_over_time(dataset, icems, config, var_name):
    """
    Extrapolate catchment data for each time step.
    Returns an xarray.Dataset with filled values.
    """
    times = dataset[var_name].coords[config.TIME_DIM]
    shape = dataset[var_name].shape
    extrap_array = np.full(shape, np.nan)

    extrap_ds = xr.DataArray(
        extrap_array,
        coords=dataset[var_name].coords,
        dims=dataset[var_name].dims,
        attrs=dataset[var_name].attrs
        
    )
    extrap_ds = xr.Dataset({var_name: extrap_ds})

    for t in range(len(times)):
        ds_data = dataset.isel({config.TIME_DIM: t}) #.rename({'x': 'x1', 'y': 'y1'})
        results = [extrapolate_catchment(ds_data, i, icems) for i in config.ICE_SHELF_REGIONS]
        merged_ds = merge_catchment_data(results)
        result_ds = copy_subset_data(ds_data, merged_ds)
        extrap_ds[var_name][t] = result_ds[var_name]
        print(f"Completed {var_name} time step {t}")
    return extrap_ds

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
        #dask="parallelized",             # Allow Dask parallelization - Dataset already parallelized when loaded
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

import xarray as xr
import numpy as np

def compute_ismip6_basal_melt(ds, gamma0, cste, scyr, rhoi):
    """
    Compute ISMIP6 basal melt (vectorized, dask-friendly).

    ds: xarray.Dataset with variables:
        - TFdraft (Time, x, y)
        - deltaT (Time, x, y)
        - basinNumber (x, y)
        - ismip6shelfMelt_offset (Time, x, y)
        - mask_floating (Time, x, y)
    gamma0, cste, scyr, rhoi: scalars

    Returns
    -------
    ds: Dataset with 'floatingBasalMassBal' (Time, x, y)
    """
    # Broadcast basinNumber for alignment with (Time, x, y)
    basinNumber = ds['basinNumber'].broadcast_like(ds['TFdraft'])

    # Stack x and y for groupby, then unstack after
    stacked = ds[['TFdraft', 'deltaT']].stack(cell=('x', 'y'))
    basin_stacked = basinNumber.stack(cell=('x', 'y'))

    # Compute mean TF per basin for each time (area weighting is trivial since area is constant)
    tf_grouped = stacked['TFdraft'].groupby(basin_stacked)
    mean_TF = tf_grouped.mean(dim='cell')  # (Time, basin)

    # Broadcast mean_TF back to (Time, x, y)
    # Prepare a DataArray aligning basins to x, y
    basin_vals = ds['basinNumber'].values.ravel()
    unique_basins = np.unique(basin_vals)
    # mean_TF: (Time, basin), basin values correspond to unique_basins
    # Map each (x, y) location to its basin index
    basin_idx_map = {b: i for i, b in enumerate(unique_basins)}
    basin_idx = np.vectorize(basin_idx_map.get)(basin_vals).reshape(ds['basinNumber'].shape)

    # Now, use advanced indexing to get (Time, x, y)
    # mean_TF_arr = mean_TF.transpose('Time', 'basin').data  # (Time, n_basins)
    mean_TF_arr = mean_TF.transpose('Time', 'basin_stacked').data  # (Time, n_basins)
    # This uses numpy/dask advanced indexing
    time_steps = ds.dims['Time']
    n_x, n_y = ds.dims['x'], ds.dims['y']
    # Use xarray's apply_ufunc for full dask-compatibility
    def select_mean_tf(mean_TF_arr, basin_idx):
        # mean_TF_arr: (Time, n_basins), basin_idx: (x, y)
        # Output: (Time, x, y)
        return mean_TF_arr[:, basin_idx]

    mean_TF_per_cell = xr.apply_ufunc(
        select_mean_tf,
        mean_TF_arr,
        basin_idx,
        input_core_dims=[['Time', 'basin_stacked'], ['x', 'y']],
        output_core_dims=[['Time', 'x', 'y']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[ds['TFdraft'].dtype],
    )

    # Now, mean_TF_per_cell: (Time, x, y)
    # Calculate coefficient
    coef = gamma0 * cste / scyr * rhoi

    # Calculate the melt (all vectorized)
    term1 = ds['TFdraft'] + ds['deltaT']
    term2 = np.abs(mean_TF_per_cell + ds['deltaT'])
    floatingBasalMassBal = -coef * term1 * term2 + ds['ismip6shelfMelt_offset']

    # Apply floating mask
    floatingBasalMassBal = floatingBasalMassBal.where(ds['mask_floating'], 0.0)

    ds['floatingBasalMassBal'] = floatingBasalMassBal
    return ds