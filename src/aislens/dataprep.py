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
from aislens.utils import (
    fill_nan_with_nearest_neighbor_vectorized,
    fill_nan_with_nearest_neighbor_vectorized_balltree,
    fill_nan_with_nearest_neighbor_ndimage,
    merge_catchment_data,
    copy_subset_data,
    write_crs,
    align_mask_to_template,
    rasterize_ice_mask,
)
from aislens.utils import compute_nearest_index_map, fill_with_index_map
from aislens.utils import draft_weight
import logging

logger = logging.getLogger(__name__)


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

def dedraft(data, draft, weights=None):
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
    # Take time-mean if Time dimension exists, otherwise use data as-is
    data_tm = data.mean(dim='Time') if 'Time' in data.dims else data
    draft_tm = draft.mean(dim='Time') if 'Time' in draft.dims else draft
    data_stack = data_tm.stack(z=('x', 'y'))
    draft_stack = draft_tm.stack(z=('x', 'y'))
    data_stack_noNaN = data_stack.fillna(0)
    draft_stack_noNaN = draft_stack.fillna(0)
    if weights is not None:
        weights_tm = weights.mean(dim='Time') if 'Time' in weights.dims else weights
        weights_stack = weights_tm.stack(z=('x', 'y'))
        w = weights_stack.fillna(0)
    else:
        w = None
    reg = LinearRegression().fit(draft_stack_noNaN.values.reshape(-1, 1), data_stack_noNaN.values.reshape(-1, 1), sample_weight=np.squeeze(w.values.reshape(-1,1)) if w is not None else None)
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
    weights=False,
    weight_power=0.25,
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
        weights (xarray.DataArray, optional): Weights for regression (binary).
        weight_power (float): Power for draft_weight (default 0.25).
        save_pred (bool): Save predicted melt (model).
        save_coefs (bool): Save regression coefficients (obs).
    """
    catchment_name = icems.name.values[i]
    # Log shelf name and bounds for easier debugging when clipping fails
    try:
        geom = icems.loc[i, 'geometry']
        bounds = getattr(geom, 'bounds', None)
    except Exception:
        geom = None
        bounds = None

    logger.info("Extracting data for catchment %s (index=%s, bounds=%s)", catchment_name, i, bounds)

    # Attempt to clip data to the catchment. If there are no raster pixels inside the
    # polygon bounds, rioxarray raises NoDataInBounds; catch it, warn, and skip this shelf.
    try:
        from rioxarray.exceptions import NoDataInBounds
    except Exception:
        NoDataInBounds = Exception

    try:
        ds = clip_data(data, i, icems)
    except NoDataInBounds:
        logger.warning(
            "No data found inside ice-shelf '%s' (index %s). Skipping dedraft for this shelf.",
            catchment_name,
            i,
        )
        # Create placeholder prediction file (zeros) so downstream merging has a file to read.
        if save_pred:
            try:
                # Choose a reference field from the provided `data` to copy spatial coords/dims
                if config.SORRM_FLUX_VAR in data.data_vars:
                    ref_var = config.SORRM_FLUX_VAR
                else:
                    ref_var = next(iter(data.data_vars))

                if config.TIME_DIM in data[ref_var].dims:
                    ref_2d = data[ref_var].mean(dim=config.TIME_DIM)
                else:
                    # take first timestep or the 2D field as-is
                    ref_2d = data[ref_var]

                # Use NaN placeholders so we don't introduce false zeros into the
                # merged dataset; downstream finalization will replace remaining
                # NaNs with zeros in the final forcing files.
                zero_da = xr.DataArray(
                    np.full(ref_2d.shape, np.nan, dtype=float),
                    coords=ref_2d.coords,
                    dims=ref_2d.dims,
                )
                zero_da.name = ref_var
                filename = Path(save_dir) / f'draftDepenModelPred_{catchment_name}.nc'
                xr.Dataset({zero_da.name: zero_da}).to_netcdf(filename)
                logger.info('Wrote placeholder zero prediction for %s -> %s', catchment_name, filename)
            except Exception as e:
                logger.warning('Failed to write placeholder prediction for %s: %s', catchment_name, e)
        return
    # Take time-mean if Time dimension exists (for saving coefficients reference)
    ds_tm = ds.mean(dim=config.TIME_DIM) if config.TIME_DIM in ds.dims else ds
    # Choose the correct variable names based on the data type
    if config.SORRM_FLUX_VAR in ds.data_vars:
        # If the data is from the model, use SORRM variables
        flux_var = config.SORRM_FLUX_VAR
        draft_var = config.SORRM_DRAFT_VAR
    elif config.SATOBS_FLUX_VAR in ds.data_vars:
        # For satellite observations, use the SATOBS variables
        flux_var = config.SATOBS_FLUX_VAR
        draft_var = config.SATOBS_DRAFT_VAR
    if weights:
        w = ds[flux_var].copy(data=draft_weight(ds[flux_var], ds[draft_var], a=weight_power))
        w = clip_data(w, i, icems)
    else:
        w = None

    print(f'Calculating draft dependent linear regression for catchment {catchment_name}')
    coef, intercept, pred = dedraft(ds[flux_var], ds[draft_var], weights=w)

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

def dedraft_catchment_comprehensive(
    i, icems, data, config, 
    save_dir, 
    weights=False,
    weight_power=0.25,
    save_pred=False, 
    save_coefs=False,
    # New parameters for comprehensive analysis
    n_bins=50,
    min_points_per_bin=5,
    ruptures_method='pelt',
    ruptures_penalty=1.0,
    min_r2_threshold=0.1,
    min_correlation=0.3,
    noisy_fallback='zero',
    model_selection='best'  # 'best', 'zero_shallow', 'mean_shallow', 'threshold_intercept'
):
    """
    Enhanced function that processes individual ice shelf catchments using changepoint detection
    and multiple piecewise linear models to create comprehensive draft dependence parameters.
    
    Args:
        i (int): Index for ice shelf catchment.
        icems: geoDataframe with ice shelf masks.
        data (xarray.DataArray): Input data containing melt and draft fields.
        config: Configuration object containing paths and attributes.
        save_dir (Path): Directory to save output.
        weights (xarray.DataArray, optional): Weights for regression (binary).
        weight_power (float): Power for draft_weight (default 0.25).
        save_pred (bool): Save predicted melt (model).
        save_coefs (bool): Save regression coefficients (obs).
        
        # New comprehensive parameters
        n_bins (int): Number of bins for draft binning (default: 50)
        min_points_per_bin (int): Minimum points required per bin (default: 5)
        ruptures_method (str): Ruptures method: 'pelt', 'binseg', or 'window' (default: 'pelt')
        ruptures_penalty (float): Penalty parameter for ruptures (default: 1.0)
        min_r2_threshold (float): Minimum R² for meaningful relationship (default: 0.1)
        min_correlation (float): Minimum correlation for meaningful relationship (default: 0.3)
        noisy_fallback (str): For noisy data: 'zero' or 'mean' (default: 'zero')
        model_selection (str): Which model to use for output (default: 'best')
    
    Returns:
        dict: Comprehensive results including all five draft dependence parameters
    """
    
    catchment_name = icems.name.values[i]
    print(f'Processing catchment {catchment_name} with comprehensive analysis')
    
    # Clip data to catchment
    ds = clip_data(data, i, icems)
    ds_tm = ds.mean(dim=config.TIME_DIM)
    
    # Choose the correct variable names based on the data type
    if config.SORRM_FLUX_VAR in ds.data_vars:
        flux_var = config.SORRM_FLUX_VAR
        draft_var = config.SORRM_DRAFT_VAR
    elif config.SATOBS_FLUX_VAR in ds.data_vars:
        flux_var = config.SATOBS_FLUX_VAR
        draft_var = config.SATOBS_DRAFT_VAR
    
    # Apply weights if specified
    if weights:
        w = ds[flux_var].copy(data=draft_weight(ds[flux_var], ds[draft_var], a=weight_power))
        w = clip_data(w, i, icems)
    else:
        w = None
    
    # Run comprehensive analysis
    result = calculate_single_shelf_comprehensive(
        ds[flux_var], ds[draft_var], catchment_name,
        n_bins=n_bins,
        min_points_per_bin=min_points_per_bin,
        ruptures_method=ruptures_method,
        ruptures_penalty=ruptures_penalty,
        min_r2_threshold=min_r2_threshold,
        min_correlation=min_correlation,
        noisy_fallback=noisy_fallback,
        weights=w
    )
    
    # Extract the five draft dependence parameters
    draft_params = extract_draft_dependence_parameters(result, model_selection)
    
    # Save results if requested
    if save_coefs:
        save_comprehensive_coefficients(
            ds_tm, draft_params, catchment_name, save_dir, config, i, icems
        )
    
    if save_pred:
        save_comprehensive_predictions(
            result, catchment_name, save_dir, ds[draft_var]
        )
    
    return {
        'catchment_name': catchment_name,
        'draft_params': draft_params,
        'full_results': result
    }

def calculate_single_shelf_comprehensive(melt_data, draft_data, shelf_name,
                                       n_bins=50, min_points_per_bin=5,
                                       ruptures_method='pelt', ruptures_penalty=1.0,
                                       min_r2_threshold=0.1, min_correlation=0.3,
                                       noisy_fallback='zero', weights=None):
    """
    Comprehensive analysis for a single ice shelf using changepoint detection.
    
    Args:
        melt_data (xarray.DataArray): Melt rate data for the shelf
        draft_data (xarray.DataArray): Draft data for the shelf
        shelf_name (str): Name of the ice shelf
        ... (other parameters as in main function)
        
    Returns:
        dict: Comprehensive results for the shelf
    """
    from scipy.stats import pearsonr
    
    # Convert to time-mean if needed
    if 'Time' in melt_data.dims:
        melt_tm = melt_data.mean(dim='Time')
        draft_tm = draft_data.mean(dim='Time')
    else:
        melt_tm = melt_data
        draft_tm = draft_data
    
    # Flatten arrays and remove NaN
    melt_vals = melt_tm.values.flatten()
    draft_vals = draft_tm.values.flatten()
    
    # Keep melt rates in original units (m/yr) - no unit conversion
    
    # Remove NaN values
    mask = ~np.isnan(melt_vals) & ~np.isnan(draft_vals)
    melt_clean = melt_vals[mask]
    draft_clean = draft_vals[mask]
    
    if len(melt_clean) < 20:
        return create_fallback_result(shelf_name, melt_clean, draft_clean, noisy_fallback)
    
    # Apply weights if provided
    if weights is not None:
        weights_clean = weights.values.flatten()[mask]
    else:
        weights_clean = None
    
    # Assess data quality
    is_meaningful, correlation, r2 = assess_data_quality(
        draft_clean, melt_clean, min_r2_threshold, min_correlation)
    
    if not is_meaningful:
        print(f"Skipping {shelf_name}: relationship too noisy (corr={correlation:.3f}, R²={r2:.3f})")
        return create_fallback_result(shelf_name, melt_clean, draft_clean, noisy_fallback, 
                                    correlation=correlation, r2=r2)
    
    # Find changepoint threshold
    threshold, binned_draft, binned_melt, _ = find_changepoint_threshold_ruptures(
        draft_clean, melt_clean, n_bins, min_points_per_bin, ruptures_method, ruptures_penalty)
    
    if np.isnan(threshold):
        print(f"Skipping {shelf_name}: insufficient data for changepoint detection")
        return create_fallback_result(shelf_name, melt_clean, draft_clean, noisy_fallback,
                                    correlation=correlation, r2=r2)
    
    # Fit piecewise models
    slope, shallow_mean, deep_intercept = fit_piecewise_models(
        draft_clean, melt_clean, threshold, weights=weights_clean)
    
    # Create predictions
    predictions = predict_piecewise_multiple(draft_clean, threshold, slope, shallow_mean, deep_intercept)
    
    return {
        'shelf_name': shelf_name,
        'threshold': threshold,
        'slope': slope,
        'shallow_mean': shallow_mean,
        'deep_intercept': deep_intercept,
        'draft_vals': draft_clean,
        'melt_vals': melt_clean,
        'predictions': predictions,
        'binned_draft': binned_draft,
        'binned_melt': binned_melt,
        'is_meaningful': True,
        'correlation': correlation,
        'r2': r2
    }

def assess_data_quality(draft_vals, melt_vals, min_r2=0.1, min_corr=0.3):
    """Assess if the draft-melt relationship is meaningful or too noisy."""
    from scipy.stats import pearsonr
    
    mask = ~np.isnan(draft_vals) & ~np.isnan(melt_vals)
    if np.sum(mask) < 20:
        return False, 0.0, 0.0
    
    draft_clean = draft_vals[mask]
    melt_clean = melt_vals[mask]
    
    # Calculate correlation
    try:
        correlation, p_value = pearsonr(draft_clean, melt_clean)
    except:
        correlation, p_value = 0.0, 1.0
    
    # Calculate R² from simple linear regression
    try:
        reg = LinearRegression()
        reg.fit(draft_clean.reshape(-1, 1), melt_clean)
        r2 = reg.score(draft_clean.reshape(-1, 1), melt_clean)
    except:
        r2 = 0.0
    
    # Check if relationship is meaningful
    is_meaningful = (abs(correlation) >= min_corr) and (r2 >= min_r2) and (p_value < 0.05)
    
    return is_meaningful, correlation, r2

def find_changepoint_threshold_ruptures(draft_vals, melt_vals, n_bins=50, min_points_per_bin=5,
                                       method='pelt', penalty=1.0):
    """Find draft threshold using ruptures changepoint detection."""
    # Remove NaN values
    mask = ~np.isnan(draft_vals) & ~np.isnan(melt_vals)
    if np.sum(mask) < 20:
        return np.nan, np.nan, np.nan, np.nan
    
    draft_clean = draft_vals[mask]
    melt_clean = melt_vals[mask]
    
    # Sort by draft
    sort_idx = np.argsort(draft_clean)
    draft_sorted = draft_clean[sort_idx]
    melt_sorted = melt_clean[sort_idx]
    
    # Bin the data
    bins = np.linspace(draft_sorted.min(), draft_sorted.max(), n_bins + 1)
    bin_indices = np.digitize(draft_sorted, bins) - 1
    
    # Calculate binned statistics
    binned_melt = []
    binned_draft = []
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) >= min_points_per_bin:
            binned_melt.append(np.mean(melt_sorted[bin_mask]))
            binned_draft.append(np.mean(draft_sorted[bin_mask]))
    
    if len(binned_melt) < 10:
        return np.nan, np.nan, np.nan, np.nan
    
    binned_melt = np.array(binned_melt)
    binned_draft = np.array(binned_draft)
    
    # Use ruptures for changepoint detection
    try:
        if method == 'pelt':
            algo = rpt.Pelt(model="rbf", min_size=2).fit(binned_melt.reshape(-1, 1))
            result = algo.predict(pen=penalty)
        elif method == 'binseg':
            algo = rpt.Binseg(model="l2", min_size=2).fit(binned_melt.reshape(-1, 1))
            result = algo.predict(n_bkps=1)
        elif method == 'window':
            algo = rpt.Window(width=5, model="l2").fit(binned_melt.reshape(-1, 1))
            result = algo.predict(n_bkps=1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if len(result) > 1:  # ruptures returns end points, so take first changepoint
            cp_idx = result[0] - 1  # Convert to 0-based index
            if 0 <= cp_idx < len(binned_draft):
                threshold_draft = binned_draft[cp_idx]
            else:
                threshold_draft = np.median(binned_draft)
        else:
            threshold_draft = np.median(binned_draft)
            
    except Exception as e:
        print(f"ruptures failed: {e}, using median")
        threshold_draft = np.median(binned_draft)
    
    return threshold_draft, binned_draft, binned_melt, (draft_clean, melt_clean)

def fit_piecewise_models(draft_vals, melt_vals, threshold, weights=None):
    """Fit multiple piecewise linear models with different shallow behaviors."""
    mask = ~np.isnan(draft_vals) & ~np.isnan(melt_vals)
    draft_clean = draft_vals[mask]
    melt_clean = melt_vals[mask]
    
    if weights is not None:
        weights_clean = weights[mask]
    else:
        weights_clean = None
    
    # Points deeper than threshold (for linear regression)
    deep_mask = draft_clean >= threshold
    shallow_mask = draft_clean < threshold
    
    # Calculate shallow statistics
    if np.sum(shallow_mask) >= 5:
        if weights_clean is not None:
            shallow_weights = weights_clean[shallow_mask]
            shallow_mean = np.average(melt_clean[shallow_mask], weights=shallow_weights)
        else:
            shallow_mean = np.mean(melt_clean[shallow_mask])
    else:
        if weights_clean is not None:
            shallow_mean = np.average(melt_clean, weights=weights_clean)
        else:
            shallow_mean = np.mean(melt_clean)  # Fallback to overall mean
    
    # Fit linear regression to deep points
    if np.sum(deep_mask) >= 5:
        X_deep = draft_clean[deep_mask].reshape(-1, 1)
        y_deep = melt_clean[deep_mask]
        
        reg = LinearRegression()
        if weights_clean is not None:
            deep_weights = weights_clean[deep_mask]
            reg.fit(X_deep, y_deep, sample_weight=deep_weights)
        else:
            reg.fit(X_deep, y_deep)
        slope = reg.coef_[0]
        deep_intercept = reg.intercept_
    else:
        # Not enough deep points, use all data
        X_all = draft_clean.reshape(-1, 1)
        y_all = melt_clean
        reg = LinearRegression()
        if weights_clean is not None:
            reg.fit(X_all, y_all, sample_weight=weights_clean)
        else:
            reg.fit(X_all, y_all)
        slope = reg.coef_[0]
        deep_intercept = reg.intercept_
        
    return slope, shallow_mean, deep_intercept

def predict_piecewise_multiple(draft_vals, threshold, slope, shallow_mean, deep_intercept):
    """Predict melt using multiple piecewise models."""
    predictions = {}
    valid_mask = ~np.isnan(draft_vals)
    shallow_mask = valid_mask & (draft_vals < threshold)
    deep_mask = valid_mask & (draft_vals >= threshold)
    
    # Model 1: Zero shallow
    pred_zero = np.full_like(draft_vals, np.nan)
    pred_zero[shallow_mask] = 0.0
    pred_zero[deep_mask] = slope * draft_vals[deep_mask] + deep_intercept
    predictions['zero_shallow'] = pred_zero
    
    # Model 2: Mean shallow
    pred_mean = np.full_like(draft_vals, np.nan)
    pred_mean[shallow_mask] = shallow_mean
    pred_mean[deep_mask] = slope * draft_vals[deep_mask] + deep_intercept
    predictions['mean_shallow'] = pred_mean
    
    # Model 3: Threshold intercept (evaluate deep line at threshold)
    threshold_melt = slope * threshold + deep_intercept
    pred_threshold = np.full_like(draft_vals, np.nan)
    pred_threshold[shallow_mask] = threshold_melt  # Constant value at threshold
    pred_threshold[deep_mask] = slope * draft_vals[deep_mask] + deep_intercept
    predictions['threshold_intercept'] = pred_threshold
    
    return predictions

def create_fallback_result(shelf_name, melt_vals, draft_vals, noisy_fallback, correlation=0.0, r2=0.0):
    """Create fallback result for noisy or insufficient data."""
    if len(melt_vals) == 0 or len(draft_vals) == 0:
        fallback_pred = np.array([])
        shallow_mean_value = 0.0
    else:
        # Calculate shallow mean for fallback, handling NaN values properly
        valid_melt = melt_vals[~np.isnan(melt_vals)] if len(melt_vals) > 0 else np.array([])
        
        if noisy_fallback == 'zero':
            fallback_pred = np.zeros_like(melt_vals)
            shallow_mean_value = 0.0
        else:  # 'mean'
            if len(valid_melt) > 0:
                mean_val = np.mean(valid_melt)
                fallback_pred = np.full_like(melt_vals, mean_val)
                shallow_mean_value = mean_val
            else:
                fallback_pred = np.zeros_like(melt_vals)
                shallow_mean_value = 0.0
    
    # Create predictions dict for consistency
    predictions = {
        'zero_shallow': fallback_pred.copy(),
        'mean_shallow': fallback_pred.copy(),
        'threshold_intercept': fallback_pred.copy()
    }
    
    return {
        'shelf_name': shelf_name,
        'threshold': np.nan,
        'slope': 0.0,
        'shallow_mean': shallow_mean_value,
        'deep_intercept': 0.0,
        'draft_vals': draft_vals,
        'melt_vals': melt_vals,
        'predictions': predictions,
        'binned_draft': None,
        'binned_melt': None,
        'is_meaningful': False,
        'correlation': correlation,
        'r2': r2
    }

def extract_draft_dependence_parameters(result, model_selection='best'):
    """
    Extract the five draft dependence parameters from comprehensive analysis results.
    
    Args:
        result (dict): Results from calculate_single_shelf_comprehensive
        model_selection (str): Which model to use ('best', 'zero_shallow', 'mean_shallow', 'threshold_intercept')
    
    Returns:
        dict: Five draft dependence parameters:
            - minDraft: threshold draft value (0 for noisy shelves)
            - constantValue: constant melt rate for shallow areas
            - paramType: 0 for linear, 1 for constant
            - alpha0: intercept (0 for noisy shelves)
            - alpha1: slope (0 for noisy shelves)
    """
    if not result['is_meaningful']:
        # Noisy shelf - return zeros/fallback values
        return {
            'minDraft': 0.0,
            'constantValue': result['shallow_mean'],  # This will be 0 or mean based on noisy_fallback
            'paramType': 1.0,  # Use constant parameterization for noisy shelves (float for NaN compatibility)
            'alpha0': 0.0,
            'alpha1': 0.0
        }
    
    # Meaningful relationship - extract parameters
    threshold = result['threshold']
    slope = result['slope']
    shallow_mean = result['shallow_mean']
    deep_intercept = result['deep_intercept']
    
    # Determine best model if requested
    if model_selection == 'best':
        # Calculate R² for each model to determine best
        melt_obs = result['melt_vals']
        best_r2 = -np.inf
        best_model = 'zero_shallow'
        
        for model_name, predicted in result['predictions'].items():
            valid_mask = ~np.isnan(melt_obs) & ~np.isnan(predicted)
            if np.sum(valid_mask) > 0:
                obs_clean = melt_obs[valid_mask]
                pred_clean = predicted[valid_mask]
                r2 = 1 - np.sum((obs_clean - pred_clean)**2) / np.sum((obs_clean - np.mean(obs_clean))**2)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_name
        
        model_selection = best_model
    
    # Extract parameters based on selected model
    if model_selection == 'zero_shallow':
        constant_value = 0.0
    elif model_selection == 'mean_shallow':
        constant_value = shallow_mean
    elif model_selection == 'threshold_intercept':
        # For threshold intercept, the constant value is the melt at threshold
        constant_value = slope * threshold + deep_intercept
    else:
        # Default to zero shallow
        constant_value = 0.0
    
    return {
        'minDraft': threshold,
        'constantValue': constant_value,
        'paramType': 0.0,  # Linear parameterization for meaningful relationships (float for NaN compatibility)
        'alpha0': deep_intercept,
        'alpha1': slope
    }

def save_comprehensive_coefficients(ds_tm, draft_params, catchment_name, save_dir, config, i, icems):
    """
    Save comprehensive draft dependence coefficients to NetCDF files.
    
    Args:
        ds_tm (xarray.Dataset): Time-averaged dataset for reference
        draft_params (dict): Draft dependence parameters
        catchment_name (str): Name of the catchment
        save_dir (Path): Directory to save files
        config: Configuration object
        i (int): Catchment index
        icems: Ice shelf geometries
    """
    # Get a reference field for spatial structure
    if config.SATOBS_FLUX_VAR in ds_tm.data_vars:
        ref_field = ds_tm[config.SATOBS_FLUX_VAR]
    elif config.SORRM_FLUX_VAR in ds_tm.data_vars:
        ref_field = ds_tm[config.SORRM_FLUX_VAR]
    else:
        # Fallback - use first available data variable
        ref_field = ds_tm[list(ds_tm.data_vars)[0]]
    
    # Create DataArrays for each parameter
    param_fields = {}
    
    # Define parameter mappings (assuming config.DATA_ATTRS has these keys)
    param_mapping = {
        'draftDepenBasalMelt_minDraft': draft_params['minDraft'],
        'draftDepenBasalMelt_constantMeltValue': draft_params['constantValue'], 
        'draftDepenBasalMelt_paramType': draft_params['paramType'],
        'draftDepenBasalMeltAlpha0': draft_params['alpha0'],
        'draftDepenBasalMeltAlpha1': draft_params['alpha1']
    }
    
    for param_name, param_value in param_mapping.items():
        if param_name in config.DATA_ATTRS:
            param_ds = setup_draft_depen_field(ref_field, param_value, param_name, i, icems)
            param_fields[param_name] = param_ds
    
    # Save individual parameter files
    for param_name, param_ds in param_fields.items():
        filename = save_dir / f'{param_name}_{catchment_name}.nc'
        param_ds.to_netcdf(filename)
        print(f'{catchment_name} {param_name} saved: {filename}')
    
    # Save combined file
    if param_fields:
        combined_ds = xr.Dataset(param_fields)
        combined_filename = save_dir / f'draftDepenBasalMelt_comprehensive_{catchment_name}.nc'
        combined_ds.to_netcdf(combined_filename)
        print(f'{catchment_name} comprehensive coefficients saved: {combined_filename}')

def save_comprehensive_predictions(result, catchment_name, save_dir, draft_reference):
    """
    Save comprehensive predictions from multiple models.
    
    Args:
        result (dict): Results from comprehensive analysis
        catchment_name (str): Name of the catchment
        save_dir (Path): Directory to save files
        draft_reference (xarray.DataArray): Reference draft array for coordinates
    """
    predictions = result['predictions']
    
    # Get the original draft and melt data to understand the spatial mapping
    draft_vals = result['draft_vals']  # These are the clean, non-NaN values
    melt_vals = result['melt_vals']    # These are the clean, non-NaN values
    
    for model_name, pred_values in predictions.items():
        # Create a DataArray matching the spatial structure
        if hasattr(draft_reference, 'Time'):
            # Take time mean if draft has time dimension
            spatial_ref = draft_reference.mean(dim='Time')
        else:
            spatial_ref = draft_reference
            
        # Create prediction field initialized with NaN
        pred_array = spatial_ref.copy()
        pred_array.values = np.full_like(spatial_ref.values, np.nan)
        
        # Map predictions back to their correct spatial locations
        # We need to find where the valid (non-NaN) draft values were in the original grid
        spatial_flat = spatial_ref.values.flatten()
        valid_mask = ~np.isnan(spatial_flat)
        
        if len(pred_values) == np.sum(valid_mask):
            # Direct mapping: predictions correspond to valid grid points
            spatial_flat[valid_mask] = pred_values
            pred_array.values = spatial_flat.reshape(spatial_ref.shape)
        else:
            # Fallback: if sizes don't match, create a uniform field
            print(f"Warning: Size mismatch for {catchment_name} {model_name}. "
                  f"Predictions: {len(pred_values)}, Valid points: {np.sum(valid_mask)}")
            if len(pred_values) > 0:
                # Use the mean prediction value for the entire ice shelf
                # Check for empty array after removing NaNs to avoid warning
                valid_pred_values = pred_values[~np.isnan(pred_values)]
                if len(valid_pred_values) > 0:
                    mean_pred = np.nanmean(pred_values)
                    if not np.isnan(mean_pred):
                        pred_array.values = np.where(~np.isnan(spatial_ref.values), mean_pred, np.nan)
                # Otherwise leave as all NaN if no valid values
            # Otherwise leave as all NaN
        
        pred_array.name = f'predicted_melt_{model_name}'
        pred_array.attrs['long_name'] = f'Predicted melt rate using {model_name} model'
        pred_array.attrs['units'] = 'm/yr'
        
        # Save to file
        filename = save_dir / f'draftDepenModelPred_{model_name}_{catchment_name}.nc'
        pred_array.to_netcdf(filename)
        print(f'{catchment_name} {model_name} prediction saved: {filename}')

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
    # Use the fast ndimage-based fill (Dask-aware) for per-catchment fills
    try:
        ds = ds.map(fill_nan_with_nearest_neighbor_ndimage, keep_attrs=True)
    except Exception:
        # Fallback to vectorized KDTree if ndimage fails for some variable
        ds = ds.map(fill_nan_with_nearest_neighbor_vectorized, keep_attrs=True)
    # Ensure dataset has CRS metadata for rioxarray.clip; some slices may lose
    # rio metadata after selection. Write CRS from the icems GeoDataFrame or
    # fall back to the configured target CRS.
    try:
        target_crs = getattr(icems, 'crs', None) or config.CRS_TARGET
        ds = write_crs(ds, target_crs)
    except Exception:
        # If writing CRS fails, allow clip to raise its own informative error
        pass
    ds = ds.rio.clip(ice_shelf_mask, icems.crs)
    return ds

def extrapolate_catchment_over_time(dataset, icems, config, var_name, use_index_map=False, index_map_cache_path=None):
    """Extrapolate catchment data for each time step.

    Implementation notes:
    - Per-catchment filling uses scipy.ndimage.distance_transform_edt (fast nearest-neighbor fill)
      via the Dask-aware helper `fill_nan_with_nearest_neighbor_ndimage` in `aislens.utils`.
    - A rasterized ice-shelf union mask is produced on the model template grid and applied to
      the merged extrapolated result to ensure ocean cells outside ice shelves are set to 0.

    Returns an xarray.Dataset with filled values.
    """

    times = dataset[var_name].coords[config.TIME_DIM]
    shape = dataset[var_name].shape
    extrap_array = np.full(shape, np.nan)

    extrap_da = xr.DataArray(
        extrap_array,
        coords=dataset[var_name].coords,
        dims=dataset[var_name].dims,
        attrs=dataset[var_name].attrs,
    )
    extrap_ds = xr.Dataset({var_name: extrap_da})

    # Prepare a spatial template (first time slice) and compute a rasterized ice mask once
    try:
        template_da = (
            dataset[var_name].isel({config.TIME_DIM: 0}).compute()
            if config.TIME_DIM in dataset[var_name].dims
            else dataset[var_name].isel({0}).compute()
        )
    except Exception:
        # Fallback: take the raw DataArray without compute
        template_da = (
            dataset[var_name].isel({config.TIME_DIM: 0})
            if config.TIME_DIM in dataset[var_name].dims
            else dataset[var_name].isel({0})
        )

    # Rasterize the full ice-shelf union onto the template grid to produce a strict mask
    try:
        ice_mask_r = rasterize_ice_mask(icems, template_da)
    except Exception:
        ice_mask_r = None
        logger.warning(
            "Could not rasterize ice-shelf geometries to template grid. Ocean zeroing will be skipped."
        )

    logger.info(
        "Extrapolation configuration: per-catchment fill=ndimage, rasterized_mask_applied=%s",
        ice_mask_r is not None,
    )

    # Optional: precompute nearest-index map for the template to apply repeatedly
    index_map = None
    if use_index_map:
        try:
            mask_for_map = np.isnan(template_da.values)
            logger.info("Computing nearest-index map for template (this may use cache)")
            index_map = compute_nearest_index_map(mask_for_map, cache_path=index_map_cache_path)
            logger.info("Index map ready; will be reused for all time slices")
        except Exception as e:
            logger.warning(
                "Failed to compute or load index_map (%s); falling back to ndimage per-slice",
                e,
            )
            index_map = None

    for t in range(len(times)):
        ds_data = dataset.isel({config.TIME_DIM: t})
        # Extrapolate each catchment (fast ndimage fills inside extrapolate_catchment)
        results = [extrapolate_catchment(ds_data, i, icems) for i in config.ICE_SHELF_REGIONS]
        merged_ds = merge_catchment_data(results)
        result_ds = copy_subset_data(ds_data, merged_ds)

        # Ensure output is aligned to the template and apply rasterized mask to zero-out ocean
        try:
            # If we precomputed an index_map, use it to fill missing values on the merged result
            if index_map is not None:
                try:
                    filled_once = fill_with_index_map(result_ds[var_name], index_map)
                    result_ds[var_name] = filled_once
                except Exception:
                    logger.debug(
                        "fill_with_index_map failed for time %d; falling back to existing filled result",
                        t,
                    )

            if ice_mask_r is not None:
                # Align mask dims/ordering if needed
                if tuple(ice_mask_r.shape) != tuple(result_ds[var_name].shape):
                    try:
                        ice_mask_al = align_mask_to_template(ice_mask_r, result_ds[var_name])
                    except Exception:
                        ice_mask_al = align_mask_to_template(ice_mask_r, template_da)
                else:
                    ice_mask_al = ice_mask_r
                # Apply mask: outside ice -> 0
                result_filled = result_ds[var_name].where(ice_mask_al, other=0)
                logger.debug("Applied rasterized mask for time %d", t)
            else:
                result_filled = result_ds[var_name]
        except Exception:
            # If anything fails, fallback to raw result
            result_filled = result_ds[var_name]

        extrap_ds[var_name][t] = result_filled
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
        deg (int): The degree of the polynomial fit (default is 1 for linear).

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

def detrend_with_breakpoints_vectorized(data, dim, deg=1, model="rbf", penalty=10):
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
        dask="parallelized",             # Allow Dask parallelization - Dataset already parallelized when loaded
        output_dtypes=[data.dtype]       # Ensure correct output dtype
    )

    return detrended_data

def detrend_with_breakpoints_ts(data, dim, deg=1, model="rbf", penalty=10):
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