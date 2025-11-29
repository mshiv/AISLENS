#!/usr/bin/env python3
"""fill_extrapolate.py

Fill NaNs in 2D spatial fields (y,x) using nearest-neighbour via
scipy.ndimage.distance_transform_edt (preferred) with an optional KDTree
fallback. Processes 3D (Time,y,x) datasets slice-by-slice to limit memory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree
try:
    import netCDF4
except Exception:
    netCDF4 = None


def _to_bool_mask(arr_like) -> np.ndarray:
    """Convert mask-like arrays to boolean (True = ice)."""
    a = np.asarray(arr_like)
    if a.dtype == bool:
        return a
    # floats with NaNs: non-NaN -> True
    if np.issubdtype(a.dtype, np.floating) and np.isnan(a).any():
        return ~np.isnan(a)
    # integers or floats with 0/1 values
    try:
        uniq = np.unique(a)
        # small arrays may include many uniques; check for 0/1 pattern
        if set(uniq.tolist()).issubset({0, 1}):
            return a != 0
    except Exception:
        pass
    # fallback: non-zero => True
    return a != 0

def _parse_ndimage_inds(res, ny: int, nx: int):
    """Extract (iy, ix) index arrays from scipy.ndimage outputs or return (None,None)."""
    iy = ix = None
    try:
        if isinstance(res, tuple):
            # common return is (iy_array, ix_array)
            if len(res) == 2 and getattr(res[0], 'ndim', None) == 2 and getattr(res[1], 'ndim', None) == 2:
                iy, ix = res[0].astype(int), res[1].astype(int)
            else:
                maybe = res[-1]
                arr = np.asarray(maybe)
                if arr.ndim == 3 and arr.shape[0] >= 2:
                    iy, ix = arr[0].astype(int), arr[1].astype(int)
                elif arr.ndim == 2 and arr.shape[0] == 2:
                    iy, ix = arr[0].astype(int), arr[1].astype(int)
        else:
            arr = np.asarray(res)
            if arr.ndim == 3 and arr.shape[0] >= 2:
                iy, ix = arr[0].astype(int), arr[1].astype(int)
    except Exception:
        return None, None

    if iy is None or ix is None:
        return None, None
    if iy.shape != (ny, nx) or ix.shape != (ny, nx):
        return None, None
    return iy, ix


def robust_fill_2d(arr2d: np.ndarray, allow_kdtree: bool = True) -> Tuple[np.ndarray, str]:
    """Fill NaNs in a 2D array using ndimage EDT; optional KDTree fallback."""
    arr = np.asarray(arr2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError("robust_fill_2d expects a 2D array")
    ny, nx = arr.shape
    valid = ~np.isnan(arr)

    if np.all(valid):
        return arr.copy(), "trivial:all-valid"
    if np.all(~valid):
        return arr.copy(), "trivial:all-nan"

    # Try ndimage distance transform
    res = ndimage.distance_transform_edt(~valid, return_indices=True)
    iy, ix = _parse_ndimage_inds(res, ny, nx)
    if iy is not None and ix is not None:
        nearest_vals = arr[iy, ix]
        filled = np.where(valid, arr, nearest_vals)
        return filled, "ndimage"

    # KDTree fallback (only if allowed)
    if allow_kdtree:
        yy, xx = np.nonzero(valid)
        if yy.size == 0:
            return arr.copy(), "kdtree-fallback:no-valid"
        vals = arr[yy, xx]
        pts = np.column_stack((yy, xx))
        tree = cKDTree(pts)
        all_pts = np.column_stack(np.nonzero(np.ones_like(arr)))
        _, idx = tree.query(all_pts, k=1)
        nearest = vals[idx].reshape((ny, nx))
        filled = np.where(valid, arr, nearest)
        return filled, "kdtree-fallback"
    # If we reach here, KDTree fallback is disabled and ndimage failed
    return arr.copy(), "ndimage-failed-no-fallback"


def extrapolate_da_nearest(
    da: xr.DataArray,
    time_dim: str = "Time",
    fill_value: Optional[float] = None,
    report_methods: bool = False,
) -> Tuple[xr.DataArray, Optional[List[str]]]:
    """Apply robust nearest-neighbour fill to an xarray.DataArray.

    Works with 2D arrays or 3D (time, y, x) arrays. Returns filled DataArray and,
    optionally, a list of per-slice method strings.
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("extrapolate_da_nearest expects an xarray.DataArray")

    # We'll process slice-by-slice to limit memory use
    arr = da
    dims = list(arr.dims)
    if time_dim in dims and arr.ndim >= 3:
        t_axis = dims.index(time_dim)
        nt = arr.sizes[time_dim]
        # prepare an empty numpy container with same dtype
        filled_np = np.full(arr.shape, np.nan, dtype=float)
        methods = []
        for t in range(nt):
            # select slice without loading entire dataset
            sl = arr.isel({time_dim: t}).compute()
            sl_np = sl.values
            filled_slice, method = robust_fill_2d(sl_np)
            methods.append(method)
            # place back into filled_np respecting axis order
            if t_axis == 0:
                filled_np[t, ...] = filled_slice
            else:
                # move axis temporarily
                filled_np[(slice(None),) * t_axis + (t,)] = filled_slice
        # build DataArray
        filled_da = xr.DataArray(filled_np, dims=arr.dims, coords={d: arr.coords[d] for d in arr.dims}, attrs=arr.attrs)
        if fill_value is not None:
            filled_da = filled_da.fillna(fill_value)
        if report_methods:
            return filled_da, methods
        return filled_da, None
    elif arr.ndim == 2:
        arr_np = arr.compute().values if hasattr(arr.data, 'compute') else arr.values
        filled_np, method = robust_fill_2d(arr_np)
        filled_da = xr.DataArray(filled_np, dims=arr.dims, coords={d: arr.coords[d] for d in arr.dims}, attrs=arr.attrs)
        if fill_value is not None:
            filled_da = filled_da.fillna(fill_value)
        if report_methods:
            return filled_da, [method]
        return filled_da, None
    else:
        # For arrays with leading non-spatial dims, process last two dims
        shape = arr.shape
        if arr.ndim < 2:
            raise ValueError("DataArray has fewer than 2 dimensions; cannot fill spatially")
        leading = shape[:-2]
        out = np.empty_like(arr.values, dtype=float)
        methods = []
        it = np.ndindex(*leading)
        for idx in it:
            # build full index
            sl = arr.values[idx + (slice(None), slice(None))]
            filled_slice, method = robust_fill_2d(sl)
            out[idx + (slice(None), slice(None))] = filled_slice
            methods.append(method)
        filled_da = xr.DataArray(out, dims=arr.dims, coords={d: arr.coords[d] for d in arr.dims}, attrs=arr.attrs)
        if fill_value is not None:
            filled_da = filled_da.fillna(fill_value)
        if report_methods:
            return filled_da, methods
        return filled_da, None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fill/extrapolate 2D spatial NaNs across time using nearest-neighbour")
    p.add_argument("--input", "-i", required=True, help="Input NetCDF file (xarray-readable)")
    p.add_argument("--output", "-o", required=True, help="Output NetCDF file to write filled dataset")
    p.add_argument("--var", "-v", default=None, help="Variable name to process (default: first data var)")
    p.add_argument("--time-dim", default="Time", help="Name of time dimension (default: Time)")
    p.add_argument("--start-index", type=int, default=None, help="Start time index (inclusive). If omitted, starts at 0")
    p.add_argument("--end-index", type=int, default=None, help="End time index (exclusive). If omitted, runs to end")
    p.add_argument("--fill-value", type=float, default=None, help="Optional fill value for any remaining NaNs (e.g., 0)")
    p.add_argument("--report-methods", action="store_true", help="Write per-slice chosen method to stdout and return with output file")
    p.add_argument("--ndimage-only", action="store_true", help="Use scipy.ndimage only and do not fall back to KDTree if ndimage fails")
    p.add_argument("--mask", default=None, help="Optional path to ice mask (raster or vector). Mask True=ice; outside will be set to --mask-fill")
    p.add_argument("--mask-fill", default="0", choices=["0", "nan"], help="Value to write for cells outside the mask: '0' or 'nan' (default: 0)")
    p.add_argument("--mask-invert", action="store_true", help="Invert the aligned mask booleans (useful if mask is 1 for ocean instead of ice)")
    p.add_argument("--verbose", "-V", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    inpath = Path(args.input)
    outpath = Path(args.output)
    if not inpath.exists():
        logging.error("Input file not found: %s", inpath)
        return 2

    logging.info("Opening %s", inpath)
    ds = xr.open_dataset(inpath)
    var_name = args.var or (list(ds.data_vars)[0] if ds.data_vars else None)
    if var_name is None:
        logging.error("No data variables found in input dataset and --var not provided")
        return 3

    da = ds[var_name]
    logging.info("Processing variable: %s", var_name)

    # Determine time slice indices
    time_dim = args.time_dim
    if time_dim in da.dims:
        ntot = da.sizes[time_dim]
        start = args.start_index if args.start_index is not None else 0
        end = args.end_index if args.end_index is not None else ntot
        # normalize negative indices
        if start < 0:
            start = max(0, ntot + start)
        if end < 0:
            end = max(0, ntot + end)
        if start >= end or start < 0 or end > ntot:
            logging.error("Invalid start/end indices: start=%s end=%s ntot=%s", start, end, ntot)
            return 5
        indices = range(start, end)
        logging.info("Processing time indices %d .. %d (n=%d)", start, end - 1, end - start)
    else:
        # no time dim; treat as single 2D
        indices = None

    # Prepare incremental netCDF writer if available
    if netCDF4 is None:
        logging.warning("netCDF4 not available; will fall back to building full xarray output in memory")

    # If no time dim, just fill and write
    if indices is None:
        filled_da, methods = extrapolate_da_nearest(da, time_dim=time_dim, fill_value=args.fill_value, report_methods=args.report_methods)
        ds_out = ds.copy(deep=True)
        ds_out[var_name] = filled_da
        logging.info("Writing output to %s", outpath)
        try:
            ds_out.to_netcdf(outpath)
        except Exception as e:
            logging.exception("Failed to write output: %s", e)
            return 4
        if args.report_methods and methods is not None:
            logging.info("Methods: %s", methods)
        logging.info("Done")
        return 0

    # Prepare output file using netCDF4 for incremental writes
    if netCDF4 is None:
        # fallback: compute full filled array in memory then write with xarray
        logging.info("Computing full filled array in memory (no netCDF4)")
        filled_da, methods = extrapolate_da_nearest(da.isel({time_dim: slice(start, end)}), time_dim=time_dim, fill_value=args.fill_value, report_methods=args.report_methods)
        ds_out = ds.copy(deep=True)
        ds_out[var_name] = filled_da
        logging.info("Writing output to %s", outpath)
        try:
            ds_out.to_netcdf(outpath)
        except Exception as e:
            logging.exception("Failed to write output: %s", e)
            return 4
        if args.report_methods and methods is not None:
            logging.info("Method counts: %s", methods)
        logging.info("Done")
        return 0

    # Use netCDF4 to create file and variables
    logging.info("Creating output NetCDF (incremental write) %s", outpath)
    # gather coords
    # spatial dims: assume last two dims are spatial
    spatial_dims = [d for d in da.dims if d != time_dim]
    if len(spatial_dims) < 2:
        logging.error("DataArray does not have at least two spatial dims")
        return 6
    y_dim, x_dim = spatial_dims[-2], spatial_dims[-1]
    y_vals = da.coords[y_dim].values
    x_vals = da.coords[x_dim].values
    time_vals = da.coords[time_dim].values[start:end]

    # open netCDF for writing
    nc = netCDF4.Dataset(str(outpath), mode='w')
    # create dims
    nc.createDimension(time_dim, len(time_vals))
    nc.createDimension(y_dim, len(y_vals))
    nc.createDimension(x_dim, len(x_vals))

    # coord vars
    # netCDF4 requires primitive dtypes; convert datetime-like -> numeric using netCDF4.date2num
    # and coerce spatial coords to float64 where needed.
    try:
        time_units = None
        time_calendar = None
        # prefer any existing units/calendar on the coord
        coord_attrs = da.coords[time_dim].attrs if time_dim in da.coords else {}
        time_units = coord_attrs.get('units') if isinstance(coord_attrs, dict) else None
        time_calendar = coord_attrs.get('calendar') if isinstance(coord_attrs, dict) else None

        numeric_time = None
        # numpy datetime64 -> convert to seconds since epoch (float seconds)
        if np.issubdtype(time_vals.dtype, np.datetime64):
            epoch = np.datetime64('1970-01-01T00:00:00Z')
            tv_ns = time_vals.astype('datetime64[ns]')
            numeric_time = (tv_ns - epoch).astype('timedelta64[ns]').astype('int64') / 1e9
            if not time_units:
                time_units = 'seconds since 1970-01-01 00:00:00'

        else:
            # handle cftime or python datetime objects using netCDF4.date2num
            py_dates = list(time_vals.tolist())
            if not time_units:
                time_units = 'seconds since 1970-01-01 00:00:00'
            if time_calendar is None:
                time_calendar = 'standard'
            try:
                numeric_time = netCDF4.date2num(py_dates, units=time_units, calendar=time_calendar)
            except Exception:
                logging.exception("Unable to convert time coordinate with netCDF4.date2num")
                nc.close()
                return 7

        # create time variable once using numeric_time
        time_var = nc.createVariable(time_dim, 'f8', (time_dim,))
        time_var[:] = numeric_time
        # attach units/calendar where possible
        try:
            if time_units:
                time_var.setncattr('units', time_units)
            if time_calendar is not None:
                time_var.setncattr('calendar', time_calendar)
        except Exception:
            pass

    except Exception:
        logging.exception("Failed to create/write time coordinate; aborting")
        nc.close()
        return 7

    # y/x coords: coerce to numeric types acceptable to netCDF4
    try:
        y_var = nc.createVariable(y_dim, 'f8', (y_dim,))
        y_var[:] = np.asarray(y_vals, dtype=float)
    except Exception:
        logging.exception("Failed to write y coordinate; aborting")
        nc.close()
        return 8

    try:
        x_var = nc.createVariable(x_dim, 'f8', (x_dim,))
        x_var[:] = np.asarray(x_vals, dtype=float)
    except Exception:
        logging.exception("Failed to write x coordinate; aborting")
        nc.close()
        return 9

    # data var
    dtype = 'f8'
    fillval = np.nan if args.fill_value is None else args.fill_value
    data_var = nc.createVariable(var_name, dtype, (time_dim, y_dim, x_dim), fill_value=fillval)
    # copy attributes if present
    for k, v in ds.attrs.items():
        try:
            nc.setncattr(k, v)
        except Exception:
            pass
    # variable attrs
    for k, v in ds[var_name].attrs.items():
        try:
            data_var.setncattr(k, v)
        except Exception:
            pass

    methods = []
    # Prepare mask if requested. Align to template grid using three-step approach similar to notebook.
    ice_mask2 = None
    mask_other = 0 if args.mask_fill == '0' else np.nan
    if args.mask:
        da_template = da.isel({time_dim: start}) if time_dim in da.dims else da
        method_used = None
        _reproj_err = _reindex_err = _rast_err = None
        # Attempt 1: rioxarray reproject_match
        try:
            import rioxarray
            # try open mask as raster
            try:
                mask_da = xr.open_dataset(args.mask)
                # if dataset, pick first data var
                if mask_da.data_vars:
                    mask_da = mask_da[list(mask_da.data_vars)[0]]
            except Exception:
                # try rasterio/rioxarray open
                try:
                    mask_da = rioxarray.open_rasterio(args.mask)
                    # open_rasterio returns (band,y,x); reduce to single band if needed
                    if mask_da.ndim == 3 and mask_da.sizes.get('band', 1) == 1:
                        mask_da = mask_da.squeeze('band', drop=True)
                except Exception:
                    raise
            if hasattr(mask_da, 'rio') and hasattr(da_template, 'rio'):
                clipped_matched = mask_da.rio.reproject_match(da_template)
                ice_bool = _to_bool_mask(clipped_matched.values)
                ice_mask2 = xr.DataArray(ice_bool, dims=da_template.dims,
                                         coords={d: da_template.coords[d] for d in da_template.dims})
                method_used = 'reproject_match'
        except Exception as _e_reproj:
            _reproj_err = _e_reproj
        # Attempt 2: reindex nearest
        if ice_mask2 is None:
            try:
                ds_mask2 = xr.open_dataset(args.mask)
                mask_var_name = list(ds_mask2.data_vars)[0] if ds_mask2.data_vars else None
                mask_da2 = ds_mask2[mask_var_name] if mask_var_name else ds_mask2
                ice_reindexed = mask_da2.reindex({
                    da_template.dims[0]: da_template.coords[da_template.dims[0]],
                    da_template.dims[1]: da_template.coords[da_template.dims[1]]
                }, method='nearest', fill_value=False)
                ice_bool = _to_bool_mask(ice_reindexed.values)
                ice_mask2 = xr.DataArray(ice_bool, dims=da_template.dims,
                                         coords={d: da_template.coords[d] for d in da_template.dims})
                method_used = 'reindex_nearest'
                method_used = 'reindex_nearest'
            except Exception as _e_reindex:
                _reindex_err = _e_reindex
        # Attempt 3: rasterize vector geometries
        if ice_mask2 is None:
            try:
                import geopandas as gpd
                from shapely.geometry import mapping
                from rasterio.features import rasterize

                geoms = gpd.read_file(args.mask)
                logging.info('Read %d geometries from mask; mask.crs=%s', len(geoms), getattr(geoms, 'crs', None))
                # ensure geometries are not empty
                geoms = geoms[~geoms.geometry.isna()]
                if geoms.empty:
                    raise RuntimeError('Mask vector contains no geometries')

                # Reproject geometries to template CRS if needed
                try:
                    tmpl_crs = da_template.rio.crs if hasattr(da_template, 'rio') else None
                except Exception:
                    tmpl_crs = None
                logging.info('Template CRS: %s', tmpl_crs)
                if getattr(geoms, 'crs', None) is not None and tmpl_crs is not None and geoms.crs != tmpl_crs:
                    geoms = geoms.to_crs(tmpl_crs)
                    logging.info('Reprojected mask geometries to template CRS')

                # diagnostic bounds
                try:
                    geom_bounds = geoms.total_bounds
                    tmpl_bounds = da_template.rio.bounds()
                    logging.info('Mask bounds: %s; template bounds: %s', geom_bounds, tmpl_bounds)
                except Exception:
                    pass

                transform = da_template.rio.transform()
                out_shape = tuple(da_template.shape)
                geom_tuples = [(mapping(g), 1) for g in geoms.geometry]
                mask_arr = rasterize(geom_tuples, out_shape=out_shape, transform=transform, fill=0, dtype='uint8')
                ice_mask2 = xr.DataArray(mask_arr.astype(bool), dims=da_template.dims,
                                         coords={dim: da_template.coords[dim] for dim in da_template.dims})
                method_used = 'rasterize'
            except Exception as _e_rast:
                _rast_err = _e_rast
        # Finalize mask
        if ice_mask2 is not None:
            # ensure ordering
            if set(ice_mask2.dims) == set(da_template.dims) and ice_mask2.dims != da_template.dims:
                ice_mask2 = ice_mask2.transpose(*da_template.dims)
            # optional invert
            if args.mask_invert:
                ice_mask2 = xr.DataArray(~ice_mask2.values, dims=ice_mask2.dims, coords=ice_mask2.coords)
            # diagnostics
            n_true = int(np.count_nonzero(ice_mask2.values))
            n_total = int(ice_mask2.size)
            logging.info('Mask alignment method used: %s', method_used)
            logging.info('Mask true count: %d / %d (%.2f%%)', n_true, n_total, 100.0 * n_true / max(1, n_total))
            if n_true == 0:
                logging.warning('Aligned mask contains zero True pixels; skipping mask application')
                ice_mask2 = None
        else:
            logging.error('Failed to align mask; reproject_match error: %s; reindex error: %s; rasterize error: %s', _reproj_err, _reindex_err, _rast_err)
            logging.warning('Proceeding without mask')
    # iterate and write each slice
    for out_t_idx, t in enumerate(indices):
        logging.info("Processing time index %d (output index %d)", t, out_t_idx)
        sl = da.isel({time_dim: t}).compute()
        filled_slice, method = robust_fill_2d(sl.values, allow_kdtree=not args.ndimage_only)
        methods.append(method)
        # apply optional fill_value override
        if args.fill_value is not None:
            filled_slice = np.where(np.isfinite(filled_slice), filled_slice, args.fill_value)
        # apply mask per-slice if present (outside -> mask_other)
        if ice_mask2 is not None:
            try:
                filled_slice = np.where(ice_mask2.values, filled_slice, mask_other)
            except Exception:
                logging.exception('Failed to apply ice mask to slice %d', t)
                nc.close()
                return 12
        # write
        data_var[out_t_idx, :, :] = filled_slice

    nc.close()

    if args.report_methods:
        from collections import Counter

        c = Counter(methods)
        logging.info("Method counts: %s", dict(c))

    logging.info("Done")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
