#!/usr/bin/env python3
"""Dask-aware detrending of ISMIP6 forcing trend components."""

import argparse
import logging
from pathlib import Path
from time import time
import xarray as xr

from aislens.config import config
from aislens.dataprep import detrend_with_breakpoints_vectorized, detrend_with_breakpoints_ts
from aislens.utils import setup_logging

logger = logging.getLogger(__name__)


def detrend_forcing_dask(forcing_file_path, output_path, method='vectorized',
                         time_chunks=12, ny_chunk=100, nx_chunk=100, ncell_chunk=100, compression=4,
                         use_dask=False, n_workers=4, threads_per_worker=1):
    """Detrend forcing data using breakpoint detection (dask-aware).

    Parameters
    - forcing_file_path: path to input netCDF
    - output_path: path to write trend netCDF
    - method: 'vectorized' or 'timeseries'
    - time_chunks: chunk size for time dimension when opening
    - ny_chunk,nx_chunk: spatial chunk sizes for output encoding
    - compression: zlib compression level (0-9)
    - use_dask: if True, start a LocalCluster with n_workers
    """
    if use_dask:
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)
        logger.info('Started local dask cluster')

    logger.info(f"Loading {forcing_file_path} as dask-backed xarray dataset with time chunk={time_chunks}...")
    time_dim = getattr(config, 'TIME_DIM', 'Time')
    ds = xr.open_dataset(forcing_file_path, chunks={time_dim: time_chunks})

    VAR = config.MALI_FLOATINGBMB_VAR
    # This subtraction is performed to reverse the sign
    # MALI outputs are negative for greater melting, while forcing variable for MALI (floatingBasalMassBalAdjustment in this case) should be positive for greater melting
    ds[VAR] = (ds[VAR].isel({time_dim: 0}) - ds[VAR])

    if method == 'vectorized':
        logger.info('Detrending (vectorized, dask-aware)')
        # ensure the Time dimension is a single dask chunk - this is required by xarray.apply_ufunc
        # when using dask='parallelized' with the time dimension.
        ds[VAR] = ds[VAR].chunk({time_dim: -1})
        detrended = detrend_with_breakpoints_vectorized(ds[VAR], dim=time_dim, deg=1, model='rbf', penalty=10)
    else:
        logger.info('Detrending (timeseries - spatial mean)')
        spatial_dims = [d for d in ds[VAR].dims if d != time_dim]
        spatial_mean_ts = ds[VAR].mean(dim=spatial_dims)
        detrended_ts = detrend_with_breakpoints_ts(spatial_mean_ts, dim=time_dim, deg=1, model='rbf', penalty=10)
        # expand back to spatial dims by broadcasting
        detrended = detrended_ts
        for sd in spatial_dims:
            detrended = detrended.expand_dims({sd: ds.coords[sd]})
        detrended = detrended.transpose(*ds[VAR].dims)

    # Build trend dataset
    trend_ds = (ds - detrended.to_dataset(name=VAR)).rename({VAR: config.AISLENS_FLOATINGBMB_VAR})

    # Prepare encoding for chunked/compressed output
    spatial_dims = [d for d in trend_ds[config.AISLENS_FLOATINGBMB_VAR].dims if d != time_dim]
    # build chunksizes tuple that matches (time, <spatial dims...>)
    if len(spatial_dims) == 1:
        # unstructured grid, e.g., dims = (Time, nCells)
        chunksizes = (1, int(ncell_chunk))
    elif len(spatial_dims) >= 2:
        # structured grid with two spatial dims (y,x)
        chunksizes = (1, int(ny_chunk), int(nx_chunk))
    else:
        # unexpected layout: fall back to time-chunking only
        chunksizes = (1,)

    enc = {}
    var_out = config.AISLENS_FLOATINGBMB_VAR
    enc[var_out] = {'zlib': True, 'complevel': int(compression), 'chunksizes': chunksizes, 'dtype': 'float32'}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Writing trend dataset to {output_path} with chunks {chunksizes} and compression={compression}')
    trend_ds.to_netcdf(output_path, encoding=enc)

    if use_dask:
        client.close()
        cluster.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dask-aware detrending of forcing trend components')
    parser.add_argument('input_file', nargs='?', type=Path, default=config.FILE_ISMIP6_SSP585_FORCING)
    parser.add_argument('output_file', nargs='?', type=Path, help='Output trend file')
    parser.add_argument('--method', choices=['vectorized', 'timeseries'], default='vectorized')
    parser.add_argument('--time-chunks', type=int, default=12, help='Time chunk size for input')
    parser.add_argument('--ny-chunk', type=int, default=100, help='Y chunk size for output encoding')
    parser.add_argument('--nx-chunk', type=int, default=100, help='X chunk size for output encoding')
    parser.add_argument('--ncell-chunk', type=int, default=100, help='nCells chunk size for unstructured outputs')
    parser.add_argument('--compression', type=int, default=4, help='zlib compression level (0-9)')
    parser.add_argument('--use-dask', action='store_true', help='Start a local dask cluster for computation')
    parser.add_argument('--n-workers', type=int, default=4, help='Number of dask workers if --use-dask')
    parser.add_argument('--threads-per-worker', type=int, default=1, help='Threads per dask worker')
    args = parser.parse_args()

    if args.output_file is None:
        input_stem = args.input_file.stem
        output_name = f"{input_stem}_TREND.nc"
        args.output_file = Path(config.DIR_MALI_ISMIP6_FORCINGS) / output_name

    setup_logging(args.output_file.parent, 'prepare_forcing_trend_components_dask')

    logger.info('DASK-AWARE FORCING TREND COMPONENT DETRENDING')
    logger.info(f'Input: {args.input_file}')
    logger.info(f'Output: {args.output_file}')
    logger.info(f'Method: {args.method}')
    logger.info(f'Time chunks: {args.time_chunks}, ny_chunk: {args.ny_chunk}, nx_chunk: {args.nx_chunk}, ncell_chunk: {args.ncell_chunk}')

    t0 = time()
    detrend_forcing_dask(args.input_file, args.output_file, method=args.method,
                         time_chunks=args.time_chunks, ny_chunk=args.ny_chunk, nx_chunk=args.nx_chunk, ncell_chunk=args.ncell_chunk,
                         compression=args.compression, use_dask=args.use_dask,
                         n_workers=args.n_workers, threads_per_worker=args.threads_per_worker)
    logger.info(f'COMPLETE ({time() - t0:.1f}s)')
