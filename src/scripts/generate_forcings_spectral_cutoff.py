#!/usr/bin/env python3
"""
Generate MALI forcing realizations using EOF decomposition and spectral-cutoff phase randomization.

This is a variant of `generate_forcings.py` that exposes spectral cutoff parameters
for the phase-randomization step (min/max period in years, retain_outside, rescale).
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import xarray as xr
import numpy as np

from aislens.generator import eof_decomposition, phase_randomization_spectral_cutoff, generate_data
from aislens.utils import setup_logging
from aislens.config import config

logger = logging.getLogger(__name__)


def load_and_prepare_data(seasonality_path=None, variability_path=None, time_chunks=36):
    seasonality_file = Path(seasonality_path) if seasonality_path else Path(config.FILE_SEASONALITY_EXTRAPL)
    variability_file = Path(variability_path) if variability_path else Path(config.FILE_VARIABILITY_EXTRAPL)

    logger.info("Loading extrapolated seasonality and variability datasets...")
    for name, path in [("Seasonality", seasonality_file), ("Variability", variability_file)]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

    seasonality = xr.open_dataset(seasonality_file, chunks={getattr(config, 'TIME_DIM', 'Time'): time_chunks})
    variability = xr.open_dataset(variability_file, chunks={getattr(config, 'TIME_DIM', 'Time'): time_chunks})

    # Rename Time -> time on the loaded datasets (assign back to ensure change)
    if 'Time' in variability.dims:
        variability = variability.rename({"Time": "time"})
        logger.debug("Renamed 'Time' to 'time' in variability dataset")
    if 'Time' in seasonality.dims:
        seasonality = seasonality.rename({"Time": "time"})
        logger.debug("Renamed 'Time' to 'time' in seasonality dataset")

    if config.SORRM_FLUX_VAR not in variability:
        raise ValueError(f"Variable '{config.SORRM_FLUX_VAR}' not found in variability dataset")

    data = variability[config.SORRM_FLUX_VAR]
    data_tmean = data.mean('time')
    data_tstd = data.std('time').where(lambda x: x > 0, 1e-10)
    data_norm = (data - data_tmean) / data_tstd

    return seasonality, variability, data_norm, data_tmean, data_tstd


def perform_eof_analysis(data_norm, load_existing=False):
    nc_path = Path(config.FILE_EOF_MODEL)

    class SimpleEOFModel:
        def __init__(self, eofs_da: xr.DataArray, pcs_da: xr.DataArray):
            self.eofs = eofs_da
            self.pcs = pcs_da

        @property
        def neofs(self):
            return int(self.eofs.sizes.get('mode', self.eofs.shape[0]))

        def reconstruct_randomized_X(self, new_pcs_arr: np.ndarray, mode_slice: slice):
            start = mode_slice.start or 1
            stop = mode_slice.stop or self.neofs
            step = mode_slice.step or 1
            py_start = max(0, int(start) - 1)
            py_stop = min(int(stop), self.neofs)
            indices = list(range(py_start, py_stop, int(step)))

            eofs_vals = self.eofs.values[indices, ...]
            pcs_sel = new_pcs_arr[:, indices]
            arr = np.tensordot(pcs_sel, eofs_vals, axes=([1], [0]))

            time_dim = self.pcs.dims[0]
            y_dim = self.eofs.dims[1] if len(self.eofs.dims) > 1 else 'y'
            x_dim = self.eofs.dims[2] if len(self.eofs.dims) > 2 else 'x'
            da = xr.DataArray(
                arr,
                dims=(time_dim, y_dim, x_dim),
                coords={time_dim: self.pcs.coords[self.pcs.dims[0]],
                        y_dim: self.eofs.coords[y_dim],
                        x_dim: self.eofs.coords[x_dim]},
            )
            return da

    if load_existing:
        if not nc_path.exists():
            raise FileNotFoundError(f"EOF NetCDF model not found: {nc_path}.")
        ds = xr.open_dataset(nc_path)
        eofs_da = ds['eofs']
        pcs_da = ds['pcs']
        nmodes = int(eofs_da.sizes.get('mode', eofs_da.shape[0]))
        model = SimpleEOFModel(eofs_da, pcs_da)
        return model, pcs_da, nmodes

    model, eofs, pcs, nmodes, varexpl = eof_decomposition(data_norm)
    return model, pcs, nmodes


def generate_and_save_forcings(model, pcs, nmodes, data_tmean, data_tstd,
                               seasonality, data, n_realizations,
                               include_seasonality=True,
                               min_period_years=None, max_period_years=None,
                               sampling_months=1.0, retain_outside=False,
                               rescale_variance=True, random_state=None):
    logger.info(f"Generating {n_realizations} ensemble realizations (spectral-cutoff)...")
    pcs_arr = pcs.values if hasattr(pcs, 'values') else np.asarray(pcs)

    new_pcs = phase_randomization_spectral_cutoff(
        pcs_arr, n_realizations,
        min_period_years=min_period_years, max_period_years=max_period_years,
        sampling_months=sampling_months, retain_outside=retain_outside,
        rescale_variance=rescale_variance, random_state=random_state
    )

    output_dir = Path(config.DIR_FORCINGS)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_realizations):
        new_data = generate_data(model, new_pcs, i, nmodes, 1)
        new_data = (new_data * data_tstd) + data_tmean
        new_data = xr.DataArray(new_data, dims=data.dims, coords=data.coords, attrs=data.attrs.copy())
        new_data.name = data.name

        if include_seasonality:
            forcing = seasonality + new_data
        else:
            forcing = new_data

        forcing.attrs.update({
            'creation_date': datetime.now().isoformat(),
            'source': 'AISLENS forcing generator (spectral cutoff)',
            'realization_number': i,
        })

        suffix = "" if forcing.attrs.get('seasonality_included', True) else "_no_ssn"
        fname = output_dir / f"forcing_realization_{i}{suffix}.nc"
        forcing.to_netcdf(fname)
        logger.info(f"Saved realization {i} -> {fname}")


def main():
    parser = argparse.ArgumentParser(description='Generate forcings (spectral cutoff)')
    parser.add_argument('--n-realizations', '-n', type=int, default=config.N_REALIZATIONS)
    parser.add_argument('--load-existing-eof', action='store_true')
    parser.add_argument('--no-seasonality', action='store_true')
    parser.add_argument('--seasonality-file', type=str, default=None)
    parser.add_argument('--variability-file', type=str, default=None)
    parser.add_argument('--min-period-years', type=float, default=None)
    parser.add_argument('--max-period-years', type=float, default=None)
    parser.add_argument('--sampling-months', type=float, default=1.0,
                        help='Sampling interval in months (default=1.0 for monthly data)')
    parser.add_argument('--retain-outside', action='store_true', help='Keep spectral content outside passband')
    parser.add_argument('--no-rescale', action='store_true', help='Do not rescale randomized PCs to match band variance')
    parser.add_argument('--random-seed', type=int, default=None)
    parser.add_argument('--time-chunks', type=int, default=36)

    args = parser.parse_args()

    output_dir = Path(config.DIR_PROCESSED)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, 'generate_forcings_spectral_cutoff')

    include_seasonality = not args.no_seasonality

    seasonality, variability, data_norm, data_tmean, data_tstd = load_and_prepare_data(
        seasonality_path=args.seasonality_file, variability_path=args.variability_file, time_chunks=args.time_chunks
    )

    model, pcs, nmodes = perform_eof_analysis(data_norm, load_existing=args.load_existing_eof)

    generate_and_save_forcings(
        model, pcs, nmodes, data_tmean, data_tstd,
        seasonality, variability[config.SORRM_FLUX_VAR], args.n_realizations,
        include_seasonality=include_seasonality,
        min_period_years=args.min_period_years, max_period_years=args.max_period_years,
        sampling_months=args.sampling_months, retain_outside=args.retain_outside,
        rescale_variance=not args.no_rescale, random_state=args.random_seed
    )


if __name__ == '__main__':
    main()
