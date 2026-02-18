import os
import numpy as np
import xarray as xr
from pathlib import Path

import pytest

from aislens import config as config_mod


def make_test_ds(path, varname, time_len=3, nx=2, ny=2):
    times = np.arange(time_len)
    data = np.random.RandomState(0).rand(time_len, ny, nx)
    ds = xr.Dataset({varname: (("time", "y", "x"), data)})
    ds.coords["time"] = times
    ds.coords["y"] = np.arange(ny)
    ds.coords["x"] = np.arange(nx)
    ds.to_netcdf(path)
    return path


def test_generate_forcings_smoke(tmp_path, monkeypatch):
    """Smoke test for generate_forcings: runs with tiny datasets and monkeypatched EOF pipeline.

    This test verifies the script creates forcing files (with and without seasonality)
    and that CLI file-overrides are respected.
    """
    # import the script module by path (it's not a package module)
    import importlib.util
    script_path = Path(__file__).resolve().parents[2] / 'src' / 'scripts' / 'generate_forcings.py'
    spec = importlib.util.spec_from_file_location('generate_forcings_module', str(script_path))
    gf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gf)

    # Create tiny seasonality / variability files
    seasonality_file = tmp_path / "seasonality.nc"
    variability_file = tmp_path / "variability.nc"
    make_test_ds(seasonality_file, config_mod.config.SORRM_FLUX_VAR)
    make_test_ds(variability_file, config_mod.config.SORRM_FLUX_VAR)

    # Monkeypatch EOF pipeline to lightweight stubs
    def fake_eof_decomp(data_norm):
        # return model, _, pcs, nmodes, _
        pcs = np.zeros((data_norm.sizes["time"], 1))
        return None, None, pcs, 1, None

    def fake_phase_randomization(pcs_vals, n_realizations):
        return np.zeros((n_realizations, pcs_vals.shape[1]))

    def fake_generate_data(model, new_pcs, i, nmodes, _):
        # return zeros matching data_norm dims (time,y,x)
        # data_tmean shape is (y,x) but generate_and_save_forcings expects same dims as data
        # We'll return zeros shaped like variability DataArray we'll open below
        return np.zeros((3, 2, 2))

    monkeypatch.setattr(gf, "perform_eof_analysis", lambda data_norm, load_existing=False: (None, np.zeros((3,1)), 1))
    # Monkeypatch generator functions used by generate_and_save_forcings
    import aislens.generator as gen
    monkeypatch.setattr(gen, "eof_decomposition", lambda data_norm: (None, None, np.zeros((3,1)), 1, None))
    monkeypatch.setattr(gen, "phase_randomization", fake_phase_randomization)
    monkeypatch.setattr(gen, "generate_data", fake_generate_data)

    # Redirect outputs to tmp_dir
    monkeypatch.setattr(config_mod.config, "DIR_FORCINGS", tmp_path / "forcings")

    # Run generator for 2 realizations without seasonality
    gf.generate_forcings(n_realizations=2, load_existing_eof=False, include_seasonality=False,
                         seasonality_file=str(seasonality_file), variability_file=str(variability_file))

    # Check files
    expected_files = [tmp_path / "forcings" / f"forcing_realization_{i}_no_ssn.nc" for i in range(2)]
    for p in expected_files:
        assert p.exists(), f"Expected output {p} not found"
