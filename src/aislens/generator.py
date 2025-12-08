import numpy as np
from xeofs.xarray import EOF

def eof_decomposition(data):
    """
    Performs EOF decomposition on the input data.

    Args:
        data (xarray.DataArray): Input data.

    Returns:
        tuple: (model, eofs, pcs, nmodes, varexpl)
    """
    model = EOF(data)
    model.solve()
    eofs = model.eofs()
    pcs = model.pcs()
    nmodes = model.n_modes
    varexpl = model.explained_variance_ratio()
    return model, eofs, pcs, nmodes, varexpl

def phase_randomization(pcs, n_realizations):
    """
    Performs phase randomization on the PCs of the EOF decomposition.

    Args:
        pcs (numpy.ndarray): Principal components (PCs).
        n_realizations (int): Number of realizations to generate.

    Returns:
        numpy.ndarray: Phase-randomized PCs.
    """
    t_length = pcs.shape[0]
    nmodes = pcs.shape[1]
    new_pcs = np.empty((n_realizations, t_length, nmodes))

    for i in range(n_realizations):
        for m in range(nmodes):
            fl = pcs[:, m]
            fl_fourier = np.fft.rfft(fl)
            random_phases = np.exp(np.random.uniform(0, 2 * np.pi, len(fl_fourier)) * 1.0j)
            fl_fourier_new = fl_fourier * random_phases
            new_pcs[i, :, m] = np.fft.irfft(fl_fourier_new)
        print(f"Calculated IFFT for realization {i}, all modes")
    return new_pcs


def phase_randomization_spectral_cutoff(pcs, n_realizations,
                                        min_period_years=None, max_period_years=None,
                                        sampling_months=1.0, retain_outside=False,
                                        rescale_variance=True, random_state=None,
                                        tiny_eps=1e-12):
    """
    Phase-randomize principal components with an optional spectral passband and variance rescaling.

    Parameters
    - pcs: ndarray (ntime, nmodes)
    - n_realizations: int
    - min_period_years / max_period_years: float or None, bounds on period (years) to retain
    - sampling_months: sampling interval in months (default 1.0 for monthly data)
    - retain_outside: if True, keep original spectrum outside passband; if False zero them
    - rescale_variance: if True, rescale each randomized realization to match original band variance
    - random_state: int or None
    - tiny_eps: small positive number to avoid divide-by-zero

    Returns
    - new_pcs: ndarray (n_realizations, ntime, nmodes)
    """
    rng = np.random.default_rng(random_state)
    pcs_arr = np.asarray(pcs)
    t_length, nmodes = pcs_arr.shape
    new_pcs = np.empty((n_realizations, t_length, nmodes), dtype=float)

    freqs = np.fft.rfftfreq(t_length, d=sampling_months)  # cycles per month
    with np.errstate(divide='ignore', invalid='ignore'):
        periods_months = np.where(freqs > 0, 1.0 / freqs, np.inf)
    periods_years = periods_months / 12.0

    # frequency mask (True => inside passband)
    mask = np.ones_like(freqs, dtype=bool)
    if min_period_years is not None:
        mask &= (periods_years >= min_period_years)
    if max_period_years is not None:
        mask &= (periods_years <= max_period_years)

    for m in range(nmodes):
        fl = pcs_arr[:, m]
        spec = np.fft.rfft(fl)
        amp = np.abs(spec)

        # original band-limited series used to compute target variance
        orig_band_spec = spec.copy()
        orig_band_spec[~mask] = 0.0
        orig_band_series = np.fft.irfft(orig_band_spec, n=t_length)
        orig_band_var = float(np.var(orig_band_series))

        # preserve DC
        dc = spec[0]

        for i in range(n_realizations):
            phases = np.exp(1.0j * rng.uniform(0, 2 * np.pi, size=spec.shape))
            new_spec = np.zeros_like(spec, dtype=complex)
            new_spec[mask] = amp[mask] * phases[mask]
            if retain_outside:
                new_spec[~mask] = spec[~mask]
            new_spec[0] = dc

            recon = np.fft.irfft(new_spec, n=t_length).real

            if rescale_variance:
                new_var = float(np.var(recon))
                if new_var > tiny_eps and orig_band_var > tiny_eps:
                    scale = (orig_band_var / new_var) ** 0.5
                    recon = recon * scale

            new_pcs[i, :, m] = recon

    return new_pcs

def generate_data(model, new_pcs, realization_idx, mode, mode_skip):
    """
    Generate synthetic data using the phase-randomized PCs.

    Args:
        model: EOF model.
        new_pcs (numpy.ndarray): Phase-randomized PCs.
        realization_idx (int): Index of the realization to generate.
        mode (int): Number of modes to use for reconstruction.
        mode_skip (int): Step size for modes.

    Returns:
        xarray.DataArray: Reconstructed data.
    """
    data_reconstr = model.reconstruct_randomized_X(new_pcs[realization_idx], slice(1, mode, mode_skip))
    return data_reconstr