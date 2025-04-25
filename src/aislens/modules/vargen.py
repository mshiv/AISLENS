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