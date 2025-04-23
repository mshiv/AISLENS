def normalize(data):
    """
    Normalize the input data.

    Args:
        data (xarray.DataArray): Input data.

    Returns:
        tuple: (normalized data, mean, std)
    """
    data_mean = data.mean("time")
    data_std = data.std("time")
    data_demeaned = data - data_mean
    data_normalized = data_demeaned / data_std
    return data_normalized, data_mean, data_std

def unnormalize(data, mean, std):
    """
    Unnormalize the data.

    Args:
        data (xarray.DataArray): Normalized data.
        mean (xarray.DataArray): Mean used for normalization.
        std (xarray.DataArray): Standard deviation used for normalization.

    Returns:
        xarray.DataArray: Unnormalized data.
    """
    return (data * std) + mean