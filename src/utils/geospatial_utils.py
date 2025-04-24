def find_ice_shelf_index(ice_shelf_name, icems):
    """
    Find the index of an ice shelf by name.

    Args:
        ice_shelf_name (str): Name of the ice shelf.
        icems (GeoDataFrame): Ice shelf geometries.

    Returns:
        int: Index of the ice shelf.
    """
    return icems[icems['name'] == ice_shelf_name].index[0]

def clip_data(total_data, basin, icems):
    """
    Clip the map to a specific domain.

    Args:
        total_data (xarray.DataArray): Input data.
        basin (str): Basin name.
        icems (GeoDataFrame): Ice shelf geometry.

    Returns:
        xarray.DataArray: Clipped data.
    """
    clipped_data = total_data.rio.clip(icems.loc[[basin], 'geometry'].apply(mapping), icems.crs)
    clipped_data = clipped_data.drop("month", errors="ignore")
    return clipped_data

def delaunay_interp_weights(xy, uv, d=2):
    """
    Compute Delaunay interpolation weights.
    Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    
    Reference: Trevor Hillebrand
    Args:
        xy (array): Input x, y coordinates.
        uv (array): Output (MPAS-LI) x, y coordinates.
        d (int): Dimensionality (default is 2).

    Returns:
        tuple: (vertices, weights, outside_indices, tree)
    """
    tri = scipy.spatial.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    weights = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    weights = np.hstack((weights, 1 - weights.sum(axis=1, keepdims=True)))
    tree = scipy.spatial.cKDTree(xy)
    return vertices, weights, tree

def nn_interp_weights(xy, uv, d=2):
    """
    Compute nearest-neighbor interpolation weights.

    Args:
        xy (array): Input x, y coordinates.
        uv (array): Output (MPAS-LI) x, y coordinates.
        d (int): Dimensionality (default is 2).

    Returns:
        array: Indices of nearest neighbors.
    """
    tree = scipy.spatial.cKDTree(xy)
    _, idx = tree.query(uv, k=1)
    return idx