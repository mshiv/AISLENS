import geopandas as gpd
from shapely.geometry import mapping
from aislens.utils import fill_nan_with_nearest_neighbor_vectorized

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

def process_ice_shelf(ds_data, iceShelfNum, icems):
    """
    Process data for a specific ice shelf.

    Args:
        ds_data (xarray.Dataset): Input dataset for a specific time step.
        iceShelfNum (int): Ice shelf index.
        icems (GeoDataFrame): Ice shelf geometries.

    Returns:
        xarray.Dataset: Processed dataset for the ice shelf.
    """
    ice_shelf_mask = icems.loc[[iceShelfNum], 'geometry'].apply(mapping)
    ds = clip_data(ds_data, iceShelfNum, icems)
    
    # Vectorized filling of NaN values
    ds = ds.map(fill_nan_with_nearest_neighbor_vectorized, keep_attrs=True)
    
    ds = ds.rio.clip(ice_shelf_mask, icems.crs)
    return ds


def read_ice_shelves_mask(file_path, target_crs="EPSG:3031"):
    """
    Read the ice shelves mask from a GeoJSON or shapefile and reproject it to the target CRS.

    Args:
        file_path (str): Path to the GeoJSON or shapefile containing the ice shelves mask.
        target_crs (str): Target coordinate reference system (CRS). Defaults to "EPSG:3031".

    Returns:
        geopandas.GeoDataFrame: Ice shelves mask reprojected to the target CRS.
    """
    # Read the mask file
    ice_shelves_mask = gpd.read_file(file_path)

    # Reproject to the target CRS
    ice_shelves_mask = ice_shelves_mask.to_crs(target_crs)

    return ice_shelves_mask
