import geopandas as gpd
from shapely.geometry import mapping

def clip_data(total_data, basin, icems):
    """
    Clip the data to a specific ice shelf.

    Args:
        total_data (xarray.DataArray): Input data.
        basin (int): Ice shelf index.
        icems (GeoDataFrame): Ice shelf geometries.

    Returns:
        xarray.DataArray: Clipped data.
    """
    clipped_data = total_data.rio.clip(icems.loc[[basin], 'geometry'].apply(mapping), icems.crs)
    clipped_data = clipped_data.drop("month")
    return clipped_data

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