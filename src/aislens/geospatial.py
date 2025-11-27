import geopandas as gpd
from shapely.geometry import mapping
from aislens.utils import fill_nan_with_nearest_neighbor_vectorized
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

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
    try:
        clipped_data = total_data.rio.clip(icems.loc[[basin], 'geometry'].apply(mapping), icems.crs)
        clipped_data = clipped_data.drop("month", errors="ignore")
        return clipped_data
    except Exception as e:
        # rioxarray raises NoDataInBounds when the polygon doesn't overlap the raster.
        # Catch that and return a NaN-filled array/dataset with the same coords/dims so
        # callers can continue. Log the basin for diagnostics.
        try:
            from rioxarray.exceptions import NoDataInBounds
        except Exception:
            NoDataInBounds = type('NoDataInBounds', (Exception,), {})

        if isinstance(e, NoDataInBounds) or 'No data found in bounds' in str(e):
            logger.warning('NoDataInBounds in clip_data for basin %s: returning NaN placeholder', basin)
            # Create NaN-filled object matching input type
            if hasattr(total_data, 'data_vars'):
                ds_nan = total_data.copy(deep=True)
                for var in ds_nan.data_vars:
                    ds_nan[var] = xr.full_like(ds_nan[var], np.nan)
                return ds_nan
            else:
                # DataArray
                return xr.full_like(total_data, np.nan)
        # otherwise re-raise
        raise

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
