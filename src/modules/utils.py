from optparse import OptionParser

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        options: Parsed options.
    """
    parser = OptionParser(description="Statistical generator for synthetic data")
    parser.add_option("-f", dest="fileInName", help="Input normalized data filename", default="SORRMv21.ISMF.FULL.nc", metavar="FILENAME")
    parser.add_option("-n", dest="nRealizations", help="Number of ensemble members to be generated", default=5, metavar="N_REALIZATIONS")
    options, _ = parser.parse_args()
    return options

def write_crs(ds, crs='epsg:3031'):
    """
    Write CRS information to an xarray dataset.

    Args:
        ds (xarray.Dataset): Dataset to update.
        crs (str): Coordinate reference system.

    Returns:
        xarray.Dataset: Updated dataset.
    """
    ds.rio.write_crs(crs, inplace=True)
    return ds