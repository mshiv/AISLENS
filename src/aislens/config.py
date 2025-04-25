from pyprojroot import here

root = here(proj_files=[".git"])
notebooks_dir = root / "notebooks"
data_dir = root / "data"
timeseries_data_dir = data_dir / "timeseries"

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory paths
DIR_EXTERNAL = BASE_DIR / "data/external"
DIR_INTERIM = BASE_DIR / "data/interim"
DIR_PROCESSED = BASE_DIR / "data/processed"
DIR_OUTPUT = BASE_DIR / "data/output"
DIR_MPAS_TOOLS = BASE_DIR / "src/MPAS-Tools"

# File paths
FILE_SORRMv21 = DIR_EXTERNAL / "SORRMv2.1.ISMF/regridded_output/Regridded_SORRMv2.1.ISMF.FULL.nc"
FILE_SORRMv21_DETREND_DESEASONALIZE = DIR_INTERIM / "SORRMv21_300-900_DETREND_DESEASONALIZE.nc"
FILE_ICE_SHELVES_SHAPE = DIR_EXTERNAL / "iceShelves.geojson"
FILE_SEASONALITY = DIR_INTERIM / "basalmelt_seasonality_model_data.nc"

# Output paths for dedraft
DEDRAFT_PARAMS_DIR = DIR_INTERIM / "dedraft/iceShelfRegions"
DEDRAFT_REGRESS_DIR = DIR_INTERIM / "dedraft/iceShelfRegions"

# Other constants
CRS_TARGET = "EPSG:3031"  # Target CRS for geospatial data
