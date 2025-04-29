from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here

@dataclass
class Config:
    # Define the root directory of the project
    BASE_DIR: Path = here()

    # Directory paths
    DIR_EXTERNAL: Path = BASE_DIR / "data/external"
    DIR_INTERIM: Path = BASE_DIR / "data/interim"
    DIR_PROCESSED: Path = BASE_DIR / "data/processed"
    DIR_TMP: Path = BASE_DIR / "data/tmp"
    DIR_ICESHELF_DEDRAFT: Path = DIR_INTERIM / "draft_dependence"
    DIR_DRAFT_DEPENDENCE: Path = DIR_PROCESSED / "draft_dependence"
    DIR_VARGENS: Path = DIR_PROCESSED / "vargen_realizations"
    DIR_FORCINGS: Path = DIR_PROCESSED / "forcings" # this is the sum of extrapolated seasonality and vargen_realizations
    DIR_MALI_FORCINGS: Path = DIR_PROCESSED / "mali_grid/forcings"
    DIR_MALI_DRAFT_DEPENDENCE: Path = DIR_PROCESSED / "mali_grid/draft_dependence"

    # File paths
    FILE_PAOLO23_SATOBS: Path = DIR_EXTERNAL / "ANT_G1920V01_IceShelfMeltDraft.nc"
    FILE_MPASO_MODEL: Path = DIR_EXTERNAL / "Regridded_SORRMv2.1.ISMF.FULL.nc"
    FILE_ICESHELFMASKS: Path = DIR_EXTERNAL / "iceShelves.geojson"

    # Processed file paths
    FILE_PAOLO23_SATOBS_PREPARED: Path = DIR_PROCESSED / "satellite_observations_prepared.nc"
    FILE_MPASO_MODEL_PREPARED: Path = DIR_PROCESSED / "model_simulation_prepared.nc"
    FILE_DRAFT_DEPENDENCE: Path = DIR_PROCESSED / "draft_dependence_params.nc"
    FILE_SEASONALITY: Path = DIR_PROCESSED / "sorrm_seasonality.nc"
    FILE_VARIABILITY: Path = DIR_PROCESSED / "sorrm_variability.nc"
    FILE_FORCING: Path = DIR_PROCESSED / "sorrm_forcing.nc"

    # Constants
    TIME_DIM: str = "Time"
    SATOBS_DRAFT_VAR: str = "draft"
    SATOBS_FLUX_VAR: str = "melt"
    SORRM_DRAFT_VAR: str = "timeMonthly"
    SORRM_FLUX_VAR: str = "freshwater_flux"
    START_YEAR: int = 2000
    END_YEAR: int = 2020
    N_REALIZATIONS: int = 10
    CRS_TARGET: str = "epsg:3031"

# Instantiate the configuration
config = Config()