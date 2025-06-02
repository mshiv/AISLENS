from dataclasses import dataclass, field
from pathlib import Path
from pyprojroot import here
import os

@dataclass
class Config:
    # Project root (code location, for source files, scripts, etc.)
    BASE_DIR: Path = here()

    # Data root, configurable by environment variable (if set, else defaults back to project location of BASE_DIR)
    # DATA_ROOT: Path = Path(
    #    os.environ.get('AISLENS_DATA_DIR', os.path.expandvars('$HOME/scratch/AISLENS'))
    #)
    DATA_ROOT: Path = Path(os.environ.get('AISLENS_DATA_DIR', BASE_DIR))

    # Directory paths (all data is stored under DATA_ROOT/data). 
    # Replace DATA_ROOT with BASE_DIR if you want to use the project code location as the data root.
    DIR_RAW: Path = DATA_ROOT / "data/raw"
    DIR_EXTERNAL: Path = DATA_ROOT / "data/external"
    DIR_INTERIM: Path = DATA_ROOT / "data/interim"
    DIR_PROCESSED: Path = DATA_ROOT / "data/processed"
    DIR_TMP: Path = DATA_ROOT / "data/tmp"

    # Alternative directory paths (comment out the above DIR_* lines if you use these)
    # Uncomment the following lines if you want to use the project code location as the data root.
    #DIR_RAW: Path = BASE_DIR / "data/raw"
    #DIR_EXTERNAL: Path = BASE_DIR / "data/external"
    #DIR_INTERIM: Path = BASE_DIR / "data/interim"
    #DIR_PROCESSED: Path = BASE_DIR / "data/processed"
    #DIR_TMP: Path = BASE_DIR / "data/tmp"

    DIR_ICESHELF_DEDRAFT_SATOBS: Path = DIR_INTERIM / "draft_dependence/satobs" # Draft dependence parameters calculated for individual ice shelves from satellite observations
    DIR_ICESHELF_DEDRAFT_MODEL: Path = DIR_INTERIM / "draft_dependence/model" # Draft dependence parameters calculated for individual ice shelves from model simulations
    DIR_DRAFT_DEPENDENCE: Path = DIR_PROCESSED / "draft_dependence" # Draft dependence parameters that are combined for entire ice sheet
    DIR_VARGENS: Path = DIR_PROCESSED / "vargen_realizations" # Generated realizations of variability
    
    DIR_FORCINGS: Path = DIR_PROCESSED / "forcings" # Location of the sum of extrapolated seasonality and vargen_realizations and trends
    DIR_MALI_FORCINGS: Path = DIR_PROCESSED / "mali_grid/forcings" # Location of the final forcing files converted to MALI grid
    DIR_MALI_DRAFT_DEPENDENCE: Path = DIR_PROCESSED / "mali_grid/draft_dependence" # Location of the final draft dependence fields on the MALI grid

    # File paths
    FILE_PAOLO23_SATOBS: Path = DIR_EXTERNAL / "ANT_G1920V01_IceShelfMeltDraft_Time.nc"
    FILE_MPASO_MODEL: Path = DIR_EXTERNAL / "Regridded_SORRMv2.1.ISMF.FULL.nc"
    FILE_ICESHELFMASKS: Path = DIR_EXTERNAL / "iceShelves.geojson"

    # Processed file paths
    FILE_PAOLO23_SATOBS_PREPARED: Path = DIR_PROCESSED / "satellite_observations_prepared.nc"
    FILE_MPASO_MODEL_PREPARED: Path = DIR_PROCESSED / "model_simulation_prepared.nc"
    FILE_DRAFT_DEPENDENCE: Path = DIR_PROCESSED / "draft_dependence_params.nc"
    FILE_SEASONALITY: Path = DIR_PROCESSED / "sorrm_seasonality.nc"
    FILE_SEASONALITY_EXTRAPL: Path = DIR_PROCESSED / "sorrm_seasonality_extrapolated.nc"
    FILE_VARIABILITY: Path = DIR_PROCESSED / "sorrm_variability.nc"
    FILE_VARIABILITY_EXTRAPL: Path = DIR_PROCESSED / "sorrm_variability_extrapolated.nc"
    FILE_FORCING: Path = DIR_PROCESSED / "sorrm_forcing.nc"

    # Constants
    TIME_DIM: str = "Time"
    SATOBS_DRAFT_VAR: str = "draft"
    SATOBS_FLUX_VAR: str = "melt"
    SORRM_DRAFT_VAR: str = "timeMonthly_avg_ssh"
    SORRM_FLUX_VAR: str = "timeMonthly_avg_landIceFreshwaterFlux"
    ICE_SHELF_REGIONS: range = range(33,133)
    SORRM_START_YEAR: int = 300
    SORRM_END_YEAR: int = 900
    N_REALIZATIONS: int = 10
    CRS_TARGET: str = "epsg:3031"
    DATA_ATTRS: dict = field(default_factory=lambda: {
        "draftDepenBasalMeltAlpha0": {
            "long_name": "Basal melt rate draft dependency coefficient (alpha0 or intercept)",
            "units": "m/yr",
        },
        "draftDepenBasalMeltAlpha1": {
            "long_name": "Basal melt rate draft dependency coefficient (alpha1 or slope)",
            "units": "m/yr/m",
        },

    })

# Instantiate the configuration
config = Config()