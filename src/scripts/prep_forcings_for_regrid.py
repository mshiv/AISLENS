#!/usr/bin/env python3

from aislens.utils import rename_dims_and_fillna, process_directory
from aislens.config import config
import cftime

def regrid_to_mali():
    # Regrid draft dependence parameters
    #rename_dims_and_fillna(config.FILE_DRAFT_DEPENDENCE, dims_to_rename={"x": "x1", "y": "y1"}, fill_value=0)
    rename_dims_and_fillna(config.FILE_FORCING_OG, dims_to_rename={"x": "x1", "y": "y1"}, fill_value=0)
    # Regrid forcing realizations
    #process_directory(config.DIR_FORCINGS, dims_to_rename={"x": "x1", "y": "y1", "time": "Time"}, fill_value=0)
    # process_directory(config.DIR_FORCINGS, fill_value=0)

if __name__ == "__main__":
    regrid_to_mali()