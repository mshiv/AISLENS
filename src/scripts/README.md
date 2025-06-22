# SCRIPTS

This directory contains example scripts that make use of the [`aislens`](https://github.com/mshiv/AISLENS/tree/refactor/src/aislens) package to prepare MALI forcing files and bash `nco` commands to manipulate MALI output files.

1. `prepare_data.py`: Loads appropriate subsets (user-defined) of the raw input dataset from SORRMv2.1 and satellite observations
2. `calculate_draft_dependence.py`: Preprocessses the satellite observations and saves the draft dependence parameterization
4. `generate_forcings.py`: Generates user-defined number of forcing realizations using the input variability dataset, extrapolates them across the entire ice sheet grid and adds them to the model's seasonality dataset (also extrapolated).
5. `regrid_to_mpas.py`: Prepares the above generated draft dependence parameters and forcing realizations for the MPAS grid and regrids them.