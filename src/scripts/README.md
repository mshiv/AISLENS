# SCRIPTS

This directory contains example scripts that make use of the [`aislens`](https://github.com/mshiv/AISLENS/tree/refactor/src/aislens) package.

1. `prepare_data.py`: Loads appropriate subsets (user-defined) of the raw input dataset from SORRMv2.1 and satellite observations
2. `calculate_draft_dependence.py`: Preprocessses the satellite observations and saves the draft dependence parameterization
3. `extract_variability.py`: Preprocesses the model simulation data and saves the variability and seasonality components
4. `generate_forcings.py`: Generates user-defined number of forcing realizations using the input dataset, and extrapolates them across the entire ice sheet grid.
5. `regrid_to_mpas.py`: Regrids the above generated draft dependence parameters and forcing realizations onto the MPAS grid.
