Outline:

/src/
├── main.py                         # Main, CLI interface
├── prepare_data.py                 # Preprocess satellite and model data
├── extract_variability.py          # Extract variability signals
├── generate_forcings.py            # Generate forcing files
├── calculate_draft_dependence.py   # Derive draft dependence parameters
├── regrid_to_mpas.py               # Regrid files to MPAS-Land Ice grid, for running MALI.
├── modules/                        # Project-specific functions
│   ├── data_preprocess.py          # detrend_dim; deseasonalize; subset_dataset
                                    # extract_seasonality; save_seasonality
│   ├── draft_dependence.py         # calc_draft_params; dedraft
│   ├── generator.py                # eof_decomposition; phase_randomization; generate_realizations
│   ├── regrid.py                   # rename_dimensions; regrid_to_mpas
│   └── ...
├── utils/                      # Generic utility functions
│   ├── logging.py
│   ├── file_utils.py           # load_dataset; save_dataset
│   └── math_utils.py           # fill_nan_with_nearest_neighbor; normalize; unnormalize
│   └── geospatial_utils.py     # mask; ...