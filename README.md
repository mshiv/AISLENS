# The Antarctic Ice Sheet Large Ensemble (AISLENS)

This is a wip codebase for the [AISLENS project](https://iceclimate.eas.gatech.edu/research/antarctic-ice-sheet-large-ensemble-project-aislens/).

More details on the ensemble generator used in this work can be found in the [aislens_emulation](https://github.com/mshiv/aislens_emulation) repo.


## DIRECTORY STRUCTURE

/aislens/
├── main.py                     # Main script to execute the workflow
├── README.md                   # Documentation of the workflow
├── data/                       # Directory for input and output data
│   ├── external/               # Raw input datasets (e.g., satellite observations, model outputs)
│   ├── interim/                # Intermediate processed datasets
│   ├── processed/              # Final processed datasets
|   ├── tmp/                    # Temporary files, treat as a scratch dir.
├── modules/                    # Core modules for reusable functionality
│   ├── __init__.py             # Makes this directory a package
│   ├── data_preprocess.py      # Preprocessing functions (detrend, deseasonalize, etc.)
│   ├── draft_dependence.py     # Functions for draft dependence calculations
│   ├── generator.py            # Statistical generator for variability realizations
│   ├── regrid.py               # Functions for regridding and renaming dimensions
│   ├── io.py                   # Input/output utility functions
│   ├── masks.py                # Functions for clipping and masking data
│   ├── interpolation.py        # Functions for filling NaN values
│   ├── extrapolation.py        # Functions for extrapolating variability
│   ├── seasonality.py          # Functions for extracting and saving seasonal signals
│   ├── normalize.py            # Functions for normalizing and unnormalizing datasets
│   └── utils.py                # Helper functions (e.g., directory setup, logging)
├── tests/                      # Unit tests for all modules
│   ├── test_data_preprocess.py
│   ├── test_draft_dependence.py
│   ├── test_generator.py
│   ├── test_regrid.py
│   ├── test_io.py
│   └── ...
├── MPAS-Tools/                 # Tools for regridding to MPAS-Land Ice grid
│   └── interpolate_to_mpasli_grid.py
└── scripts/                    # SLURM job scripts for HPC workflows
    ├── preprocess.sbatch
    ├── generate_forcings.sbatch
    └── regrid_to_mpas.sbatch







## Workflow

This assumes you have access to the datasets to prepare initial condition and forcing files: i.e., satellite observations of AIS basal melt rates (`basalmelt_satobs_data.nc`) and a high resolution ocean model output of melt rates (`basalmelt_model_data.nc`) in the `data/external` directory.

- `basalmelt_satobs_data.nc`: Freshwater flux at the ice ocean interface, ice draft
- `basalmelt_model_data.nc` : 1000 year long simulation of freshwater flux across the ice sheet (SORRMv2.1)

Refer to the [zenodo link](zenodo data link) for access to these files.

We extract a mean current basal melt rate field from the satellite observations, which will be used to derive the draft dependent parameterization of basal melt. We also extract seasonality and variability signals from the model simulation data. The former will serve as an initial condition for the ice sheet model, while the latter is the forcing. Note that the model is forced directly by freshwater flux, not thermal forcing.

* Load `basalmelt_satobs_data.nc`
    * Obtain a time mean of `basalmelt_satobs_data.freshwater_flux`.
    * Dedraft the dataset with `basalmelt_satobs_data.draft`: this makes use of the data_preprocessing and draft_dependence modules:
        * Dedrafting is done on an ice-shelf by ice-shelf basis, i.e., the dataset is clipped to each ice shelf, and draft dependence linear parameters are calculated for each ice shelf.
        * Next, this value of the draft dependence parameters is defined at each grid point within each ice shelf, so that we have a `draftDependentBasalMeltAlpha0` and `draftDependentBasalMeltAlpha1` with the same spatial dimensions as the original data.
    * Save `draftDependenceBasalMeltAlpha0` and `draftDependenceBasalMeltAlpha1` in `data/processed/draft_dependence_params.nc`
* Load `basalmelt_model_data.nc`: this includes a `freshwaterFlux` and `draft` variable along `(Time, x, y)` dimensions.
    * Detrend `basalmelt_model_data` to get `detrended_basalmelt_model_data`
    * Deseasonalize `detrended_basalmelt_model_data` to get `deseasonalized_detrended_basalmelt_model_data`.
    * Save the seasonal signal as `basalmelt_seasonality_model_data`. Ensure that this seasonal signal dataset does not have a mean value, i.e., that it is centered around `0`, since the mean is provided by the draft dependence parameterization.
    * Dedraft `deseasonalized_detrended_basalmelt_model_data` to get `dedrafted_deseasonalized_detrended_basalmelt_model_data`.
    * This is the variability signal, i.e., `dedrafted_deseasonalized_detrended_basalmelt_model_data` = `basalmelt_variability_model_data`
* Run `basalmelt_variability_model_data` through the generator for a specified number (user provided, "`n_gen`") of generated output datasets. This involves the following steps:
    * Normalize `basalmelt_variability_model_data` to obtain `normalized_basalmelt_variability_model_data`
    * Perform an EOF decomposition of `normalized_basalmelt_variability_model_data`.
    * Perform a Fourier phase randomization of the principal components in the above decomposition `n_gen` number of times to obtain that many realizations.
    * Generate `n_gen` variability realizations by reconstructing the dataset with the randomized principal components.
    * We now have `basalmelt_variability_model_data`, which is the original model simulation data, and `n_gen` realizations of `basalmelt_variability_generated_data`. 
* Create the forcing files for the ice sheet model and save them in appropriate subdirectories (based on the experiment/ensemble) of `data/processed/forcings/`.
    * Do this by adding the seasonal signal saved earlier (`basalmelt_seasonality_model_data`) to each of the `basalmelt_variability` datasets to get different forcing realizations of `basalmelt_forcing_anomaly`.
* Regrid the `basalmelt_forcing_anomaly` files and the `draft_dependence_params` file to the MPAS-Land Ice grid. 
    * Rename the `x,y,Time` dimensions in these files to the required names (usually `x1,y1, ...` ). 
    * Make use of `MPAS-Tools/interpolate_to_mpasli_grid.py` to perform the regridding.



------------
/src:

`generator.py`: Statistical generator of ocean variability. Takes MPAS-Ocean dataset as input. //
`forcing_extrapolate.py`: Extrapolate forcing file (i.e., ocean variability dataset provided by `generator.py` to the entire ice sheet domain. //
`data_preprocess.py`: Preprocess the input dataset to extract variability signal. //
`data_reshape.py`: //
`data_seasonality.py`: // 
`interpolate_to_mpasli_grid_mod.py`: //
`rename_dims_to_CISM.py`: //
`rename_time_dimension.py`: //

/scripts/slurm-nco: SLURM job scripts that make use of NCO commands to modify /data

`aislens_preprocess_ncks_misc.sbatch`: //
`aislens_preprocess_forcing_components.sbatch`: //
`prep_MALI_forcings.sbatch`: //
`prep_MALI_forcings_test.sbatch`: // 
`rm_nan_forcings.sbatch`: //

Workflow with job arrays on slurm:

`interp_to_mpas_jobarray.sbatch`: //
`add_xtime_jobarray.sbatch`: //
`setup-aislens-ctrl-mini.sbatch`: // 

