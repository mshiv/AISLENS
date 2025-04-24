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