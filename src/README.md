# Workflow

1. `detrend.py`: Remove linear trend from the dataset.
2. `deseasonalize.py`: Remove annual climatologies from the detrended dataset.
3. `dedraft.py`: Fit a freshwater flux vs. draft function for different regions.
4. `merge_regions.py`: Merge the draft-flux function prediction defined for each region into a single datafile.
5. `make_clean_data.py`: Subtract the draft-flux prediction from the detrended, deseasonalized data to generate the detrended, deseasonalized and dedrafted dataset. This is our 'variability' dataset, input to the generator.
6. `plot_ts.py`: Make plots of spatially averaged time seies
7. `resample.py`: Make annual resolution from monthly
7. `normalize.py`: Normalization of clean dataset
8. `generator.py`: Generator
    `EOF_decompose.py`: Decompose normalized dataset into EOF modes, i.e., spatial and temporal patterns
    `phase_randomize.py`: Fourier phase randomization of temporal patterns
    `generate_realization.py`: Generate new realizations of dataset
    `unnormalize.py`: