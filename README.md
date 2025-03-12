# The Antarctic Ice Sheet Large Ensemble (AISLENS)

This is a wip codebase for the [AISLENS project](https://iceclimate.eas.gatech.edu/research/antarctic-ice-sheet-large-ensemble-project-aislens/).

More details on the ensemble generator used in this work can be found in the [aislens_emulation](https://github.com/mshiv/aislens_emulation) repo.

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

