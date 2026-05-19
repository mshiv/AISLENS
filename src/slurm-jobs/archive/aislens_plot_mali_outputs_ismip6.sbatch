#!/bin/bash
#SBATCH  --job-name=plot_vars_ctrl_ensemble
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=4G
#SBATCH  --output=plot_ctrl_ensemble_%A_%a.out
#SBATCH  --error=plot_ctrl_ensemble_%A_%a.err
#SBATCH  --exclusive
#SBATCH  --time=8:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}


# Define base directory and ensemble directories
#BASE_DIR="${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL"
#ENSEMBLE_DIRS=("CTRL-EM0" "CTRL-EM1" "CTRL-EM2" "CTRL-EM3" "CTRL-EM4" "CTRL-EM5" "CTRL-EM6" "CTRL-EM7" "CTRL-EM8" "CTRL-EM9")

# Get the current ensemble directory based on array task ID
#CURRENT_ENSEMBLE=${ENSEMBLE_DIRS[$SLURM_ARRAY_TASK_ID]}
#OUTPUT_DIR="${BASE_DIR}/${CURRENT_ENSEMBLE}/output/"

OUTPUT_DIR="${HOME}/scratch/ISMIP6-SSP585-4KM/output/"

#echo "Processing ensemble: $CURRENT_ENSEMBLE"
echo "Output directory: $OUTPUT_DIR"

# Load environment
#source /path/to/shared/Slurm/load_mali_env.sh
cd ${AISLENS_REPO}/src/MPAS-Tools/
module load anaconda3
conda activate mpas-analysis

# Run the plotting script
python3 plot_var_maps.py -r "$OUTPUT_DIR" -v thickness
python3 plot_var_maps.py -r "$OUTPUT_DIR" -v dHdt
python3 plot_var_maps.py -r "$OUTPUT_DIR" -v surfaceSpeed
python3 plot_var_maps.py -r "$OUTPUT_DIR" -v floatingBasalMassBalApplied 


#echo "Completed processing for $CURRENT_ENSEMBLE"


#python3 plot_var_maps.py -r ${HOME}/scratch/malitest/meanSatObsForcing_8km_noSMB/landice/ismip6_run/ismip6_ais_proj2300/hist_08/output/ -v thickness 
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6_obs23_IC/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v surfaceSpeed
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6_obs23_IC/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v sfcMassBalApplied
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6_obs23_IC/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v groundedBasalMassBalApplied
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6_obs23_IC/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v fluxAcrossGroundingLineOnCells
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6_obs23_IC/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v dHdt
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1101-obs23/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v thickness
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1101-obs23/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v floatingBasalMassBalApplied
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1101-obs23/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v surfaceSpeed
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1101-obs23/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v sfcMassBalApplied
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1101-obs23/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v groundedBasalMassBalApplied 
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1101-obs23/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v fluxAcrossGroundingLineOnCells
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1101-obs23/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v dHdt
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-relax/landice/ismip6_run/ismip6_ais_proj2300/expAE01_04/output/ -v thickness
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-relax/landice/ismip6_run/ismip6_ais_proj2300/expAE01_04/output/ -v floatingBasalMassBalApplied
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-relax/landice/ismip6_run/ismip6_ais_proj2300/expAE01_04/output/ -v surfaceSpeed
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-relax/landice/ismip6_run/ismip6_ais_proj2300/expAE01_04/output/ -v sfcMassBalApplied
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-relax/landice/ismip6_run/ismip6_ais_proj2300/expAE01_04/output/ -v groundedBasalMassBalApplied
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-relax/landice/ismip6_run/ismip6_ais_proj2300/expAE01_04/output/ -v fluxAcrossGroundingLineOnCells
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-relax/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output/ -v dHdt
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1108/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_04/output/ -v thickness
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1108/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_04/output/ -v floatingBasalMassBalApplied
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1108/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_04/output/ -v surfaceSpeed
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1108/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_04/output/ -v sfcMassBalApplied
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1108/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_04/output/ -v groundedBasalMassBalApplied
#python3 plot_var_maps.py -r ${HOME}/scratch/ismip6-1108/landice/ismip6_run/ismip6_ais_proj2300/ctrlAE_04/output/ -v fluxAcrossGroundingLineOnCells
#python3 plot_var_maps.py -r ${HOME}/scratch/malitest/meanSatObsForcing_8km_noSMB/landice/ismip6_run/ismip6_ais_proj2300/hist_08/output -v floatingBasalMassBalApplied
