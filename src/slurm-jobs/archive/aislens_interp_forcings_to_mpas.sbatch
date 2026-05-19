#!/bin/bash
#SBATCH --job-name=interp_mpasli
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=8G
#SBATCH --output=interp_mpasli_%A_%a.out
#SBATCH --error=interp_mpasli_%A_%a.err
#SBATCH --exclusive
#SBATCH --time=15:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}
#SBATCH --array=0-9

# Array of file suffixes matching your pattern
suffixes=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")

# Select the current suffix based on the array task ID
current_suffix=${suffixes[$SLURM_ARRAY_TASK_ID]}

# Source and destination paths
SOURCE_FILE="${HOME}/data/projects/MALI_projects/ISMIP6-2300/initial_conditions/AIS_4to20km_20230105/AIS_4to20km_r01_20220907_Forcing_TMP.nc"
DEST_DIR="${AISLENS_DATA_DIR}/processed/vargen_realizations/"
DEST_FILE="${DEST_DIR}AIS_4to20km_r01_20220907_AISLENS-Forcing_${current_suffix}.nc"

# Copy source file to destination with new name
# Uncomment this command when running the script for a new ensemble/directory the first time.
# cp "$SOURCE_FILE" "$DEST_FILE"

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

cd ${AISLENS_REPO}/src/MPAS-Tools

python interpolate_to_mpasli_grid_mod.py \
    -s ${AISLENS_DATA_DIR}/processed/forcings/forcing_realization_"$current_suffix"_no_ssn.nc \
    -d "$DEST_FILE" \
    --timestart 0 --timeend 3599 \
    -v floatingBasalMassBalAdjustment
