#!/bin/bash
#SBATCH --job-name=replace_nan
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=replace_nan_%A_%a.out
#SBATCH --error=replace_nan_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}
#SBATCH --array=0-10

# Array of file suffixes matching your pattern
suffixes=("OG" "0" "1" "2" "3" "4" "5" "6" "7" "8" "9")

# Select the current suffix based on the array task ID
current_suffix=${suffixes[$SLURM_ARRAY_TASK_ID]}

# Variable name to process
VAR_NAME="timeMonthly_avg_landIceFreshwaterFlux"

# Construct the filename
FILENAME="forcing_realization_${current_suffix}.nc"

# Load required modules
source ${HOME}/shared/MALI-RHEL9/load_mali_env.sh

# Change to the directory containing your NetCDF files
# Adjust this path to match your file location
cd ${AISLENS_DATA_DIR}/processed/forcings

echo "Processing file: $FILENAME"
echo "Variable: $VAR_NAME"
echo "Job array task ID: $SLURM_ARRAY_TASK_ID"
echo "----------------------------------------"

# Check if file exists
if [ ! -f "$FILENAME" ]; then
    echo "Error: File $FILENAME not found!"
    exit 1
fi

# Use ncap2 to replace NaN values with 0, overwriting the original file
echo "Replacing NaN values with 0 in $VAR_NAME..."
ncap2 -O -s "where($VAR_NAME != $VAR_NAME) $VAR_NAME = 0" "$FILENAME" "$FILENAME"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "✓ Successfully processed $FILENAME"
    echo "NaN values in $VAR_NAME have been replaced with 0"
else
    echo "✗ Error processing $FILENAME"
    exit 1
fi

echo "Job completed successfully!"
