#!/bin/bash
#SBATCH  --job-name=mali-thickness-plot
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.mali-thickness-plot.%j
#SBATCH  --error=err.mali-thickness-plot.%j
#SBATCH  --time=4:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis # this loads nco and python environment

# Configuration - Update these paths as needed
BASE_DATA_PATH="${AISLENS_DATA_DIR}/MALI"
BASE_FIGURES_PATH="${AISLENS_DATA_DIR}/figures/MALI"
PYTHON_SCRIPT_PATH="${AISLENS_REPO}/src/MPAS-Tools/plot_output_maps_masked.py"

# Define the directories to process
DIRECTORIES=(
    "${AISLENS_DATA_DIR}/MALI/ISMIP6/SSP585/output"
    "${AISLENS_DATA_DIR}/MALI/ISMIP6/SSP126/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM0/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM1/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM2/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM3/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM4/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM5/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM6/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM7/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM8/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM9/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM-NO-BMB-ADJ/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM3/output/TF-INITIAL-ONLY"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM4/output/TF-INITIAL-ONLY"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM5/output/TF-INITIAL-ONLY"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM1/output/TF-INITIAL-ONLY"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM2/output/TF-INITIAL-ONLY"
)

# Function to create figure directory structure
create_figure_dir() {
    local data_dir="$1"
    local figure_dir="${data_dir/$BASE_DATA_PATH/$BASE_FIGURES_PATH}"
    mkdir -p "$figure_dir"
    echo "$figure_dir"
}

# Function to plot thickness for a single directory
process_thickness_plotting() {
    local dir="$1"
    echo "=== Processing directory: $dir ==="

    # Change to the directory
    cd "$dir" || { echo "Failed to change to directory $dir"; return 1; }

    # Find all netcdf files matching the pattern and extract years
    years=($(ls output_flux_all_timesteps_2*.nc 2>/dev/null | \
             sed 's/output_flux_all_timesteps_\([0-9]\+\)\.nc/\1/' | \
             sort -n))

    # Check if we found any files
    if [ ${#years[@]} -eq 0 ]; then
        echo "No matching files found in $dir"
        return 1
    fi

    # Get the last (most recent) year
    LAST_YEAR=${years[-1]}
    LAST_FILE="output_flux_all_timesteps_${LAST_YEAR}.nc"

    echo "Found years: ${years[*]}"
    echo "Last year: $LAST_YEAR"
    echo "Last file: $LAST_FILE"

    # Check if the last file exists
    if [ ! -f "$LAST_FILE" ]; then
        echo "Error: Last file $LAST_FILE not found in $dir"
        return 1
    fi

    # Get the parent directory for figure saving
    local data_parent_dir=$(dirname "$dir")
    
    # Create figure directory
    local figure_dir=$(create_figure_dir "$data_parent_dir")
    local save_name="$figure_dir/thickness_${LAST_YEAR}"

    echo "Output figure: $save_name.png"

    # Change to the parent data directory before running Python script
    cd "$data_parent_dir" || {
        echo "Error: Failed to change to directory $data_parent_dir"
        return 1
    }

    # Run the Python plotting script for thickness
    python3 "$PYTHON_SCRIPT_PATH" \
        -v thickness \
        -r "$LAST_FILE" \
        -m "$LAST_FILE" \
        -s "$save_name"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        # Handle potential _2000 suffix in filename (if the script adds it)
        if [ -f "${save_name}_${LAST_YEAR}.png" ]; then
            mv "${save_name}_${LAST_YEAR}.png" "${save_name}.png"
            echo "Renamed figure to remove year suffix"
        elif [ -f "${save_name}_2000.png" ]; then
            mv "${save_name}_2000.png" "${save_name}.png"
            echo "Renamed figure to remove _2000 suffix"
        fi
        echo "Successfully completed thickness plotting for: $dir"
        return 0
    else
        echo "Error in thickness plotting for: $dir (exit code: $exit_code)"
        return 1
    fi
}

# Main script execution
echo "Starting MALI thickness plotting job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Base data path: $BASE_DATA_PATH"
echo "Base figures path: $BASE_FIGURES_PATH"
echo "Python script: $PYTHON_SCRIPT_PATH"
echo ""

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT_PATH' not found"
    echo "Please update PYTHON_SCRIPT_PATH in the script configuration"
    exit 1
fi

# Check if base data path exists
if [ ! -d "$BASE_DATA_PATH" ]; then
    echo "Error: Base data path '$BASE_DATA_PATH' does not exist"
    exit 1
fi

# Process each directory
success_count=0
failure_count=0

echo "Processing ${#DIRECTORIES[@]} directories for thickness plotting..."
echo ""

for dir in "${DIRECTORIES[@]}"; do
    if [ -d "$dir" ]; then
        if process_thickness_plotting "$dir"; then
            ((success_count++))
        else
            ((failure_count++))
            echo "Failed to process directory: $dir"
        fi
    else
        echo "Directory does not exist: $dir"
        ((failure_count++))
    fi
    echo ""
done

# Final summary
echo "========================================="
echo "Job completed at $(date)"
echo "Successfully processed: $success_count directories"
echo "Failed to process: $failure_count directories"
echo "Figures saved to: $BASE_FIGURES_PATH"

if [ $failure_count -gt 0 ]; then
    echo "Some directories failed to process. Check the error log for details."
    exit 1
else
    echo "All directories processed successfully!"
    exit 0
fi
