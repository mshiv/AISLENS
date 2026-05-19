#!/bin/bash
#SBATCH  --job-name=mali-plotting
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.mali-plotting.%j
#SBATCH  --error=err.mali-plotting.%j
#SBATCH  --time=6:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis # this loads nco and python environment

# Configuration - Update these paths as needed
BASE_DATA_PATH="${AISLENS_DATA_DIR}/MALI"
BASE_FIGURES_PATH="${AISLENS_DATA_DIR}/figures/MALI"
PYTHON_SCRIPT_PATH="${AISLENS_REPO}/src/MPAS-Tools/plot_output_maps_masked.py"

# Function to find all directories containing NetCDF files
find_data_directories() {
    local base_path="$1"
    find "$base_path" -type d -name "output" | while read output_dir; do
        parent_dir=$(dirname "$output_dir")
        # Check if this directory contains the required files
        if ls "$output_dir"/output_flux_all_timesteps_dhdt_*yr.nc >/dev/null 2>&1; then
            echo "$parent_dir"
        fi
    done
}

# Function to get the END_YEAR from available files (same logic as NCO script)
get_end_year() {
    local output_dir="$1"
    local years=($(ls "$output_dir"/output_flux_all_timesteps_2*.nc 2>/dev/null | \
                   sed 's/.*output_flux_all_timesteps_\([0-9]\+\)\.nc/\1/' | \
                   sort -n))

    if [ ${#years[@]} -lt 2 ]; then
        echo ""
        return 1
    fi

    # Return the penultimate (second to last) year
    echo "${years[-2]}"
}

# Function to create figure directory structure
create_figure_dir() {
    local data_dir="$1"
    local figure_dir="${data_dir/$BASE_DATA_PATH/$BASE_FIGURES_PATH}"
    mkdir -p "$figure_dir"
    echo "$figure_dir"
}

# Function to process a single directory
process_directory() {
    local data_dir="$1"
    local output_dir="$data_dir/output"

    echo "Processing directory: $data_dir"

    # Get END_YEAR for this directory
    local end_year=$(get_end_year "$output_dir")
    if [ -z "$end_year" ]; then
        echo "Error: Could not determine END_YEAR for $data_dir"
        return 1
    fi

    local nyears=$((end_year - 2000))
    echo "  END_YEAR: $end_year, NYEARS: $nyears"

    # Define file paths
    local dhdt_file="$output_dir/output_flux_all_timesteps_dhdt_${nyears}yr.nc"
    local mesh_2000="$output_dir/output_flux_all_timesteps_2000_tAvg.nc"
    local mesh_end="$output_dir/output_flux_all_timesteps_${end_year}_tAvg.nc"

    # Check if required files exist
    if [ ! -f "$dhdt_file" ]; then
        echo "  Error: dhdt file not found: $dhdt_file"
        return 1
    fi

    if [ ! -f "$mesh_2000" ]; then
        echo "  Error: 2000 mesh file not found: $mesh_2000"
        return 1
    fi

    if [ ! -f "$mesh_end" ]; then
        echo "  Error: $end_year mesh file not found: $mesh_end"
        return 1
    fi

    # Create figure directory
    local figure_dir=$(create_figure_dir "$data_dir")
    # Modified save_name to remove _2000 suffix - just use dhdt_${nyears}yr
    local save_name="$figure_dir/dhdt_${nyears}yr"

    echo "  Output figure: $save_name.png"

    # Change to the data directory before running Python script
    cd "$data_dir"

    # Run the Python plotting script
    python3 "$PYTHON_SCRIPT_PATH" \
        -v dhdt \
        -r "$dhdt_file" \
        -m "$mesh_2000" \
        -m2 "$mesh_end" \
        -s "$save_name"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        # Rename file to remove _2000 suffix if it exists
        if [ -f "${save_name}_2000.png" ]; then
            mv "${save_name}_2000.png" "${save_name}.png"
            echo "  Renamed figure to remove _2000 suffix"
        fi
        echo "  Successfully processed: $data_dir"
        return 0
    else
        echo "  Error processing: $data_dir (exit code: $exit_code)"
        return 1
    fi
}

# Main execution
echo "Starting MALI plotting job at $(date)"
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

# Find all directories with processed data
echo "Searching for directories with dhdt files..."
directories=($(find_data_directories "$BASE_DATA_PATH"))

if [ ${#directories[@]} -eq 0 ]; then
    echo "No directories with dhdt files found under $BASE_DATA_PATH"
    exit 1
fi

echo "Found ${#directories[@]} directories to process:"
printf '%s\n' "${directories[@]}"
echo ""

# Process each directory
success_count=0
failure_count=0

for data_dir in "${directories[@]}"; do
    echo "========================================="
    if process_directory "$data_dir"; then
        ((success_count++))
    else
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
