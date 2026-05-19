#!/bin/bash
#SBATCH  --job-name=mali-process-plot
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.mali-process-plot.%j
#SBATCH  --error=err.mali-process-plot.%j
#SBATCH  --time=8:00:00
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
    #"${AISLENS_DATA_DIR}/MALI/ISMIP6/SSP585/output"
    #"${AISLENS_DATA_DIR}/MALI/ISMIP6/SSP126/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM0/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM1/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM2/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM3/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM4/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM5/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM6/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM7/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM8/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM9/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM-NO-BMB-ADJ/output"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM3/output/TF-INITIAL-ONLY"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM4/output/TF-INITIAL-ONLY"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM5/output/TF-INITIAL-ONLY"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM1/output/TF-INITIAL-ONLY"
    #"${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM2/output/TF-INITIAL-ONLY"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CHANGEPOINT-CFL_0.8_cpu-small/output"
    "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM8-2PARAM-LATEST/output"
)

# Function to create figure directory structure
create_figure_dir() {
    local data_dir="$1"
    local figure_dir="${data_dir/$BASE_DATA_PATH/$BASE_FIGURES_PATH}"
    mkdir -p "$figure_dir"
    echo "$figure_dir"
}

# Function to process NCO operations for a single directory
process_nco_directory() {
    local dir="$1"
    echo "=== NCO Processing directory: $dir ==="

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

    # Get the penultimate (second to last) year as END_YEAR
    if [ ${#years[@]} -lt 2 ]; then
        echo "Need at least 2 files to determine END_YEAR in $dir"
        return 1
    fi

    END_YEAR=${years[-2]}  # Second to last element
    NYEARS=$((END_YEAR - 2000))

    echo "Found years: ${years[*]}"
    echo "END_YEAR: $END_YEAR"
    echo "NYEARS: $NYEARS"

    # Check if dhdt file already exists
    if [ -f "output_flux_all_timesteps_dhdt_${NYEARS}yr.nc" ]; then
        echo "dhdt file already exists, skipping NCO processing"
        return 0
    fi

    # Check if required files exist
    if [ ! -f "output_flux_all_timesteps_2000.nc" ]; then
        echo "Error: output_flux_all_timesteps_2000.nc not found in $dir"
        return 1
    fi

    if [ ! -f "output_flux_all_timesteps_${END_YEAR}.nc" ]; then
        echo "Error: output_flux_all_timesteps_${END_YEAR}.nc not found in $dir"
        return 1
    fi

    echo "Starting NCO processing..."

    # Step 1: Time average for year 2000
    echo "Step 1: Time averaging for year 2000..."
    ncwa -a Time -O output_flux_all_timesteps_2000.nc output_flux_all_timesteps_2000_tAvg.nc
    if [ $? -ne 0 ]; then
        echo "Error in ncwa for year 2000"
        return 1
    fi

    # Step 2: Time average for END_YEAR
    echo "Step 2: Time averaging for year $END_YEAR..."
    ncwa -a Time -O output_flux_all_timesteps_${END_YEAR}.nc output_flux_all_timesteps_${END_YEAR}_tAvg.nc
    if [ $? -ne 0 ]; then
        echo "Error in ncwa for year $END_YEAR"
        return 1
    fi

    # Step 3: Add Time dimension to year 2000 file
    echo "Step 3: Adding Time dimension to year 2000 file..."
    ncecat -O -u Time output_flux_all_timesteps_2000_tAvg.nc output_flux_all_timesteps_2000_tAvg.nc
    if [ $? -ne 0 ]; then
        echo "Error in ncecat for year 2000"
        return 1
    fi

    # Step 4: Add Time dimension to END_YEAR file
    echo "Step 4: Adding Time dimension to year $END_YEAR file..."
    ncecat -O -u Time output_flux_all_timesteps_${END_YEAR}_tAvg.nc output_flux_all_timesteps_${END_YEAR}_tAvg.nc
    if [ $? -ne 0 ]; then
        echo "Error in ncecat for year $END_YEAR"
        return 1
    fi

    # Step 5: Calculate thickness difference
    echo "Step 5: Calculating thickness difference..."
    ncdiff -O -v thickness output_flux_all_timesteps_${END_YEAR}_tAvg.nc output_flux_all_timesteps_2000_tAvg.nc output_flux_all_timesteps_thickness_${END_YEAR}-2000_diff.nc
    if [ $? -ne 0 ]; then
        echo "Error in ncdiff"
        return 1
    fi

    # Step 6: Calculate dhdt and set units to m/yr and long_name
    echo "Step 6: Calculating dhdt with units m/yr and long_name..."
    ncap2 -O -s "dhdt=thickness/${NYEARS}" -s 'dhdt@units="m/yr"' -s 'dhdt@long_name="rate of ice thickness change"' output_flux_all_timesteps_thickness_${END_YEAR}-2000_diff.nc output_flux_all_timesteps_dhdt_${NYEARS}yr.nc
    if [ $? -ne 0 ]; then
        echo "Error in ncap2"
        return 1
    fi

    echo "Successfully completed NCO processing for: $dir"
    echo "Final output file: output_flux_all_timesteps_dhdt_${NYEARS}yr.nc"

    return 0
}

# Function to create plots for a single directory
process_plotting_directory() {
    local data_dir="$1"
    local output_dir="$data_dir/"

    echo "=== Plotting Processing directory: $data_dir ==="

    # Get END_YEAR for this directory
    cd "$output_dir" || { echo "Failed to change to directory $output_dir"; return 1; }
    
    local years=($(ls output_flux_all_timesteps_2*.nc 2>/dev/null | \
                   sed 's/.*output_flux_all_timesteps_\([0-9]\+\)\.nc/\1/' | \
                   sort -n))

    if [ ${#years[@]} -lt 2 ]; then
        echo "Error: Could not determine END_YEAR for $data_dir"
        return 1
    fi

    local end_year=${years[-2]}
    local nyears=$((end_year - 2000))
    echo "END_YEAR: $end_year, NYEARS: $nyears"

    # Define file paths
    local dhdt_file="$output_dir/output_flux_all_timesteps_dhdt_${nyears}yr.nc"
    local mesh_2000="$output_dir/output_flux_all_timesteps_2000_tAvg.nc"
    local mesh_end="$output_dir/output_flux_all_timesteps_${end_year}_tAvg.nc"

    # Check if required files exist
    if [ ! -f "$dhdt_file" ]; then
        echo "Error: dhdt file not found: $dhdt_file"
        return 1
    fi

    if [ ! -f "$mesh_2000" ]; then
        echo "Error: 2000 mesh file not found: $mesh_2000"
        return 1
    fi

    if [ ! -f "$mesh_end" ]; then
        echo "Error: $end_year mesh file not found: $mesh_end"
        return 1
    fi

    # Create figure directory
    local figure_dir=$(create_figure_dir "$data_dir")
    local save_name="$figure_dir/dhdt_${nyears}yr"

    echo "Output figure: $save_name.png"

    # Change to the data directory before running Python script
    cd "$data_dir" || {
        echo "Error: Failed to change to directory $data_dir"
        return 1
    }

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
            echo "Renamed figure to remove _2000 suffix"
        fi
        echo "Successfully completed plotting for: $data_dir"
        return 0
    else
        echo "Error in plotting for: $data_dir (exit code: $exit_code)"
        return 1
    fi
}

# Function to process a single directory (NCO + Plotting)
process_directory() {
    local dir="$1"
    local data_dir=$(dirname "$dir")
    
    echo "========================================="
    echo "Processing: $dir"
    echo "Data directory: $data_dir"
    
    # Step 1: NCO Processing
    if process_nco_directory "$dir"; then
        echo "NCO processing completed successfully"
    else
        echo "NCO processing failed for $dir"
        return 1
    fi
    
    # Step 2: Plotting
    if process_plotting_directory "$data_dir"; then
        echo "Plotting completed successfully"
    else
        echo "Plotting failed for $data_dir"
        return 1
    fi
    
    echo "Successfully completed all processing for: $dir"
    return 0
}

# Main script execution
echo "Starting combined MALI processing and plotting job at $(date)"
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
nco_success_count=0
nco_failure_count=0
plot_success_count=0
plot_failure_count=0
total_success_count=0
total_failure_count=0

echo "Processing ${#DIRECTORIES[@]} directories..."
echo ""

for dir in "${DIRECTORIES[@]}"; do
    if [ -d "$dir" ]; then
        if process_directory "$dir"; then
            ((total_success_count++))
        else
            ((total_failure_count++))
            echo "Failed to process directory: $dir"
        fi
    else
        echo "Directory does not exist: $dir"
        ((total_failure_count++))
    fi
    echo ""
done

# Final summary
echo "========================================="
echo "Job completed at $(date)"
echo "Successfully processed: $total_success_count directories"
echo "Failed to process: $total_failure_count directories"
echo "Figures saved to: $BASE_FIGURES_PATH"

if [ $total_failure_count -gt 0 ]; then
    echo "Some directories failed to process. Check the error log for details."
    exit 1
else
    echo "All directories processed successfully!"
    exit 0
fi
