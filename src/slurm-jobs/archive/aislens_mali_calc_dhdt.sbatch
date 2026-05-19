#!/bin/bash
#SBATCH  --job-name=nco-processing
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=8G
#SBATCH  --output=out.nco-processing.%j
#SBATCH  --error=err.nco-processing.%j
#SBATCH  --exclusive
#SBATCH  --time=2:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis # this loads nco

# Function to process a single directory
process_directory() {
    local dir="$1"
    echo "Processing directory: $dir"

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

    echo "Successfully processed directory: $dir"
    echo "Final output file: output_flux_all_timesteps_dhdt_${NYEARS}yr.nc"
    echo "----------------------------------------"

    return 0
}

# Main script execution
echo "Starting NCO processing job at $(date)"

# Define the directories to process
# Modify this array to include your actual directory paths
DIRECTORIES=(
    "${AISLENS_DATA_DIR}/MALI/draft-depen-paramset-tests/fb_A/output"
)

# Alternative: If all directories are in a parent directory, you can use:
# DIRECTORIES=($(find /path/to/parent/directory -maxdepth 1 -type d -name "pattern*"))

# Process each directory
success_count=0
failure_count=0

for dir in "${DIRECTORIES[@]}"; do
    if [ -d "$dir" ]; then
        if process_directory "$dir"; then
            ((success_count++))
        else
            ((failure_count++))
            echo "Failed to process directory: $dir"
        fi
    else
        echo "Directory does not exist: $dir"
        ((failure_count++))
    fi
done

echo "----------------------------------------"
echo "Job completed at $(date)"
echo "Successfully processed: $success_count directories"
echo "Failed to process: $failure_count directories"

if [ $failure_count -gt 0 ]; then
    exit 1
else
    exit 0
fi
