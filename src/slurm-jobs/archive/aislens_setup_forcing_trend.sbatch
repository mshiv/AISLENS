#!/bin/bash
#SBATCH --job-name=ssp585-forcing
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=8G
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --time=05:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis

# NCO processing script for floatingBasalMassBalApplied variable
# Description: Extract time-averaged floatingBasalMassBalApplied from yearly files
#              and create concatenated file with 12 repetitions per year

# Set base directory and variable name
#BASE_DIR="${HOME}/scratch/ISMIP6/landice/ismip6_run/ismip6_ais_proj2300/hist_04/output"
BASE_DIR="${HOME}/scratch/ISMIP6-SSP585-4KM/output"
SUBFOLDER="floatingBMB"
VAR_NAME="floatingBasalMassBalApplied"
FINAL_FILE="floatingBasalMassBalApplied_SSP585_Trend_2000-2270.nc"

# Create subfolder if it doesn't exist
mkdir -p "${BASE_DIR}/${SUBFOLDER}"

echo "Starting processing of floatingBasalMassBalApplied files..."

# Step 1: Extract time-averaged variable from each yearly file
echo "Step 1: Extracting time-averaged ${VAR_NAME} from each yearly file..."

for year in {2000..2270}; do
    input_file="${BASE_DIR}/output_flux_all_timesteps_${year}.nc"
    output_file="${BASE_DIR}/${SUBFOLDER}/floatingBasalMassBalApplied_${year}.nc"
    
    if [ -f "$input_file" ]; then
        echo "Processing year ${year}..."
        
        # Extract the variable and compute time average
        ncwa -a Time -v ${VAR_NAME} "$input_file" "$output_file"
        
        echo "  Created: $output_file"
    else
        echo "Warning: Input file not found: $input_file"
    fi
done

echo "Step 1 completed."

# Step 2: Create monthly repetitions and concatenate
echo "Step 2: Creating monthly repetitions and concatenating files..."

# Array to store temporary monthly files for concatenation
temp_files=()

for year in {2000..2270}; do
    input_file="${BASE_DIR}/${SUBFOLDER}/floatingBasalMassBalApplied_${year}.nc"
    
    if [ -f "$input_file" ]; then
        echo "Creating 12 monthly repetitions for year ${year}..."
        
        # Create 12 copies of the yearly file and reintroduce Time dimension
        for month in {01..12}; do
            monthly_file="${BASE_DIR}/${SUBFOLDER}/temp_${year}_${month}.nc"
            
            # Copy the time-averaged file and add Time dimension back
            cp "$input_file" "$monthly_file"
            
            # Add unlimited Time dimension (since ncwa removed it)
            ncecat -O -u Time "$monthly_file" "$monthly_file"
            
            # Add to array for concatenation
            temp_files+=("$monthly_file")
        done
    else
        echo "Warning: Processed file not found: $input_file"
    fi
done

# Step 3: Concatenate all monthly files
echo "Step 3: Concatenating all monthly files..."
final_output="${BASE_DIR}/${SUBFOLDER}/${FINAL_FILE}"

if [ ${#temp_files[@]} -gt 0 ]; then
    # Concatenate all files along Time dimension
    ncrcat "${temp_files[@]}" "$final_output"
    
    echo "Final concatenated file created: $final_output"
    echo "Total timesteps: $(ncdump -h "$final_output" | grep "Time = " | grep -o '[0-9]*')"
    
    # Clean up temporary files
    echo "Cleaning up temporary files..."
    for temp_file in "${temp_files[@]}"; do
        rm -f "$temp_file"
    done
    
    echo "Processing completed successfully!"
    echo "Output files location: ${BASE_DIR}/${SUBFOLDER}/"
    echo "Final concatenated file: $final_output"
    
else
    echo "Error: No temporary files were created. Check input files and processing."
    exit 1
fi

# Optional: Display summary information
echo ""
echo "=== Processing Summary ==="
echo "Base directory: $BASE_DIR"
echo "Subfolder: $SUBFOLDER"
echo "Variable processed: $VAR_NAME"
echo "Years processed: 2000-2015 (16 years)"
echo "Monthly repetitions per year: 12"
echo "Total expected timesteps: 192"
echo "Final output file: $final_output"

# Verify the final file
if [ -f "$final_output" ]; then
    echo ""
    echo "=== Final File Information ==="
    ncdump -h "$final_output" | head -20
fi
