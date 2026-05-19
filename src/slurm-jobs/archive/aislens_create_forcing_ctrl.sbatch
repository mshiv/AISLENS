#!/bin/bash
#SBATCH --job-name=ctrl-forcing-processing
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --time=05:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis

# Define base directory to process
BASE_DIRECTORY="${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL/CTRL-EM8/output"

# Set other variables
SUBFOLDER="floatingBMB"
VAR_NAME="floatingBasalMassBalApplied"

echo "Starting processing of floatingBasalMassBalApplied files..."
echo "Processing directory: $BASE_DIRECTORY"
echo ""

# Check if base directory exists
if [ ! -d "$BASE_DIRECTORY" ]; then
    echo "Error: Base directory does not exist: $BASE_DIRECTORY"
    echo "Exiting..."
    exit 1
fi

# Extract scenario name from path (CTRL-EM8)
SCENARIO_NAME=$(echo "$BASE_DIRECTORY" | grep -oE 'CTRL-EM[0-9]+' | head -1)
if [ -z "$SCENARIO_NAME" ]; then
    SCENARIO_NAME="CTRL"
fi

# Create subfolder if it doesn't exist
mkdir -p "${BASE_DIRECTORY}/${SUBFOLDER}"

echo "Base directory: $BASE_DIRECTORY"
echo "Scenario: $SCENARIO_NAME"

# Step 0: Dynamically determine year range from available files
echo "Step 0: Determining available years from input files..."

# Find all files matching the pattern and extract years
available_years=()
for file in "${BASE_DIRECTORY}"/output_flux_all_timesteps_*.nc; do
    if [ -f "$file" ]; then
        # Extract year from filename
        year=$(basename "$file" | grep -oE '[0-9]{4}' | head -1)
        if [[ "$year" =~ ^[0-9]{4}$ ]]; then
            available_years+=("$year")
        fi
    fi
done

# Check if any files were found
if [ ${#available_years[@]} -eq 0 ]; then
    echo "Error: No output_flux_all_timesteps_*.nc files found in $BASE_DIRECTORY"
    echo "Exiting..."
    exit 1
fi

# Sort years and get min/max
IFS=$'\n' sorted_years=($(sort -n <<<"${available_years[*]}"))
unset IFS

START_YEAR=${sorted_years[0]}
END_YEAR=${sorted_years[-1]}

echo "Found ${#available_years[@]} files"
echo "Year range: $START_YEAR to $END_YEAR"

# Set final filename with year range
FINAL_FILE="floatingBasalMassBalApplied_${SCENARIO_NAME}_Trend_${START_YEAR}-${END_YEAR}.nc"

# Step 1: Extract time-averaged variable from each yearly file
echo "Step 1: Extracting time-averaged ${VAR_NAME} from each yearly file..."

processed_years=()
skipped_years=()
yearly_files=()

for year in "${sorted_years[@]}"; do
    input_file="${BASE_DIRECTORY}/output_flux_all_timesteps_${year}.nc"
    output_file="${BASE_DIRECTORY}/${SUBFOLDER}/floatingBasalMassBalApplied_${year}.nc"

    # Check if output file already exists
    if [ -f "$output_file" ]; then
        echo "Year ${year}: Output file already exists, skipping..."
        echo "  Existing file: $output_file"
        skipped_years+=("$year")
        processed_years+=("$year")  # Still add to processed list for concatenation
        yearly_files+=("$output_file")
    elif [ -f "$input_file" ]; then
        echo "Processing year ${year}..."

        # Extract the variable and compute time average
        ncwa -a Time -v ${VAR_NAME} "$input_file" "$output_file"

        if [ $? -eq 0 ]; then
            echo "  Created: $output_file"
            processed_years+=("$year")
            yearly_files+=("$output_file")
        else
            echo "  Error processing year $year"
        fi
    else
        echo "Warning: Input file not found: $input_file"
    fi
done

echo "Step 1 completed. Total years available: ${#processed_years[@]}"
echo "  Newly processed: $((${#processed_years[@]} - ${#skipped_years[@]}))"
echo "  Skipped (already existed): ${#skipped_years[@]}"
if [ ${#skipped_years[@]} -gt 0 ]; then
    echo "  Skipped years: ${skipped_years[*]}"
fi

# Step 2: Add Time dimension back to yearly files and concatenate
echo "Step 2: Adding Time dimension and concatenating yearly files..."

# Array to store temporary files with Time dimension
temp_files=()

for i in "${!yearly_files[@]}"; do
    yearly_file="${yearly_files[$i]}"
    year="${processed_years[$i]}"
    
    if [ -f "$yearly_file" ]; then
        echo "Adding Time dimension to year ${year}..."
        
        # Create temporary file with Time dimension
        temp_file="${BASE_DIRECTORY}/${SUBFOLDER}/temp_${year}_with_time.nc"
        
        # Add unlimited Time dimension (since ncwa removed it)
        ncecat -O -u Time "$yearly_file" "$temp_file"
        
        # Add to array for concatenation
        temp_files+=("$temp_file")
    else
        echo "Warning: Yearly file not found: $yearly_file"
    fi
done

# Step 3: Concatenate all yearly files
echo "Step 3: Concatenating all yearly files..."
final_output="${BASE_DIRECTORY}/${SUBFOLDER}/${FINAL_FILE}"

if [ ${#temp_files[@]} -gt 0 ]; then
    # Concatenate all files along Time dimension
    ncrcat "${temp_files[@]}" "$final_output"

    if [ $? -eq 0 ]; then
        echo "Final concatenated file created: $final_output"
        echo "Total timesteps: $(ncdump -h "$final_output" | grep "Time = " | grep -o '[0-9]*')"

        # Clean up temporary files
        echo "Cleaning up temporary files..."
        for temp_file in "${temp_files[@]}"; do
            rm -f "$temp_file"
        done

        echo "Processing completed successfully for $SCENARIO_NAME!"
        echo "Output files location: ${BASE_DIRECTORY}/${SUBFOLDER}/"
        echo "Final concatenated file: $final_output"
    else
        echo "Error: Failed to concatenate files for $SCENARIO_NAME"
        exit 1
    fi
else
    echo "Error: No temporary files were created for $SCENARIO_NAME. Check input files and processing."
    exit 1
fi

# Display summary information
echo ""
echo "=== Processing Summary for $SCENARIO_NAME ==="
echo "Base directory: $BASE_DIRECTORY"
echo "Scenario: $SCENARIO_NAME"
echo "Subfolder: $SUBFOLDER"
echo "Variable processed: $VAR_NAME"
echo "Years processed: $START_YEAR-$END_YEAR (${#processed_years[@]} years)"
echo "Years newly processed: $((${#processed_years[@]} - ${#skipped_years[@]}))"
echo "Years skipped (already existed): ${#skipped_years[@]}"
echo "Final output file: $final_output"

# Verify the final file
if [ -f "$final_output" ]; then
    echo ""
    echo "=== Final File Information for $SCENARIO_NAME ==="
    ncdump -h "$final_output" | head -20
fi

echo ""
echo "Processing completed!"
echo "========================================"
