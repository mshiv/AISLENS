#!/bin/bash
#SBATCH --job-name=ssp-forcing-processing
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

# Define base directories to process
BASE_DIRECTORIES=(
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

# Set other variables
SUBFOLDER="floatingBMB"
VAR_NAME="floatingBasalMassBalApplied"

echo "Starting processing of floatingBasalMassBalApplied files..."
echo "Will process ${#BASE_DIRECTORIES[@]} base directories"
echo ""

# Loop through each base directory
for BASE_DIR in "${BASE_DIRECTORIES[@]}"; do
    echo "========================================"
    echo "Processing directory: $BASE_DIR"
    echo "========================================"
    
    # Check if base directory exists
    if [ ! -d "$BASE_DIR" ]; then
        echo "Warning: Base directory does not exist: $BASE_DIR"
        echo "Skipping this directory..."
        echo ""
        continue
    fi

    # Extract scenario name from path (e.g., SSP585, SSP126)
    SCENARIO_NAME=$(echo "$BASE_DIR" | grep -oE 'SSP[0-9]+' | head -1)
    if [ -z "$SCENARIO_NAME" ]; then
        SCENARIO_NAME="UNKNOWN"
    fi

    # Create subfolder if it doesn't exist
    mkdir -p "${BASE_DIR}/${SUBFOLDER}"

    echo "Base directory: $BASE_DIR"
    echo "Scenario: $SCENARIO_NAME"

    # Step 0: Dynamically determine year range from available files
    echo "Step 0: Determining available years from input files..."

    # Find all files matching the pattern and extract years
    available_years=()
    for file in "${BASE_DIR}"/output_flux_all_timesteps_*.nc; do
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
        echo "Warning: No output_flux_all_timesteps_*.nc files found in $BASE_DIR"
        echo "Skipping this directory..."
        echo ""
        continue
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
    for year in "${sorted_years[@]}"; do
        input_file="${BASE_DIR}/output_flux_all_timesteps_${year}.nc"
        output_file="${BASE_DIR}/${SUBFOLDER}/floatingBasalMassBalApplied_${year}.nc"

        if [ -f "$input_file" ]; then
            echo "Processing year ${year}..."

            # Extract the variable and compute time average
            ncwa -a Time -v ${VAR_NAME} "$input_file" "$output_file"

            if [ $? -eq 0 ]; then
                echo "  Created: $output_file"
                processed_years+=("$year")
            else
                echo "  Error processing year $year"
            fi
        else
            echo "Warning: Input file not found: $input_file"
        fi
    done

    echo "Step 1 completed. Successfully processed ${#processed_years[@]} years."

    # Step 2: Create monthly repetitions and concatenate
    echo "Step 2: Creating monthly repetitions and concatenating files..."

    # Array to store temporary monthly files for concatenation
    temp_files=()

    for year in "${processed_years[@]}"; do
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

        if [ $? -eq 0 ]; then
            echo "Final concatenated file created: $final_output"
            echo "Total timesteps: $(ncdump -h "$final_output" | grep "Time = " | grep -o '[0-9]*')"

            # Clean up temporary files
            echo "Cleaning up temporary files..."
            for temp_file in "${temp_files[@]}"; do
                rm -f "$temp_file"
            done

            echo "Processing completed successfully for $SCENARIO_NAME!"
            echo "Output files location: ${BASE_DIR}/${SUBFOLDER}/"
            echo "Final concatenated file: $final_output"
        else
            echo "Error: Failed to concatenate files for $SCENARIO_NAME"
        fi
    else
        echo "Error: No temporary files were created for $SCENARIO_NAME. Check input files and processing."
    fi

    # Display summary information for this directory
    echo ""
    echo "=== Processing Summary for $SCENARIO_NAME ==="
    echo "Base directory: $BASE_DIR"
    echo "Scenario: $SCENARIO_NAME"
    echo "Subfolder: $SUBFOLDER"
    echo "Variable processed: $VAR_NAME"
    echo "Years processed: $START_YEAR-$END_YEAR (${#processed_years[@]} years)"
    echo "Monthly repetitions per year: 12"
    echo "Total expected timesteps: $((${#processed_years[@]} * 12))"
    echo "Final output file: $final_output"

    # Verify the final file
    if [ -f "$final_output" ]; then
        echo ""
        echo "=== Final File Information for $SCENARIO_NAME ==="
        ncdump -h "$final_output" | head -20
    fi
    
    echo ""
    echo "Finished processing $SCENARIO_NAME"
    echo ""

done

echo "========================================"
echo "All directories processed!"
echo "========================================"
