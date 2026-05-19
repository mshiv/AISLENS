#!/bin/bash
#SBATCH  --job-name=mali-robust-process
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.mali-robust-process.%j
#SBATCH  --error=err.mali-robust-process.%j
#SBATCH  --time=8:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

# Load modules and environment
module load anaconda3
conda activate mpas-analysis  # this loads nco and python environment

# Configuration
BASE_DATA_PARENT="${AISLENS_DATA_DIR}/MALI"
BASE_FIGURES_PARENT="${AISLENS_DATA_DIR}/figures/MALI"
PYTHON_SCRIPT_PATH="${AISLENS_REPO}/src/MPAS-Tools/plot_output_maps_masked.py"
LOG_FILE="mali_process_log_$(date +%Y%m%d_%H%M%S).txt"
MIN_YEAR_DIFF=2  # Minimum year difference to avoid division issues

# Initialize log
echo "Starting MALI processing at $(date)" | tee -a "$LOG_FILE"
echo "Script run by user: $USER" | tee -a "$LOG_FILE"

# Optional target year from command line
TARGET_YEAR=""
if [ $# -eq 1 ]; then
    TARGET_YEAR="$1"
    echo "Using specified target year: $TARGET_YEAR" | tee -a "$LOG_FILE"
fi

# Find all output directories
run_dirs=($(find "$BASE_DATA_PARENT" -type d -name "output"))
echo "Found ${#run_dirs[@]} output directories to process" | tee -a "$LOG_FILE"

# Process each directory
for output_dir in "${run_dirs[@]}"; do
    data_dir=$(dirname "$output_dir")
    rel_dir=${data_dir#"$BASE_DATA_PARENT"/}
    figures_dir="$BASE_FIGURES_PARENT/$rel_dir/output"
    
    echo "=========================================" | tee -a "$LOG_FILE"
    echo "Processing: $rel_dir" | tee -a "$LOG_FILE"
    
    mkdir -p "$figures_dir"
    cd "$output_dir" || { 
        echo "[$rel_dir] ERROR: Failed to enter directory $output_dir" | tee -a "$LOG_FILE"
        continue
    }

    # Find available years
    years=($(ls output_flux_all_timesteps_2*.nc 2>/dev/null | \
             sed 's/output_flux_all_timesteps_\([0-9]\+\)\.nc/\1/' | sort -n))
    
    if [ ${#years[@]} -lt 2 ]; then
        echo "[$rel_dir] Not enough files to process (need at least 2)" | tee -a "$LOG_FILE"
        continue
    fi
    
    echo "[$rel_dir] Available years: ${years[*]}" | tee -a "$LOG_FILE"
    
    # Determine year to process
    YEAR_TO_PROCESS="$TARGET_YEAR"
    if [ -z "$YEAR_TO_PROCESS" ]; then
        YEAR_TO_PROCESS=${years[-2]}  # Use penultimate year by default
    fi
    NYEARS=$((YEAR_TO_PROCESS - 2000))
    
    # Check if year difference is sufficient
    if [ $NYEARS -lt $MIN_YEAR_DIFF ]; then
        echo "[$rel_dir] ERROR: Year difference too small ($NYEARS years). Need at least $MIN_YEAR_DIFF years." | tee -a "$LOG_FILE"
        continue
    fi
    
    echo "[$rel_dir] Processing year $YEAR_TO_PROCESS (NYEARS=$NYEARS)" | tee -a "$LOG_FILE"
    
    # Define file paths
    dhdt_file="$output_dir/output_flux_all_timesteps_dhdt_${NYEARS}yr.nc"
    mesh_2000="$output_dir/output_flux_all_timesteps_2000_tAvg.nc"
    mesh_end="$output_dir/output_flux_all_timesteps_${YEAR_TO_PROCESS}_tAvg.nc"
    flux_2000="$output_dir/output_flux_all_timesteps_2000.nc"
    flux_end="$output_dir/output_flux_all_timesteps_${YEAR_TO_PROCESS}.nc"
    figure_file="$figures_dir/dhdt_${NYEARS}yr.png"
    
    # Skip if both dhdt file and figure already exist
    if [ -f "$dhdt_file" ] && [ -f "$figure_file" ]; then
        echo "[$rel_dir] SKIPPING: dhdt file and figure already exist" | tee -a "$LOG_FILE"
        continue
    fi
    
    # Check if required source files exist
    if [ ! -f "$flux_2000" ] || [ ! -f "$flux_end" ]; then
        echo "[$rel_dir] ERROR: Missing required source files:" | tee -a "$LOG_FILE"
        [ ! -f "$flux_2000" ] && echo "  - $flux_2000" | tee -a "$LOG_FILE"
        [ ! -f "$flux_end" ] && echo "  - $flux_end" | tee -a "$LOG_FILE"
        continue
    fi
    
    # 1. Time average for year 2000
    if [ ! -f "$mesh_2000" ]; then
        echo "[$rel_dir] Creating time average for year 2000..." | tee -a "$LOG_FILE"
        ncwa -a Time -O "$flux_2000" "$mesh_2000"
        if [ $? -ne 0 ]; then
            echo "[$rel_dir] ERROR: Failed to create time average for 2000" | tee -a "$LOG_FILE"
            continue
        fi
    fi
    
    # 2. Time average for end year
    if [ ! -f "$mesh_end" ]; then
        echo "[$rel_dir] Creating time average for year $YEAR_TO_PROCESS..." | tee -a "$LOG_FILE"
        ncwa -a Time -O "$flux_end" "$mesh_end"
        if [ $? -ne 0 ]; then
            echo "[$rel_dir] ERROR: Failed to create time average for $YEAR_TO_PROCESS" | tee -a "$LOG_FILE"
            continue
        fi
    fi
    
    # 3. Add Time dimension to both files (handling xtime issue)
    for mesh_file in "$mesh_2000" "$mesh_end"; do
        # Check if Time dimension already exists
        if ! ncdump -h "$mesh_file" | grep -q "Time = "; then
            echo "[$rel_dir] Adding Time dimension to $(basename "$mesh_file")..." | tee -a "$LOG_FILE"
            
            # Create a temporary file without xtime
            temp_file="${mesh_file}.temp"
            
            # Check if xtime variable exists
            if ncdump -h "$mesh_file" | grep -q "xtime"; then
                echo "[$rel_dir] Removing xtime variable before adding Time dimension" | tee -a "$LOG_FILE"
                ncks -x -v xtime -O "$mesh_file" "$temp_file"
                if [ $? -ne 0 ]; then
                    echo "[$rel_dir] ERROR: Failed to remove xtime from $(basename "$mesh_file")" | tee -a "$LOG_FILE"
                    rm -f "$temp_file"
                    continue 2  # Skip to next directory
                fi
            else
                # Just copy the file if no xtime
                cp "$mesh_file" "$temp_file"
            fi
            
            # Add Time dimension
            ncecat -O -u Time "$temp_file" "$mesh_file"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] ERROR: Failed to add Time dimension to $(basename "$mesh_file")" | tee -a "$LOG_FILE"
                rm -f "$temp_file"
                continue 2  # Skip to next directory
            fi
            
            # Clean up
            rm -f "$temp_file"
        fi
    done
    
    # 4. Calculate dhdt if needed
    if [ ! -f "$dhdt_file" ]; then
        echo "[$rel_dir] Calculating dhdt..." | tee -a "$LOG_FILE"
        
        # Calculate thickness difference
        echo "[$rel_dir] Running ncdiff to calculate thickness difference..." | tee -a "$LOG_FILE"
        ncdiff "$mesh_end" "$mesh_2000" "out_dhdt_temp.nc"
        if [ $? -ne 0 ]; then
            echo "[$rel_dir] ERROR: Failed to calculate thickness difference" | tee -a "$LOG_FILE"
            rm -f "out_dhdt_temp.nc"
            continue
        fi
        
        # Check if the difference file was created successfully
        if [ ! -f "out_dhdt_temp.nc" ]; then
            echo "[$rel_dir] ERROR: ncdiff did not create the temporary file" | tee -a "$LOG_FILE"
            continue
        fi
        
        # Inspect temp file header for debugging
        echo "[$rel_dir] Temporary file created successfully, checking content..." | tee -a "$LOG_FILE"
        if ! ncdump -h "out_dhdt_temp.nc" > /dev/null 2>&1; then
            echo "[$rel_dir] ERROR: Temporary file is not a valid NetCDF file" | tee -a "$LOG_FILE"
            rm -f "out_dhdt_temp.nc"
            continue
        fi
        
        # Calculate dhdt with proper units - breaking into two steps for robustness
        echo "[$rel_dir] Calculating dhdt using ncap2 (NYEARS=$NYEARS)..." | tee -a "$LOG_FILE"
        
        # Step 1: Create a simple division operation (more likely to succeed)
        ncap2 -O -s "dhdt=thickness/$NYEARS" "out_dhdt_temp.nc" "$dhdt_file" 2>&1 | tee -a "$LOG_FILE"
        if [ $? -ne 0 ]; then
            echo "[$rel_dir] ERROR: Failed to calculate dhdt with ncap2" | tee -a "$LOG_FILE"
            
            # Try alternative method
            echo "[$rel_dir] Trying alternative calculation method..." | tee -a "$LOG_FILE"
            ncap2 -O -s "dhdt=thickness*1.0/$NYEARS" "out_dhdt_temp.nc" "$dhdt_file" 2>&1 | tee -a "$LOG_FILE"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] ERROR: Both calculation methods failed" | tee -a "$LOG_FILE"
                rm -f "out_dhdt_temp.nc"
                continue
            fi
        fi
        
        # Step 2: Add attributes separately (less likely to fail)
        echo "[$rel_dir] Adding metadata attributes to dhdt variable..." | tee -a "$LOG_FILE"
        ncatted -O -a units,dhdt,o,c,"m/yr" -a long_name,dhdt,o,c,"rate of thickness change" "$dhdt_file"
        if [ $? -ne 0 ]; then
            echo "[$rel_dir] WARNING: Failed to add attributes to dhdt, but calculation succeeded" | tee -a "$LOG_FILE"
            # Continue anyway since we have the main data
        fi
        
        # Clean up
        rm -f "out_dhdt_temp.nc"
        
        # Verify dhdt file exists and is valid
        if [ ! -f "$dhdt_file" ]; then
            echo "[$rel_dir] ERROR: dhdt file was not created" | tee -a "$LOG_FILE"
            continue
        fi
        
        if ! ncdump -h "$dhdt_file" | grep -q "dhdt" ; then
            echo "[$rel_dir] ERROR: dhdt file was created but doesn't contain dhdt variable" | tee -a "$LOG_FILE"
            continue
        fi
        
        echo "[$rel_dir] Successfully created dhdt file" | tee -a "$LOG_FILE"
    fi
    
    # 5. Generate plot if needed
    if [ ! -f "$figure_file" ]; then
        echo "[$rel_dir] Generating plot..." | tee -a "$LOG_FILE"
        
        # Check all required files exist for plotting
        if [ ! -f "$dhdt_file" ] || [ ! -f "$mesh_2000" ] || [ ! -f "$mesh_end" ]; then
            echo "[$rel_dir] ERROR: Missing files needed for plotting:" | tee -a "$LOG_FILE"
            [ ! -f "$dhdt_file" ] && echo "  - $dhdt_file" | tee -a "$LOG_FILE"
            [ ! -f "$mesh_2000" ] && echo "  - $mesh_2000" | tee -a "$LOG_FILE"
            [ ! -f "$mesh_end" ] && echo "  - $mesh_end" | tee -a "$LOG_FILE"
            continue
        fi
        
        # Run plotting script
        echo "[$rel_dir] Running Python plotting script..." | tee -a "$LOG_FILE"
        python3 "$PYTHON_SCRIPT_PATH" \
            -v dhdt \
            -r "$dhdt_file" \
            -m "$mesh_2000" \
            -m2 "$mesh_end" \
            -s "${figure_file%.*}" 2>&1 | tee -a "$LOG_FILE"
        
        if [ $? -ne 0 ]; then
            echo "[$rel_dir] ERROR: Plotting script failed" | tee -a "$LOG_FILE"
            continue
        fi
        
        # Handle file naming (remove _2000 suffix if present)
        if [ -f "${figure_file%.*}_2000.png" ]; then
            mv "${figure_file%.*}_2000.png" "$figure_file"
            echo "[$rel_dir] Renamed figure file to remove _2000 suffix" | tee -a "$LOG_FILE"
        fi
        
        if [ -f "$figure_file" ]; then
            echo "[$rel_dir] Successfully created plot: $figure_file" | tee -a "$LOG_FILE"
        else
            echo "[$rel_dir] WARNING: Plot file not found after plotting script completed" | tee -a "$LOG_FILE"
        fi
    fi
    
    echo "[$rel_dir] Successfully processed!" | tee -a "$LOG_FILE"
done

echo "=========================================" | tee -a "$LOG_FILE"
echo "Processing completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to $LOG_FILE"
exit 0
