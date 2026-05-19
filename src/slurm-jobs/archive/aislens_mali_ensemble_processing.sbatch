#!/bin/bash
#SBATCH  --job-name=mali-ensemble-process
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.mali-ensemble-process.%j
#SBATCH  --error=err.mali-ensemble-process.%j
#SBATCH  --time=24:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

# Load modules and environment
module load anaconda3
conda activate mpas-analysis  # this loads nco and python environment

# Configuration
BASE_DATA_PARENT="${AISLENS_DATA_DIR}/MALI/ENSEMBLES/"
BASE_FIGURES_PARENT="${AISLENS_DATA_DIR}/figures/MALI/"
PYTHON_SCRIPT_PATH="${AISLENS_REPO}/src/MPAS-Tools/plot_ensemble_maps.py"
LOG_FILE="mali_ensemble_log_$(date +%Y%m%d_%H%M%S).txt"
MIN_YEAR_DIFF=2  # Minimum year difference to avoid division issues
GL_LINEWIDTH=0.15


# >>>>>>>>>>>> USER: Edit this list to restrict processing <<<<<<<<<<<<
# List your relative parent directories here (leave empty to process all):
RELATIVE_PARENT_PATHS=(
    "CTRL-SSN/CTRL00"
    "CTRL-SSN/CTRL01"
    "CTRL-SSN/CTRL02"
    "CTRL-SSN/CTRL03"
    "CTRL-SSN/CTRL04"
    "CTRL-SSN/CTRL05"
    "CTRL-SSN/CTRL06"
    "CTRL-SSN/CTRL07"
    "CTRL-SSN/CTRL08"
    "CTRL-SSN/CTRL09"
    #"CTRL/CTRL-EM2-5PARAM"
    #"CTRL/CTRL-EM4-5PARAM"
    #"CTRL/CTRL-EM6-5PARAM"
    #"CTRL/CTRL-EM8-5PARAM"
    #"CTRL/CTRL-EM1-5PARAM"
    #"CTRL/CTRL-EM-NO-BMB-ADJ-5PARAM"
    #"SSP585/SSP585-EM1"
    #"SSP585/SSP585-EM2"
    #"SSP585/SSP585-EM4"
    #"SSP585/SSP585-EM6"
    #"SSP585/SSP585-EM8"
)

# Variables to process for ensemble statistics
VARIABLES=("thickness") # "surfaceSpeed" "dhdt")

# Years to process (25 year increments from 2000)
START_YEAR=2000
END_YEAR=2250
YEAR_INCREMENT=25
YEARS=()
for ((year=$START_YEAR+$YEAR_INCREMENT; year<=$END_YEAR; year+=$YEAR_INCREMENT)); do
    YEARS+=($year)
done
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Initialize log
echo "Starting MALI ensemble processing at $(date)" | tee -a "$LOG_FILE"
echo "Script run by user: $USER" | tee -a "$LOG_FILE"
echo "Processing years: ${YEARS[*]}" | tee -a "$LOG_FILE"
echo "Variables: ${VARIABLES[*]}" | tee -a "$LOG_FILE"

# Collect run directories
run_dirs=()
if [ "${#RELATIVE_PARENT_PATHS[@]}" -gt 0 ]; then
    echo "Processing only the following relative paths under \$BASE_DATA_PARENT:" | tee -a "$LOG_FILE"
    for rel_path in "${RELATIVE_PARENT_PATHS[@]}"; do
        echo "  $rel_path" | tee -a "$LOG_FILE"
        parent_dir="${BASE_DATA_PARENT}/${rel_path}"
        output_dir="${parent_dir}/output"
        if [ -d "$output_dir" ]; then
            run_dirs+=("$output_dir")
        else
            echo "WARNING: '$output_dir' does not exist, skipping." | tee -a "$LOG_FILE"
        fi
    done
else
    run_dirs=($(find "$BASE_DATA_PARENT" -type d -name "output"))
fi

echo "Found ${#run_dirs[@]} ensemble members to process" | tee -a "$LOG_FILE"

# Create ensemble statistics directory
ensemble_stats_dir="$BASE_FIGURES_PARENT/ensemble_stats"
mkdir -p "$ensemble_stats_dir"

# Process each run directory to create time-averaged files and dhdt files
echo "=========================================" | tee -a "$LOG_FILE"
echo "Processing individual runs..." | tee -a "$LOG_FILE"

for output_dir in "${run_dirs[@]}"; do
    data_dir=$(dirname "$output_dir")
    rel_dir=${data_dir#"$BASE_DATA_PARENT"/}
    echo "Processing run: $rel_dir" | tee -a "$LOG_FILE"

    cd "$output_dir" || {
        echo "ERROR: Failed to enter directory $output_dir" | tee -a "$LOG_FILE"
        continue
    }

    # Create time-averaged files for reference year and target years
    ref_year_file="output_flux_all_timesteps_${START_YEAR}.nc"
    ref_avg_file="output_flux_all_timesteps_${START_YEAR}_tAvg.nc"

    if [ -f "$ref_year_file" ] && [ ! -f "$ref_avg_file" ]; then
        echo "Creating time average for reference year ${START_YEAR}..." | tee -a "$LOG_FILE"
        ncwa -a Time -O "$ref_year_file" "$ref_avg_file"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to create time average for ${START_YEAR}" | tee -a "$LOG_FILE"
            continue
        fi
    fi

    for year in "${YEARS[@]}"; do
        year_file="output_flux_all_timesteps_${year}.nc"
        avg_file="output_flux_all_timesteps_${year}_tAvg.nc"
        dhdt_file="output_flux_all_timesteps_dhdt_$((year - START_YEAR))yr.nc"

        # Create time-averaged file if it doesn't exist
        if [ -f "$year_file" ] && [ ! -f "$avg_file" ]; then
            echo "Creating time average for year ${year}..." | tee -a "$LOG_FILE"
            ncwa -a Time -O "$year_file" "$avg_file"
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed to create time average for $year" | tee -a "$LOG_FILE"
                continue
            fi
        fi

        # Add Time dimension to averaged files if needed
        for file in "$ref_avg_file" "$avg_file"; do
            if [ -f "$file" ] && ! ncdump -h "$file" | grep -q "Time = "; then
                echo "Adding Time dimension to $(basename "$file")..." | tee -a "$LOG_FILE"
                temp_file="${file}.temp"
                if ncdump -h "$file" | grep -q "xtime"; then
                    ncks -x -v xtime -O "$file" "$temp_file"
                else
                    cp "$file" "$temp_file"
                fi
                ncecat -O -u Time "$temp_file" "$file"
                rm -f "$temp_file"
            fi
        done

        # Calculate dhdt if needed
        if [ -f "$avg_file" ] && [ -f "$ref_avg_file" ] && [ ! -f "$dhdt_file" ]; then
            NYEARS=$((year - START_YEAR))
            if [ $NYEARS -ge $MIN_YEAR_DIFF ]; then
                echo "Calculating dhdt for year $year (${NYEARS} years)..." | tee -a "$LOG_FILE"
                ncdiff "$avg_file" "$ref_avg_file" "temp_dhdt.nc"
                ncap2 -O -s "dhdt=thickness/$NYEARS" "temp_dhdt.nc" "$dhdt_file"
                ncatted -O -a units,dhdt,o,c,"m/yr" -a long_name,dhdt,o,c,"rate of thickness change" "$dhdt_file"
                rm -f "temp_dhdt.nc"

                # Add Time dimension to dhdt file
                if ! ncdump -h "$dhdt_file" | grep -q "Time = "; then
                    ncecat -O -u Time "$dhdt_file" "${dhdt_file}.temp"
                    mv "${dhdt_file}.temp" "$dhdt_file"
                fi
            fi
        fi
    done

    echo "Completed processing run: $rel_dir" | tee -a "$LOG_FILE"
done

# Calculate ensemble statistics for each year and variable
echo "=========================================" | tee -a "$LOG_FILE"
echo "Calculating ensemble statistics..." | tee -a "$LOG_FILE"

for year in "${YEARS[@]}"; do
    echo "Processing ensemble statistics for year $year..." | tee -a "$LOG_FILE"
    stats_file="$ensemble_stats_dir/ensembleStats_${year}.nc"
    
    # Skip if stats file already exists and we're not forcing recreation
    if [ -f "$stats_file" ]; then
        echo "Ensemble stats file for year $year already exists, skipping creation..." | tee -a "$LOG_FILE"
        continue
    fi

    # Clean up any leftover temporary files from previous runs
    rm -f tmp_*.nc tmp_rms.nc tmp_avg.nc

    # Initialize with the first run's COMPLETE file (including mesh)
    first_run_file="${run_dirs[0]}/output_flux_all_timesteps_${year}_tAvg.nc"
    if [ -f "$first_run_file" ]; then
        # Simply copy the entire first run file as base
        cp "$first_run_file" "$stats_file"
        echo "Initialized ensemble stats file with complete file from: $first_run_file" | tee -a "$LOG_FILE"
    else
        echo "ERROR: First run file $first_run_file not found" | tee -a "$LOG_FILE"
        continue
    fi

    for variable in "${VARIABLES[@]}"; do
        echo "Processing variable: $variable for year $year" | tee -a "$LOG_FILE"

        # Create temporary working directory to avoid conflicts
        temp_work_dir="$ensemble_stats_dir/temp_${variable}_${year}"
        mkdir -p "$temp_work_dir"
        cd "$temp_work_dir" || {
            echo "ERROR: Failed to create/enter temp directory $temp_work_dir" | tee -a "$LOG_FILE"
            continue
        }

        # Collect all files for this variable and year
        input_files=()
        for output_dir in "${run_dirs[@]}"; do
            if [ "$variable" == "dhdt" ]; then
                file_path="$output_dir/output_flux_all_timesteps_dhdt_$((year - START_YEAR))yr.nc"
            else
                file_path="$output_dir/output_flux_all_timesteps_${year}_tAvg.nc"
            fi

            if [ -f "$file_path" ]; then
                input_files+=("$file_path")
            else
                echo "WARNING: Missing file: $file_path" | tee -a "$LOG_FILE"
            fi
        done

        if [ ${#input_files[@]} -lt 2 ]; then
            echo "WARNING: Not enough files for ensemble statistics (found ${#input_files[@]})" | tee -a "$LOG_FILE"
            continue
        fi

        echo "Found ${#input_files[@]} files for ensemble statistics" | tee -a "$LOG_FILE"

        # Calculate ensemble statistics using NCO
        file_list=$(IFS=' '; echo "${input_files[*]}")

        # Create temporary files for statistics
        tmp_min="tmp_${variable}_min.nc"
        tmp_max="tmp_${variable}_max.nc"
        tmp_mean="tmp_${variable}_mean.nc"
        tmp_std="tmp_${variable}_std.nc"
        tmp_range="tmp_${variable}_range.nc"

        # Calculate min, max, mean
        ncra -O -y min -v "$variable" $file_list "$tmp_min"
        ncra -O -y max -v "$variable" $file_list "$tmp_max"
        ncra -O -y avg -v "$variable" $file_list "$tmp_mean"

        # Calculate standard deviation (need RMS and mean)
        ncra -O -y rms -v "$variable" $file_list "tmp_rms.nc"
        ncra -O -y avg -v "$variable" $file_list "tmp_avg.nc"
        ncap2 -O -s "${variable}_std=sqrt(${variable}-${variable}*${variable})" "tmp_rms.nc" "$tmp_std"

        # Calculate range (max - min)
        ncdiff -O "$tmp_max" "$tmp_min" "$tmp_range"

        # Rename variables and add to stats file
        ncrename -v "$variable","${variable}_mean" "$tmp_mean"
        ncrename -v "$variable","${variable}_min" "$tmp_min"
        ncrename -v "$variable","${variable}_max" "$tmp_max"
        ncrename -v "$variable","${variable}_range" "$tmp_range"

        # Remove original variable from stats file first (if it exists)
        if ncdump -h "$stats_file" | grep -q "$variable("; then
            ncks -x -v "$variable" -O "$stats_file" "${stats_file}.tmp"
            mv "${stats_file}.tmp" "$stats_file"
        fi

        # Add statistics to the main file
        ncks -A -v "${variable}_mean" "$tmp_mean" "$stats_file"
        ncks -A -v "${variable}_min" "$tmp_min" "$stats_file"
        ncks -A -v "${variable}_max" "$tmp_max" "$stats_file"
        ncks -A -v "${variable}_range" "$tmp_range" "$stats_file"
        ncks -A -v "${variable}_std" "$tmp_std" "$stats_file"

        # Clean up temporary files
        rm -f $tmp_min $tmp_max $tmp_mean $tmp_std $tmp_range tmp_rms.nc tmp_avg.nc

        # Return to base directory and clean up temp work directory
        cd "$ensemble_stats_dir"
        rm -rf "$temp_work_dir"
    done

    # Simple cleanup: just remove xtime if it exists
    if ncdump -h "$stats_file" | grep -q "xtime"; then
        echo "Removing xtime variable from stats file..." | tee -a "$LOG_FILE"
        ncks -x -v xtime -O "$stats_file" "${stats_file}.tmp"
        mv "${stats_file}.tmp" "$stats_file"
    fi

    # Verify mesh variables are present
    echo "Verifying mesh variables in stats file..." | tee -a "$LOG_FILE"
    missing_vars=()
    for var in xCell yCell dcEdge; do
        if ! ncdump -h "$stats_file" | grep -q "$var"; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo "ERROR: Missing mesh variables: ${missing_vars[*]}" | tee -a "$LOG_FILE"
    else
        echo "All mesh variables verified present in stats file." | tee -a "$LOG_FILE"
    fi

    echo "Completed ensemble statistics for year $year" | tee -a "$LOG_FILE"
done

# Verify ensemble stats files exist before plotting
echo "=========================================" | tee -a "$LOG_FILE"
echo "Verifying ensemble statistics files..." | tee -a "$LOG_FILE"

valid_years=()
ensemble_stats_files=()
for year in "${YEARS[@]}"; do
    stats_file="$ensemble_stats_dir/ensembleStats_${year}.nc"
    if [ -f "$stats_file" ]; then
        valid_years+=($year)
        ensemble_stats_files+=("$stats_file")
        echo "Found: $(basename "$stats_file")" | tee -a "$LOG_FILE"
    else
        echo "Missing: ensembleStats_${year}.nc" | tee -a "$LOG_FILE"
    fi
done

if [ ${#valid_years[@]} -eq 0 ]; then
    echo "ERROR: No valid ensemble statistics files found. Cannot proceed with plotting." | tee -a "$LOG_FILE"
    exit 1
fi

echo "Will plot ${#valid_years[@]} years: ${valid_years[*]}" | tee -a "$LOG_FILE"

# REVISED SECTION: Prepare year-specific mesh files for grounding line overlays
echo "Preparing year-specific mesh files for grounding line overlays..." | tee -a "$LOG_FILE"

# Create a mapping of years to mesh files for each run
run_names=()
for output_dir in "${run_dirs[@]}"; do
    data_dir=$(dirname "$output_dir")
    rel_dir=${data_dir#"$BASE_DATA_PARENT"/}
    run_name="$(basename "$rel_dir")"
    run_names+=("$run_name")
done

# Prepare arguments for Python plotting script
stats_files_str=$(IFS=','; echo "${ensemble_stats_files[*]}")
years_str=$(IFS=','; echo "${valid_years[*]}")
variables_str=$(IFS=','; echo "${VARIABLES[*]}")
run_names_str=$(IFS=','; echo "${run_names[*]}")
save_base="$BASE_FIGURES_PARENT/ensemble/ensemble_maps"

# Create run_dirs string for Python script
run_dirs_str=$(IFS=','; echo "${run_dirs[*]}")

echo "=========================================" | tee -a "$LOG_FILE"
echo "Calling Python plotting script..." | tee -a "$LOG_FILE"
echo "Stats files: $(echo "$stats_files_str" | tr ',' '\n' | wc -l) files" | tee -a "$LOG_FILE"
echo "Years: $years_str" | tee -a "$LOG_FILE"
echo "Variables: $variables_str" | tee -a "$LOG_FILE"
echo "Run directories: $(echo "$run_dirs_str" | tr ',' '\n' | wc -l) directories" | tee -a "$LOG_FILE"
echo "Run names: $run_names_str" | tee -a "$LOG_FILE"

mkdir -p "$(dirname "$save_base")"

# REVISED: Pass run directories instead of pre-selected mesh files
python3 "$PYTHON_SCRIPT_PATH" \
    --ensemble_files "$stats_files_str" \
    --years "$years_str" \
    --variables "$variables_str" \
    --run_dirs "$run_dirs_str" \
    --run_names "$run_names_str" \
    --gl_linewidth $GL_LINEWIDTH
    --save_base "$save_base" \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "Successfully generated ensemble plots" | tee -a "$LOG_FILE"
else
    echo "ERROR: Plotting script failed" | tee -a "$LOG_FILE"
fi

echo "=========================================" | tee -a "$LOG_FILE"
echo "Ensemble processing completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to $LOG_FILE"
exit 0
