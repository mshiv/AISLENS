#!/bin/bash
#SBATCH  --job-name=mali-robust-process
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.mali-robust-process.%j
#SBATCH  --error=err.mali-robust-process.%j
#SBATCH  --time=24:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

# Load modules and environment
module load anaconda3
conda activate mpas-analysis  # this loads nco and python environment

# Configuration
BASE_DATA_PARENT="${AISLENS_DATA_DIR}/MALI/ENSEMBLES"
BASE_FIGURES_PARENT="${AISLENS_DATA_DIR}/figures/MALI"
PYTHON_SCRIPT_PATH="${AISLENS_REPO}/src/MPAS-Tools/plot_output_maps_masked_animation.py"
LOG_FILE="mali_process_log_$(date +%Y%m%d_%H%M%S).txt"
MIN_YEAR_DIFF=2  # Minimum year difference to avoid division issues

# >>>>>>>>>>>> USER: Edit this list to restrict processing <<<<<<<<<<<<
# List your relative parent directories here (leave empty to process all):
RELATIVE_PARENT_PATHS=(
    "CTRL/CTRL-EM2-5PARAM"
    "CTRL/CTRL-EM4-5PARAM"
    "CTRL/CTRL-EM6-5PARAM"
    "CTRL/CTRL-EM8-5PARAM"
    "CTRL/CTRL-EM1-5PARAM"
    "CTRL/CTRL-EM-NO-BMB-ADJ-5PARAM"
    # "case3"
)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Initialize log
echo "Starting MALI processing at $(date)" | tee -a "$LOG_FILE"
echo "Script run by user: $USER" | tee -a "$LOG_FILE"

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
    # No paths specified: find all output directories under BASE_DATA_PARENT
    run_dirs=($(find "$BASE_DATA_PARENT" -type d -name "output"))
fi
echo "Found ${#run_dirs[@]} output directories to process" | tee -a "$LOG_FILE"

for output_dir in "${run_dirs[@]}"; do
    data_dir=$(dirname "$output_dir")
    rel_dir=${data_dir#"$BASE_DATA_PARENT"/}
    figures_dir="$BASE_FIGURES_PARENT/$rel_dir/output/all-years"

    echo "=========================================" | tee -a "$LOG_FILE"
    echo "Processing: $rel_dir" | tee -a "$LOG_FILE"

    mkdir -p "$figures_dir"
    cd "$output_dir" || {
        echo "[$rel_dir] ERROR: Failed to enter directory $output_dir" | tee -a "$LOG_FILE"
        continue
    }

    # Find available years
    years=($(ls output_flux_all_timesteps_[0-9][0-9][0-9][0-9].nc 2>/dev/null | \
             sed -E 's/output_flux_all_timesteps_([0-9]{4})\.nc/\1/' | sort -n))
    echo "[$rel_dir] All years detected: ${years[*]}" | tee -a "$LOG_FILE"

    # Filter for five-year intervals, skipping 2000 (reference)
    five_years=()
    for year in "${years[@]}"; do
        if [[ "$year" =~ ^[0-9]+$ ]] && [ "$year" -gt 2000 ] && [ $((year % 5)) -eq 0 ]; then
            five_years+=("$year")
        fi
    done
    echo "[$rel_dir] Five-year intervals to process: ${five_years[*]}" | tee -a "$LOG_FILE"

    if [ ${#five_years[@]} -lt 1 ]; then
        echo "[$rel_dir] No five-year interval files to process (need at least one after 2000)" | tee -a "$LOG_FILE"
        continue
    fi

    # Process dhdt files for five-year intervals only
    dhdt_files=()
    mesh_files=()
    figure_files=()
    all_years_to_process=()

    for year in "${five_years[@]}"; do
        NYEARS=$((year - 2000))

        # Check if year difference is sufficient
        if [ $NYEARS -lt $MIN_YEAR_DIFF ]; then
            echo "[$rel_dir] Skipping year $year: Year difference too small ($NYEARS years)" | tee -a "$LOG_FILE"
            continue
        fi

        all_years_to_process+=($year)

        # Define file paths
        dhdt_file="$output_dir/output_flux_all_timesteps_dhdt_${NYEARS}yr.nc"
        mesh_2000="$output_dir/output_flux_all_timesteps_2000_tAvg.nc"
        mesh_end="$output_dir/output_flux_all_timesteps_${year}_tAvg.nc"
        flux_2000="$output_dir/output_flux_all_timesteps_2000.nc"
        flux_end="$output_dir/output_flux_all_timesteps_${year}.nc"
        figure_file="$figures_dir/dhdt_${NYEARS}yr.png"

        dhdt_files+=("$dhdt_file")
        mesh_files+=("$mesh_end")
        figure_files+=("$figure_file")

        echo "[$rel_dir] Processing year $year (NYEARS=$NYEARS)" | tee -a "$LOG_FILE"

        # Check if required source files exist
        if [ ! -f "$flux_2000" ] || [ ! -f "$flux_end" ]; then
            echo "[$rel_dir] ERROR: Missing required source files for year $year:" | tee -a "$LOG_FILE"
            [ ! -f "$flux_2000" ] && echo "  - $flux_2000" | tee -a "$LOG_FILE"
            [ ! -f "$flux_end" ] && echo "  - $flux_end" | tee -a "$LOG_FILE"
            continue
        fi

        # 1. Time average for year 2000 (only create once)
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
            echo "[$rel_dir] Creating time average for year $year..." | tee -a "$LOG_FILE"
            ncwa -a Time -O "$flux_end" "$mesh_end"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] ERROR: Failed to create time average for $year" | tee -a "$LOG_FILE"
                continue
            fi
        fi

        # 3. Add Time dimension to both files (handling xtime issue)
        for mesh_file in "$mesh_2000" "$mesh_end"; do
            if ! ncdump -h "$mesh_file" | grep -q "Time = "; then
                echo "[$rel_dir] Adding Time dimension to $(basename "$mesh_file")..." | tee -a "$LOG_FILE"
                temp_file="${mesh_file}.temp"
                if ncdump -h "$mesh_file" | grep -q "xtime"; then
                    echo "[$rel_dir] Removing xtime variable before adding Time dimension" | tee -a "$LOG_FILE"
                    ncks -x -v xtime -O "$mesh_file" "$temp_file"
                    if [ $? -ne 0 ]; then
                        echo "[$rel_dir] ERROR: Failed to remove xtime from $(basename "$mesh_file")" | tee -a "$LOG_FILE"
                        rm -f "$temp_file"
                        continue 2
                    fi
                else
                    cp "$mesh_file" "$temp_file"
                fi
                ncecat -O -u Time "$temp_file" "$mesh_file"
                if [ $? -ne 0 ]; then
                    echo "[$rel_dir] ERROR: Failed to add Time dimension to $(basename "$mesh_file")" | tee -a "$LOG_FILE"
                    rm -f "$temp_file"
                    continue 2
                fi
                rm -f "$temp_file"
            fi
        done

        # 4. Calculate dhdt if needed
        if [ ! -f "$dhdt_file" ]; then
            echo "[$rel_dir] Calculating dhdt for year $year..." | tee -a "$LOG_FILE"
            echo "[$rel_dir] Running ncdiff to calculate thickness difference..." | tee -a "$LOG_FILE"
            ncdiff "$mesh_end" "$mesh_2000" "out_dhdt_temp.nc"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] ERROR: Failed to calculate thickness difference" | tee -a "$LOG_FILE"
                rm -f "out_dhdt_temp.nc"
                continue
            fi

            if [ ! -f "out_dhdt_temp.nc" ]; then
                echo "[$rel_dir] ERROR: ncdiff did not create the temporary file" | tee -a "$LOG_FILE"
                continue
            fi

            echo "[$rel_dir] Temporary file created successfully, checking content..." | tee -a "$LOG_FILE"
            if ! ncdump -h "out_dhdt_temp.nc" > /dev/null 2>&1; then
                echo "[$rel_dir] ERROR: Temporary file is not a valid NetCDF file" | tee -a "$LOG_FILE"
                rm -f "out_dhdt_temp.nc"
                continue
            fi

            echo "[$rel_dir] Calculating dhdt using ncap2 (NYEARS=$NYEARS)..." | tee -a "$LOG_FILE"
            ncap2 -O -s "dhdt=thickness/$NYEARS" "out_dhdt_temp.nc" "$dhdt_file" 2>&1 | tee -a "$LOG_FILE"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] ERROR: Failed to calculate dhdt with ncap2" | tee -a "$LOG_FILE"
                echo "[$rel_dir] Trying alternative calculation method..." | tee -a "$LOG_FILE"
                ncap2 -O -s "dhdt=thickness*1.0/$NYEARS" "out_dhdt_temp.nc" "$dhdt_file" 2>&1 | tee -a "$LOG_FILE"
                if [ $? -ne 0 ]; then
                    echo "[$rel_dir] ERROR: Both calculation methods failed" | tee -a "$LOG_FILE"
                    rm -f "out_dhdt_temp.nc"
                    continue
                fi
            fi

            echo "[$rel_dir] Adding metadata attributes to dhdt variable..." | tee -a "$LOG_FILE"
            ncatted -O -a units,dhdt,o,c,"m/yr" -a long_name,dhdt,o,c,"rate of thickness change" "$dhdt_file"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] WARNING: Failed to add attributes to dhdt, but calculation succeeded" | tee -a "$LOG_FILE"
            fi

            rm -f "out_dhdt_temp.nc"

            if [ ! -f "$dhdt_file" ]; then
                echo "[$rel_dir] ERROR: dhdt file was not created" | tee -a "$LOG_FILE"
                continue
            fi

            if ! ncdump -h "$dhdt_file" | grep -q "dhdt" ; then
                echo "[$rel_dir] ERROR: dhdt file was created but doesn't contain dhdt variable" | tee -a "$LOG_FILE"
                continue
            fi

            echo "[$rel_dir] Successfully created dhdt file for year $year" | tee -a "$LOG_FILE"
        fi
    done

    # Generate plots for five-year intervals with consistent colorbar using the Python script
    if [ ${#dhdt_files[@]} -gt 0 ]; then
        echo "[$rel_dir] Generating plots for five-year intervals with consistent colorbar..." | tee -a "$LOG_FILE"
        dhdt_files_str=$(IFS=','; echo "${dhdt_files[*]}")
        mesh_files_str=$(IFS=','; echo "${mesh_files[*]}")
        mesh_2000="$output_dir/output_flux_all_timesteps_2000_tAvg.nc"
        figure_base="$figures_dir/dhdt"

        echo "[$rel_dir] Running Python plotting script for five-year intervals..." | tee -a "$LOG_FILE"
        python3 "$PYTHON_SCRIPT_PATH" \
            -v dhdt \
            -r "$dhdt_files_str" \
            -m "$mesh_2000" \
            -m2 "$mesh_files_str" \
            -s "$figure_base" \
            --all_years "${all_years_to_process[*]}" 2>&1 | tee -a "$LOG_FILE"

        if [ $? -ne 0 ]; then
            echo "[$rel_dir] ERROR: Plotting script failed" | tee -a "$LOG_FILE"
            continue
        fi

        echo "[$rel_dir] Successfully created plots for five-year intervals" | tee -a "$LOG_FILE"
    else
        echo "[$rel_dir] No dhdt files were successfully created for five-year intervals" | tee -a "$LOG_FILE"
    fi

    echo "[$rel_dir] Successfully processed!" | tee -a "$LOG_FILE"
done

echo "=========================================" | tee -a "$LOG_FILE"
echo "Processing completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to $LOG_FILE"
exit 0
