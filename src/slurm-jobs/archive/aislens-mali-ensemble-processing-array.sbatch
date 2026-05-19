#!/bin/bash
#SBATCH  --job-name=mali-ensemble-process
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.mali-ensemble-process.%A_%a.out
#SBATCH  --error=err.mali-ensemble-process.%A_%a.err
#SBATCH  --time=24:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}
# EDIT the next line to the correct array range before submitting (see notes)
#SBATCH --array=0-9

set -euo pipefail
set -x

# Load modules and environment
module load anaconda3
conda activate mpas-analysis  # this loads nco and python environment

# --------------------
# Configuration (edit)
BASE_DATA_PARENT="${AISLENS_DATA_DIR}/MALI/ENSEMBLES"
BASE_FIGURES_PARENT="${AISLENS_DATA_DIR}/figures/MALI"
PYTHON_SCRIPT_PATH="${AISLENS_REPO}/src/MPAS-Tools/plot_ensemble_maps.py"
LOG_FILE="mali_ensemble_log_$(date +%Y%m%d_%H%M%S).txt"
MIN_YEAR_DIFF=2  # Minimum year difference to avoid division issues

# >>>>>>>>>>>> USER: Edit this list to restrict processing <<<<<<<<<<<<
# If non-empty: this defines the ordered list of parents that will be mapped to array indices.
# Example entries: "CTRL/CTRL-EM2-5PARAM" "CTRL/CTRL-EM4-5PARAM" ...
RELATIVE_PARENT_PATHS=(
    "CTRL/CTRL-EM0"
    "CTRL/CTRL-EM1"
    "CTRL/CTRL-EM2"
    "CTRL/CTRL-EM3"
    "CTRL/CTRL-EM4"
    "CTRL/CTRL-EM5"
    "CTRL/CTRL-EM6"
    "CTRL/CTRL-EM7"
    "CTRL/CTRL-EM8"
    "CTRL/CTRL-EM9"
)
# If RELATIVE_PARENT_PATHS is empty, the script will discover all matching parents under BASE_DATA_PARENT.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Variables to process for ensemble statistics
VARIABLES=("thickness") # e.g. ("thickness" "surfaceSpeed" "dhdt")

# Years to process (25 year increments from 2000)
START_YEAR=2000
END_YEAR=2030
YEAR_INCREMENT=5
YEARS=()
for ((year=$START_YEAR+$YEAR_INCREMENT; year<=$END_YEAR; year+=$YEAR_INCREMENT)); do
    YEARS+=($year)
done
# --------------------

echo "Starting MALI ensemble processing at $(date)" | tee -a "$LOG_FILE"
echo "Script run by user: $USER" | tee -a "$LOG_FILE"
echo "Processing years: ${YEARS[*]}" | tee -a "$LOG_FILE"
echo "Variables: ${VARIABLES[*]}" | tee -a "$LOG_FILE"

# Build list of candidate relative parent paths if not provided
parents=()
if [ "${#RELATIVE_PARENT_PATHS[@]}" -gt 0 ]; then
    parents=("${RELATIVE_PARENT_PATHS[@]}")
else
    # Discover directories under BASE_DATA_PARENT that contain an 'output' subdir;
    # convert to paths relative to BASE_DATA_PARENT
    while IFS= read -r -d $'\0' d; do
        rel=${d#"$BASE_DATA_PARENT"/}
        parents+=("$rel")
    done < <(find "${BASE_DATA_PARENT}" -mindepth 1 -maxdepth 3 -type d -name "output" -print0 | xargs -0 -n1 dirname -z)
fi

num_parents=${#parents[@]}
echo "Discovered ${num_parents} parent(s) to process" | tee -a "$LOG_FILE"

# Check SLURM_ARRAY_TASK_ID presence
if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array or set SBATCH --array accordingly." | tee -a "$LOG_FILE"
    exit 1
fi

idx=${SLURM_ARRAY_TASK_ID}
if [ "$idx" -ge "$num_parents" ] || [ "$idx" -lt 0 ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID ($idx) out of range [0..$((num_parents-1))]" | tee -a "$LOG_FILE"
    echo "Parents list (index:name):" | tee -a "$LOG_FILE"
    for i in "${!parents[@]}"; do
        echo "  $i : ${parents[$i]}" | tee -a "$LOG_FILE"
    done
    exit 2
fi

# Pick the parent for this array task
REL_PARENT="${parents[$idx]}"
PARENT_DIR="${BASE_DATA_PARENT}/${REL_PARENT}"
OUTPUT_DIR="${PARENT_DIR}/output"

echo "Array task $idx processing parent: ${REL_PARENT}" | tee -a "$LOG_FILE"
echo "Parent dir: ${PARENT_DIR}" | tee -a "$LOG_FILE"
echo "Output dir expected at: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: Expected output dir does not exist: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
    exit 3
fi

# Create ensemble statistics directory (per-parent)
ensemble_stats_dir="${BASE_FIGURES_PARENT}/${REL_PARENT}/ensemble_stats"
mkdir -p "$ensemble_stats_dir"

# Process this single parent (same logic as original script but scoped to one parent)
echo "=========================================" | tee -a "$LOG_FILE"
echo "Processing runs under parent ${REL_PARENT}..." | tee -a "$LOG_FILE"

# In the original script, run_dirs was a list of output dirs; here we process the single parent's output directory
run_dirs=("${OUTPUT_DIR}")

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

        # Add Time dimension to averaged files if needed (robust remove of xtime, preserve debug on failure)
        for file in "$ref_avg_file" "$avg_file"; do
            if [ -f "$file" ] && ! ncdump -h "$file" | grep -q "Time = "; then
                echo "Adding Time dimension to $(basename "$file")..." | tee -a "$LOG_FILE"
                temp_file="${file}.temp"

                if ncdump -h "$file" | grep -q "xtime"; then
                    # Try to create a copy without xtime
                    if ncks -C -x -v xtime "$file" "$temp_file" 2>>"$LOG_FILE"; then
                        echo "Removed xtime into $temp_file" | tee -a "$LOG_FILE"
                    else
                        echo "WARNING: ncks failed to remove xtime directly; attempting fallback" | tee -a "$LOG_FILE"
                        # fallback: copy then try remove on the copy
                        cp "$file" "$temp_file"
                        if ncks -C -x -v xtime "$temp_file" "${temp_file}.noxtime" 2>>"$LOG_FILE"; then
                            mv "${temp_file}.noxtime" "$temp_file"
                            echo "Fallback removal succeeded" | tee -a "$LOG_FILE"
                        else
                            echo "ERROR: unable to remove xtime from $file; preserving debug files and skipping Time-add" | tee -a "$LOG_FILE"
                            mkdir -p "${ensemble_stats_dir}/debug_${rel_dir}_${year}"
                            cp "$file" "${ensemble_stats_dir}/debug_${rel_dir}_${year}/"
                            [ -f temp_dhdt.nc ] && cp temp_dhdt.nc "${ensemble_stats_dir}/debug_${rel_dir}_${year}/"
                            echo "Saved debug files to ${ensemble_stats_dir}/debug_${rel_dir}_${year}" | tee -a "$LOG_FILE"
                            continue
                        fi
                    fi
                else
                    cp "$file" "$temp_file"
                fi

                # Run ncecat and capture stderr; if it fails, preserve debug files
                if ! ncecat -O -u Time "$temp_file" "$file" 2>>"$LOG_FILE"; then
                    echo "ERROR: ncecat failed when adding Time to $file; preserving debug files" | tee -a "$LOG_FILE"
                    mkdir -p "${ensemble_stats_dir}/debug_${rel_dir}_${year}"
                    mv "$temp_file" "${ensemble_stats_dir}/debug_${rel_dir}_${year}/"
                    [ -f temp_dhdt.nc ] && mv temp_dhdt.nc "${ensemble_stats_dir}/debug_${rel_dir}_${year}/"
                    echo "Saved debug files to ${ensemble_stats_dir}/debug_${rel_dir}_${year}" | tee -a "$LOG_FILE"
                    continue
                fi

                rm -f "$temp_file"
            fi
        done

        # Calculate dhdt if needed (robust sequence with checks and logging)
        if [ -f "$avg_file" ] && [ -f "$ref_avg_file" ] && [ ! -f "$dhdt_file" ]; then
            NYEARS=$((year - START_YEAR))
            if [ $NYEARS -ge $MIN_YEAR_DIFF ]; then
                echo "Calculating dhdt for year $year (${NYEARS} years)..." | tee -a "$LOG_FILE"

                # Create the diff file
                if ! ncdiff "$avg_file" "$ref_avg_file" "temp_dhdt.nc"; then
                    echo "ERROR: ncdiff failed for $avg_file vs $ref_avg_file" | tee -a "$LOG_FILE"
                    rm -f "temp_dhdt.nc"
                    continue
                fi

                # Verify thickness exists in the diff
                if ! ncdump -h "temp_dhdt.nc" 2>/dev/null | grep -q "thickness"; then
                    echo "ERROR: 'thickness' not found in temp_dhdt.nc — cannot compute dhdt" | tee -a "$LOG_FILE"
                    echo "--- temp_dhdt.nc header (first 200 lines) ---" >> "$LOG_FILE"
                    ncdump -h "temp_dhdt.nc" | sed -n '1,200p' >> "$LOG_FILE" 2>&1 || true
                    rm -f "temp_dhdt.nc"
                    continue
                fi

                # Compute dhdt -> write to temporary output then move into place
                tmp_out="${dhdt_file}.tmp"
                if ! ncap2 -O -s "dhdt=thickness/${NYEARS}; del(thickness)" "temp_dhdt.nc" "${tmp_out}" 2>>"$LOG_FILE"; then
                    echo "ERROR: ncap2 failed to create dhdt (see log)" | tee -a "$LOG_FILE"
                    echo "--- temp_dhdt.nc header (for debug) ---" >> "$LOG_FILE"
                    ncdump -h "temp_dhdt.nc" | sed -n '1,200p' >> "$LOG_FILE" 2>&1 || true
                    rm -f "temp_dhdt.nc" "${tmp_out}"
                    continue
                fi

                # Add metadata/attributes
                ncatted -O -a units,dhdt,o,c,"m/yr" -a long_name,dhdt,o,c,"rate of thickness change" "${tmp_out}" 2>>"$LOG_FILE" || true

                # Move into place atomically and cleanup
                mv -f "${tmp_out}" "$dhdt_file"
                rm -f "temp_dhdt.nc"

                # Add Time dimension to dhdt file if downstream expects it
                if ! ncdump -h "$dhdt_file" | grep -q "Time = "; then
                    ncecat -O -u Time "$dhdt_file" "${dhdt_file}.temp"
                    mv "${dhdt_file}.temp" "$dhdt_file"
                fi
            fi
        fi
    done

    echo "Completed processing run: $rel_dir" | tee -a "$LOG_FILE"
done

# Calculate ensemble statistics for this parent (across runs under it)
echo "=========================================" | tee -a "$LOG_FILE"
echo "Calculating ensemble statistics for parent ${REL_PARENT}..." | tee -a "$LOG_FILE"

for year in "${YEARS[@]}"; do
    echo "Processing ensemble statistics for year $year..." | tee -a "$LOG_FILE"
    stats_file="$ensemble_stats_dir/ensembleStats_${year}.nc"
    
    # Skip if stats file already exists and we're not forcing recreation
    if [ -f "$stats_file" ]; then
        echo "Ensemble stats file for year $year already exists, skipping creation..." | tee -a "$LOG_FILE"
        continue
    fi

    # Collect input files across runs (here just the single parent's output dir(s))
    input_files=()
    for output_dir in "${run_dirs[@]}"; do
        file_path="$output_dir/output_flux_all_timesteps_${year}_tAvg.nc"
        if [ -f "$file_path" ]; then
            input_files+=("$file_path")
        else
            echo "WARNING: Missing file: $file_path" | tee -a "$LOG_FILE"
        fi
    done

    if [ ${#input_files[@]} -lt 1 ]; then
        echo "WARNING: Not enough files for ensemble statistics under this parent (found ${#input_files[@]})" | tee -a "$LOG_FILE"
        continue
    fi

    # If there's only one run under a parent, produce a copy (mean=that run) so downstream plotting can still run.
    # Initialize stats file by copying the first available input file
    cp "${input_files[0]}" "$stats_file"
    echo "Initialized ensemble stats file with: ${input_files[0]}" | tee -a "$LOG_FILE"

    # If multiple variables, compute stats; otherwise copy/rename single-run var to *_mean
    for variable in "${VARIABLES[@]}"; do
        if [ ${#input_files[@]} -eq 1 ]; then
            # Single-run case: rename variable -> variable_mean and add to stats file
            tmp_mean="tmp_${variable}_mean.nc"
            ncrename -v "$variable","${variable}_mean" "${input_files[0]}"
            # Ensure the renamed variable is added (ncrename edited in-place if possible)
            ncks -A -v "${variable}_mean" "${input_files[0]}" "$stats_file" || true
            # continue to next variable
            continue
        fi

        # Multiple-runs case: compute min,max,mean,std,range similar to original script
        temp_work_dir="$ensemble_stats_dir/temp_${variable}_${year}"
        mkdir -p "$temp_work_dir"
        cd "$temp_work_dir" || { echo "ERROR: cannot enter $temp_work_dir" | tee -a "$LOG_FILE"; continue; }

        file_list=$(IFS=' '; echo "${input_files[*]}")
        tmp_min="tmp_${variable}_min.nc"
        tmp_max="tmp_${variable}_max.nc"
        tmp_mean="tmp_${variable}_mean.nc"
        tmp_std="tmp_${variable}_std.nc"
        tmp_range="tmp_${variable}_range.nc"

        ncra -O -y min -v "$variable" $file_list "$tmp_min"
        ncra -O -y max -v "$variable" $file_list "$tmp_max"
        ncra -O -y avg -v "$variable" $file_list "$tmp_mean"
        ncra -O -y rms -v "$variable" $file_list "tmp_rms.nc"
        ncra -O -y avg -v "$variable" $file_list "tmp_avg.nc"
        ncap2 -O -s "${variable}_std=sqrt(${variable}-${variable}*${variable})" "tmp_rms.nc" "$tmp_std"
        ncdiff -O "$tmp_max" "$tmp_min" "$tmp_range"

        ncrename -v "$variable","${variable}_mean" "$tmp_mean"
        ncrename -v "$variable","${variable}_min" "$tmp_min"
        ncrename -v "$variable","${variable}_max" "$tmp_max"
        ncrename -v "$variable","${variable}_range" "$tmp_range"

        # Remove original variable from stats file first (if it exists)
        if ncdump -h "$stats_file" | grep -q "$variable("; then
            ncks -x -v "$variable" -O "$stats_file" "${stats_file}.tmp"
            mv "${stats_file}.tmp" "$stats_file"
        fi

        ncks -A -v "${variable}_mean" "$tmp_mean" "$stats_file"
        ncks -A -v "${variable}_min" "$tmp_min" "$stats_file"
        ncks -A -v "${variable}_max" "$tmp_max" "$stats_file"
        ncks -A -v "${variable}_range" "$tmp_range" "$stats_file"
        ncks -A -v "${variable}_std" "$tmp_std" "$stats_file"

        # cleanup
        rm -f $tmp_min $tmp_max $tmp_mean $tmp_std $tmp_range tmp_rms.nc tmp_avg.nc
        cd "$ensemble_stats_dir"
        rm -rf "$temp_work_dir"
    done

    # remove xtime if present
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

# Plotting for this parent (if any stats files created)
echo "=========================================" | tee -a "$LOG_FILE"
echo "Preparing to call Python plotting script for parent ${REL_PARENT}..." | tee -a "$LOG_FILE"

# Gather created stats files
ensemble_stats_files=()
valid_years=()
for year in "${YEARS[@]}"; do
    stats_file="$ensemble_stats_dir/ensembleStats_${year}.nc"
    if [ -f "$stats_file" ]; then
        ensemble_stats_files+=("$stats_file")
        valid_years+=($year)
    fi
done

if [ ${#ensemble_stats_files[@]} -eq 0 ]; then
    echo "No ensemble stats files created for ${REL_PARENT}; skipping plotting." | tee -a "$LOG_FILE"
    exit 0
fi

stats_files_str=$(IFS=','; echo "${ensemble_stats_files[*]}")
years_str=$(IFS=','; echo "${valid_years[*]}")
variables_str=$(IFS=','; echo "${VARIABLES[*]}")
run_dirs_str="${OUTPUT_DIR}"  # single parent
run_names_str="$(basename "$PARENT_DIR")"
save_base="${BASE_FIGURES_PARENT}/${REL_PARENT}/ensemble/ensemble_maps"
mkdir -p "$(dirname "$save_base")"

python3 "$PYTHON_SCRIPT_PATH" \
    --ensemble_files "$stats_files_str" \
    --years "$years_str" \
    --variables "$variables_str" \
    --run_dirs "$run_dirs_str" \
    --run_names "$run_names_str" \
    --save_base "$save_base" \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "Successfully generated ensemble plots for ${REL_PARENT}" | tee -a "$LOG_FILE"
else
    echo "ERROR: Plotting script failed for ${REL_PARENT}" | tee -a "$LOG_FILE"
fi

echo "=========================================" | tee -a "$LOG_FILE"
echo "Ensemble processing for ${REL_PARENT} completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to $LOG_FILE"
exit 0
