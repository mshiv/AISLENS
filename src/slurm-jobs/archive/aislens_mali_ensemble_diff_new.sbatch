#!/bin/bash
#SBATCH --job-name=mali-ensemble-stats
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.mali-ensemble-stats.%j
#SBATCH --error=err.mali-ensemble-stats.%j
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

set -e

module load anaconda3
conda activate mpas-analysis

BASE_DATA_PARENT="${AISLENS_DATA_DIR}/MALI/ENSEMBLES"
BASE_FIGURES_PARENT="${AISLENS_DATA_DIR}/figures/MALI/ENSEMBLES/CTRL-SSN"
PYTHON_SCRIPT_PATH="${AISLENS_REPO}/src/MPAS-Tools/plot_ensembleDiffs.py"
LOG_FILE="mali_ensemble_log_$(date +%Y%m%d_%H%M%S).txt"

# ---- USER: SET YOUR RUN LIST HERE ----
run_dirs=(CTRL-SSN/CTRL00 CTRL-SSN/CTRL01 CTRL-SSN/CTRL02 CTRL-SSN/CTRL03 CTRL-SSN/CTRL04 CTRL-SSN/CTRL05 CTRL-SSN/CTRL06 CTRL-SSN/CTRL07 CTRL-SSN/CTRL08 CTRL-SSN/CTRL09)  # Just subdir names
# --------------------------------------

# ---- USER: SET VARIABLES AND YEARS ----
VARIABLES=("thickness") # "surfaceSpeed" "dhdt")
INCREMENT=25
START_YEAR=2000
LAST_YEAR=2250
# ---------------------------------------

echo "Starting MALI ensemble stats and maps at $(date)" | tee -a "$LOG_FILE"
echo "Run directories to process: ${run_dirs[*]}" | tee -a "$LOG_FILE"

# Find available years from first run
years=()
for ((y=$START_YEAR; y<=$LAST_YEAR; y+=$INCREMENT)); do
    exists=0
    for run in "${run_dirs[@]}"; do
        if [ -f "$BASE_DATA_PARENT/$run/output/output_flux_all_timesteps_${y}.nc" ]; then
            exists=1
            break
        fi
    done
    if [ $exists -eq 1 ]; then
        years+=($y)
    fi
done
echo "Years to process: ${years[*]}" | tee -a "$LOG_FILE"

# For each run and year, create tAvg and dhdt if needed
for run in "${run_dirs[@]}"; do
    echo "Processing run $run" | tee -a "$LOG_FILE"
    outdir="$BASE_DATA_PARENT/$run/output"
    cd "$outdir" || { echo "Failed to enter $outdir" | tee -a "$LOG_FILE"; continue; }
    # Make tAvg files
    for year in "${years[@]}"; do
        src="output_flux_all_timesteps_${year}.nc"
        tavg="output_flux_all_timesteps_${year}_tAvg.nc"
        if [ ! -f "$tavg" ] && [ -f "$src" ]; then
            echo "Creating $tavg from $src" | tee -a "$LOG_FILE"
            ncwa -a Time -O "$src" "$tavg"
            # Remove xtime
            if ncdump -h "$tavg" | grep -q "xtime"; then
                ncks -x -v xtime -O "$tavg" "${tavg}.tmp" && mv "${tavg}.tmp" "$tavg"
            fi
            # Ensure Time dim
            if ! ncdump -h "$tavg" | grep -q "Time = "; then
                ncecat -O -u Time "$tavg" "${tavg}.tmp" && mv "${tavg}.tmp" "$tavg"
            fi
        fi
    done
    # Make dhdt files for each year (except first year)
    for i in $(seq 1 $((${#years[@]}-1))); do
        year=${years[i]}
        prev_year=${years[0]}
        NYEARS=$((year - prev_year))
        dhdt_file="output_flux_all_timesteps_dhdt_${NYEARS}yr.nc"
        tavg_start="output_flux_all_timesteps_${prev_year}_tAvg.nc"
        tavg_end="output_flux_all_timesteps_${year}_tAvg.nc"
        if [ ! -f "$dhdt_file" ] && [ -f "$tavg_start" ] && [ -f "$tavg_end" ]; then
            echo "Creating $dhdt_file for $run ($prev_year to $year)" | tee -a "$LOG_FILE"
            ncdiff "$tavg_end" "$tavg_start" "dhdt_diff_tmp.nc"
            ncap2 -O -s "dhdt=thickness/$NYEARS" "dhdt_diff_tmp.nc" "$dhdt_file"
            ncatted -O -a units,dhdt,o,c,"m/yr" -a long_name,dhdt,o,c,"rate of thickness change" "$dhdt_file"
            rm -f "dhdt_diff_tmp.nc"
            # Remove xtime, ensure Time dim
            if ncdump -h "$dhdt_file" | grep -q "xtime"; then
                ncks -x -v xtime -O "$dhdt_file" "${dhdt_file}.tmp" && mv "${dhdt_file}.tmp" "$dhdt_file"
            fi
            if ! ncdump -h "$dhdt_file" | grep -q "Time = "; then
                ncecat -O -u Time "$dhdt_file" "${dhdt_file}.tmp" && mv "${dhdt_file}.tmp" "$dhdt_file"
            fi
        fi
    done
done

# Ensemble statistics for each year
ensemble_stats_dir="${BASE_DATA_PARENT}/ensemble_stats"
mkdir -p "$ensemble_stats_dir"
for year in "${years[@]}"; do
    echo "Calculating ensemble stats for year $year" | tee -a "$LOG_FILE"
    files=()
    dhdt_files=()
    for run in "${run_dirs[@]}"; do
        files+=("$BASE_DATA_PARENT/$run/output/output_flux_all_timesteps_${year}_tAvg.nc")
        if [ "$year" -eq "${years[0]}" ]; then
            dhdt_files+=("")
        else
            NYEARS=$((year - ${years[0]}))
            dhdt_files+=("$BASE_DATA_PARENT/$run/output/output_flux_all_timesteps_dhdt_${NYEARS}yr.nc")
        fi
    done
    stats_file="${ensemble_stats_dir}/ensembleStats_${year}.nc"
    # For each variable, stack the data and calculate stats
    tmp_min="tmp_min.nc"
    tmp_max="tmp_max.nc"
    tmp_mean="tmp_mean.nc"
    tmp_std="tmp_std.nc"
    tmp_range="tmp_range.nc"
    rm -f $tmp_min $tmp_max $tmp_mean $tmp_std $tmp_range $stats_file

    # For non-dhdt variables
    for var in thickness surfaceSpeed; do
        input_files=()
        for f in "${files[@]}"; do
            if ncdump -h "$f" | grep -qw "$var"; then
                input_files+=("$f")
            fi
        done
        if [ ${#input_files[@]} -gt 0 ]; then
            ncra -O -v $var "${input_files[@]}" $tmp_mean
            ncra -O -y min -v $var "${input_files[@]}" $tmp_min
            ncra -O -y max -v $var "${input_files[@]}" $tmp_max
            ncdiff -O $tmp_max $tmp_min $tmp_range
            ncra -O -y rms -v $var "${input_files[@]}" tmp_rms.nc
            ncra -O -y avg -v $var "${input_files[@]}" tmp_avg.nc
            # Rename variables so ncap2 can see <var>_rms and <var>_avg in the same file
            if ncdump -h tmp_rms.nc | grep -qw "${var}"; then
                ncrename -v ${var},${var}_rms tmp_rms.nc
            fi
            if ncdump -h tmp_avg.nc | grep -qw "${var}"; then
                ncrename -v ${var},${var}_avg tmp_avg.nc
            fi
            # Merge avg into rms file so both variables are present
            ncks -A tmp_avg.nc tmp_rms.nc
            # Now compute std using the renamed variables
            ncap2 -O -s "${var}_std=sqrt(${var}_rms-${var}_avg*${var}_avg)" tmp_rms.nc $tmp_std
            ncrename -v $var,${var}_mean $tmp_mean
            ncrename -v $var,${var}_min $tmp_min
            ncrename -v $var,${var}_max $tmp_max
            ncrename -v $var,${var}_range $tmp_range
            # tmp_std already contains variable ${var}_std from ncap2; no rename needed
            cp $tmp_mean $stats_file
            ncks -A -v ${var}_min $tmp_min $stats_file
            ncks -A -v ${var}_max $tmp_max $stats_file
            ncks -A -v ${var}_range $tmp_range $stats_file
            ncks -A -v ${var}_std $tmp_std $stats_file
            rm -f $tmp_min $tmp_max $tmp_mean $tmp_std $tmp_range tmp_rms.nc tmp_avg.nc
        fi
    done
    # dhdt variable:
    if [ "$year" -ne "${years[0]}" ]; then
        input_files=()
        for f in "${dhdt_files[@]}"; do
            if [ -f "$f" ] && ncdump -h "$f" | grep -qw "dhdt"; then
                input_files+=("$f")
            fi
        done
        if [ ${#input_files[@]} -gt 0 ]; then
            ncra -O -v dhdt "${input_files[@]}" $tmp_mean
            ncra -O -y min -v dhdt "${input_files[@]}" $tmp_min
            ncra -O -y max -v dhdt "${input_files[@]}" $tmp_max
            ncdiff -O $tmp_max $tmp_min $tmp_range
            ncra -O -y rms -v dhdt "${input_files[@]}" tmp_rms.nc
            ncra -O -y avg -v dhdt "${input_files[@]}" tmp_avg.nc
            # Rename and merge so dhtd_rms and dhdt_avg exist together
            if ncdump -h tmp_rms.nc | grep -qw "dhdt"; then
                ncrename -v dhdt,dhdt_rms tmp_rms.nc
            fi
            if ncdump -h tmp_avg.nc | grep -qw "dhdt"; then
                ncrename -v dhdt,dhdt_avg tmp_avg.nc
            fi
            ncks -A tmp_avg.nc tmp_rms.nc
            ncap2 -O -s "dhdt_std=sqrt(dhdt_rms-dhdt_avg*dhdt_avg)" tmp_rms.nc $tmp_std
            ncrename -v dhdt,dhdt_mean $tmp_mean
            ncrename -v dhdt,dhdt_min $tmp_min
            ncrename -v dhdt,dhdt_max $tmp_max
            ncrename -v dhdt,dhdt_range $tmp_range
            # tmp_std already contains variable dhdt_std from ncap2; no rename needed
            ncks -A -v dhdt_mean $tmp_mean $stats_file
            ncks -A -v dhdt_min $tmp_min $stats_file
            ncks -A -v dhdt_max $tmp_max $stats_file
            ncks -A -v dhdt_range $tmp_range $stats_file
            ncks -A -v dhdt_std $tmp_std $stats_file
            rm -f $tmp_min $tmp_max $tmp_mean $tmp_std $tmp_range tmp_rms.nc tmp_avg.nc
        fi
    fi
    # Remove xtime/ensure Time dim for stats_file
    if ncdump -h "$stats_file" | grep -q "xtime"; then
        ncks -x -v xtime -O "$stats_file" "${stats_file}.tmp" && mv "${stats_file}.tmp" "$stats_file"
    fi
    if ! ncdump -h "$stats_file" | grep -q "Time = "; then
        ncecat -O -u Time "$stats_file" "${stats_file}.tmp" && mv "${stats_file}.tmp" "$stats_file"
    fi
done

# Gather mesh files for ensemble GL overlays (one per run, for reference)
mesh_files=()
for run in "${run_dirs[@]}"; do
    mesh_files+=("$BASE_DATA_PARENT/$run/output/output_flux_all_timesteps_${START_YEAR}_tAvg.nc")
done
mesh_files_str=$(IFS=','; echo "${mesh_files[*]}")

# Call plotting script
ensemble_stats_files=()
for year in "${years[@]}"; do
    ensemble_stats_files+=("${ensemble_stats_dir}/ensembleStats_${year}.nc")
done
stats_files_str=$(IFS=','; echo "${ensemble_stats_files[*]}")
years_str=$(IFS=','; echo "${years[*]}")
variables_str=$(IFS=','; echo "${VARIABLES[*]}")
save_base="$BASE_FIGURES_PARENT/ensemble"

python3 "$PYTHON_SCRIPT_PATH" \
    --ensemble_files "$stats_files_str" \
    --years "$years_str" \
    --variables "$variables_str" \
    --mesh_files "$mesh_files_str" \
    --run_names "$(IFS=','; echo "${run_dirs[*]}")" \
    --save_base "$save_base" \
    2>&1 | tee -a "$LOG_FILE"

echo "Processing completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to $LOG_FILE"
exit 0

