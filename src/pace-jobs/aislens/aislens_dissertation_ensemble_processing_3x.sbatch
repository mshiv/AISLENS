#!/bin/bash
#SBATCH  --job-name=mali-ensemble-process-fast
#SBATCH  --account=gts-arobel3-atlas
#SBATCH  --nodes=2 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.mali-ensemble-process-fast.%A_%a.%j
#SBATCH  --error=err.mali-ensemble-process-fast.%A_%a.%j
#SBATCH  --time=24:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=smurugan9@gatech.edu
# Array job: one array index per target year (2010..2300 in 10-year increments).
# Set the sbatch array range to match the YEARS list used in the script
# (update this line if you change START_YEAR/END_YEAR/YEAR_INCREMENT)
#SBATCH  --array=0-29   # YEARS = 2010..2300 step 10 -> 30 entries

################################################################################
# faster ensemble processing
# - Uses local scratch for heavy I/O
# - Unique temp dirs to avoid collisions
# - Parallel per-run processing (background)
# - Correct std computation (rms^2 - mean^2)
# - Optionally supports region subsetting (disabled by default)
# - Supports SLURM array mode: one job per year if $SLURM_ARRAY_TASK_ID set
################################################################################

module load anaconda3
conda activate mpas-analysis

##### USER CONFIGURATION (edit as needed) ####################################
BASE_DATA_PARENT="/storage/home/hcoda1/6/smurugan9/scratch/AISLENS/data/MALI/ENSEMBLES"
BASE_FIGURES_PARENT="/storage/home/hcoda1/6/smurugan9/scratch/AISLENS/data/figures/MALI"
ENSEMBLE_DIR="SSP585-3X"
PYTHON_SCRIPT_PATH="/storage/home/hcoda1/6/smurugan9/data/dev/AISLENS/src/MPAS-Tools/plot_ensemble_maps.py"
GL_LINEWIDTH=0.85

# Performance / behavior toggles
# This script uses the shared filesystem for temporary work by default.
# Remove node-local scratch logic for clusters without /scratch access.
MAX_CONCURRENT_JOBS=8                  # per-node background concurrency for per-run tasks
FORCE_STATS=0                          # 1 to force recreation of ensemble stats even if present
PROCESS_PLOTTING=1                     # 1 to call Python plotting at the end

# Region subsetting (disabled by default). If enabled, set a subset bbox or indices
REGION_SUBSET=0                        # 0: full grid (default). 1: use subset below
# Example index bounds (uncomment/use when REGION_SUBSET=1)
#SUBSET_DIM="nCells"                  # or use relevant dimension name
#SUBSET_START=0
#SUBSET_END=99999

RELATIVE_PARENT_PATHS=(
    # SSP585-3X_00 is excluded from dissertation statistics because its
    # Shackleton--Denman transition coincides with concentrated Albany
    # non-convergence. Retain the raw run, but do not include it in ensemble means.
    "SSP585-3X/SSP585-3X_01"
    "SSP585-3X/SSP585-3X_02"
    "SSP585-3X/SSP585-3X_03"
    "SSP585-3X/SSP585-3X_04"
    "SSP585-3X/SSP585-3X_05"
    "SSP585-3X/SSP585-3X_06"
    "SSP585-3X/SSP585-3X_07"
    "SSP585-3X/SSP585-3X_08"
    "SSP585-3X/SSP585-3X_09"
)

VARIABLES=("thickness" "dhdt")

START_YEAR=2000
END_YEAR=2300
YEAR_INCREMENT=10
MIN_YEAR_DIFF=2

LOG_FILE="mali_ensemble_fast_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Starting optimized MALI ensemble processing at $(date)" | tee -a "$LOG_FILE"

###############################################################################
# Build YEARS array
YEARS=()
for ((year=$START_YEAR+$YEAR_INCREMENT; year<=$END_YEAR; year+=$YEAR_INCREMENT)); do
    YEARS+=($year)
done

# If this is an array job, limit YEARS to the single-year corresponding to SLURM_ARRAY_TASK_ID
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    idx=$SLURM_ARRAY_TASK_ID   # --array is 0-based; do NOT subtract 1
    if [ $idx -lt 0 ] || [ $idx -ge ${#YEARS[@]} ]; then
        echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range" | tee -a "$LOG_FILE"
        exit 2
    fi
    YEAR_TO_PROCESS=${YEARS[$idx]}
    echo "Array mode: processing single year $YEAR_TO_PROCESS (array id $SLURM_ARRAY_TASK_ID)" | tee -a "$LOG_FILE"
    YEARS=($YEAR_TO_PROCESS)
fi

echo "Years to process: ${YEARS[*]}" | tee -a "$LOG_FILE"

###############################################################################
# Find run output directories (same logic as original)
run_dirs=()
if [ "${#RELATIVE_PARENT_PATHS[@]}" -gt 0 ]; then
    for rel_path in "${RELATIVE_PARENT_PATHS[@]}"; do
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

echo "Found ${#run_dirs[@]} ensemble members" | tee -a "$LOG_FILE"

###############################################################################
# No node-local scratch on this cluster: use shared storage for temp work
LOCAL_SCRATCH=""

###############################################################################
# Helper: run up to $MAX_CONCURRENT_JOBS background jobs
function run_with_semaphore() {
    local maxjobs=$1; shift
    local cmd=("$@")
    while [ "$(jobs -rp | wc -l)" -ge "$maxjobs" ]; do sleep 1; done
    "${cmd[@]}" &
}

###############################################################################
# Per-run processing function (creates tAvg and dhdt for all YEARS for a single run)
process_run() {
    local output_dir="$1"
    local data_dir
    data_dir=$(dirname "$output_dir")
    local rel_dir=${data_dir#"$BASE_DATA_PARENT"/}
    echo "Processing run (fast): $rel_dir" | tee -a "$LOG_FILE"

    # --- reference-year time average -------------------------------------------
    # Every year's dhdt is differenced against START_YEAR, but in ARRAY mode each
    # task sees only its own year, so no task except the START_YEAR one would ever
    # create this file -- and that task races everyone else. Create it here in
    # every task, idempotently and atomically (write unique temp, then mv -n).
    local ref0_year_file="$output_dir/output_flux_all_timesteps_${START_YEAR}.nc"
    local ref0_avg_file="$output_dir/output_flux_all_timesteps_${START_YEAR}_tAvg.nc"
    if [ -f "$ref0_year_file" ] && [ ! -f "$ref0_avg_file" ]; then
        local tmpref="${ref0_avg_file}.tmp.$$.${SLURM_ARRAY_TASK_ID:-0}"
        echo "Creating REFERENCE time average ${START_YEAR} for ${rel_dir}..." | tee -a "$LOG_FILE"
        if ncwa -a Time -O "$ref0_year_file" "$tmpref" 2>>"$LOG_FILE"; then
            if ncdump -h "$tmpref" | grep -qi "xtime"; then
                ncks -x -v xtime -O "$tmpref" "${tmpref}.b" && mv "${tmpref}.b" "$tmpref"
            fi
            ncdump -h "$tmpref" | grep -q "Time = " || \
                { ncecat -O -u Time "$tmpref" "${tmpref}.b" && mv "${tmpref}.b" "$tmpref"; }
            mv -n "$tmpref" "$ref0_avg_file" 2>/dev/null
        fi
        rm -f "$tmpref" "${tmpref}.b"
    fi

    for year in "${YEARS[@]}"; do
        local year_file="$output_dir/output_flux_all_timesteps_${year}.nc"
        local avg_file="$output_dir/output_flux_all_timesteps_${year}_tAvg.nc"
        local ref_year_file="$ref0_year_file"
        local ref_avg_file="$ref0_avg_file"
        local dhdt_file="$output_dir/output_flux_all_timesteps_dhdt_$((year - START_YEAR))yr.nc"

        # Create tAvg locally then move back
        if [ -f "$year_file" ] && [ ! -f "$avg_file" ]; then
            echo "Creating time average for ${rel_dir} year ${year}..." | tee -a "$LOG_FILE"
            # Use shared filesystem (no node-local scratch available)
            # optional region subset: select variables or indices early to reduce I/O
            if [ "$REGION_SUBSET" -eq 1 ]; then
                ncks -d ${SUBSET_DIM},${SUBSET_START},${SUBSET_END} -O "$year_file" "sub_${year_file}" || true
                ncwa -a Time -O "sub_${year_file}" "$avg_file" || { echo "ncwa failed" | tee -a "$LOG_FILE"; rm -f "sub_${year_file}"; continue; }
                rm -f "sub_${year_file}"
            else
                ncwa -a Time -O "$year_file" "$avg_file" || { echo "ncwa failed" | tee -a "$LOG_FILE"; continue; }
            fi
        fi

        # Ensure Time dim exists for avg files (do locally if using scratch)
        for f in "$ref_avg_file" "$avg_file"; do
            if [ -f "$f" ] && ! ncdump -h "$f" | grep -q "Time = "; then
                echo "Adding Time dim to $(basename "$f")..." | tee -a "$LOG_FILE"
                # Use shared filesystem flow: remove xtime if present, then add Time dim with ncecat
                if ncdump -h "$f" | grep -qi "xtime"; then
                    ncks -x -v xtime -O "$f" "${f}.tmp" && mv "${f}.tmp" "$f"
                fi
                ncecat -O -u Time "$f" "${f}.tmp" && mv "${f}.tmp" "$f"
            fi
        done

        # Calculate dhdt (using local scratch to speed I/O)
        if [ -f "$avg_file" ] && [ -f "$ref_avg_file" ] && [ ! -f "$dhdt_file" ]; then
            NYEARS=$((year - START_YEAR))
            if [ $NYEARS -ge $MIN_YEAR_DIFF ]; then
                echo "Calculating dhdt for ${rel_dir} year $year ($NYEARS yrs)" | tee -a "$LOG_FILE"
                # Shared-fs flow: compute diff and dhdt in-place
                ncdiff -O "$avg_file" "$ref_avg_file" "temp_dhdt.nc" || continue
                ncap2 -O -s "dhdt=thickness/$NYEARS" "temp_dhdt.nc" "$dhdt_file" || { rm -f temp_dhdt.nc; continue; }
                ncatted -O -a units,dhdt,o,c,"m/yr" -a long_name,dhdt,o,c,"rate of thickness change" "$dhdt_file"
                rm -f "temp_dhdt.nc"
                if ! ncdump -h "$dhdt_file" | grep -q "Time = "; then
                    ncecat -O -u Time "$dhdt_file" "${dhdt_file}.temp" && mv "${dhdt_file}.temp" "$dhdt_file"
                fi
            fi
        fi
    done
}

###############################################################################
# Process runs in parallel (bounded concurrency)
echo "Starting per-run processing with max concurrency $MAX_CONCURRENT_JOBS" | tee -a "$LOG_FILE"
for output_dir in "${run_dirs[@]}"; do
    # background the job using run_with_semaphore wrapper
    run_with_semaphore $MAX_CONCURRENT_JOBS process_run "$output_dir"
done
wait
echo "Completed per-run processing" | tee -a "$LOG_FILE"

###############################################################################
# Ensemble stats per year (can be run in array-mode to do one year per job)
ensemble_stats_dir="$BASE_FIGURES_PARENT/$ENSEMBLE_DIR/ensemble_stats"
mkdir -p "$ensemble_stats_dir"

for year in "${YEARS[@]}"; do
    echo "Building ensemble stats for year $year" | tee -a "$LOG_FILE"
    stats_file="$ensemble_stats_dir/ensembleStats_${year}.nc"
    if [ -f "$stats_file" ] && [ "$FORCE_STATS" -eq 0 ]; then
        echo "Stats file exists and FORCE_STATS=0 -> skipping recreation: $stats_file" | tee -a "$LOG_FILE"
        # If plotting is enabled, still call the plotting script on the existing stats file
        if [ "$PROCESS_PLOTTING" -eq 1 ]; then
            echo "Plotting existing stats for year $year" | tee -a "$LOG_FILE"
            mkdir -p "$BASE_FIGURES_PARENT/$ENSEMBLE_DIR/ensemble_maps"
            python3 "$PYTHON_SCRIPT_PATH" \
                --ensemble_files "$stats_file" \
                --years "$year" \
                --gl_linewidth $GL_LINEWIDTH \
                --variables "$(IFS=,; echo "${VARIABLES[*]}")" \
                --run_dirs "$(IFS=,; echo "${run_dirs[*]}")" \
                --run_names "$(IFS=,; echo "${run_dirs[*]##*/}")" \
                --single_gl \
                --gl_run_index 0 \
                --save_base "$BASE_FIGURES_PARENT/$ENSEMBLE_DIR/ensemble_maps" \
                --log_cbar \
                2>&1 | tee -a "$LOG_FILE"
            if [ $? -eq 0 ]; then
                echo "Plotting succeeded for existing stats year $year" | tee -a "$LOG_FILE"
            else
                echo "ERROR: plotting failed for existing stats year $year" | tee -a "$LOG_FILE"
            fi
        fi
        continue
    fi

    # Create minimal base with mesh variables only (faster than copying full file)
    # Use a single comma-separated string for -v to avoid shell token-splitting issues
    mesh_vars="xCell,yCell,dcEdge,xEdge,yEdge"  # adjust as needed
    first_run_file="${run_dirs[0]}/output_flux_all_timesteps_${year}_tAvg.nc"
    if [ -f "$first_run_file" ]; then
        # Quick sanity check: ensure the candidate file is a valid netCDF
        if ! ncdump -h "$first_run_file" >/dev/null 2>&1; then
            echo "ERROR: first run file is not a valid netCDF: $first_run_file" | tee -a "$LOG_FILE"
            continue
        fi
        ncks -O -v "$mesh_vars" "$first_run_file" "$stats_file" || { echo "ERROR: failed to create base stats file" | tee -a "$LOG_FILE"; continue; }
        echo "Initialized stats base with mesh variables from $first_run_file" | tee -a "$LOG_FILE"
    else
        echo "ERROR: first run file not found: $first_run_file" | tee -a "$LOG_FILE"
        continue
    fi

    for variable in "${VARIABLES[@]}"; do
        echo "Processing variable $variable for year $year" | tee -a "$LOG_FILE"
        # place temporary work under ensemble_stats to avoid /scratch usage
        tmp_work_dir="$ensemble_stats_dir/tmp_${variable}_${year}_${SLURM_JOB_ID:-$$}"
        mkdir -p "$tmp_work_dir"
        pushd "$tmp_work_dir" > /dev/null || continue

        # Gather input files for this var/year
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
                echo "WARNING: missing file for var $variable: $file_path" | tee -a "$LOG_FILE"
            fi
        done

        if [ ${#input_files[@]} -lt 2 ]; then
            echo "WARNING: not enough files for ensemble stats for $variable (found ${#input_files[@]})" | tee -a "$LOG_FILE"
            popd > /dev/null
            rm -rf "$tmp_work_dir"
            continue
        fi

        file_list=$(IFS=' '; echo "${input_files[*]}")

        # Compute min, max, mean (ncra handles many files efficiently)
        ncra -O -y min -v "$variable" $file_list "tmp_min.nc" || { echo "ERROR: ncra -y min failed for $variable year $year" | tee -a "$LOG_FILE"; popd >/dev/null; rm -rf "$tmp_work_dir"; continue; }
        ncra -O -y max -v "$variable" $file_list "tmp_max.nc" || { echo "ERROR: ncra -y max failed for $variable year $year" | tee -a "$LOG_FILE"; popd >/dev/null; rm -rf "$tmp_work_dir"; continue; }
        ncra -O -y avg -v "$variable" $file_list "tmp_mean.nc" || { echo "ERROR: ncra -y avg failed for $variable year $year" | tee -a "$LOG_FILE"; popd >/dev/null; rm -rf "$tmp_work_dir"; continue; }

        # Compute std robustly: compute rms and mean, rename and merge, then std = sqrt(rms^2 - mean^2)
        ncra -O -y rms -v "$variable" $file_list "tmp_rms.nc"
        ncra -O -y avg -v "$variable" $file_list "tmp_mean_forstd.nc"
        # rename variables to distinct names
        ncrename -v "$variable","${variable}_rms" "tmp_rms.nc"
        ncrename -v "$variable","${variable}_mean" "tmp_mean_forstd.nc"
        # merge mean into rms file
        ncks -A "tmp_mean_forstd.nc" "tmp_rms.nc"
        # compute std (guard against tiny negative values from floating-point cancellation)
        # If the inner expression is negative (due to roundoff), set it to 0 before sqrt.
        # ncap2's `where` is a STATEMENT BLOCK, not a ternary function -- the old
        # form where(cond, a, b) is numpy/xarray syntax and simply fails, which is
        # why _std was the only statistic missing while mean/min/max/range worked.
        #
        # ncra -y rms and -y avg reduce over the N member files, so
        #     rms^2 - mean^2  =  E[x^2] - E[x]^2  =  POPULATION variance (/N).
        # Multiply by N/(N-1) to get the SAMPLE variance, matching ddof=1 in
        # freeze_results_table.py so map sigma and table sigma are the same quantity.
        NMEM=${#input_files[@]}
        ncap2 -O -s "v2=${variable}_rms*${variable}_rms-${variable}_mean*${variable}_mean; where(v2<0.0) v2=0.0; ${variable}_std=sqrt(v2*${NMEM}.0/(${NMEM}.0-1.0));" "tmp_rms.nc" "tmp_std.nc" \
            || { echo "ERROR: ncap2 std failed for $variable year $year" | tee -a "$LOG_FILE"; popd >/dev/null; rm -rf "$tmp_work_dir"; continue; }
        echo "  computed ${variable}_std from N=$NMEM members (sample, ddof=1)" | tee -a "$LOG_FILE"

        # Compute range = max - min
        ncdiff -O "tmp_max.nc" "tmp_min.nc" "tmp_range.nc"

        # Rename outputs to suffixes and add to stats file
        ncrename -v "$variable","${variable}_mean" "tmp_mean.nc"
        ncrename -v "$variable","${variable}_min" "tmp_min.nc"
        ncrename -v "$variable","${variable}_max" "tmp_max.nc"
        ncrename -v "$variable","${variable}_range" "tmp_range.nc"
        ncrename -v "${variable}_std","${variable}_std" "tmp_std.nc" || true

        # Remove variable from stats file first (if it exists)
        if ncdump -h "$stats_file" | grep -q "$variable("; then
            ncks -x -v "$variable" -O "$stats_file" "${stats_file}.tmp" && mv "${stats_file}.tmp" "$stats_file"
        fi

        # Append stats into main file
        ncks -A -v "${variable}_mean" "tmp_mean.nc" "$stats_file"
        ncks -A -v "${variable}_min" "tmp_min.nc" "$stats_file"
        ncks -A -v "${variable}_max" "tmp_max.nc" "$stats_file"
        ncks -A -v "${variable}_range" "tmp_range.nc" "$stats_file"
        ncks -A -v "${variable}_std" "tmp_std.nc" "$stats_file" \
            || echo "ERROR: could not append ${variable}_std for year $year" | tee -a "$LOG_FILE"

        # Cleanup temp workdir
        popd > /dev/null
        rm -rf "$tmp_work_dir"
    done

    # Remove xtime if present (keep file tidy)
    if ncdump -h "$stats_file" | grep -qi "xtime"; then
        echo "Removing xtime from $stats_file" | tee -a "$LOG_FILE"
        ncks -x -v xtime -O "$stats_file" "${stats_file}.tmp" && mv "${stats_file}.tmp" "$stats_file"
    fi

    # Verify mesh variables
    missing=()
    for v in xCell yCell dcEdge; do
        if ! ncdump -h "$stats_file" | grep -q "$v"; then missing+=("$v"); fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: mesh vars missing in $stats_file: ${missing[*]}" | tee -a "$LOG_FILE"
    else
        echo "Mesh variables present in $stats_file" | tee -a "$LOG_FILE"
    fi

    # Optionally call plotting script (pass the single-year stats file so python need not change)
    if [ "$PROCESS_PLOTTING" -eq 1 ]; then
        echo "Calling Python plotting script for year $year" | tee -a "$LOG_FILE"
        python3 "$PYTHON_SCRIPT_PATH" \
            --ensemble_files "$stats_file" \
            --years "$year" \
            --gl_linewidth $GL_LINEWIDTH \
            --variables "$(IFS=,; echo "${VARIABLES[*]}")" \
            --run_dirs "$(IFS=,; echo "${run_dirs[*]}")" \
            --run_names "$(IFS=,; echo "${run_dirs[*]##*/}")" \
            --single_gl \
	    --gl_run_index 0 \
	    --save_base "$BASE_FIGURES_PARENT/$ENSEMBLE_DIR/ensemble_maps" \
            --log_cbar \
            2>&1 | tee -a "$LOG_FILE"
        if [ $? -eq 0 ]; then
            echo "Plotting succeeded for year $year" | tee -a "$LOG_FILE"
        else
            echo "ERROR: plotting failed for year $year" | tee -a "$LOG_FILE"
        fi
    fi
done

echo "All done at $(date)" | tee -a "$LOG_FILE"
exit 0
