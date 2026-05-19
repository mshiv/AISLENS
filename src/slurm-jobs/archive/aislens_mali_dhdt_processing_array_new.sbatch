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
BASE_DATA_PARENT="${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP126"
BASE_FIGURES_PARENT="${AISLENS_DATA_DIR}/figures/MALI/ENSEMBLES/SSP126"
PYTHON_SCRIPT_PATH="${AISLENS_REPO}/src/MPAS-Tools/plot_output_maps_masked.py"
LOG_FILE="mali_process_log_$(date +%Y%m%d_%H%M%S).txt"
MIN_YEAR_DIFF=2  # Minimum year difference to avoid division issues

# Initialize log
echo "Starting MALI processing at $(date)" | tee -a "$LOG_FILE"
echo "Script run by user: $USER" | tee -a "$LOG_FILE"

# Optional target year from command line
TARGET_YEAR=""
# Save original args for possible re-submission
ORIGINAL_ARGS=("$@")

set -euo pipefail
set -x

# --------- CLI / array helper options ----------
print_usage() {
    cat <<EOF
Usage: $0 [--dirs DIR1,DIR2,...] [--dir-file FILE] [--array-index N] [--no-resubmit] [TARGET_YEAR]

Options:
  --dirs        Comma-separated list of data parent directories (or their 'output' subdirs)
  --dir-file    File containing one directory per line
  --array-index Explicit index to process (overrides SLURM_ARRAY_TASK_ID)
  --no-resubmit Do not auto-submit an sbatch array; run sequentially in this process
  TARGET_YEAR   Optional positional argument used as target year (existing behavior)

Behavior:
  - If neither --dirs nor --dir-file is provided the script will auto-discover all 'output' dirs
    under BASE_DATA_PARENT and will re-submit itself as an sbatch array (one task per directory).
  - If run as an array task (SLURM_ARRAY_TASK_ID set) the task will process exactly one directory.
  - Use --array-index to override which directory index to run when testing interactively.
EOF
}

# Parse simple CLI flags (positional TARGET_YEAR still supported)
DIRS_CSV=""
DIR_FILE=""
ARRAY_INDEX=""
NO_RESUBMIT=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dirs)
            DIRS_CSV="$2"
            shift 2
            ;;
        --dir-file)
            DIR_FILE="$2"
            shift 2
            ;;
        --array-index)
            ARRAY_INDEX="$2"
            shift 2
            ;;
        --no-resubmit)
            NO_RESUBMIT=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -* )
            echo "Unknown option: $1" >&2; print_usage; exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Restore positional TARGET_YEAR if provided
if [ ${#POSITIONAL[@]} -ge 1 ]; then
    TARGET_YEAR="${POSITIONAL[0]}"
fi

# Resolve directories according to CLI or discovery
resolve_run_dirs() {
    local -n _out_arr=$1
    _out_arr=()

    if [ -n "$DIRS_CSV" ]; then
        IFS=',' read -ra supplied <<< "$DIRS_CSV"
        for d in "${supplied[@]}"; do
            # Accept either data parent or explicit output dir
            if [ -d "$d" ]; then
                if [ "$(basename "$d")" = "output" ]; then
                    _out_arr+=("$d")
                elif [ -d "$d/output" ]; then
                    _out_arr+=("$d/output")
                else
                    echo "Warning: provided dir exists but no 'output' subdir found: $d"
                fi
            else
                echo "Warning: provided dir does not exist: $d"
            fi
        done
        return 0
    fi

    if [ -n "$DIR_FILE" ]; then
        if [ ! -f "$DIR_FILE" ]; then
            echo "Directory file not found: $DIR_FILE"; return 1
        fi
        while IFS= read -r line || [ -n "$line" ]; do
            [ -z "$line" ] && continue
            if [ -d "$line" ]; then
                if [ "$(basename "$line")" = "output" ]; then
                    _out_arr+=("$line")
                elif [ -d "$line/output" ]; then
                    _out_arr+=("$line/output")
                else
                    echo "Warning: line refers to dir but no 'output' subdir found: $line"
                fi
            else
                echo "Warning: line refers to non-existing dir: $line"
            fi
        done < "$DIR_FILE"
        return 0
    fi

    # Default discovery
    while IFS= read -r outdir; do
        _out_arr+=("$outdir")
    done < <(find "$BASE_DATA_PARENT" -type d -name "output" -print)
}

# Build run_dirs array
run_dirs=()
if ! resolve_run_dirs run_dirs; then
    echo "Failed to resolve directories"; exit 2
fi

num_dirs=${#run_dirs[@]}
if [ $num_dirs -eq 0 ]; then
    echo "No directories found under $BASE_DATA_PARENT (or via --dirs/--dir-file)"; exit 0
fi

# If not running inside an array task and user didn't ask to run sequentially, re-submit
if [ -z "${SLURM_ARRAY_TASK_ID+x}" ] && [ -z "$ARRAY_INDEX" ] && [ $NO_RESUBMIT -eq 0 ]; then
    echo "Found $num_dirs directories. Will submit array 0..$((num_dirs-1))."
    printf '%3s  %s\n' "IDX" "DIRECTORY"
    for i in "${!run_dirs[@]}"; do
        printf '%3d: %s\n' "$i" "${run_dirs[$i]}"
    done
    echo "Submitting sbatch --array=0-$((num_dirs-1)) $0 -- ${ORIGINAL_ARGS[*]}"
    sbatch --array=0-$((num_dirs-1)) "$0" -- "${ORIGINAL_ARGS[@]}"
    echo "Submission complete; exiting wrapper process."
    exit 0
fi

# Determine which single index to run (if any). Prefer explicit --array-index over SLURM_ARRAY_TASK_ID
SELECT_INDEX=""
if [ -n "$ARRAY_INDEX" ]; then
    SELECT_INDEX=$ARRAY_INDEX
elif [ -n "${SLURM_ARRAY_TASK_ID+x}" ]; then
    SELECT_INDEX=${SLURM_ARRAY_TASK_ID}
fi

if [ -n "$SELECT_INDEX" ]; then
    if [ "$SELECT_INDEX" -lt 0 ] || [ "$SELECT_INDEX" -ge "$num_dirs" ]; then
        echo "Requested index ($SELECT_INDEX) out of range [0..$((num_dirs-1))]"; exit 2
    fi
    # Narrow run_dirs to single entry
    run_dirs=("${run_dirs[$SELECT_INDEX]}")
    echo "Array/task mode: processing index $SELECT_INDEX -> ${run_dirs[0]}"
else
    echo "Running sequentially over ${num_dirs} directories (no array index provided)"
fi

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
    
    # 3. Ensure t-averaged files have a proper Time record dimension and NO xtime
    #    Strategy: immediately strip any 'xtime' variable from the t-averaged files
    #    then use ncecat -u Time to add a Time record dimension (clean, single-step).
    for mesh_file in "$mesh_2000" "$mesh_end"; do
        # If Time already exists, still remove xtime if present so no xtime remains
        if ncdump -h "$mesh_file" | grep -q "xtime"; then
            echo "[$rel_dir] Stripping xtime from $(basename \"$mesh_file\")" | tee -a "$LOG_FILE"
            tmp_no_xtime="${mesh_file}.no_xtime"
            ncks -x -v xtime -O "$mesh_file" "$tmp_no_xtime"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] ERROR: Failed to remove xtime from $(basename \"$mesh_file\")" | tee -a "$LOG_FILE"
                rm -f "$tmp_no_xtime"
                continue 2
            fi
            mv "$tmp_no_xtime" "$mesh_file"
        fi

        # If Time dimension not present, add it with ncecat (safe: file now has no xtime)
        if ! ncdump -h "$mesh_file" | grep -q "Time = "; then
            echo "[$rel_dir] Adding Time record dimension to $(basename \"$mesh_file\") using ncecat" | tee -a "$LOG_FILE"
            tmp_for_ncecat="${mesh_file}.for_ncecat"
            # Work on a temporary copy to avoid in-place surprises
            cp "$mesh_file" "$tmp_for_ncecat"
            ncecat -O -u Time "$tmp_for_ncecat" "$mesh_file"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] ERROR: ncecat failed for $(basename \"$mesh_file\"); preserving debug copy" | tee -a "$LOG_FILE"
                debug_dir="${output_dir}/debug_time_fix_$(date +%Y%m%d_%H%M%S)"
                mkdir -p "$debug_dir"
                mv "$tmp_for_ncecat" "$debug_dir/"
                continue 2
            fi
            rm -f "$tmp_for_ncecat"
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
        
        # Step 1: Create a dhdt variable into a temp file (keep original temp intact)
        echo "[$rel_dir] Creating dhdt variable (temporary file)..." | tee -a "$LOG_FILE"
        tmp_dhdt_with_vars="out_dhdt_withvars.nc"
        ncap2 -O -s "dhdt=thickness/$NYEARS" "out_dhdt_temp.nc" "$tmp_dhdt_with_vars" 2>&1 | tee -a "$LOG_FILE"
        if [ $? -ne 0 ]; then
            echo "[$rel_dir] ERROR: Failed to calculate dhdt with ncap2 into temp file" | tee -a "$LOG_FILE"
            echo "[$rel_dir] Trying alternative calculation expression..." | tee -a "$LOG_FILE"
            ncap2 -O -s "dhdt=thickness*1.0/$NYEARS" "out_dhdt_temp.nc" "$tmp_dhdt_with_vars" 2>&1 | tee -a "$LOG_FILE"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] ERROR: Both calculation methods failed" | tee -a "$LOG_FILE"
                rm -f "out_dhdt_temp.nc" "$tmp_dhdt_with_vars"
                continue
            fi
        fi

        # Step 2: Extract a minimal dhdt-only file to avoid unrelated variables (e.g., angleEdge)
        echo "[$rel_dir] Extracting dhdt-only file to avoid incompatible variables..." | tee -a "$LOG_FILE"
        ncks -O -v dhdt "$tmp_dhdt_with_vars" "$dhdt_file" 2>&1 | tee -a "$LOG_FILE"
        if [ $? -ne 0 ]; then
            echo "[$rel_dir] ERROR: Failed to extract dhdt variable into final file" | tee -a "$LOG_FILE"
            rm -f "out_dhdt_temp.nc" "$tmp_dhdt_with_vars"
            continue
        fi

        # Step 3: Append mesh variables from the end-year mesh into the dhdt file
        # so plotting scripts that expect mesh variables (angleEdge, etc.) have them.
        # Exclude the thickness variable to avoid conflicts.
        if [ -f "$mesh_end" ]; then
            echo "[$rel_dir] Appending mesh variables from $mesh_end into $dhdt_file" | tee -a "$LOG_FILE"
            ncks -A -x -v thickness "$mesh_end" "$dhdt_file" 2>&1 | tee -a "$LOG_FILE"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] WARNING: Failed to append mesh variables from $mesh_end into $dhdt_file" | tee -a "$LOG_FILE"
            fi
        else
            echo "[$rel_dir] WARNING: mesh_end file not found; plot may fail if mesh vars missing" | tee -a "$LOG_FILE"
        fi

        # Add attributes separately (less likely to fail)
        echo "[$rel_dir] Adding metadata attributes to dhdt variable..." | tee -a "$LOG_FILE"
        ncatted -O -a units,dhdt,o,c,"m/yr" -a long_name,dhdt,o,c,"rate of thickness change" "$dhdt_file" 2>&1 | tee -a "$LOG_FILE" || echo "[$rel_dir] WARNING: Failed to add attributes to dhdt, but calculation succeeded" | tee -a "$LOG_FILE"

        # Ensure dhdt file has no xtime and has a Time record dimension (use ncecat)
        if ncdump -h "$dhdt_file" | grep -q "xtime"; then
            echo "[$rel_dir] Stripping xtime from $(basename \"$dhdt_file\") before adding Time" | tee -a "$LOG_FILE"
            tmp_no_xtime="${dhdt_file}.no_xtime"
            ncks -x -v xtime -O "$dhdt_file" "$tmp_no_xtime"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] ERROR: Failed to remove xtime from $(basename \"$dhdt_file\")" | tee -a "$LOG_FILE"
                rm -f "$tmp_no_xtime" "$tmp_dhdt_with_vars" "out_dhdt_temp.nc"
                continue
            fi
            mv "$tmp_no_xtime" "$dhdt_file"
        fi

        # Add Time record dimension to dhdt (if missing) using the dhdt-only file
        if ! ncdump -h "$dhdt_file" | grep -q "Time = "; then
            echo "[$rel_dir] Adding Time record dimension to $(basename \"$dhdt_file\") using ncecat" | tee -a "$LOG_FILE"
            tmp_for_ncecat="${dhdt_file}.for_ncecat"
            cp "$dhdt_file" "$tmp_for_ncecat"
            ncecat -O -u Time "$tmp_for_ncecat" "$dhdt_file" 2>&1 | tee -a "$LOG_FILE"
            if [ $? -ne 0 ]; then
                echo "[$rel_dir] ERROR: ncecat failed for dhdt; preserving debug copy" | tee -a "$LOG_FILE"
                debug_dir="${output_dir}/debug_dhdt_timefix_$(date +%Y%m%d_%H%M%S)"
                mkdir -p "$debug_dir"
                mv "$tmp_for_ncecat" "$debug_dir/"
                rm -f "out_dhdt_temp.nc" "$tmp_dhdt_with_vars"
                continue
            fi
            rm -f "$tmp_for_ncecat"
        fi

        # Clean up
        rm -f "out_dhdt_temp.nc" "$tmp_dhdt_with_vars"

        # Verify dhdt file exists and contains dhdt variable
        if [ ! -f "$dhdt_file" ]; then
            echo "[$rel_dir] ERROR: dhdt file was not created" | tee -a "$LOG_FILE"
            continue
        fi

        if ! ncdump -h "$dhdt_file" | grep -q "dhdt" ; then
            echo "[$rel_dir] ERROR: dhdt file was created but doesn't contain dhdt variable" | tee -a "$LOG_FILE"
            continue
        fi

        echo "[$rel_dir] Successfully created dhdt file (dhdt-only, no xtime, Time record present)" | tee -a "$LOG_FILE"
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

