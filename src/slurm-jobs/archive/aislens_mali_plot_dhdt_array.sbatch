#!/bin/bash
#SBATCH --job-name=mali-plotting-array
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.mali-plotting.%A_%a.out
#SBATCH --error=err.mali-plotting.%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}
# NOTE: no --array here; the script will re-submit itself as an array when run.

set -euo pipefail
set -x

module load anaconda3
conda activate mpas-analysis   # ensure this env contains nco and python

# ------------ USER CONFIG -------------
BASE_DATA_PATH="${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP585"
BASE_FIGURES_PATH="${AISLENS_DATA_DIR}/figures/MALI/ENSEMBLES/SSP585"
PYTHON_SCRIPT_PATH="${AISLENS_REPO}/src/MPAS-Tools/plot_output_maps_masked.py"
# --------------------------------------

# functions copied from your original script
find_data_directories() {
    local base_path="$1"
    find "$base_path" -type d -name "output" | while read output_dir; do
        parent_dir=$(dirname "$output_dir")
        if ls "$output_dir"/output_flux_all_timesteps_dhdt_*yr.nc >/dev/null 2>&1; then
            echo "$parent_dir"
        fi
    done
}

get_end_year() {
    local output_dir="$1"
    local years=($(ls "$output_dir"/output_flux_all_timesteps_2*.nc 2>/dev/null | \
                   sed 's/.*output_flux_all_timesteps_\([0-9]\+\)\.nc/\1/' | \
                   sort -n))
    if [ ${#years[@]} -lt 2 ]; then
        echo ""
        return 1
    fi
    echo "${years[-2]}"
}

create_figure_dir() {
    local data_dir="$1"
    local figure_dir="${data_dir/$BASE_DATA_PATH/$BASE_FIGURES_PATH}"
    mkdir -p "$figure_dir"
    echo "$figure_dir"
}

process_directory() {
    local data_dir="$1"
    local output_dir="$data_dir/output"
    echo "Processing directory: $data_dir"
    local end_year
    end_year=$(get_end_year "$output_dir")
    if [ -z "$end_year" ]; then
        echo "Error: Could not determine END_YEAR for $data_dir"; return 1
    fi
    local nyears=$((end_year - 2000))
    echo "  END_YEAR: $end_year, NYEARS: $nyears"
    local dhdt_file="$output_dir/output_flux_all_timesteps_dhdt_${nyears}yr.nc"
    local mesh_2000="$output_dir/output_flux_all_timesteps_2000_tAvg.nc"
    local mesh_end="$output_dir/output_flux_all_timesteps_${end_year}_tAvg.nc"
    if [ ! -f "$dhdt_file" ]; then echo "  Error: dhdt file not found: $dhdt_file"; return 1; fi
    if [ ! -f "$mesh_2000" ]; then echo "  Error: 2000 mesh file not found: $mesh_2000"; return 1; fi
    if [ ! -f "$mesh_end" ]; then echo "  Error: $end_year mesh file not found: $mesh_end"; return 1; fi
    local figure_dir
    figure_dir=$(create_figure_dir "$data_dir")
    local save_name="$figure_dir/dhdt_${nyears}yr"
    echo "  Output figure: $save_name.png"
    cd "$data_dir"
    python3 "$PYTHON_SCRIPT_PATH" \
        -v dhdt \
        -r "$dhdt_file" \
        -m "$mesh_2000" \
        -m2 "$mesh_end" \
        -s "$save_name"
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        if [ -f "${save_name}_2000.png" ]; then
            mv "${save_name}_2000.png" "${save_name}.png"
            echo "  Renamed figure to remove _2000 suffix"
        fi
        echo "  Successfully processed: $data_dir"
        return 0
    else
        echo "  Error processing: $data_dir (exit code: $exit_code)"
        return 1
    fi
}

# ---------------- main logic ----------------
echo "Starting array-friendly plotting script at $(date)"
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then echo "Python plotting script not found at $PYTHON_SCRIPT_PATH"; exit 1; fi
if [ ! -d "$BASE_DATA_PATH" ]; then echo "Base data path not found: $BASE_DATA_PATH"; exit 1; fi

# discover directories
mapfile -t directories < <(find_data_directories "$BASE_DATA_PATH")
num_dirs=${#directories[@]}
if [ $num_dirs -eq 0 ]; then
    echo "No directories with dhdt files found under $BASE_DATA_PATH"; exit 1
fi

# If not running as an array task, re-submit with the correct array range
if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
    echo "Found $num_dirs directories. Will submit array 0..$((num_dirs-1))."
    printf '%3s  %s\n' "IDX" "DIRECTORY"
    for i in "${!directories[@]}"; do
        printf '%3d: %s\n' "$i" "${directories[$i]}"
    done
    echo "Submitting sbatch --array=0-$((num_dirs-1)) $0"
    sbatch --array=0-$((num_dirs-1)) "$0"
    echo "Submission complete; exiting wrapper process."
    exit 0
fi

# running as array task -> pick one dir
task_id=${SLURM_ARRAY_TASK_ID}
if [ $task_id -lt 0 ] || [ $task_id -ge $num_dirs ]; then
    echo "SLURM_ARRAY_TASK_ID ($task_id) out of range [0..$((num_dirs-1))]"; exit 2
fi

data_dir="${directories[$task_id]}"
echo "Array task ${task_id} processing: ${data_dir}"
if process_directory "$data_dir"; then
    echo "Task ${task_id} success"; exit 0
else
    echo "Task ${task_id} failed"; exit 1
fi
