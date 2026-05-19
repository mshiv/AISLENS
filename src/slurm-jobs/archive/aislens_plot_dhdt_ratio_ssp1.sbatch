#!/bin/bash
#SBATCH --job-name=mali-ensemble-ratio
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=2 --ntasks-per-node=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.mali-ensemble-ratio.%j
#SBATCH --error=err.mali-ensemble-ratio.%j
#SBATCH --time=02:00:00

# This SLURM script calls the standalone ratio plotting script.
# Expectations:
# - ensembleStats_{year}.nc files exist in ENSEMBLE_STATS_DIR
# - the stats files contain `thickness_range` and `dhdt_mean` variables
# - run output directories are provided for grounding-line overlays

module load anaconda3
conda activate mpas-analysis

# USER CONFIGURATION
ENSEMBLE_STATS_DIR="${AISLENS_DATA_DIR}/figures/MALI/ensemble_stats/CTRL-SSN-10"
PYTHON_SCRIPT="${AISLENS_REPO}/src/MPAS-Tools/plot_ensemble_maps_ratio.py"
RUN_DIRS=("${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP585-SSNTRND/SSP12600/output" \
          "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP585-SSNTRND/SSP12601/output" \
          "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP585-SSNTRND/SSP12602/output" \ 
          "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP585-SSNTRND/SSP12603/output" \
          "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP585-SSNTRND/SSP12604/output" \
          "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP585-SSNTRND/SSP12605/output" \
          "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP585-SSNTRND/SSP12606/output" \
          "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP585-SSNTRND/SSP12607/output" \
	  "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP585-SSNTRND/SSP12609/output")
RUN_NAMES=("SSP12600" "SSP12601" "SSP12602" "SSP12603" "SSP12604" "SSP12605" "SSP12606" "SSP12607" "SSP12608" "SSP12609")
SAVE_BASE="${AISLENS_DATA_DIR}/figures/MALI/ensemble_ratio/SSP126"
YEARS=(2150 2125 2125 2100 2075 2050 2025)
NUMERATORS=(thickness_range) #,dhdt_range,dhdt_std,thickness_std)
DENOMINATORS=(abs_thickness_change)
GL_LINEWIDTH=0.55

mkdir -p "$SAVE_BASE"

# Build stats file list for years
STATS_FILES=()
for y in "${YEARS[@]}"; do
    sf="$ENSEMBLE_STATS_DIR/ensembleStats_${y}.nc"
    if [ -f "$sf" ]; then
        STATS_FILES+=("$sf")
    else
        echo "WARNING: stats file missing for year $y: $sf"
    fi
done

if [ ${#STATS_FILES[@]} -eq 0 ]; then
    echo "No stats files found. Exiting."; exit 1
fi

# join arrays into comma-separated strings
IFS=','; STATS_STR="${STATS_FILES[*]}"; unset IFS
IFS=','; RUN_DIRS_STR="${RUN_DIRS[*]}"; unset IFS
IFS=','; RUN_NAMES_STR="${RUN_NAMES[*]}"; unset IFS
IFS=','; YEARS_STR="${YEARS[*]}"; unset IFS
IFS=','; VARS_STR="${VARIABLES[*]}"; unset IFS

python3 "$PYTHON_SCRIPT" \
  --stats_files "$STATS_STR" \
  --years "$YEARS_STR" \
  --variables "$VARS_STR" \
  --numerators "$NUMERATORS" \
  --denominators "$DENOMINATORS" \
  --run_dirs "$RUN_DIRS_STR" \
  --run_names "$RUN_NAMES_STR" \
  --save_base "$SAVE_BASE" \
  --initial_thickness_file "${AISLENS_DATA_DIR}/MALI/ENSEMBLES/CTRL-SSN/CTRL00/output/output_flux_all_timesteps_2000.nc" \
  --cmap "PuBuGn" \
  --gl_colors "black" \
  --log_cbar \
  --initial_thickness_time_index 0 \
  --gl_linewidth "$GL_LINEWIDTH" \
  --mask_ratio_below 1e-3

echo "Job finished"

