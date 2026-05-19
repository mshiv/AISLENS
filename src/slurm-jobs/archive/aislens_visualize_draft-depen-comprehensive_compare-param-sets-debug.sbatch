#!/bin/bash
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=3 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.viz-dd-aislens.%j
#SBATCH  --error=err.viz-dd-aislens.%j
#SBATCH  --exclusive
#SBATCH  --time=6:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}
#SBATCH  --dependency=afterok:2434074

# REMOVE DEPENDENCY IN FUTURE!!!!

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

cd ${AISLENS_REPO}/src/scripts 

# THIS RAN FIRST
#python compare_parameter_sets_debug.py \
#  --parameter-sets standard permissive strict sensitive_changepoint robust_changepoint fine_binning \
#  --base-dir ${AISLENS_DATA_DIR}/interim/parameter-tests \
#  --processed-dir ${AISLENS_DATA_DIR}/processed/draft_dependence_parameter-tests \
#  --output-dir ${AISLENS_DATA_DIR}/processed/visualizations/comprehensive-parallel-compare-paramsets \
#  --plot-all-shelves

# Run below to visualize fine_binning and sensitive_changepoint variant paramsets.
python compare_parameter_sets_debug.py \
  --parameter-sets fb_A fb_B fb_C sc_A sc_B\
  --base-dir ${AISLENS_DATA_DIR}/interim/parameter-tests \
  --processed-dir ${AISLENS_DATA_DIR}/processed/draft_dependence_parameter-tests \
  --output-dir ${AISLENS_DATA_DIR}/processed/visualizations/comprehensive-parallel-compare-paramsets \
  --plot-all-shelves

# Compare only those shelves (use compare script's flag for specific shelves)
#python compare_parameter_sets_debug.py \
#  --parameter-sets standard permissive strict \
#  --base-dir ${AISLENS_DATA_DIR}/interim/parameter-tests \
#  --processed-dir ${AISLENS_DATA_DIR}/processed/draft_dependence_parameter-tests \
#  --output-dir ${AISLENS_DATA_DIR}/processed/visualizations/comprehensive-parallel-compare-paramsets \
#  --plot-shelves Amery Eastern_Ross Thwaites Filchner Ronne

