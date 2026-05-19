#!/bin/bash
#SBATCH  --job-name=aislens-dd
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=3 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.calc-dd-aislens.%j
#SBATCH  --error=err.calc-dd-aislens.%j
#SBATCH  --exclusive
#SBATCH  --time=36:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

cd ${AISLENS_REPO}/ 


#python src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py \
#    --interim-dir ${AISLENS_DATA_DIR}/interim/parameter-tests \
#    --processed-dir ${AISLENS_DATA_DIR}/processed/draft_dependence_parameter-tests \
#    --no-skip-existing --test-all-sets

# Run experiment for new variant paramsets based on fine_binning and sensitive_changepoint paramsets of original
python src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py \
  --interim-dir ${AISLENS_DATA_DIR}/interim/parameter-tests \
  --processed-dir ${AISLENS_DATA_DIR}/processed/draft_dependence_parameter-tests \
  --parameter-sets fb_A fb_B fb_C sc_A sc_B \
  --n-workers 8 \
  --no-skip-existing

# Rerun only a small list of shelves (inline list)
#python src/scripts/calculate_draft_dependence_comprehensive_parallel_debug.py \
#  --interim-dir ${AISLENS_DATA_DIR}/interim/parameter-tests \
#  --processed-dir ${AISLENS_DATA_DIR}/processed/draft_dependence_parameter-tests \
#  --no-skip-existing \
#  --test-all-sets \
#  --rerun-shelves Amery Eastern_Ross Thwaites Filchner Ronne
