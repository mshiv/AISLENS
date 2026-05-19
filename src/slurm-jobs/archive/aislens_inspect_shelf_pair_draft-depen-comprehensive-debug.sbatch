#!/bin/bash
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=2 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.viz-dd-aislens.%j
#SBATCH  --error=err.viz-dd-aislens.%j
#SBATCH  --exclusive
#SBATCH  --time=6:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

cd ${AISLENS_REPO} 

python src/scripts/inspect_shelf_pair.py \
  --shelf Amery \
  --param-set standard \
  --fast-dir ${AISLENS_DATA_DIR}/interim/draft_dependence/satobs-fast/comprehensive \
  --parallel-dir ${AISLENS_DATA_DIR}/interim/parameter-tests \
  --out ${AISLENS_DATA_DIR}/processed/visualizations/comprehensive-parallel-compare-paramsets/brahms_inspect.json

