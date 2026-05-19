#!/bin/bash
#SBATCH  --job-name=aislens-viz-dd
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

cd ${AISLENS_REPO}/src/scripts 

#python prepare_data.py
python -u visualize_draft_dependence_map.py --parameter_set original --max_shelves 3 --start_index 33
