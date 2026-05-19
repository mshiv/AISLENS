#!/bin/bash
#SBATCH  --job-name=forcing-trend-ssp585
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.trend-ssp585.%j
#SBATCH  --error=err.trend-ssp585.%j
#SBATCH  --exclusive
#SBATCH  --time=2:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis

cd ${AISLENS_DATA_DIR}/MALI/ISMIP6/outputs_from_Trevor/expAE10/expAE10/output/floatingBMB
ncwa -a nCells floatingBasalMassBalApplied_UNKNOWN_Trend_2015-2300.nc floatingBasalMassBalApplied_expAE10_Trend_2015-2300_ts.nc

cd ${AISLENS_DATA_DIR}/MALI/ISMIP6/outputs_from_Trevor/expAE05/output/floatingBMB
ncwa -a nCells floatingBasalMassBalApplied_UNKNOWN_Trend_2015-2300.nc floatingBasalMassBalApplied_expAE05_Trend_2015-2300_ts.nc
