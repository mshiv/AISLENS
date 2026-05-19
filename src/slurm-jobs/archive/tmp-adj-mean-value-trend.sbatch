#!/bin/bash
#SBATCH  --job-name=forcing-dedrafted-ts
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.ddrft-tr.%j
#SBATCH  --error=err.ddrft-tr.%j
#SBATCH  --exclusive
#SBATCH  --time=2:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis

cd ${AISLENS_DATA_DIR}/processed/forcing-trend-scenarios_test

ncap2 -O -s 'floatingBasalMassBalAdjustment=floatingBasalMassBalAdjustment-floatingBasalMassBalAdjustment(0,:)' trend_dedrafted_extracted_flux_expAE10_2015-2300_with_state.nc forcing_trend_expAE10_SSP126_2015-2300_Adj.nc

ncap2 -O -s 'floatingBasalMassBalAdjustment=floatingBasalMassBalAdjustment-floatingBasalMassBalAdjustment(0,:)' trend_dedrafted_extracted_flux_expAE05_2015-2300_with_state.nc forcing_trend_expAE05_SSP585_2015-2300_Adj.nc

#ncwa -a nCells dedrafted_extracted_flux_expAE10_2015-2300_with_state.nc dedrafted_extracted_flux_expAE10_2015-2300_with_state_ts.nc

#ncwa -a nCells dedrafted_extracted_flux_expAE05_2015-2300_with_state.nc dedrafted_extracted_flux_expAE05_2015-2300_with_state_ts.nc

