#!/bin/bash
#SBATCH  --job-name=ismip6-forc
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=ismip6-ssp585-forc-bmb.out
#SBATCH  --error=ismip6-ssp585-forc-bmb.err
#SBATCH  --exclusive
#SBATCH  --time=15:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis # this loads nco and python environment

cd ${AISLENS_DATA_DIR}/MALI/ISMIP6/ISMIP6_outputs_from_Trevor/ISMIP6_output_for_Shiva/expAE10/expAE10/output

# Extract and concatenate floatingBasalMassBalApplied along Time
# Extract variable
for f in output_flux_all_timesteps_2*.nc; do
  ncks -v floatingBasalMassBalApplied "$f" "tmp_${f}"
done

# Concatenate along Time
ncrcat $(ls tmp_output_flux_all_timesteps_2*.nc | sort) ismip6-ssp126-floatingBasalMassBalApplied.nc

# Remove temp files
rm tmp_output_flux_all_timesteps_2*.nc
