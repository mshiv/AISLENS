#!/bin/bash
#SBATCH  --job-name=gen-aislens
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=2 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.rename-dims.%j
#SBATCH  --error=err.rename-dims.%j
#SBATCH  --exclusive
#SBATCH  --time=3:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

cd ${AISLENS_REPO}/src/scripts

python ${HOME}/data/projects/AISLENS/src/MPAS-Tools/interpolate_to_mpasli_grid_mod.py -s ${HOME}/data/projects/AISLENS/data/processed/forcing_components/satObsPaolo23/draftDepen/draftDepenBasalMeltAlpha_noNaN.nc -d ${HOME}/data/MALI_projects/ISMIP6-2300/initial_conditions/AIS_8to30km_20221027/AIS_8to30km_r01_20220906.smooth3.basinsFineTuned_carvedRonne_CIRWIP_noSMB_muMax2000_draftDepen.nc
