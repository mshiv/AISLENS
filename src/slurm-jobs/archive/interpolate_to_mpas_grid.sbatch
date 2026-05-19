#!/bin/bash
#SBATCH  --job-name=interp_mpasli
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=interp_mpasli%j.out
#SBATCH  --error=interp_mpasli%j.err
#SBATCH  --exclusive
#SBATCH  --time=12:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

cd ${HOME}/data/projects/AISLENS/src/
module load anaconda3/2022.05.0.1
conda activate mpas-analysis

# python ${HOME}/data/projects/AISLENS/src/MPAS-Tools/interpolate_to_mpasli_grid_mod.py -s ${HOME}/data/projects/AISLENS/data/processed/forcing_components/ensemble/OG/sorrm_varclim_MOD.nc -d ${HOME}/data/MALI_projects/ISMIP6-2300/initial_conditions/AIS_8to30km_20221027/AIS_8to30km_r01_20220906.smooth3.basinsFineTuned_carvedRonne_CIRWIP_noSMB_muMax2000_draftDepen_TMP.nc --timestart 0 --timeend 3599 -v floatingBasalMassBalAdjustment

python ${HOME}/data/projects/AISLENS/src/MPAS-Tools/interpolate_to_mpasli_grid_mod.py -s ${HOME}/data/projects/AISLENS/data/processed/forcing_components/satObsPaolo23/draftDepen/draftDepenBasalMeltAlpha_noNaN.nc -d ${HOME}/data/MALI_projects/ISMIP6-2300/initial_conditions/AIS_8to30km_20221027/AIS_8to30km_r01_20220906.smooth3.basinsFineTuned_carvedRonne_CIRWIP_noSMB_muMax2000_draftDepen.nc

# python ${AISLENS_DATA_DIR%/data}/src/interpolate_to_mpasli_grid_mod.py -s ${HOME}/data/MALI_projects/ISMIP6-2300/initial_conditions/AIS_4to20km_20230105/AIS_4to20km_DraftDepen.nc -d ${HOME}/data/MALI_projects/ISMIP6-2300/initial_conditions/AIS_4to20km_20230105/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_meanSatObsBMB_Paolo2023_draftDepen.nc
