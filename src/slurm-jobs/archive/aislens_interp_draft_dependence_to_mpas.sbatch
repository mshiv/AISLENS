#!/bin/bash
#SBATCH --job-name=interp_mpasli
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=8G
#SBATCH --output=interp_mpasli_%A_%a.out
#SBATCH --error=interp_mpasli_%A_%a.err
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

# Source and destination paths
DEST_DIR="${AISLENS_DATA_DIR}/processed/"
DEST_FILE="${DEST_DIR}AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_meanSatObsBMB_Paolo2023_draftDepen.nc"

module load anaconda3
conda activate mpas-analysis
export AISLENS_DATA_DIR=${AISLENS_DATA_DIR%/data}

cd ${AISLENS_REPO}/src/MPAS-Tools

python interpolate_to_mpasli_grid_mod.py -s ${AISLENS_DATA_DIR}/processed/draft_dependence_params.nc -d "$DEST_FILE"
python interpolate_to_mpasli_grid_mod.py -s ${AISLENS_DATA_DIR}/processed/satellite_observations_ic_field.nc -d "$DEST_FILE"
