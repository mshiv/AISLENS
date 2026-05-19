#!/bin/bash
#SBATCH  --job-name=prep-MALI-forcings
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=1 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=prepmalif.out
#SBATCH  --error=prepmalif.err
#SBATCH  --exclusive
#SBATCH  --time=2:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

#cd ${AISLENS_DATA_DIR}/processed/vargen_realizations
cd ${HOME}/scratch/ISMIP6-SSP585-4KM/output/floatingBMB
module load anaconda3
conda activate mpas-analysis

# Process the files

# Directory containing the NetCDF files
#input_dir="${AISLENS_DATA_DIR}/processed/vargen_realizations"
#input_dir="${AISLENS_DATA_DIR}/processed/forcings"
input_dir="${HOME}/scratch/ISMIP6-SSP585-4KM/output/floatingBMB"

# Output directory (optional, can be the same as input directory)
#output_dir="${AISLENS_DATA_DIR}/processed/vargen_realizations"
#output_dir="${AISLENS_DATA_DIR}/processed/forcings"
output_dir="${HOME}/scratch/ISMIP6-SSP585-4KM/output/floatingBMB"

# Loop through the NetCDF files matching the pattern
for file in ${input_dir}/floatingBasalMassBalApplied_SSP585_Trend_*.nc; do
    # for file in ${input_dir}/AIS_4to20km_r01_20220907_AISLENS-Forcing_?.nc; do
    # Extract the filename without path
    filename=$(basename "$file")

    # Create output filename (append 'ts' to original filename)
    output_file="${output_dir}/${filename%.*}_ts.nc"

    # Use ncwa to compute the mean along nCells dimension
    # -a nCells: average along nCells dimension
    # -O: overwrite output file if it exists
    #ncwa -a x1,y1 -O "$file" "$output_file"
    ncwa -a nCells -O "$file" "$output_file"
    echo "Processed $filename -> $output_file"
done
