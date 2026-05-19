#!/bin/bash
#SBATCH  --job-name=forcing-trend-ssp585
#SBATCH  --account=${SLURM_ACCOUNT}
#SBATCH  --nodes=2 --ntasks-per-node=24
#SBATCH  --mem-per-cpu=12G
#SBATCH  --output=out.trend-ssp585.%j
#SBATCH  --error=err.trend-ssp585.%j
#SBATCH  --exclusive
#SBATCH  --time=24:00:00
#SBATCH  --mail-type=BEGIN,END,FAIL
#SBATCH  --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis

# Script to add floatingBasalMassBalAdjustment from trend file to ensemble files
# using NCO (NetCDF Operators) and save them in appropriate SSP585 ensemble directories

# Define paths
VARGEN_DIR="${AISLENS_DATA_DIR}/processed/vargen_realizations"
TREND_FILE="${AISLENS_DATA_DIR}/MALI/ISMIP6/SSP585_TREND_2000-2270.nc"
OUTPUT_BASE="${AISLENS_DATA_DIR}/MALI/ENSEMBLES/SSP585"

# Check if trend file exists
if [ ! -f "$TREND_FILE" ]; then
    echo "Error: Trend file $TREND_FILE not found!"
    exit 1
fi

# Check if vargen directory exists
if [ ! -d "$VARGEN_DIR" ]; then
    echo "Error: Vargen directory $VARGEN_DIR not found!"
    exit 1
fi

# Create output directories if they don't exist
for i in {0..9}; do
    mkdir -p "${OUTPUT_BASE}/SSP585-EM${i}"
done

# Get time dimension length from trend file
TREND_TIME_DIM=$(ncdump -h "$TREND_FILE" | grep "Time = " | grep -o '[0-9]\+')
echo "Trend file has $TREND_TIME_DIM time steps"

# Process files 0-9
for i in {0..9}; do
    INPUT_FILE="${VARGEN_DIR}/AIS_4to20km_r01_20220907_AISLENS-Forcing_${i}.nc"
    OUTPUT_FILE="${OUTPUT_BASE}/SSP585-EM${i}/AIS_4to20km_r01_20220907_AISLENS-Forcing_${i}_with_trend.nc"
    
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: Input file $INPUT_FILE not found, skipping..."
        continue
    fi
    
    echo "Processing ensemble member $i..."
    echo "  Input: $INPUT_FILE"
    echo "  Output: $OUTPUT_FILE"
    
    # Get time dimension length from input file
    INPUT_TIME_DIM=$(ncdump -h "$INPUT_FILE" | grep "Time = " | grep -o '[0-9]\+')
    echo "  Input file has $INPUT_TIME_DIM time steps"
    
    # Copy the original file to preserve all variables and attributes
    cp "$INPUT_FILE" "$OUTPUT_FILE"
    
    # Create temporary files for processing
    TEMP_TREND="temp_trend_${i}.nc"
    
    if [ "$INPUT_TIME_DIM" -le "$TREND_TIME_DIM" ]; then
        # Extract first INPUT_TIME_DIM records from trend file
        echo "  Extracting first $INPUT_TIME_DIM time steps from trend file"
        ncks -d Time,0,$((INPUT_TIME_DIM-1)) -v floatingBasalMassBalAdjustment "$TREND_FILE" "$TEMP_TREND"
    else
        # Need to extend trend data to match input time dimension
        echo "  Extending trend data from $TREND_TIME_DIM to $INPUT_TIME_DIM time steps"
        
        # First, extract all trend data
        ncks -v floatingBasalMassBalAdjustment "$TREND_FILE" "$TEMP_TREND"
        
        # Get the last time record from trend
        LAST_RECORD_FILE="temp_last_record_${i}.nc"
        ncks -d Time,$((TREND_TIME_DIM-1)) -v floatingBasalMassBalAdjustment "$TREND_FILE" "$LAST_RECORD_FILE"
        
        # Calculate how many additional records we need
        ADDITIONAL_RECORDS=$((INPUT_TIME_DIM - TREND_TIME_DIM))
        
        # Extend the trend file by repeating the last record
        for ((j=1; j<=ADDITIONAL_RECORDS; j++)); do
            # Create a temporary file with the last record but new time index
            TEMP_RECORD="temp_record_${i}_${j}.nc"
            cp "$LAST_RECORD_FILE" "$TEMP_RECORD"
            
            # Modify the time dimension to be unlimited for concatenation
            ncks --mk_rec_dmn Time "$TEMP_RECORD" "${TEMP_RECORD}_rec"
            mv "${TEMP_RECORD}_rec" "$TEMP_RECORD"
            
            # Concatenate this record to the trend file
            ncrcat "$TEMP_TREND" "$TEMP_RECORD" "temp_extended_${i}.nc"
            mv "temp_extended_${i}.nc" "$TEMP_TREND"
            
            rm "$TEMP_RECORD"
        done
        
        rm "$LAST_RECORD_FILE"
    fi
    
    # Add the trend to the original floatingBasalMassBalAdjustment using ncap2
    echo "  Adding trend to floatingBasalMassBalAdjustment..."
    
    # First, add the trend variable to the output file with a temporary name
    ncks -A -v floatingBasalMassBalAdjustment "$TEMP_TREND" "$OUTPUT_FILE"
    
    # Rename the added variable to avoid conflicts
    ncrename -v floatingBasalMassBalAdjustment,trend_floatingBasalMassBalAdjustment "$OUTPUT_FILE"
    
    # Use ncap2 to add the trend to the original variable
    ncap2 -s 'floatingBasalMassBalAdjustment=floatingBasalMassBalAdjustment+trend_floatingBasalMassBalAdjustment' "$OUTPUT_FILE" "temp_output_${i}.nc"
    
    # Remove the temporary trend variable and keep only the updated original
    ncks -x -v trend_floatingBasalMassBalAdjustment "temp_output_${i}.nc" "$OUTPUT_FILE"
    
    # Clean up temporary files
    rm "$TEMP_TREND" "temp_output_${i}.nc"
    
    echo "  Completed ensemble member $i"
    echo ""
done

# Handle the OG file separately if it exists
OG_INPUT="${VARGEN_DIR}/AIS_4to20km_r01_20220907_AISLENS-Forcing_OG.nc"
if [ -f "$OG_INPUT" ]; then
    echo "Processing OG file..."
    # Create a separate OG directory
    mkdir -p "${OUTPUT_BASE}/SSP585-OG"
    OG_OUTPUT="${OUTPUT_BASE}/SSP585-OG/AIS_4to20km_r01_20220907_AISLENS-Forcing_OG_with_trend.nc"
    
    # Get time dimension length from OG file
    OG_TIME_DIM=$(ncdump -h "$OG_INPUT" | grep "Time = " | grep -o '[0-9]\+')
    echo "  OG file has $OG_TIME_DIM time steps"
    
    cp "$OG_INPUT" "$OG_OUTPUT"
    
    TEMP_TREND_OG="temp_trend_OG.nc"
    
    if [ "$OG_TIME_DIM" -le "$TREND_TIME_DIM" ]; then
        echo "  Extracting first $OG_TIME_DIM time steps from trend file"
        ncks -d Time,0,$((OG_TIME_DIM-1)) -v floatingBasalMassBalAdjustment "$TREND_FILE" "$TEMP_TREND_OG"
    else
        echo "  Extending trend data from $TREND_TIME_DIM to $OG_TIME_DIM time steps"
        
        ncks -v floatingBasalMassBalAdjustment "$TREND_FILE" "$TEMP_TREND_OG"
        
        LAST_RECORD_FILE="temp_last_record_OG.nc"
        ncks -d Time,$((TREND_TIME_DIM-1)) -v floatingBasalMassBalAdjustment "$TREND_FILE" "$LAST_RECORD_FILE"
        
        ADDITIONAL_RECORDS=$((OG_TIME_DIM - TREND_TIME_DIM))
        
        for ((j=1; j<=ADDITIONAL_RECORDS; j++)); do
            TEMP_RECORD="temp_record_OG_${j}.nc"
            cp "$LAST_RECORD_FILE" "$TEMP_RECORD"
            
            ncks --mk_rec_dmn Time "$TEMP_RECORD" "${TEMP_RECORD}_rec"
            mv "${TEMP_RECORD}_rec" "$TEMP_RECORD"
            
            ncrcat "$TEMP_TREND_OG" "$TEMP_RECORD" "temp_extended_OG.nc"
            mv "temp_extended_OG.nc" "$TEMP_TREND_OG"
            
            rm "$TEMP_RECORD"
        done
        
        rm "$LAST_RECORD_FILE"
    fi
    
    echo "  Adding trend to OG floatingBasalMassBalAdjustment..."
    
    ncks -A -v floatingBasalMassBalAdjustment "$TEMP_TREND_OG" "$OG_OUTPUT"
    ncrename -v floatingBasalMassBalAdjustment,trend_floatingBasalMassBalAdjustment "$OG_OUTPUT"
    ncap2 -s 'floatingBasalMassBalAdjustment=floatingBasalMassBalAdjustment+trend_floatingBasalMassBalAdjustment' "$OG_OUTPUT" "temp_output_OG.nc"
    ncks -x -v trend_floatingBasalMassBalAdjustment "temp_output_OG.nc" "$OG_OUTPUT"
    
    rm "$TEMP_TREND_OG" "temp_output_OG.nc"
    
    echo "  Completed OG file"
fi

echo "All files processed successfully!"
echo "Modified files are saved in: $OUTPUT_BASE"
