#!/bin/bash
#SBATCH --job-name=ssp585-forcing-norename
#SBATCH --account=gt-pace-pi-account
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=8G
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --time=05:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=email@gatech.edu

module load anaconda3
conda activate mpas-analysis

# Script to combine SSP585 trend component with AISLENS forcing
# Modified version: NO variable renaming step - assumes trend file already has floatingBasalMassBalAdjustment
# Adds trend values to forcing values with proper time alignment
# Now processes multiple ensemble members automatically

# Note: Removed 'set -e' to allow processing to continue even if one ensemble member fails

# Base directory containing all ensemble subdirectories
PARENT_DIR="/path/to/scratch/AISLENS/data/MALI/ENSEMBLES/SSP585"

# List of ensemble members to process
ENSEMBLE_MEMBERS=("SSP585-EM0" "SSP585-EM1" "SSP585-EM2" "SSP585-EM3" "SSP585-EM4" "SSP585-EM5")

echo "=== SSP585 Trend + AISLENS Forcing Combination Script (Multi-Ensemble, No Rename) ==="
echo "Parent directory: $PARENT_DIR"
echo "Processing ${#ENSEMBLE_MEMBERS[@]} ensemble members:"
for ensemble in "${ENSEMBLE_MEMBERS[@]}"; do
    echo "  - $ensemble"
done
echo "NOTE: Trend file assumed to already have floatingBasalMassBalAdjustment variable"
echo

# Function to process a single ensemble member
process_ensemble() {
    local ENSEMBLE_NAME=$1
    local ENSEMBLE_NUM=$(echo $ENSEMBLE_NAME | sed 's/SSP585-EM//')
    
    echo
    echo "=========================================="
    echo "Processing $ENSEMBLE_NAME (Ensemble Member $ENSEMBLE_NUM)"
    echo "=========================================="
    
    # Base directory for this ensemble
    BASE_DIR="$PARENT_DIR/$ENSEMBLE_NAME"
    
    # Input files - trend file is in ISMIP6 output directory, forcing file is in ensemble directory
    TREND_FILE="/path/to/data/projects/MALI_projects/forcing-trends/SSP585_Trend_2015-2300_ForcingComponent.nc"
    FORCING_FILE="$BASE_DIR/AIS_4to20km_r01_20220907_AISLENS-Forcing_${ENSEMBLE_NUM}.nc"
    OUTPUT_FILE="$BASE_DIR/AIS_4to20km_r01_20220907_AISLENS-Forcing_${ENSEMBLE_NUM}_combined.nc"

    # Temporary directory and files (created as subdirectory of base)
    TEMP_DIR="$BASE_DIR/temp_combine"
    mkdir -p "$TEMP_DIR"

    TEMP_TREND_SUBSET="$TEMP_DIR/temp_trend_subset.nc"
    TEMP_FORCING_SUBSET="$TEMP_DIR/temp_forcing_subset.nc"
    TEMP_COMBINED_SUBSET="$TEMP_DIR/temp_combined_subset.nc"

    echo "Ensemble directory: $BASE_DIR"
    echo "Temporary directory: $TEMP_DIR"
    echo "Input files:"
    echo "  Trend file: $TREND_FILE"
    echo "  Forcing file: $FORCING_FILE"
    echo "Output file: $OUTPUT_FILE"
    echo

    # Check if input files exist
    if [ ! -f "$TREND_FILE" ]; then
        echo "Error: Trend file $TREND_FILE not found!"
        return 1
    fi

    if [ ! -f "$FORCING_FILE" ]; then
        echo "Error: Forcing file $FORCING_FILE not found!"
        return 1
    fi

    # Step 1: Extract overlapping time period from trend file (all 3432 timesteps)
    echo "Step 1: Extracting full time series from trend file..."
    # Copy trend file directly to temp (no renaming needed)
    cp "$TREND_FILE" "$TEMP_TREND_SUBSET"
    echo "  Extracted 3432 timesteps from trend file"

    # Step 2: Extract corresponding time period from forcing file (timesteps 168-3599)
    echo "Step 2: Extracting timesteps 168-3599 from forcing file..."
    # 168 corresponds to Jan 2015 (year 15 * 12 months + 0-based indexing)
    # 3599 corresponds to Dec 2299 (last timestep we can match)
    ncks -O -d Time,168,3599 "$FORCING_FILE" "$TEMP_FORCING_SUBSET"
    echo "  Extracted 3432 timesteps from forcing file (Time indices 168-3599)"

    # Step 3: Verify dimensions match
    echo "Step 3: Verifying dimensions match..."
    TREND_TIME_SIZE=$(ncdump -h "$TEMP_TREND_SUBSET" | grep "Time = " | sed 's/.*Time = \([0-9]*\).*/\1/')
    FORCING_TIME_SIZE=$(ncdump -h "$TEMP_FORCING_SUBSET" | grep "Time = " | sed 's/.*Time = \([0-9]*\).*/\1/')

    echo "  Trend file Time dimension: $TREND_TIME_SIZE"
    echo "  Forcing file Time dimension: $FORCING_TIME_SIZE"

    if [ "$TREND_TIME_SIZE" != "$FORCING_TIME_SIZE" ]; then
        echo "Error: Time dimensions don't match!"
        echo "  Trend: $TREND_TIME_SIZE, Forcing: $FORCING_TIME_SIZE"
        return 1
    fi

    echo "  ✓ Time dimensions match: $TREND_TIME_SIZE timesteps"

    # Step 4: Verify both files have floatingBasalMassBalAdjustment variable
    echo "Step 4: Verifying variable names..."
    if ! ncdump -h "$TEMP_TREND_SUBSET" | grep -q "floatingBasalMassBalAdjustment"; then
        echo "  ✗ Error: floatingBasalMassBalAdjustment variable not found in trend file"
        echo "  Available variables in trend file:"
        ncdump -h "$TEMP_TREND_SUBSET" | grep "float\|double" | head -5
        return 1
    fi

    if ! ncdump -h "$TEMP_FORCING_SUBSET" | grep -q "floatingBasalMassBalAdjustment"; then
        echo "  ✗ Error: floatingBasalMassBalAdjustment variable not found in forcing file"
        echo "  Available variables in forcing file:"
        ncdump -h "$TEMP_FORCING_SUBSET" | grep "float\|double" | head -5
        return 1
    fi

    echo "  ✓ floatingBasalMassBalAdjustment variable found in both files"

    # Step 5: Add the variables
    echo "Step 5: Adding floatingBasalMassBalAdjustment variables..."
    ncbo -O -o "$TEMP_COMBINED_SUBSET" --op_typ=add "$TEMP_FORCING_SUBSET" "$TEMP_TREND_SUBSET"
    echo "  Variables added successfully"

    # Step 6: Create final output by combining unmodified early period with modified period
    echo "Step 6: Creating final output file..."

    # First, extract the early period (Time 0-167) from original forcing file
    TEMP_EARLY_PERIOD="$TEMP_DIR/temp_early_period.nc"
    ncks -O -d Time,0,167 "$FORCING_FILE" "$TEMP_EARLY_PERIOD"
    echo "  Extracted early period (timesteps 0-167)"

    # Concatenate early period with combined period
    ncrcat -O "$TEMP_EARLY_PERIOD" "$TEMP_COMBINED_SUBSET" "$OUTPUT_FILE"
    echo "  Final file created with concatenated periods"

    # Step 7: Verify final output
    echo "Step 7: Verifying final output..."
    FINAL_TIME_SIZE=$(ncdump -h "$OUTPUT_FILE" | grep "Time = " | sed 's/.*Time = \([0-9]*\).*/\1/')
    echo "  Final file Time dimension: $FINAL_TIME_SIZE"

    if [ "$FINAL_TIME_SIZE" = "3600" ]; then
        echo "  ✓ Final file has correct Time dimension (3600)"
    else
        echo "  ⚠ Warning: Expected 3600 timesteps, got $FINAL_TIME_SIZE"
    fi

    # Check that the variable exists
    if ncdump -h "$OUTPUT_FILE" | grep -q "floatingBasalMassBalAdjustment"; then
        echo "  ✓ floatingBasalMassBalAdjustment variable present in output"
    else
        echo "  ✗ Error: floatingBasalMassBalAdjustment variable not found in output"
    fi

    # Step 8: Clean up temporary files
    echo "Step 8: Cleaning up temporary files..."
    rm -f "$TEMP_TREND_SUBSET" "$TEMP_FORCING_SUBSET" "$TEMP_COMBINED_SUBSET"
    # Also remove the early period temporary file
    rm -f "$TEMP_EARLY_PERIOD"
    # Remove temporary directory if empty
    rmdir "$TEMP_DIR" 2>/dev/null || echo "  Note: Temporary directory not empty, keeping it"
    echo "  Temporary files removed"

    echo
    echo "=== $ENSEMBLE_NAME COMPLETION SUMMARY ==="
    echo "✓ Time alignment: Trend 2015-2300 added to Forcing 2015-2300 period"
    echo "✓ Output file: $OUTPUT_FILE"
    echo "✓ Final Time dimension: $FINAL_TIME_SIZE timesteps"
    echo
    echo "Time period breakdown:"
    echo "  Years 2000-2014 (months 0-167):   Original forcing values only"
    echo "  Years 2015-2299 (months 168-3599): Original forcing + SSP585 trend"
    echo "  Year 2300 (month 3600):            Original forcing values only"
    echo
    echo "$ENSEMBLE_NAME processing completed successfully!"
    
    return 0
}

# Main processing loop
SUCCESSFUL_COUNT=0
FAILED_COUNT=0
FAILED_ENSEMBLES=()

for ENSEMBLE in "${ENSEMBLE_MEMBERS[@]}"; do
    if process_ensemble "$ENSEMBLE"; then
        ((SUCCESSFUL_COUNT++))
        echo "✓ $ENSEMBLE: SUCCESS"
    else
        ((FAILED_COUNT++))
        FAILED_ENSEMBLES+=("$ENSEMBLE")
        echo "✗ $ENSEMBLE: FAILED"
    fi
done

echo
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo "Total ensemble members processed: ${#ENSEMBLE_MEMBERS[@]}"
echo "Successful: $SUCCESSFUL_COUNT"
echo "Failed: $FAILED_COUNT"

if [ $FAILED_COUNT -gt 0 ]; then
    echo
    echo "Failed ensemble members:"
    for failed_ensemble in "${FAILED_ENSEMBLES[@]}"; do
        echo "  - $failed_ensemble"
    done
    echo
    echo "Please check the error messages above for details."
    exit 1
else
    echo
    echo "All ensemble members processed successfully!"
    exit 0
fi

