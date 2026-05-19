#!/bin/bash

# AISLENS Ensemble Setup Script
# This script sets up the CTRL ensemble directories with necessary files

set -e  # Exit on any error

# Define paths
SOURCE_DIR="/path/to/scratch/aislens-debug-tests/4km/4km_N8"
DEST_BASE="/path/to/scratch/AISLENS/data/MALI/ENSEMBLES/CTRL"
FORCING_DIR="/path/to/scratch/AISLENS/data/processed/vargen_realizations"
INITIAL_CONDITION="/path/to/scratch/AISLENS/data/processed/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_meanSatObsBMB_Paolo2023_draftDepen.nc"

# Define common files to copy
COMMON_FILES=(
    "AIS_4to20km_r01_20220907_RACMO2.3p2_ANT27_smb_climatology_1995-2017_minus1_bare_land.nc"
    "AIS_4to20km_r01_20220907.regionMask_ismip6.nc"
    "albany_input.yaml"
    "graph.info"
    "job_script.sh"
    "landice_model"
    "namelist.landice"
    "streams.landice"
)

# Define ensemble directories
ENSEMBLE_DIRS=(
    "CTRL-EM0" "CTRL-EM1" "CTRL-EM2" "CTRL-EM3" "CTRL-EM4"
    "CTRL-EM5" "CTRL-EM6" "CTRL-EM7" "CTRL-EM8" "CTRL-EM9"
)

echo "Starting AISLENS ensemble setup..."

# Step 1: Copy common files to all ensemble directories
echo "Step 1: Copying common files to ensemble directories..."
for ensemble in "${ENSEMBLE_DIRS[@]}"; do
    dest_dir="${DEST_BASE}/${ensemble}"
    
    # Create directory if it doesn't exist
    mkdir -p "$dest_dir"
    echo "  Created/verified directory: $dest_dir"
    
    # Copy common files
    for file in "${COMMON_FILES[@]}"; do
        if [[ -f "${SOURCE_DIR}/${file}" ]]; then
            cp "${SOURCE_DIR}/${file}" "$dest_dir/"
            echo "    Copied $file to $ensemble"
        else
            echo "    WARNING: Source file not found: ${SOURCE_DIR}/${file}"
        fi
    done
done

# Step 2: Copy forcing files to respective directories
echo ""
echo "Step 2: Copying forcing files to respective directories..."
for i in {0..9}; do
    ensemble="CTRL-EM${i}"
    forcing_file="AIS_4to20km_r01_20220907_AISLENS-Forcing_${i}.nc"
    source_forcing="${FORCING_DIR}/${forcing_file}"
    dest_dir="${DEST_BASE}/${ensemble}"
    
    if [[ -f "$source_forcing" ]]; then
        cp "$source_forcing" "$dest_dir/"
        echo "  Copied $forcing_file to $ensemble"
    else
        echo "  WARNING: Forcing file not found: $source_forcing"
    fi
done

# Step 3: Create symlinks to initial condition file
echo ""
echo "Step 3: Creating symlinks to initial condition file..."
ic_basename=$(basename "$INITIAL_CONDITION")
for ensemble in "${ENSEMBLE_DIRS[@]}"; do
    dest_dir="${DEST_BASE}/${ensemble}"
    symlink_path="${dest_dir}/${ic_basename}"
    
    # Remove existing symlink or file if it exists
    if [[ -L "$symlink_path" ]] || [[ -f "$symlink_path" ]]; then
        rm "$symlink_path"
    fi
    
    if [[ -f "$INITIAL_CONDITION" ]]; then
        ln -s "$INITIAL_CONDITION" "$symlink_path"
        echo "  Created symlink in $ensemble"
    else
        echo "  WARNING: Initial condition file not found: $INITIAL_CONDITION"
    fi
done

# Step 4: Update streams.landice files
echo ""
echo "Step 4: Updating streams.landice files..."
for i in {0..9}; do
    ensemble="CTRL-EM${i}"
    streams_file="${DEST_BASE}/${ensemble}/streams.landice"
    forcing_filename="AIS_4to20km_r01_20220907_AISLENS-Forcing_${i}.nc"
    
    if [[ -f "$streams_file" ]]; then
        # Create backup
        cp "$streams_file" "${streams_file}.bak"
        
        # Update the filename_template line
        sed -i 's/filename_template="AIS_4to20km_r01_20220907_Forcing_00\.nc"/filename_template="'"$forcing_filename"'"/' "$streams_file"
        
        # Verify the change was made
        if grep -q "filename_template=\"$forcing_filename\"" "$streams_file"; then
            echo "  Updated streams.landice in $ensemble with $forcing_filename"
        else
            echo "  WARNING: Failed to update streams.landice in $ensemble"
            echo "  Please manually check the file and update the aislens_TF input stream"
        fi
    else
        echo "  WARNING: streams.landice not found in $ensemble"
    fi
done

echo ""
echo "Setup complete!"
echo ""
echo "Summary:"
echo "- Copied common files to ${#ENSEMBLE_DIRS[@]} ensemble directories"
echo "- Copied forcing files to respective directories (0-9)"
echo "- Created symlinks to initial condition file in all directories"
echo "- Updated streams.landice files with correct forcing filenames"
echo ""
echo "Please verify the setup by checking a few directories and the streams.landice files."
echo "Backup files (streams.landice.bak) have been created before modifications."
