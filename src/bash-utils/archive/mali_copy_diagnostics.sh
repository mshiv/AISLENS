#!/bin/bash

# Base paths
BASE_PATH="/path/to/scratch/AISLENS/data/MALI"
DIAGNOSTICS_PATH="${BASE_PATH}/diagnostics"

# Files to symlink
FILES=("globalStats.nc" "regionalStats.nc")

# Create the diagnostics directory if it doesn't exist
mkdir -p "$DIAGNOSTICS_PATH"

echo "Copying NetCDF files..."

# Handle ENSEMBLES/CTRL/CTRL-EM* directories
ENSEMBLES_PATH="${BASE_PATH}/ENSEMBLES/CTRL"
if [ -d "$ENSEMBLES_PATH" ]; then
    echo "Processing ENSEMBLES/CTRL directories..."
    
    # Create the ENSEMBLES/CTRL structure in diagnostics
    mkdir -p "${DIAGNOSTICS_PATH}/ENSEMBLES/CTRL"
    
    # Process each CTRL-EM* directory
    for ctrl_dir in "${ENSEMBLES_PATH}"/CTRL-EM*; do
        if [ -d "$ctrl_dir" ]; then
            # Get the directory name (e.g., CTRL-EM0, CTRL-EM1, etc.)
            dir_name=$(basename "$ctrl_dir")
            
            # Create corresponding directory in diagnostics
            target_dir="${DIAGNOSTICS_PATH}/ENSEMBLES/CTRL/${dir_name}"
            mkdir -p "$target_dir"
            
            echo "  Processing $dir_name..."
            
            # Copy each file
            for file in "${FILES[@]}"; do
                source_file="${ctrl_dir}/output/${file}"
                target_file="${target_dir}/${file}"
                
                if [ -f "$source_file" ]; then
                    cp "$source_file" "$target_file"
                    echo "    Copied: $source_file -> $target_file"
                else
                    echo "    Warning: $source_file not found"
                fi
            done
        fi
    done
else
    echo "Warning: ENSEMBLES/CTRL directory not found at $ENSEMBLES_PATH"
fi

# Handle ISMIP6 directories
ISMIP6_SCENARIOS=("SSP126" "SSP585")

for scenario in "${ISMIP6_SCENARIOS[@]}"; do
    ISMIP6_PATH="${BASE_PATH}/ISMIP6/${scenario}"
    
    if [ -d "$ISMIP6_PATH" ]; then
        echo "Processing ISMIP6/$scenario directory..."
        
        # Create the ISMIP6/scenario structure in diagnostics
        mkdir -p "${DIAGNOSTICS_PATH}/ISMIP6/${scenario}"
        
        # Copy each file
        for file in "${FILES[@]}"; do
            source_file="${ISMIP6_PATH}/output/${file}"
            target_file="${DIAGNOSTICS_PATH}/ISMIP6/${scenario}/${file}"
            
            if [ -f "$source_file" ]; then
                cp "$source_file" "$target_file"
                echo "  Copied: $source_file -> $target_file"
            else
                echo "  Warning: $source_file not found"
            fi
        done
    else
        echo "Warning: ISMIP6/$scenario directory not found at $ISMIP6_PATH"
    fi
done

echo "Done! Files copied to $DIAGNOSTICS_PATH"

# Optional: Display the created structure
echo ""
echo "Created directory structure:"
find "$DIAGNOSTICS_PATH" -type d | sort
echo ""
echo "Copied files:"
find "$DIAGNOSTICS_PATH" -name "*.nc" | sort
