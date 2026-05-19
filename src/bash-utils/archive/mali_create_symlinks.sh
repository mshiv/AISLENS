#!/bin/bash

# Source directory containing the original files
SOURCE_DIR="/path/to/scratch/AISLENS/data/MALI/ISMIP6/HIST/output"

# Target directories where symlinks will be created
TARGET_DIRS=(
    "/path/to/scratch/AISLENS/data/MALI/ISMIP6/SSP585/output"
    "/path/to/scratch/AISLENS/data/MALI/ISMIP6/SSP126/output"
)

# Create symlinks in each target directory
for TARGET_DIR in "${TARGET_DIRS[@]}"; do
    echo "Processing target directory: $TARGET_DIR"
    
    # Create target directory if it doesn't exist
    mkdir -p "$TARGET_DIR"
    
    # Create symlinks for all output_flux_all_timesteps_*.nc files
    for file in "$SOURCE_DIR"/output_flux_all_timesteps_*.nc; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "  Creating symlink for $filename"
            ln -sf "$file" "$TARGET_DIR/$filename"
        fi
    done
    
    echo "  Completed symlinks for: $TARGET_DIR"
    echo ""
done

echo "All symlink creation completed!"
