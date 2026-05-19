#!/bin/bash

# Base path to the CTRL directory
base_path="/path/to/scratch/AISLENS/data/MALI/ENSEMBLES/CTRL"

# Array of subdirectories
dirs=("CTRL-EM0" "CTRL-EM1" "CTRL-EM2" "CTRL-EM3" "CTRL-EM4" "CTRL-EM5" "CTRL-EM6" "CTRL-EM7" "CTRL-EM8" "CTRL-EM9")

# Loop through each directory
for dir in "${dirs[@]}"; do
    full_path="$base_path/$dir"
    echo "Processing $full_path..."
    
    # Check if the directory exists
    if [ -d "$full_path" ]; then
        # Check if output directory exists
        if [ -d "$full_path/output" ]; then
            # Create figures directory if it doesn't exist
            mkdir -p "$full_path/output/figures"
            
            # Move all PNG files to figures directory
            if ls "$full_path/output"/*.png 1> /dev/null 2>&1; then
                mv "$full_path/output"/*.png "$full_path/output/figures/"
                echo "  Moved PNG files from $full_path/output to $full_path/output/figures"
            else
                echo "  No PNG files found in $full_path/output"
            fi
        else
            echo "  Warning: $full_path/output directory not found"
        fi
    else
        echo "  Warning: $full_path directory not found"
    fi
done

echo "Script completed!"
