#!/bin/bash

# Base paths
BASE_PATH="/path/to/scratch/AISLENS/data/MALI"
DIAGNOSTICS_PATH="${BASE_PATH}/diagnostics"

# Files to copy
FILES=("globalStats.nc" "regionalStats.nc")

# Create the diagnostics directory if it doesn't exist
mkdir -p "$DIAGNOSTICS_PATH"

echo "Copying NetCDF files from all subdirectories..."

# Find all subdirectories under BASE_PATH (excluding 'diagnostics' itself)
find "$BASE_PATH" -mindepth 1 -type d ! -path "$DIAGNOSTICS_PATH*" | while read -r src_dir; do
    # Skip the diagnostics directory
    if [[ "$src_dir" == "$DIAGNOSTICS_PATH"* ]]; then
        continue
    fi

    # Get the relative path from BASE_PATH for the source directory
    rel_path="${src_dir#$BASE_PATH/}"

    # Create the corresponding target directory in diagnostics
    target_dir="${DIAGNOSTICS_PATH}/${rel_path}"
    mkdir -p "$target_dir"

    echo "  Processing $rel_path..."

    # Look for the output subdirectory and copy files if present
    output_dir="${src_dir}/output"
    if [ -d "$output_dir" ]; then
        for file in "${FILES[@]}"; do
            source_file="${output_dir}/${file}"
            target_file="${target_dir}/${file}"

            if [ -f "$source_file" ]; then
                cp "$source_file" "$target_file"
                echo "    Copied: $source_file -> $target_file"
            else
                echo "    Warning: $source_file not found"
            fi
        done
    else
        echo "    Warning: No output/ directory found in $rel_path"
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
