#!/bin/bash

ENSEMBLE_DIR="/path/to/scratch/AISLENS/data/MALI/ENSEMBLES/SSP585"
EXPERIMENTS="SSP585-EM1 SSP585-EM2 SSP585-EM4 SSP585-EM6 SSP585-EM8"

echo "Checking files for thickness analysis..."
echo ""

for EXP in $EXPERIMENTS; do
    echo "=== $EXP ==="
    EXP_DIR="${ENSEMBLE_DIR}/${EXP}/output/"
    
    if [ ! -d "$EXP_DIR" ]; then
        echo "  ERROR: Directory not found"
        continue
    fi
    
    # Check for output files
    FILES=$(ls ${EXP_DIR}/output*.nc 2>/dev/null | wc -l)
    echo "  Found $FILES output files"
    
    if [ $FILES -gt 0 ]; then
        LATEST=$(ls -t ${EXP_DIR}/output*.nc | head -1)
        echo "  Latest: $(basename $LATEST)"
        
        # Check for thickness variable
        if ncdump -h "$LATEST" 2>/dev/null | grep -q "float thickness"; then
            echo "  ✓ Has 'thickness' variable"
        else
            echo "  ✗ Missing 'thickness' variable"
            echo "  Available variables:"
            ncdump -h "$LATEST" 2>/dev/null | grep "float" | head -5
        fi
        
        # Check time dimension
        TIME_DIM=$(ncdump -h "$LATEST" 2>/dev/null | grep "Time = " | sed 's/.*Time = \([0-9]*\).*/\1/')
        echo "  Time dimension: $TIME_DIM"
    fi
    echo ""
done

