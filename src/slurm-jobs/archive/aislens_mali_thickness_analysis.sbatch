#!/bin/bash
#SBATCH --job-name=thickness-analysis
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=24G
#SBATCH --output=thickness_analysis_%j.out
#SBATCH --error=thickness_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL}

module load anaconda3
conda activate mpas-analysis

# =========================================================================

# Output directory for plots and analysis data
# Option 1: Keep it with ensemble data (current setup)
OUTPUT_DIR="${DATA_ROOT}/${ENSEMBLE_NAME}/thickness_analysis"

# Option 2: Central analysis directory
# OUTPUT_DIR="${AISLENS_DATA_DIR%/data}/analysis/thickness/${ENSEMBLE_NAME}"

# Option 3: User scratch space
# OUTPUT_DIR="${HOME}/scratch/analysis/thickness_${ENSEMBLE_NAME}"

# Option 4: Specific custom location
# OUTPUT_DIR="${HOME}/analysis_results/thickness_variability"

# Time indices to analyze (space-separated list)
# -1 means last timestep, or specify specific indices like "120 240 360"
TIME_INDICES="120 240 -1"

# Script location
SCRIPT_DIR="${HOME}/research/aislens/AISLENS/src/MPAS-Tools"

# ============================================================================
# End Configuration
# ============================================================================

echo "================================================================"
echo "Ensemble Thickness Analysis Job Started"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Started at: $(date)"
echo ""
echo "Configuration:"
echo "  Ensemble: ${ENSEMBLE_NAME}"
echo "  Experiments: ${EXPERIMENT_LIST}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Time indices: ${TIME_INDICES}"
echo "================================================================"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Loop through time indices
for TIME_IDX in ${TIME_INDICES}; do
    echo "================================================================"
    echo "Analyzing time index: ${TIME_IDX}"
    echo "================================================================"
    
    python ${SCRIPT_DIR}/plot_ensemble_thickness_analysis.py \
        --root ${DATA_ROOT} \
        --base ${ENSEMBLE_NAME} \
        --experiments ${EXPERIMENT_LIST} \
        --mesh ${MESH_FILE} \
        --time-index ${TIME_IDX} \
        --save \
        --output-dir ${OUTPUT_DIR} \
        --dpi 300
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed analysis for time index ${TIME_IDX}"
    else
        echo "✗ Failed to complete analysis for time index ${TIME_IDX}"
    fi
    echo ""
done

echo "================================================================"
echo "All analyses complete!"
echo "Finished at: $(date)"
echo "Output saved to: ${OUTPUT_DIR}"
echo "================================================================"
echo "All analyses complete!"
echo "Finished at: $(date)"
echo "Output saved to: ${OUTPUT_DIR}"
echo "================================================================"

