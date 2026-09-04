#!/bin/bash
#SBATCH --job-name=aislens-spatial-figs
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=4G
#SBATCH --output=spatial-figs.%j.out
#SBATCH --error=spatial-figs.%j.err
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smurugan9@gatech.edu

# =============================================================================
# AISLENS Spatial Figure Generation — HPC Batch Script
# =============================================================================
# Run spatial analysis scripts that require output_state files.
#
# USAGE:
#   cd /storage/home/hcoda1/6/smurugan9/data/jobs/aislens
#   sbatch run_spatial_figures.sbatch
#
# Output_state dir structure (per member):
#   .../ENSEMBLES/<scenario>/<member>/output/output_state_YYYY.nc
#
# OUTPUT: Figures saved to $FIGURES_ROOT/spatial/
# =============================================================================

module load anaconda3
conda activate aislens

CODE_ROOT="/storage/home/hcoda1/6/smurugan9/data/dev/AISLENS"
ENSEMBLES_ROOT="/storage/home/hcoda1/6/smurugan9/scratch/AISLENS/data/MALI/ENSEMBLES"
FIGURES_ROOT="/storage/home/hcoda1/6/smurugan9/scratch/AISLENS/data/figures/MALI"
SCRIPTS_DIR="$CODE_ROOT/src/scripts"

mkdir -p "$FIGURES_ROOT/spatial"

run_script() {
    local script="$1"; shift
    python "$SCRIPTS_DIR/$script" --root "$ENSEMBLES_ROOT" "$@"
}

echo "============================================================"
echo "AISLENS Spatial Figure Generation"
echo "Start: $(date)"
echo "ENSEMBLES_ROOT: $ENSEMBLES_ROOT"
echo "============================================================"

# --- Script 1: Velocity Evolution ---
echo "[1] Velocity evolution..."

run_script fig_velocity_evolution.py \
    --ensemble SSP585 --include '^SSP585_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/velocity_evolution_SSP585.png"

run_script fig_velocity_evolution.py \
    --ensemble SSP585_varScaled10x --include '^SSP585_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/velocity_evolution_varScaled10x.png"

run_script fig_velocity_evolution.py \
    --ensemble SSP126 --include '^SSP126_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/velocity_evolution_SSP126.png"

run_script fig_velocity_evolution.py \
    --ensemble CTRL --include '^CTRL_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/velocity_evolution_CTRL.png"

# --- Script 2: Basal Temperature & Beta (saves 2 files per call) ---
echo "[2] Basal temperature & till strength..."

run_script fig_basal_temperature_beta.py \
    --ensemble SSP585 --include '^SSP585_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/basal_temperature_beta_SSP585.png"

run_script fig_basal_temperature_beta.py \
    --ensemble SSP126 --include '^SSP126_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/basal_temperature_beta_SSP126.png"

run_script fig_basal_temperature_beta.py \
    --ensemble CTRL --include '^CTRL_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/basal_temperature_beta_CTRL.png"

# --- Script 3: dH/dt Maps ---
echo "[3] Thickness change rate maps..."

run_script fig_thickness_dhdt_maps.py \
    --ensemble SSP585 --include '^SSP585_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/dhdt_maps_SSP585.png"

run_script fig_thickness_dhdt_maps.py \
    --ensemble SSP585_varScaled10x --include '^SSP585_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/dhdt_maps_varScaled10x.png"

# --- Script 4: GL Flux Spatial ---
echo "[4] Grounding line flux spatial..."

run_script fig_gl_flux_spatial.py \
    --ensemble SSP585 --include '^SSP585_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/gl_flux_spatial_SSP585.png"

# --- Script 5: Thickness-weighted proxy ---
echo "[5] Thickness-weighted proxy..."

run_script fig_driving_stress.py \
    --ensemble SSP585 --include '^SSP585_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/thickness_weighted_proxy_SSP585.png"

# --- Script 6: Ice Flux Divergence ---
echo "[6] Ice flux divergence..."

run_script fig_ice_flux_divergence.py \
    --ensemble SSP585 --include '^SSP585_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/ice_flux_divergence_SSP585.png"

run_script fig_ice_flux_divergence.py \
    --ensemble SSP126 --include '^SSP126_\d+$' \
    --year-start 0 --year-end 300 \
    --out "$FIGURES_ROOT/spatial/ice_flux_divergence_SSP126.png"

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "Figures in: $FIGURES_ROOT/spatial/"
ls -lh "$FIGURES_ROOT/spatial/" 2>/dev/null || echo "(empty)"
echo "============================================================"
