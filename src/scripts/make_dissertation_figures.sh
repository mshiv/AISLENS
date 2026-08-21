#!/usr/bin/env bash
# =============================================================================
# make_dissertation_figures.sh -- build every LOCAL Chapter 3 figure into one tree.
#
# Runs only what works from globalStats.nc / regionalStats.nc, which is all that
# exists on the laptop. Spatial (per-cell) figures need output_state/output_flux
# and are produced on the HPC -- see src/pace-jobs/aislens/aislens_dissertation_*.sbatch.
#
# Layout mirrors the figure plan so numbering is unambiguous:
#   reports/dissertation/figures/
#       tierA/   F4  F5  F6      the chapter cannot exist without these
#       tierB/   F7  F8  F9  F10 mechanism figures
#       tierC/   F11             convergence / supporting
#       tables/  frozen results table
#       _logs/   per-figure stdout+stderr
#
# Every figure is run independently; one failure never stops the run. A summary
# of OK/FAIL is printed at the end and each failure keeps its log.
#
# Usage:  bash src/scripts/make_dissertation_figures.sh [OUTROOT]
# =============================================================================
set -uo pipefail

ROOT="${AISLENS_ENSEMBLES_ROOT:-data/MALI/diagnostics/ENSEMBLES}"
OUT="${1:-reports/dissertation/figures}"
SCRIPTS="src/scripts"
LOGS="$OUT/_logs"

mkdir -p "$OUT"/{tierA,tierB,tierC,tables} "$LOGS"

PASS=(); FAIL=()

run() {                     # run <label> <logname> <command...>
    local label="$1"; local log="$2"; shift 2
    printf '  %-46s ' "$label"
    if "$@" > "$LOGS/$log.log" 2>&1; then
        echo "OK";  PASS+=("$label")
    else
        echo "FAIL  -> $LOGS/$log.log"; FAIL+=("$label")
    fi
}

echo "============================================================"
echo "AISLENS dissertation figures"
echo "  ensembles : $ROOT"
echo "  output    : $OUT"
echo "============================================================"

if [[ ! -d "$ROOT" ]]; then
    echo "ERROR: ensembles root not found: $ROOT"; exit 1
fi

# ------------------------------------------------------------------ Tier A
echo
echo "-- Tier A: essential --"

run "F4  global response, all ensembles" f4_topline \
    python3 "$SCRIPTS/fig_full_ensemble_topline.py" --root "$ROOT" --out-dir "$OUT/tierA"

run "F4b mean + spread time series" f4b_percentile \
    python3 "$SCRIPTS/fig_percentile_band.py" --root "$ROOT" --out-dir "$OUT/tierA"

run "F5  variability axis 1x vs 10x" f5_amplification \
    python3 "$SCRIPTS/fig_spread_amplification.py" --root "$ROOT" --out-dir "$OUT/tierA"

run "F5b sigma vs mean (both axes)" f5b_std_vs_mean \
    python3 "$SCRIPTS/fig_std_vs_mean.py" --root "$ROOT" --out-dir "$OUT/tierA"

run "F6  mean axis, 3x melt trend" f6_melt3x \
    python3 "$SCRIPTS/fig_melt3x_calibration.py" --root "$ROOT" \
        --out "$OUT/tierA/F6_melt3x_calibration.png"

# ------------------------------------------------------------------ Tier B
echo
echo "-- Tier B: mechanism --"

run "F7  basin covariance (Eq 3.13)" f7_covariance \
    python3 "$SCRIPTS/fig_basin_covariance.py" --root "$ROOT" --ensemble SSP585 \
        --out "$OUT/tierB/F7_basin_covariance_SSP585.png"

run "F7b spread budget by basin" f7b_spread_budget \
    python3 "$SCRIPTS/fig_spread_budget.py" --root "$ROOT" --out-dir "$OUT/tierB"

run "F8  dynamic gating, SSP585" f8_gating \
    python3 "$SCRIPTS/fig_dynamic_gating.py" --root "$ROOT" --ensemble SSP585 \
        --out "$OUT/tierB/F8_dynamic_gating_SSP585.png"

run "F8b grounding-line migration" f8b_gl_migration \
    python3 "$SCRIPTS/fig_gl_migration.py" --root "$ROOT" --out-dir "$OUT/tierB"

run "F9  metric robustness (rate)" f9_rate \
    python3 "$SCRIPTS/fig_rate_of_change.py" --root "$ROOT" --out-dir "$OUT/tierB"

run "F10 Jourdain model universe" f10_jourdain \
    python3 "$SCRIPTS/fig_jourdain_model_universe.py"

# ------------------------------------------------------------------ Tier C
echo
echo "-- Tier C: supporting --"

run "F11 distribution snapshots" f11_distributions \
    python3 "$SCRIPTS/fig_distribution_snapshots.py" --root "$ROOT" --out-dir "$OUT/tierC"

for E in CTRL SSP126 SSP585 SSP585_varScaled10x; do
    run "F11b convergence/drift: $E" "f11b_analyze_$E" \
        python3 "$SCRIPTS/analyze_ensemble.py" --root "$ROOT" --ensemble "$E" \
            --as-sle --out-fig-dir "$OUT/tierC" 
done

# ------------------------------------------------------------------ tables
echo
echo "-- Tables and diagnostics --"

run "frozen results table (markdown)" frozen_table \
    python3 "$SCRIPTS/freeze_results_table.py" --root "$ROOT" \
        --out "$OUT/tables/frozen_results.md"

{
  echo "# Chapter 3 readiness diagnostics"
  echo '```'
  python3 "$SCRIPTS/chapter3_readiness_diagnostics.py" --root "$ROOT" --section ABCDE 2>&1
  echo '```'
} > "$OUT/tables/readiness_diagnostics.md" 2>/dev/null \
  && { echo "  readiness diagnostics                          OK"; PASS+=("readiness diagnostics"); } \
  || { echo "  readiness diagnostics                          FAIL"; FAIL+=("readiness diagnostics"); }

# ------------------------------------------------------------------ summary
echo
echo "============================================================"
echo "PASS: ${#PASS[@]}    FAIL: ${#FAIL[@]}"
if ((${#FAIL[@]})); then
    printf '  failed: %s\n' "${FAIL[@]}"
    echo "  (logs in $LOGS)"
fi
echo
echo "Figures written under $OUT:"
find "$OUT" -name '*.png' -newermt '-10 minutes' 2>/dev/null | sort | sed 's/^/  /'
echo "============================================================"
