#!/bin/bash
# prune_flux_outputs.sh — thin out per-year MALI flux snapshots to save disk.
#
# KEEP  output_flux_all_timesteps_YYYY*.nc  when  (YYYY % 2 == 0)  OR  (YYYY % 5 == 0)
#       i.e. every even year (2000,2002,2004,...) PLUS the odd 5-year marks (2005,2015,2025,...)
# DELETE the rest (odd years not divisible by 5: 2001,2003,2007,2009,2011,2013,...).
#
# DRY RUN BY DEFAULT — prints what it *would* delete and the space freed. Pass --apply to delete.
# Only touches files whose name has a 4-digit year immediately after the prefix, so derived products
# (…_dhdt_75yr.nc, …_thickness_2075-2000_diff.nc, …_tAvg without a leading year, etc.) are left alone.
#
# Usage:
#   bash prune_flux_outputs.sh --root <dir> [--apply] [--pattern GLOB] [--path-contains STR]
# Examples:
#   bash prune_flux_outputs.sh --root .../ENSEMBLES/SSP585                 # dry run, whole ensemble
#   bash prune_flux_outputs.sh --root .../ENSEMBLES --path-contains SSP585_0   # one member subset
#   bash prune_flux_outputs.sh --root .../ENSEMBLES/SSP585 --apply         # actually delete

set -uo pipefail

ROOT=""
APPLY=0
PATTERN="output_flux_all_timesteps_[0-9][0-9][0-9][0-9]*.nc"
PATH_CONTAINS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)          ROOT="$2"; shift 2 ;;
    --apply)         APPLY=1; shift ;;
    --pattern)       PATTERN="$2"; shift 2 ;;
    --path-contains) PATH_CONTAINS="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$ROOT" ]] && { echo "ERROR: --root <dir> is required" >&2; exit 2; }
[[ ! -d "$ROOT" ]] && { echo "ERROR: root not a directory: $ROOT" >&2; exit 2; }

if [[ "$APPLY" -eq 1 ]]; then
  echo "=== APPLY MODE: files WILL be deleted ==="
else
  echo "=== DRY RUN (default): nothing deleted. Add --apply to delete. ==="
fi
echo "root: $ROOT"
echo "pattern: $PATTERN   ${PATH_CONTAINS:+(path must contain: $PATH_CONTAINS)}"
echo

n_keep=0; n_del=0; bytes_del=0
# NUL-delimited to survive spaces in paths
while IFS= read -r -d '' f; do
  [[ -n "$PATH_CONTAINS" && "$f" != *"$PATH_CONTAINS"* ]] && continue
  base=$(basename "$f")
  # 4-digit year immediately after the prefix; skip if not present
  year=$(sed -nE 's/^output_flux_all_timesteps_([0-9]{4}).*/\1/p' <<<"$base")
  [[ -z "$year" ]] && continue
  y=$((10#$year))                              # force base-10 (avoid octal on 2008/2009)
  if (( y % 2 == 0 || y % 5 == 0 )); then
    n_keep=$((n_keep+1))
  else
    sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    bytes_del=$((bytes_del + sz))
    n_del=$((n_del+1))
    if [[ "$APPLY" -eq 1 ]]; then
      rm -f -- "$f" && echo "DELETED  $f"
    else
      echo "WOULD DELETE  $f  (year $y)"
    fi
  fi
done < <(find "$ROOT" -type f -name "$PATTERN" -print0)

echo
printf "keep: %d   %s: %d   space %s: %.2f GB\n" \
  "$n_keep" "$([[ $APPLY -eq 1 ]] && echo deleted || echo to-delete)" "$n_del" \
  "$([[ $APPLY -eq 1 ]] && echo freed || echo would-free)" \
  "$(echo "$bytes_del" | awk '{printf "%.6f", $1/1073741824}')"
