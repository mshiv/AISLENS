#!/bin/bash
#SBATCH --job-name=forcing-ts-extract
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=out.forcing-ts-extract.%j
#SBATCH --error=err.forcing-ts-extract.%j
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smurugan9@gatech.edu

# forcing_ts_extract.sh — reduce ensemble forcing files to a 1-D (Time) time series by collapsing nCells
# with NCO, for a quick cross-ensemble magnitude/variability look.
#
# For each --dir it randomly samples N members and writes a small time-series .nc per member to
#   <out-root>/<dir-basename>/<member>.nc     (xtime is preserved for a real year axis)
# Then plot/compare them with forcing_ts_plot.py.
#
# op=sum  -> ncwa -y ttl  (domain-integrated adjustment; best for comparing MAGNITUDE, since most cells are 0)
# op=avg  -> ncwa -y avg  (mean over ALL nCells; diluted by the many zero/non-shelf cells -- smaller number)
#
# Submit with:
#   sbatch forcing_ts_extract.sh --dir DIR [--dir DIR2 ...] --n 5 --op sum [--var V] [--seed 1] [--out-root DIR]
# Or run locally for quick debugging:
#   bash forcing_ts_extract.sh --dir DIR --n 5 --op sum

set -euo pipefail
shopt -s nullglob

module load anaconda3 2>/dev/null || true
conda activate mpas-analysis
export HDF5_USE_FILE_LOCKING=FALSE

VAR=floatingBasalMassBalAdjustment
N=5; OP=sum; SEED=""; OUT_ROOT=""; PATTERN="*.nc"
DIRS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)      DIRS+=("$2"); shift 2;;
    --n)        N="$2"; shift 2;;
    --op)       OP="$2"; shift 2;;
    --var)      VAR="$2"; shift 2;;
    --seed)     SEED="$2"; shift 2;;
    --pattern)  PATTERN="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    -h|--help)  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ ${#DIRS[@]} -eq 0 ]] && { echo "ERROR: need at least one --dir" >&2; exit 2; }
command -v ncwa >/dev/null || { echo "ERROR: ncwa (NCO) not on PATH -- 'module load nco' or activate the env" >&2; exit 2; }
command -v shuf >/dev/null || { echo "ERROR: shuf not on PATH -- load coreutils or use an environment that provides it" >&2; exit 2; }

case "$OP" in
  sum)            YOP=ttl;;
  avg|mean)       YOP=avg;;
  *) echo "ERROR: --op must be sum or avg" >&2; exit 2;;
esac

: "${AISLENS_DATA_DIR:=/storage/home/hcoda1/6/smurugan9/scratch/AISLENS}"
[[ -z "$OUT_ROOT" ]] && OUT_ROOT="$AISLENS_DATA_DIR/data/processed/forcings_diagnostics/ts"

# optional reproducible shuffle
_rand() { openssl enc -aes-256-ctr -pass pass:"$SEED" -nosalt </dev/zero 2>/dev/null; }

for d in "${DIRS[@]}"; do
  label=$(basename "$d")
  outdir="$OUT_ROOT/$label"
  mkdir -p "$outdir"
  files=("$d"/$PATTERN)
  [[ ${#files[@]} -eq 0 ]] && { echo "WARN: no $PATTERN in $d -- skipping"; continue; }
  if [[ -n "$SEED" ]]; then
    mapfile -t pick < <(printf '%s\n' "${files[@]}" | shuf --random-source=<( _rand ) -n "$N")
  else
    mapfile -t pick < <(printf '%s\n' "${files[@]}" | shuf -n "$N")
  fi
  echo "== ${label}: ${#pick[@]}/${#files[@]} members, op=${OP} (ncwa -y ${YOP} -a nCells) =="
  for f in "${pick[@]}"; do
    out="$outdir/$(basename "$f")"
    if ncwa -O -y "$YOP" -a nCells "$f" "$out" 2>/dev/null; then
      echo "  $(basename "$f") -> $out"
    else
      echo "  FAILED: $(basename "$f")" >&2
    fi
  done
done
echo
echo "done. Compare with:"
echo "  python $(dirname "$0")/forcing_ts_plot.py --root $OUT_ROOT --var $VAR"
