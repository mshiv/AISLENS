#!/usr/bin/env bash
# ensemble_job_script.sh
# Submit job_script.sh in each ensemble member directory by cd'ing into each
# member directory before calling `sbatch job_script.sh`.
#
# Usage:
#   ensemble_job_script.sh -p PREFIX -d PARENT_DIR [-n N] [-l LIST] [-s START] [-z PAD] [-D DELAY]
#
# Modes (mutually exclusive):
# 1) -n N (and optional -s START) : submit members from START .. START+N-1
# 2) -l LIST                     : submit members listed by comma-separated indices and ranges (e.g. 0,3,5-7)
#
# Examples:
#   ensemble_job_script.sh -p CTRL -d /.../ENSEMBLES/CTRL-SSN -n 10
#   ensemble_job_script.sh -p CTRL -d /.../ENSEMBLES/CTRL-SSN -l 0,3,7
#
set -euo pipefail
IFS=$'\n\t'

usage(){
  sed -n '1,160p' <<'USG'
Usage: ensemble_job_script.sh -p PREFIX -d PARENT_DIR [-n N] [-l LIST] [-s START] [-z PAD] [-D DELAY]

Required:
  -p PREFIX       Ensemble name prefix (e.g. CTRL)
  -d PARENT_DIR   Parent directory that contains member subdirectories (where the script will be placed/run)

Selection (one of):
  -n N            Ensemble size: submit all members from START .. START+N-1 (default START=0)
  -l LIST         Comma-separated list of indices and ranges (e.g. 0,2,5-7)

Optional:
  -s START        Start index for -n mode (default 0)
  -z PAD          Zero-pad width for indices (default 2 -> 00, 01, ...)
  -D DELAY        Seconds to sleep between sbatch submissions (default 0)
  -h              Show help

Notes:
  - For each index i the member directory name is constructed as: <PREFIX><idx_padded>,
    where idx_padded is index zero-padded to width PAD.
  - The script `cd` into each member directory and runs: sbatch job_script.sh
  - The wrapper prints the sbatch submission response and continues to the next member.
USG
}

# Parse args
PREFIX=""
PARENT_DIR=""
MODE_N=""
MODE_LIST=""
START=0
PAD=2
DELAY=0

while getopts ":p:d:n:l:s:z:D:h" opt; do
  case ${opt} in
    p) PREFIX="${OPTARG}" ;;
    d) PARENT_DIR="${OPTARG}" ;;
    n) MODE_N="${OPTARG}" ;;
    l) MODE_LIST="${OPTARG}" ;;
    s) START="${OPTARG}" ;;
    z) PAD="${OPTARG}" ;;
    D) DELAY="${OPTARG}" ;;
    h) usage; exit 0 ;;
    :) echo "Error: -${OPTARG} requires an argument." >&2; usage; exit 2 ;;
    \?) echo "Invalid option: -${OPTARG}" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${PREFIX}" || -z "${PARENT_DIR}" ]]; then
  echo "Error: -p and -d are required" >&2; usage; exit 2
fi

if [[ -n "${MODE_N}" && -n "${MODE_LIST}" ]]; then
  echo "Error: supply only one of -n or -l" >&2; usage; exit 2
fi

# expand list (comma/range) into array of indices
expand_list(){
  local list_str="$1"
  local -n out_arr=$2
  IFS=',' read -ra parts <<< "${list_str}"
  for part in "${parts[@]}"; do
    if [[ "${part}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      a=${BASH_REMATCH[1]}
      b=${BASH_REMATCH[2]}
      for ((i=a; i<=b; i++)); do out_arr+=("${i}"); done
    elif [[ "${part}" =~ ^[0-9]+$ ]]; then
      out_arr+=("${part}")
    elif [[ -z "${part}" ]]; then
      continue
    else
      echo "Warning: ignoring invalid list item: ${part}" >&2
    fi
  done
}

# Build members array
members=()
if [[ -n "${MODE_N}" ]]; then
  # MODE_N provided: iterate START .. START+N-1
  n=${MODE_N}
  for ((i=START; i<START + n; i++)); do
    members+=("${i}")
  done
elif [[ -n "${MODE_LIST}" ]]; then
  expand_list "${MODE_LIST}" members
else
  echo "Error: one of -n or -l must be provided" >&2; usage; exit 2
fi

if [[ ${#members[@]} -eq 0 ]]; then
  echo "No members to submit"; exit 0
fi

# Main loop: cd into each member dir and submit
submitted=0
failed=0
for idx in "${members[@]}"; do
  idx_padded=$(printf "%0${PAD}d" "${idx}")
  member_name="${PREFIX}${idx_padded}"
  member_dir="${PARENT_DIR%/}/${member_name}"

  if [[ ! -d "${member_dir}" ]]; then
    echo "Skipping ${member_name}: directory not found: ${member_dir}" >&2
    failed=$((failed+1))
    continue
  fi

  if [[ ! -f "${member_dir}/job_script.sh" ]]; then
    echo "Skipping ${member_name}: job_script.sh not found in ${member_dir}" >&2
    failed=$((failed+1))
    continue
  fi

  echo "Submitting ${member_name} from ${member_dir} ..."
  pushd "${member_dir}" > /dev/null
  # Submit the job_script.sh from inside the member directory
  if output=$(sbatch job_script.sh 2>&1); then
    echo "  sbatch output: ${output}"
    submitted=$((submitted+1))
  else
    echo "  sbatch failed: ${output}" >&2
    failed=$((failed+1))
  fi
  popd > /dev/null

  # optional delay between submissions
  if [[ "${DELAY}" -gt 0 ]]; then sleep "${DELAY}"; fi
done

echo "Done. submitted=${submitted}, failed=${failed}"

