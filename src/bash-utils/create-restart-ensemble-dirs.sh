#!/usr/bin/env bash
# create_restart_ensemble_dirs.sh
# Create ensemble member directories that use per-member restart directories.
#
# Usage (example):
# src/scripts/create_restart_ensemble_dirs.sh \
#   -R "/storage/.../ENSEMBLES/CTRL-SSN/CTRL%02d" \
#   -f /path/to/forcings_dir \
#   -t /path/to/template_dir \
#   -p CTRL- -n 10 -d /scratch/ensembles/CTRL -e /path/to/landice_exec \
#   -b "AIS_4to20km_r01_20220907_AISLENS-Forcing_%02d.nc"
#
# Notes:
# - The restart dir template should expand to a directory that contains a file like rst.2015-01-01.nc.
# - The script will symlink the first rst.*.nc found in that restart directory into the member directory.
# - It will NOT change the <immutable_stream name="input" ...> block in the copied streams.landice.
#   Only the variability-forcing filename_template will be updated.

#set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOU'
Usage: create_restart_ensemble_dirs.sh -R RESTART_DIR_TEMPLATE -f FORCING_DIR -t TEMPLATE_DIR \
  -p ENSEMBLE -n N -d DEST_DIR -e ICE_EXEC -b FORCING_BASENAME [ -s START_INDEX ]

Required:
  -R RESTART_DIR_TEMPLATE  Template for per-member restart directories. Supports %d/%02d or {i}.
  -f FORCING_DIR          Directory with forcing files
  -t TEMPLATE_DIR         Directory with template files (job_script.sh, streams.landice, ...)
  -p ENSEMBLE             Ensemble prefix (e.g. CTRL-)
  -n N                    Number of members
  -d DEST_DIR             Destination parent directory
  -e ICE_EXEC             Path to landice executable to symlink
  -b FORCING_BASENAME     Forcing filename pattern (supports %d, %02d or {i})

Optional:
  -s START_INDEX          Start index (default 0)
  -r RESTART_FILENAME     Specific restart filename in each restart dir (default: rst.2015-01-01.nc)
  -h                      Show help
EOU
}

START_INDEX=0
RESTART_TEMPLATE=""
RESTART_FILENAME="rst.2015-01-01.nc"

while getopts ":R:f:t:p:n:d:e:b:s:r:h" opt; do
  case $opt in
    R) RESTART_TEMPLATE="$OPTARG" ;;
    r) RESTART_FILENAME="$OPTARG" ;;
    f) FORCING_DIR="$OPTARG" ;;
    t) TEMPLATE_DIR="$OPTARG" ;;
    p) ENSEMBLE="$OPTARG" ;;
    n) N_MEMBERS="$OPTARG" ;;
    d) DEST_DIR="$OPTARG" ;;
    e) ICE_EXEC="$OPTARG" ;;
    b) FORCING_BASENAME="$OPTARG" ;;
    s) START_INDEX="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Missing arg for -$OPTARG" >&2; usage; exit 2 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

: "${RESTART_TEMPLATE:?}" "${FORCING_DIR:?}" "${TEMPLATE_DIR:?}" "${ENSEMBLE:?}" "${N_MEMBERS:?}" "${DEST_DIR:?}" "${ICE_EXEC:?}" "${FORCING_BASENAME:?}"

FORCING_DIR=$(realpath "${FORCING_DIR}")
TEMPLATE_DIR=$(realpath "${TEMPLATE_DIR}")
DEST_DIR=$(realpath "${DEST_DIR}")
ICE_EXEC=$(realpath "${ICE_EXEC}")
mkdir -p "${DEST_DIR}"

# Data symlinks (same as your original)
TEMPLATE_SYMLINKS=(
  "AIS_4to20km_r01_20220907_RACMO2.3p2_ANT27_smb_climatology_1995-2017_minus1_bare_land.nc"
  "AIS_4to20km_r01_20220907.regionMask_ismip6.nc"
  "albany_input.yaml"
  "graph.info"
)

TEMPLATE_COPY=(
  "job_script.sh"
  "namelist.landice"
  "streams.landice"
)

format_forcing_name() {
  local idx="$1"; local idx_padded="${2:-$(printf "%02d" "$idx")}"
  if [[ "${FORCING_BASENAME}" == *"%d"* ]]; then
    if [[ "${FORCING_BASENAME}" == *"%02d"* ]]; then
      # Try printf expansion first; if it leaves an unexpanded '%' fallback to replacement
      formatted=$(printf "${FORCING_BASENAME}" "${idx}")
      if [[ "${formatted}" == *"%"* ]]; then
        formatted="${FORCING_BASENAME//%02d/${idx_padded}}"
      fi
      printf "%s" "${formatted}"
    else
      echo "${FORCING_BASENAME//%d/${idx_padded}}"
    fi
  elif [[ "${FORCING_BASENAME}" == *"{i}"* ]]; then
    echo "${FORCING_BASENAME//\{i\}/${idx_padded}}"
  else
    if [[ "${FORCING_BASENAME}" == *.nc ]]; then
      base="${FORCING_BASENAME%.nc}"
      echo "${base}-${idx_padded}.nc"
    else
      echo "${FORCING_BASENAME}-${idx_padded}.nc"
    fi
  fi
}

format_restart_dir() {
  local template="$1"
  local idx="$2"
  local idx_padded
  idx_padded=$(printf "%02d" "${idx}")

  # If the template contains printf-style placeholders (%d, %02d, etc.), use printf
  if [[ "${template}" == *"%d"* ]]; then
    formatted=$(printf "${template}" "${idx}")
    # If printf left '%' sequences unexpanded (bad template or escaping), fall back
    if [[ "${formatted}" == *"%"* ]]; then
      # Try replacing common patterns
      formatted="${template//%02d/${idx_padded}}"
      formatted="${formatted//%d/${idx_padded}}"
    fi
    printf "%s" "${formatted}"
  elif [[ "${template}" == *"{i}"* ]]; then
    echo "${template//\{i\}/${idx_padded}}"
  else
    # Fallback: append padded index
    echo "${template}${idx_padded}"
  fi
}

end_index=$((START_INDEX + N_MEMBERS - 1))
created=0

for idx in $(seq ${START_INDEX} ${end_index}); do
  idx_padded=$(printf "%02d" "${idx}")
  member_name="${ENSEMBLE}${idx_padded}"
  member_dir="${DEST_DIR}/${member_name}"
  echo "Creating member: ${member_name} -> ${member_dir}"
  mkdir -p "${member_dir}"

  # symlink large template files
  for tfile in "${TEMPLATE_SYMLINKS[@]}"; do
    src="${TEMPLATE_DIR}/${tfile}"
    if [[ -e "${src}" ]]; then
      ln -sf "${src}" "${member_dir}/$(basename "${tfile}")"
    else
      echo "Warning: template symlink source not found: ${src}" >&2
    fi
  done

  # copy small runtime files
  for cfile in "${TEMPLATE_COPY[@]}"; do
    src="${TEMPLATE_DIR}/${cfile}"
    if [[ -e "${src}" ]]; then
      cp -f "${src}" "${member_dir}/$(basename "${cfile}")"
    else
      echo "Error: required template file missing: ${src}" >&2
    fi
  done

  # link restart file from per-member restart dir
  restart_dir=$(format_restart_dir "${RESTART_TEMPLATE}" "${idx}")
  if [[ -d "${restart_dir}" ]]; then
    # Link the specific restart file expected for each member (rst.2015-01-01.nc)
    specific_rst="${restart_dir}/${RESTART_FILENAME}"
    if [[ -e "${specific_rst}" ]]; then
      ln -sf "${specific_rst}" "${member_dir}/$(basename "${specific_rst}")"
      echo "  Linked restart file: $(basename "${specific_rst}")"
    else
      echo "Error: expected restart file not found: ${specific_rst}" >&2
    fi
  else
    echo "Error: restart dir not found: ${restart_dir}" >&2
  fi

  # copy restart_timestamp file from template dir into member (if present)
  restart_timestamp_src="${TEMPLATE_DIR}/restart_timestamp"
  if [[ -e "${restart_timestamp_src}" ]]; then
    cp -f "${restart_timestamp_src}" "${member_dir}/restart_timestamp"
    echo "  Copied restart_timestamp into ${member_name}"
  else
    echo "Warning: restart_timestamp not found in template dir: ${restart_timestamp_src}" >&2
  fi

  # symlink model executable
  if [[ -e "${ICE_EXEC}" ]]; then
    ln -sf "${ICE_EXEC}" "${member_dir}/$(basename "${ICE_EXEC}")"
  else
    echo "Error: ice executable not found: ${ICE_EXEC}" >&2
  fi

  # symlink forcing file (try padded then unpadded)
  forcing_name=$(format_forcing_name "${idx}" "${idx_padded}")
  forcing_src="${FORCING_DIR}/${forcing_name}"
  if [[ -e "${forcing_src}" ]]; then
    ln -sf "${forcing_src}" "${member_dir}/${forcing_name}"
  else
    forcing_name_unpadded=$(format_forcing_name "${idx}" "${idx}")
    forcing_src_unpadded="${FORCING_DIR}/${forcing_name_unpadded}"
    if [[ -e "${forcing_src_unpadded}" ]]; then
      ln -sf "${forcing_src_unpadded}" "${member_dir}/${forcing_name_unpadded}"
      forcing_name="${forcing_name_unpadded}"
    else
      echo "Error: forcing file not found for member ${member_name}: ${forcing_src} (tried padded) and ${forcing_src_unpadded} (tried unpadded)" >&2
    fi
  fi

  # update job_script.sh SBATCH job-name (replace or prepend)
  jobfile="${member_dir}/job_script.sh"
  jobname="${member_name}"
  if [[ -f "${jobfile}" ]]; then
    cp -f "${jobfile}" "${jobfile}.bak"
    sed -E '/--job-name/ s/--job-name[= ]*[^[:space:]]+/--job-name='"${jobname}"'/' "${jobfile}" > "${jobfile}.tmp" || true
    if [ -f "${jobfile}.tmp" ] && grep -q -- --job-name "${jobfile}.tmp"; then
      mv "${jobfile}.tmp" "${jobfile}"
      echo "  Updated job_script.sh job name in ${member_name}"
    else
      printf "#SBATCH --job-name=%s\n" "${jobname}" > "${jobfile}.tmp2"
      cat "${jobfile}" >> "${jobfile}.tmp2"
      mv "${jobfile}.tmp2" "${jobfile}"
      echo "  Prepended SBATCH job name in ${member_name}"
    fi
  else
    echo "Warning: job_script.sh not found in ${member_dir}; skipping job customization." >&2
  fi

  # update streams.landice: DO NOT change the input immutable_stream; only update variability-forcing
  streamsfile="${member_dir}/streams.landice"
  if [[ -f "${streamsfile}" ]]; then
    cp -f "${streamsfile}" "${streamsfile}.bak"
    awk -v forcing="${forcing_name}" '
      BEGIN { RS=">"; ORS=">" }
      {
        record = $0
        if (record ~ /name="variability-forcing"/) {
          gsub(/filename_template="[^"]*"/, "filename_template=\"" forcing "\"", record)
        }
        print record
      }
    ' "${streamsfile}" > "${streamsfile}.tmp" || echo "Warning: streams customization failed for ${streamsfile}" >&2

    if [ -f "${streamsfile}.tmp" ]; then
      perl -0777 -pe 's/>\z//' "${streamsfile}.tmp" > "${streamsfile}.tmp2" && mv "${streamsfile}.tmp2" "${streamsfile}.tmp" || true
    fi

    if grep -q -- "${forcing_name}" "${streamsfile}.tmp"; then
      mv "${streamsfile}.tmp" "${streamsfile}"
      echo "  Updated streams.landice (variability-forcing) in ${member_name}"
    else
      rm -f "${streamsfile}.tmp"
      echo "  WARNING: Failed to update streams.landice in ${member_name}" >&2
    fi
  else
    echo "Warning: streams.landice not found in ${member_dir}; skipping streams customization." >&2
  fi

  created=$((created+1))
done

echo "Created ${created}/${N_MEMBERS} ensemble member directories in ${DEST_DIR}"
