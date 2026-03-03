#!/usr/bin/env bash
# create_ensemble_dirs.sh
# Create ensemble experiment directories from a template, copy/link files,
# and customize job_script and streams files per-member.
#
# Usage:
#   create_ensemble_dirs.sh \
#     -i /path/to/init_file.nc \
#     -f /path/to/forcing_dir \
#     -t /path/to/template_dir \
#     -p ENSEMBLE (prefix string to identify the scenario, e.g. CTRL_, SSP126_) \
#     -n 10 \
#     -d /path/to/destination_dir \
#     -e /path/to/landice_model \
#     -b forcing_basename
#
# forcing_basename semantics:
# - If it contains a '%d' it will be used with printf to format the index.
# - If it contains '{i}' it will be replaced with the index.
# - Otherwise it will be treated as a base and the script will form: "${basename}-${idx}.nc"
#
# The script will:
# - create member directories named "<ENSEMBLE><NN>" (index zero-padded to 2 digits: 00, 01, ...)
# - copy small run-time files (job_script.sh, namelist.landice, streams.landice)
# - symlink the landice executable into each member dir
# - symlink the appropriate forcing file from the forcing directory into each member dir
# - customize job script (job name) and streams file (forcing filename and init file)

set -euo pipefail
IFS=$'\n\t'

usage() {
  sed -n '1,200p' <<'USG'
Usage: create_ensemble_dirs.sh -i INIT_FILE -f FORCING_DIR -t TEMPLATE_DIR \
  -p ENSEMBLE -n N -d DEST_DIR -e ICE_EXEC -b FORCING_BASENAME [ -s START_INDEX ]

Required:
  -i INIT_FILE            Path to initialization file to reference in streams
  -f FORCING_DIR          Directory that contains forcing files (some-string-i.nc)
  -t TEMPLATE_DIR         Directory containing template files
  -p ENSEMBLE             Ensemble name prefix for member dirs (e.g. CTRL_, SSP126_)
  -n N                    Number of ensemble members to create
  -d DEST_DIR             Destination directory to hold ensemble subdirectories
  -e ICE_EXEC             Path to landice_model executable to symlink into each dir
  -b FORCING_BASENAME     Forcing filename pattern (see header for semantics)

Optional:
  -s START_INDEX          Start index for members (default: 0)
  -h                      Show this help and exit

Example:
  create_ensemble_dirs.sh \
    -i /data/init/initial.nc -f /data/forcings -t /templates/run_template \
    -p run_ -n 12 -d /scratch/ensembles -e /usr/local/bin/landice_model -b "forcing-%d.nc"
USG
}

# Defaults
START_INDEX=0

# Parse args
while getopts ":i:f:t:p:n:d:e:b:s:h" opt; do
  case ${opt} in
    i) INIT_FILE="${OPTARG}" ;;
    f) FORCING_DIR="${OPTARG}" ;;
    t) TEMPLATE_DIR="${OPTARG}" ;;
  p) ENSEMBLE="${OPTARG}" ;;
    n) N_MEMBERS="${OPTARG}" ;;
    d) DEST_DIR="${OPTARG}" ;;
    e) ICE_EXEC="${OPTARG}" ;;
    b) FORCING_BASENAME="${OPTARG}" ;;
    s) START_INDEX="${OPTARG}" ;;
    h) usage; exit 0 ;;
    :) echo "Error: -${OPTARG} requires an argument." >&2; usage; exit 2 ;;
    \?) echo "Invalid option: -${OPTARG}" >&2; usage; exit 2 ;;
  esac
done

# Validate required variables
: "${INIT_FILE:?}" "${FORCING_DIR:?}" "${TEMPLATE_DIR:?}" "${ENSEMBLE:?}" "${N_MEMBERS:?}" "${DEST_DIR:?}" "${ICE_EXEC:?}" "${FORCING_BASENAME:?}"

# Resolve to absolute paths where possible
INIT_FILE=$(realpath "${INIT_FILE}")
FORCING_DIR=$(realpath "${FORCING_DIR}")
TEMPLATE_DIR=$(realpath "${TEMPLATE_DIR}")
DEST_DIR=$(realpath "${DEST_DIR}")
ICE_EXEC=$(realpath "${ICE_EXEC}")

mkdir -p "${DEST_DIR}"

# Files to symlink from template (large files / data):
# (Using the exact names you requested)
TEMPLATE_SYMLINKS=(
  "AIS_4to20km_r01_20220907_RACMO2.3p2_ANT27_smb_climatology_1995-2017_minus1_bare_land.nc"
  "AIS_4to20km_r01_20220907.regionMask_ismip6.nc"
  "albany_input.yaml"
  "graph.info"
)

# Files to copy (small runtime files)
TEMPLATE_COPY=(
  "job_script.sh"
  "namelist.landice"
  "streams.landice"
)

# Helper: format forcing filename given index
format_forcing_name() {
  # args: idx (integer), idx_padded (e.g. 02)
  local idx="$1"
  local idx_padded="${2:-$(printf "%02d" "$idx")}"

  # If user used printf-style formats
  if [[ "${FORCING_BASENAME}" == *"%d"* ]]; then
    if [[ "${FORCING_BASENAME}" == *"%02d"* ]]; then
      # user already requested padding via %02d; honor it
      printf "%s" "$(printf "${FORCING_BASENAME}" "${idx}")"
    else
      # replace %d placeholders with our padded string to enforce two-digit indices
      echo "${FORCING_BASENAME//%d/${idx_padded}}"
    fi
  elif [[ "${FORCING_BASENAME}" == *"{i}"* ]]; then
    echo "${FORCING_BASENAME//\{i\}/${idx_padded}}"
  else
    # default: basename-<padded>.nc
    if [[ "${FORCING_BASENAME}" == *.nc ]]; then
      base="${FORCING_BASENAME%.nc}"
      echo "${base}-${idx_padded}.nc"
    else
      echo "${FORCING_BASENAME}-${idx_padded}.nc"
    fi
  fi
}

# Note: in-place text editing is handled below with perl one-liners; no helper function required

# Create each member directory
end_index=$((START_INDEX + N_MEMBERS - 1))
created=0
for idx in $(seq ${START_INDEX} ${end_index}); do
  # zero-pad the numeric index to two digits for naming (00, 01, ...)
  idx_padded=$(printf "%02d" "${idx}")
  member_name="${ENSEMBLE}${idx_padded}"
  member_dir="${DEST_DIR}/${member_name}"
  echo "Creating member: ${member_name} -> ${member_dir}"
  mkdir -p "${member_dir}"

  # 1) Symlink large template files if present
  for tfile in "${TEMPLATE_SYMLINKS[@]}"; do
    src="${TEMPLATE_DIR}/${tfile}"
    if [[ -e "${src}" ]]; then
      ln -sf "${src}" "${member_dir}/$(basename "${tfile}")"
    else
      echo "Warning: template symlink source not found: ${src}" >&2
    fi
  done

  # 2) Copy small runtime files
  for cfile in "${TEMPLATE_COPY[@]}"; do
    src="${TEMPLATE_DIR}/${cfile}"
    if [[ -e "${src}" ]]; then
      cp -f "${src}" "${member_dir}/$(basename "${cfile}")"
    else
      echo "Error: required template file missing: ${src}" >&2
    fi
  done

  # 2b) Symlink the initialization file into the member directory so streams can
  # reference a local basename (this makes the streams replacement reliable).
  if [[ -e "${INIT_FILE}" ]]; then
    ln -sf "${INIT_FILE}" "${member_dir}/$(basename "${INIT_FILE}")"
  else
    echo "Error: init file not found: ${INIT_FILE}" >&2
  fi

  # 3) Symlink ice sheet model executable
  if [[ -e "${ICE_EXEC}" ]]; then
    ln -sf "${ICE_EXEC}" "${member_dir}/$(basename "${ICE_EXEC}")"
  else
    echo "Error: ice executable not found: ${ICE_EXEC}" >&2
  fi

  # 4) Symlink the forcing file for this member.
  # Try the zero-padded variant first (default), then fall back to a non-padded
  # variant if present in the forcing directory. This avoids needing to rename
  # your existing files which may be single-digit numbered.
  forcing_name=$(format_forcing_name "${idx}" "${idx_padded}")
  forcing_src="${FORCING_DIR}/${forcing_name}"
  if [[ -e "${forcing_src}" ]]; then
    ln -sf "${forcing_src}" "${member_dir}/${forcing_name}"
  else
    # try unpadded index (e.g. 1 instead of 01)
    forcing_name_unpadded=$(format_forcing_name "${idx}" "${idx}")
    forcing_src_unpadded="${FORCING_DIR}/${forcing_name_unpadded}"
    if [[ -e "${forcing_src_unpadded}" ]]; then
      ln -sf "${forcing_src_unpadded}" "${member_dir}/${forcing_name_unpadded}"
      # update forcing_name variable so later steps (streams editing) reference the
      # actual filename we linked
      forcing_name="${forcing_name_unpadded}"
    else
      echo "Error: forcing file not found for member ${member_name}: ${forcing_src} (tried padded) and ${forcing_src_unpadded} (tried unpadded)" >&2
    fi
  fi

  # 5) Customize job_script.sh
  jobfile="${member_dir}/job_script.sh"
  jobname="${member_name}"
  if [[ -f "${jobfile}" ]]; then
    # Backup and replace job name in SBATCH header. Works for lines like:
    # #SBATCH --job-name=oldname  or  #SBATCH --job-name oldname
    cp -f "${jobfile}" "${jobfile}.bak"
    # Use sed to replace the --job-name value on the SBATCH line; if not present, prepend it.
    # This targets lines containing --job-name and substitutes the value as `--job-name=NAME`.
    sed -E '/--job-name/ s/--job-name[= ]*[^[:space:]]+/--job-name='"${jobname}"'/' "${jobfile}" > "${jobfile}.tmp" || true
    if [ -f "${jobfile}.tmp" ] && grep -q -- "--job-name" "${jobfile}.tmp"; then
      mv "${jobfile}.tmp" "${jobfile}"
      echo "  Updated job_script.sh job name in ${member_name}"
    else
      # Prepend SBATCH line if replacement did not find an existing line
      printf "#SBATCH --job-name=%s\n" "${jobname}" > "${jobfile}.tmp2"
      cat "${jobfile}" >> "${jobfile}.tmp2"
      mv "${jobfile}.tmp2" "${jobfile}"
      echo "  Prepended SBATCH job name in ${member_name}"
    fi
  else
    echo "Warning: job_script.sh not found in ${member_dir}; skipping job customization." >&2
  fi

  # 6) Customize streams.landice: replace placeholders or attempt heuristics
  streamsfile="${member_dir}/streams.landice"
  if [[ -f "${streamsfile}" ]]; then
    # Backup streams file
    cp -f "${streamsfile}" "${streamsfile}.bak"
    INIT_BASENAME="$(basename "${INIT_FILE}")"
    # Update filename_template for the input immutable_stream to the init basename, and
    # for the variability-forcing stream to the per-member forcing filename.
    # Use awk with RS='>' to treat multi-line XML-style start-tags as single records.
    # This allows `name="..."` and `filename_template="..."` to be on different lines.
    awk -v init="${INIT_BASENAME}" -v forcing="${forcing_name}" '
    BEGIN { RS=">"; ORS=">" }
    {
      record = $0
      if (record ~ /name="input"/) {
        gsub(/filename_template="[^"]*"/, "filename_template=\"" init "\"", record)
      }
      if (record ~ /name="variability-forcing"/) {
        gsub(/filename_template="[^"]*"/, "filename_template=\"" forcing "\"", record)
      }
  # use print so awk appends ORS (">") automatically and preserves tag endings
  print record
    }
    ' "${streamsfile}" > "${streamsfile}.tmp" || echo "Warning: streams customization failed for ${streamsfile}" >&2
    # The RS='>' / ORS='>' trick re-adds '>' after each record and can leave
    # a trailing '>' at the end of the file. Remove a single trailing '>' if
    # present. Using perl with slurp mode is robust on multi-line files.
    if [ -f "${streamsfile}.tmp" ]; then
      perl -0777 -pe 's/>\z//' "${streamsfile}.tmp" > "${streamsfile}.tmp2" && mv "${streamsfile}.tmp2" "${streamsfile}.tmp" || true
    fi
    if grep -q -- "${forcing_name}" "${streamsfile}.tmp" || grep -q -- "${INIT_BASENAME}" "${streamsfile}.tmp"; then
      mv "${streamsfile}.tmp" "${streamsfile}"
      echo "  Updated streams.landice in ${member_name}"
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
