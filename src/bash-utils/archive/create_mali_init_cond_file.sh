
SRC="/path/to/data/projects/MALI_projects/ISMIP6-2300/initial_conditions/AIS_4to20km_20230105/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_meanSatObsBMB_Paolo2023_draftDepen.nc"
DEST_DIR="/path/to/data/projects/MALI_projects/ISMIP6-2300/initial_conditions/AIS_4to20km_20230105/"
BASE_OUTNAME="AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_meanSatObsBMB_Paolo2023_draftDepenPiecewise.nc"

# 0) load nco if required (example)
# module load nco            # or activate conda env that has nco
# conda activate mpas-analysis
# loadaislens

# 1) copy the source to the new destination (work in-place on the copy)
cp "$SRC" "${DEST_DIR}${BASE_OUTNAME}"

# 2) create three new variables by copying draftDepenBasalMeltAlpha0 as a template
#    This will create variables with the same dimensions and values as Alpha0.
#    - draftDepenBasalMelt_minDraft
#    - draftDepenBasalMelt_constantMeltValue
#    - draftDepenBasalMelt_paramType
ncap2 -O -s \
'draftDepenBasalMelt_minDraft=draftDepenBasalMeltAlpha0;
 draftDepenBasalMelt_constantMeltValue=draftDepenBasalMeltAlpha0;
 draftDepenBasalMelt_paramType=draftDepenBasalMeltAlpha0' \
"${DEST_DIR}${BASE_OUTNAME}" "${DEST_DIR}${BASE_OUTNAME}"

# 3) set attributes (units, description). Use ncatted to add/overwrite attributes.
#    Adjust attribute text if you want different units/metadata.
ncatted -O \
  -a units,draftDepenBasalMelt_minDraft,o,c,"m" \
  -a description,draftDepenBasalMelt_minDraft,o,c,"Minimum draft threshold for piecewise linear basal melt parameterization" \
  -a units,draftDepenBasalMelt_constantMeltValue,o,c,"kg m^-2 s^-1" \
  -a description,draftDepenBasalMelt_constantMeltValue,o,c,"Constant basal melt rate for shallow areas (draft < minDraft)" \
  -a units,draftDepenBasalMelt_paramType,o,c,"dimensionless" \
  -a description,draftDepenBasalMelt_paramType,o,c,"parameterization type selector (0=linear,1=constant)" \
  "${DEST_DIR}${BASE_OUTNAME}"

# (optional) ensure alpha units also set (if needed)
ncatted -O \
  -a units,draftDepenBasalMeltAlpha0,o,c,"kg m^-2 s^-1" \
  -a description,draftDepenBasalMeltAlpha0,o,c,"Basal melt rate draft dependency coefficient (alpha0 or intercept)" \
  -a units,draftDepenBasalMeltAlpha1,o,c,"kg m^-3 s^-1" \
  -a description,draftDepenBasalMeltAlpha1,o,c,"Basal melt rate draft dependency coefficient (alpha1 or slope)" \
  "${DEST_DIR}${BASE_OUTNAME}"

# 4) quick verify header
ncdump -h "${DEST_DIR}${BASE_OUTNAME}" | sed -n '1,200p'
