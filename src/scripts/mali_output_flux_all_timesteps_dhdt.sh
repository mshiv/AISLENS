# This command calculates the average of output flux data variables 
# along the Time dimension, so that we are left with a single time slice for each output year
# Repeat it as required for all output_flux_all_timesteps_****.nc files.
# TODO: Loop through all such files in a given directory
ncwa -a Time -O output_flux_all_timesteps_2000.nc output_flux_all_timesteps_2000_tAvg.nc
ncwa -a Time -O output_flux_all_timesteps_2270.nc output_flux_all_timesteps_2270_tAvg.nc
ncwa -a Time -O output_flux_all_timesteps_2090.nc output_flux_all_timesteps_2090_tAvg.nc
ncwa -a Time -O output_flux_all_timesteps_2100.nc output_flux_all_timesteps_2100_tAvg.nc

# This command takes the difference in thickness between year 300 (or end of simulation) and year 0 (start of simulation)
# and writes the thickness difference to a new file. Note that the variable is not renamed yet.
ncdiff -v thickness output_flux_all_timesteps_2300_tAvg.nc output_flux_all_timesteps_2000_tAvg.nc output_flux_all_timesteps_thickness_2300-2000_diff.nc
ncdiff -v thickness output_flux_all_timesteps_2270_tAvg.nc output_flux_all_timesteps_2000_tAvg.nc output_flux_all_timesteps_thickness_2270-2000_diff.nc
ncdiff -v thickness output_flux_all_timesteps_2090_tAvg.nc output_flux_all_timesteps_2000_tAvg.nc output_flux_all_timesteps_thickness_2090-2000_diff.nc
ncdiff -v thickness output_flux_all_timesteps_2100_tAvg.nc output_flux_all_timesteps_2000_tAvg.nc output_flux_all_timesteps_thickness_2100-2000_diff.nc

# This command calculates dhdt across the number of simulation years (in this case, 300) 
# based on the thickness difference calculated above
ncap2 -s dhdt=thickness/300 output_flux_all_timesteps_thickness_2300-2000_diff.nc output_flux_all_timesteps_dhdt_300yr.nc
ncap2 -s dhdt=thickness/270 output_flux_all_timesteps_thickness_2270-2000_diff.nc output_flux_all_timesteps_dhdt_270yr.nc
ncap2 -s dhdt=thickness/90 output_flux_all_timesteps_thickness_2090-2000_diff.nc output_flux_all_timesteps_dhdt_090yr.nc
ncap2 -s dhdt=thickness/100 output_flux_all_timesteps_thickness_2100-2000_diff.nc output_flux_all_timesteps_dhdt_100yr.nc

# The above ncap2 command removes Time dimension from the output_flux files, so we add back a record Time dimension using one of the following methods:

##### METHOD 1: More commands
ncap2 -s 'defdim("Time", $UNLIMITED); Time[Time]=0' output_flux_all_timesteps_dhdt_270yr.nc output_flux_all_timesteps_dhdt_270yr_tmp.nc
ncap2 -s 'defdim("Time", 1); Time[Time]=0' output_flux_all_timesteps_dhdt_270yr.nc output_flux_all_timesteps_dhdt_270yr_tmp.nc
ncks -O --mk_rec_dmn Time output_flux_all_timesteps_dhdt_270yr_tmp.nc output_flux_all_timesteps_dhdt_270yr_tmp_rec.nc
ncap2 -s 'dhdt[Time,nCells]=dhdt' output_flux_all_timesteps_dhdt_270yr_tmp_rec.nc output_flux_all_timesteps_dhdt_270yr_UPDATED.nc

## METHOD 2: ncecat is a cleaner way, the below command overwrites the original file

ncecat -O -u Time output_flux_all_timesteps_dhdt_270yr.nc output_flux_all_timesteps_dhdt_270yr.nc
ncecat -O -u Time output_flux_all_timesteps_dhdt_100yr.nc output_flux_all_timesteps_dhdt_100yr.nc
ncecat -O -u Time output_flux_all_timesteps_dhdt_090yr.nc output_flux_all_timesteps_dhdt_090yr.nc

ncecat -O -u Time output_flux_all_timesteps_2000_tAvg.nc output_flux_all_timesteps_2000_tAvg.nc
ncecat -O -u Time output_flux_all_timesteps_2270_tAvg.nc output_flux_all_timesteps_2270_tAvg.nc
ncecat -O -u Time output_flux_all_timesteps_2090_tAvg.nc output_flux_all_timesteps_2090_tAvg.nc
ncecat -O -u Time output_flux_all_timesteps_2100_tAvg.nc output_flux_all_timesteps_2100_tAvg.nc

# -------------------


python plot_output_maps_masked.py -v dhdt -r ../../../../../../DATA/AISLENS/MALI-outputs/draftDepen/output_flux_all_timesteps_dhdt_300yr.nc -m ../../../../../../DATA/AISLENS/MALI-outputs/draftDepen/output_flux_all_timesteps_2000.nc -m2 ../../../../../../DATA/AISLENS/MALI-outputs/draftDepen/output_flux_all_timesteps_2000.nc
python plot_output_maps_masked.py -v dhdt -r ../../data/MALI/ISMIP6/SSP585-RIGNOT2013/output_flux_all_timesteps_dhdt_270yr.nc -m ../../data/MALI/ISMIP6/SSP585-RIGNOT2013/output_flux_all_timesteps_2000_tAvg.nc -m2 ../../data/MALI/ISMIP6/SSP585-RIGNOT2013/output_flux_all_timesteps_2270_tAvg.nc

python plot_output_maps_masked.py -v dhdt -r ../../data/MALI/ENSEMBLES/CTRL/EM2/output_flux_all_timesteps_dhdt_100yr.nc -m ../../data/MALI/ENSEMBLES/CTRL/EM2/output_flux_all_timesteps_2000_tAvg.nc -m2 ../../data/MALI/ENSEMBLES/CTRL/EM2/output_flux_all_timesteps_2100_tAvg.nc
python plot_output_maps_masked.py -v dhdt -r ../../../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_dhdt_090yr.nc -m ../../../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_2000_tAvg.nc -m2 ../../../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_2090_tAvg.nc

python plot_output_maps_masked.py -v dhdt -r ../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_dhdt_090yr.nc -m ../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_2000_tAvg.nc -m2 ../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_2090_tAvg.nc

-n /Users/smurugan9/research/aislens/aislens_emulation/data/external/MALI_projects/ISMIP6_2300/initial_conditions/AIS_4to20km_20230105/AIS_4to20km_r01_20220907.regionMask_ismip6.nc

