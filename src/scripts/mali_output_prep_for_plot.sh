#!/bin/bash
# This command calculates the average of output flux data variables 
# along the Time dimension, so that we are left with a single time slice for each output year
# Repeat it as required for all output_flux_all_timesteps_****.nc files.
# TODO: Loop through all such files in a given directory
cd /Users/smurugan9/research/aislens/AISLENS/data/MALI/ISMIP6/SSP585-RIGNOT2013
END_YEAR = '2100'
NYEARS = END_YEAR - 2000

ncwa -a Time -O output_flux_all_timesteps_2000.nc output_flux_all_timesteps_2000_tAvg.nc
ncwa -a Time -O output_flux_all_timesteps_{$END_YEAR}.nc output_flux_all_timesteps_{$END_YEAR}_tAvg.nc
ncecat -O -u Time output_flux_all_timesteps_2000_tAvg.nc output_flux_all_timesteps_2000_tAvg.nc
ncecat -O -u Time output_flux_all_timesteps_{$END_YEAR}_tAvg.nc output_flux_all_timesteps_{$END_YEAR}_tAvg.nc


# This command takes the difference in thickness between year 300 (or end of simulation) and year 0 (start of simulation)
# and writes the thickness difference to a new file. Note that the variable is not renamed yet.
ncdiff -v thickness output_flux_all_timesteps_{$END_YEAR}_tAvg.nc output_flux_all_timesteps_2000_tAvg.nc output_flux_all_timesteps_thickness_{$END_YEAR}-2000_diff.nc

# This command calculates dhdt across the number of simulation years (in this case, 300) 
# based on the thickness difference calculated above
ncap2 -s dhdt=thickness/{$NYEARS} output_flux_all_timesteps_thickness_{$END_YEAR}-2000_diff.nc output_flux_all_timesteps_dhdt_{$NYEARS}yr.nc

#cd /Users/smurugan9/research/aislens/AISLENS/src/MPAS-Tools/plot_output_maps_masked.py -v dhdt -r 
# -------------------


#python plot_output_maps_masked.py -v dhdt -r ../../../../../../DATA/AISLENS/MALI-outputs/draftDepen/output_flux_all_timesteps_dhdt_300yr.nc -m ../../../../../../DATA/AISLENS/MALI-outputs/draftDepen/output_flux_all_timesteps_2000.nc -m2 ../../../../../../DATA/AISLENS/MALI-outputs/draftDepen/output_flux_all_timesteps_2000.nc
#python plot_output_maps_masked.py -v dhdt -r ../../data/MALI/ISMIP6/SSP585-RIGNOT2013/output_flux_all_timesteps_dhdt_270yr.nc -m ../../data/MALI/ISMIP6/SSP585-RIGNOT2013/output_flux_all_timesteps_2000_tAvg.nc -m2 ../../data/MALI/ISMIP6/SSP585-RIGNOT2013/output_flux_all_timesteps_2270_tAvg.nc


#python plot_output_maps_masked.py -v dhdt -r ../../data/MALI/ISMIP6/SSP585/output_flux_all_timesteps_dhdt_75yr.nc -m ../../data/MALI/ISMIP6/SSP585/output_flux_all_timesteps_2000_tAvg.nc -m2 ../../data/MALI/ISMIP6/SSP585/output_flux_all_timesteps_2075_tAvg.nc

#python plot_output_maps_masked.py -v dhdt -r ../../../../data/MALI/ENSEMBLES/CTRL/EM2/output_flux_all_timesteps_dhdt_100yr.nc -m ../../../../data/MALI/ENSEMBLES/CTRL/EM2/output_flux_all_timesteps_2000_tAvg.nc -m2 ../../../../data/MALI/ENSEMBLES/CTRL/EM2/output_flux_all_timesteps_2100_tAvg.nc
#python plot_output_maps_masked.py -v dhdt -r ../../../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_dhdt_090yr.nc -m ../../../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_2000_tAvg.nc -m2 ../../../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_2090_tAvg.nc

#python plot_output_maps_masked.py -v dhdt -r ../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_dhdt_090yr.nc -m ../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_2000_tAvg.nc -m2 ../../data/MALI/ENSEMBLES/CTRL/EM3/output_flux_all_timesteps_2090_tAvg.nc



ncwa -a Time -O output_flux_all_timesteps_2000.nc output_flux_all_timesteps_2000_tAvg.nc
ncwa -a Time -O output_flux_all_timesteps_2225.nc output_flux_all_timesteps_2225_tAvg.nc
ncecat -O -u Time output_flux_all_timesteps_2000_tAvg.nc output_flux_all_timesteps_2000_tAvg.nc
ncecat -O -u Time output_flux_all_timesteps_2200_tAvg.nc output_flux_all_timesteps_2200_tAvg.nc
ncdiff -O -v thickness output_flux_all_timesteps_2200_tAvg.nc output_flux_all_timesteps_2000_tAvg.nc output_flux_all_timesteps_thickness_2200-2000_diff.nc
ncap2 -O -s dhdt=thickness/200 output_flux_all_timesteps_thickness_2200-2000_diff.nc output_flux_all_timesteps_dhdt_200yr.nc

python plot_output_maps_masked.py -v dhdt -r ../../data/MALI/ISMIP6/SSP585-RIGNOT2013/output_flux_all_timesteps_dhdt_200yr.nc -m ../../data/MALI/ISMIP6/SSP585-RIGNOT2013/output_flux_all_timesteps_2000_tAvg.nc -m2 ../../data/MALI/ISMIP6/SSP585-RIGNOT2013/output_flux_all_timesteps_2200_tAvg.nc