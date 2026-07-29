#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import netCDF4
import numpy as np

#f = netCDF4.Dataset('/storage/home/hcoda1/6/smurugan9/data/MALI_projects/ISMIP6-2300/initial_conditions/AIS_4to20km_20230105/AIS_4to20km_r01_20220907_ICV_SORRMv21_extrapl_2300_Forcing.nc', 'r+')

f = netCDF4.Dataset('/storage/home/hcoda1/6/smurugan9/scratch/AISLENS/data/processed/vargen_realizations/AIS_4to20km_r01_20220907_AISLENS-Forcing_0.nc', 'r+')
nt = len(f.dimensions['Time'])

# It seems much faster for large files to use NCO to add xtime, like:
# ncap2 -s 'defdim("StrLen",64); xtime[$Time,$StrLen]=" "' TF_NorESM_2.6_1995-2100.nc temp.nc
StrLen=64
if 'xtime' in f.variables:
    xtime = f.variables['xtime']
else:
    if not 'StrLen' in f.dimensions:
       print("adding StrLen")
       f.createDimension('StrLen', StrLen)
    print("adding xtime")
    xtime = f.createVariable('xtime','c', ('Time', 'StrLen'))

startYear = 2000
for i in range(300):
    for m in range(12):
        time_string = f.variables['xtime'][i*12+m,:]
        time_string = "{:04}-{:02}-01_00:00:00".format(i+startYear, m+1)
        print(time_string)
        f.variables['xtime'][i*12+m,:] = list(time_string.ljust(StrLen))

f.close()
