#!/usr/bin/env python
'''
Script to plot common time-series from one or more landice globalStats files for diagnostics.
This script is a modified version of the original plot_globalStats.py script, allowing for plotting different variables individually.

Created on Jul 05, 2024

@author: Shivaprakash Muruganandham
'''
import sys
import os
os.environ['USE_PYGEOS'] = '0'
import gc
import collections
from pathlib import Path
from optparse import OptionParser

import matplotlib.pyplot as plt
# import seaborn as sns

import xarray as xr
from netCDF4 import Dataset
import pandas as pd
import scipy
from scipy import signal
import cftime

parser = OptionParser(description=__doc__)
parser.add_option("-v", dest="variable", help="input variable to plot", default="totalFloatingBasalMassBal")
parser.add_option("--tStart", dest="timeLevelStart", help="first time level to plot", default=0, type=int)
parser.add_option("--tEnd", dest="timeLevelEnd", help="last time level to plot", default=1000, type=int)
parser.add_option("-1", dest="file1inName", help="input filename", default="globalStats.nc", metavar="FILENAME")
parser.add_option("-2", dest="file2inName", help="input filename", metavar="FILENAME")
parser.add_option("-3", dest="file3inName", help="input filename", metavar="FILENAME")
parser.add_option("-4", dest="file4inName", help="input filename", metavar="FILENAME")
options, args = parser.parse_args()

plt.figure(figsize=(25,8))

def plotStat(fname, variable, timeLevelStart, timeLevelEnd):
    print("Reading and plotting file: {}".format(fname))

    name = fname
    f = xr.open_dataset(fname)

    # f = Dataset(fname, 'r')
    var = f[variable][timeLevelStart:timeLevelEnd]

    #yr = f.variables['daysSinceStart'][:]/365.0
    #yr = yr-yr[0]

    var.plot(label=name)
    plt.xlabel('Model Simulation Time Level')
    plt.ylabel(variable)

plotStat(options.file1inName, options.variable, options.timeLevelStart, options.timeLevelEnd)

if options.file2inName:
    plotStat(options.file2inName, options.variable, options.timeLevelStart, options.timeLevelEnd)

if options.file3inName:
    plotStat(options.file3inName, options.variable, options.timeLevelStart, options.timeLevelEnd)

if options.file4inName:
    plotStat(options.file4inName, options.variable, options.timeLevelStart, options.timeLevelEnd)

plt.legend()
print("Generating plot")
plt.tight_layout()
plt.show()