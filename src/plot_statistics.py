# To plot exploratory statistics and maps of the raw and processed datasets
# TODO: Convert to class and add more methods

"""
Functions:
0. clip_data: Clip the map to a specific domain (using rio.clip to set the domain mask)
    Calling this function is optional. If the user does not call this function, the entire ice sheet will be plotted.
    If the user calls this function, the map will be clipped to the specified domain.
    It will need to be called before any of the other functions, when used.
   
1. scatter: Plot the dependence of melt rate on draft for a given ice shelf domain
2. histogram: Plot the histogram of the melt rate for a given ice shelf domain
3. time_series: Plot a spatially averaged time series of the melt rate over a defined domain (can be full ice sheet, or specific regions)
4. power_spectrum: Plot the power spectrum of the melt rate time series
5. autocorrelation: Plot the autocorrelation of the melt rate time series
6. map_variance: Plot the spatial distribution of the time variance of the melt rate over a defined domain
7. map_trend: Plot the spatial distribution of the linear trend of the melt rate over a defined domain
8. map_seasonal: Plot the spatial distribution of the seasonal cycle of the melt rate over a defined domain
9. map_residual: Plot the spatial distribution of the residual melt rate after removing the linear trend and seasonal cycle
10. map_autocorrelation: Plot the spatial distribution of the autocorrelation of the melt rate over a defined domain
11. map_power_spectrum: Plot the spatial distribution of the power spectrum of the melt rate over a defined domain
12. map_standard_deviation: Plot the spatial distribution of the standard deviation of the melt rate over a defined domain
13. map_skewness: Plot the spatial distribution of the skewness of the melt rate over a defined domain
14. map_kurtosis: Plot the spatial distribution of the kurtosis of the melt rate over a defined domain
15. map_percentile: Plot the spatial distribution of the 95th percentile of the melt rate over a defined domain
"""
# Import necessary libraries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import signal
from shapely.geometry import mapping
import geopandas as gpd
import rioxarray

# Load the ice shelf geometry

def clip_data(data, domain):
    """
    Clip the map to a specific domain
    data: input data (xarray DataArray)
    domain: domain name (string), as defined in the ice shelf geometry file (icems)
    """
    # TODO: Include a step here to convert from domain name string to the domain index number used in the ice shelf geometry file
    clipped_data = data.rio.clip(icems.loc[domain,'geometry'].apply(mapping))
    #clipped_data = clipped_data.dropna('time',how='all')
    #clipped_data = clipped_data.dropna('y',how='all')
    #clipped_data = clipped_data.dropna('x',how='all')
    #clipped_data = clipped_data.drop("month")
    return clipped_data

def scatter(melt, draft):
    """
    Plot the dependence of melt rate on draft
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(draft, melt, s=1, c='k')
    plt.xlabel('Draft (m)')
    plt.ylabel('Melt rate (m/yr)')
    plt.title('Melt rate vs. draft')
    plt.show()

def histogram(melt):
    """
    Plot the histogram of the melt rate
    """
    plt.figure(figsize=(8, 6))
    plt.hist(melt, bins=100, color='k')
    plt.xlabel('Melt rate (m/yr)')
    plt.ylabel('Frequency')
    plt.title('Melt rate histogram')
    plt.show()

def time_series(melt, domain):
    """
    Plot a spatially averaged time series of the melt rate over a defined domain
    """
    plt.figure(figsize=(8, 6))
    melt.mean(dim=domain).plot()
    plt.xlabel('Time')
    plt.ylabel('Melt rate (m/yr)')
    plt.title('Melt rate time series')
    plt.show()

def power_spectrum(melt):
    """
    Plot the power spectrum of the melt rate time series
    """
    plt.figure(figsize=(8, 6))
    melt = melt - melt.mean()
    f, Pxx = signal.periodogram(melt, fs=1)
    plt.loglog(f, Pxx, color='k')
    plt.xlabel('Frequency (1/yr)')
    plt.ylabel('Power')
    plt.title('Power spectrum of melt rate')
    plt.show()

def autocorrelation(melt):
    """
    Plot the autocorrelation of the melt rate time series
    """
    plt.figure(figsize=(8, 6))
    ac = signal.correlate(melt, melt, mode='full')
    ac = ac / np.max(ac)
    plt.plot(ac, color='k')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation of melt rate')
    plt.show()

def map_variance(melt, domain):
    """
    Plot the spatial distribution of the time variance of the melt rate over a defined domain
    """
    plt.figure(figsize=(8, 6))
    melt.var(dim=domain).plot()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Variance of melt rate')
    plt.show()

def map_trend(melt, domain):
    """
    Plot the spatial distribution of the linear trend of the melt rate over a defined domain
    """
    plt.figure(figsize=(8, 6))
    melt = melt - melt.mean(dim='Time')
    p = melt.polyfit(dim='Time', deg=1)
    trend = xr.polyval(melt.Time, p.polyfit_coefficients)
    trend.plot()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Trend of melt rate')
    plt.show()

def map_seasonal(melt, domain):
    """
    Plot the spatial distribution of the seasonal cycle of the melt rate over a defined domain
    """
    plt.figure(figsize=(8, 6))
    melt = melt - melt.mean(dim='Time')
    p = melt.polyfit(dim='Time', deg=1)
    seasonal = melt - xr.polyval(melt.Time, p.polyfit_coefficients)
    seasonal = seasonal.groupby('Time.month').mean(dim='Time')
    seasonal.plot()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Seasonal cycle of melt rate')
    plt.show()

def map_residual(melt, domain):
    """
    Plot the spatial distribution of the residual melt rate after removing the linear trend and seasonal cycle
    """
    plt.figure(figsize=(8, 6))
    melt = melt - melt.mean(dim='Time')
    p = melt.polyfit(dim='Time', deg=1)
    trend = xr.polyval(melt.Time, p.polyfit_coefficients)
    seasonal = melt - trend
    seasonal = seasonal.groupby('Time.month').mean(dim='Time')
    residual = melt - trend - seasonal
    residual.plot()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Residual melt rate')
    plt.show()