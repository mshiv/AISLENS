import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.colorbar import Colorbar
from netCDF4 import Dataset
import seaborn as sns

# ----------------------------------------
# Global and Regional Statistics
# ----------------------------------------

def plot_global_stats(file_path, variable, output_path, time_start=0, time_end=None):
    """
    Plot global statistics from a NetCDF file.

    Args:
        file_path (str): Path to the NetCDF file.
        variable (str): Variable to plot.
        output_path (str): Path to save the plot.
        time_start (int): Start time index. Defaults to 0.
        time_end (int): End time index. Defaults to None.
    """
    f = xr.open_dataset(file_path)
    var = f[variable][time_start:time_end]
    time = f['daysSinceStart'][time_start:time_end] / 365.0

    ax = setup_plot(f"Global {variable}", "Year", variable)
    ax.plot(time, var, label=variable)
    ax.legend()

    save_plot(ax.figure, output_path)

def plot_regional_stats(file_path, variable, region_name, output_path, time_start=0, time_end=None):
    """
    Plot regional statistics for a specific region.

    Args:
        file_path (str): Path to the NetCDF file.
        variable (str): Variable to plot.
        region_name (str): Name of the region.
        output_path (str): Path to save the plot.
        time_start (int): Start time index. Defaults to 0.
        time_end (int): End time index. Defaults to None.
    """
    f = xr.open_dataset(file_path)
    var = f[variable][time_start:time_end]
    time = f['daysSinceStart'][time_start:time_end] / 365.0

    ax = setup_plot(f"Regional {variable}: {region_name}", "Year", variable)
    ax.plot(time, var, label=region_name)
    ax.legend()

    save_plot(ax.figure, output_path)

# ----------------------------------------
# Draft Dependence
# ----------------------------------------

def plot_draft_dependence(draft_data, melt_flux_data, output_path):
    """
    Plot draft dependence of melt flux.

    Args:
        draft_data (array-like): Draft data.
        melt_flux_data (array-like): Melt flux data.
        output_path (str): Path to save the plot.
    """
    ax = setup_plot("Draft Dependence of Melt Flux", "Draft (m)", "Melt Flux (m³/s)")
    ax.scatter(draft_data, melt_flux_data, label="Data Points", s=10, c="blue")
    ax.legend()

    save_plot(ax.figure, output_path)

# ----------------------------------------
# Map Plotting
# ----------------------------------------

def plot_map(data, x, y, title, output_path, cmap="viridis", vmin=None, vmax=None):
    """
    Plot a 2D map of data.

    Args:
        data (2D array): Data to plot.
        x (1D array): X-coordinates.
        y (1D array): Y-coordinates.
        title (str): Title of the plot.
        output_path (str): Path to save the plot.
        cmap (str): Colormap. Defaults to "viridis".
        vmin (float): Minimum value for the color scale. Defaults to None.
        vmax (float): Maximum value for the color scale. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.pcolormesh(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(c, ax=ax, label="Value")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    save_plot(fig, output_path)

# ----------------------------------------
# Seasonal and Trend Analysis
# ----------------------------------------

def plot_seasonal_cycle(melt, domain, output_path):
    """
    Plot the spatial distribution of the seasonal cycle of the melt rate over a defined domain.

    Args:
        melt (xarray.DataArray): Melt rate data.
        domain (str): Domain name.
        output_path (str): Path to save the plot.
    """
    melt = melt - melt.mean(dim='Time')
    p = melt.polyfit(dim='Time', deg=1)
    seasonal = melt - xr.polyval(melt.Time, p.polyfit_coefficients)
    seasonal = seasonal.groupby('Time.month').mean(dim='Time')

    fig, ax = plt.subplots(figsize=(8, 6))
    seasonal.plot(ax=ax)
    ax.set_title(f"Seasonal Cycle of Melt Rate ({domain})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    save_plot(fig, output_path)

def plot_trend(melt, domain, output_path):
    """
    Plot the spatial distribution of the linear trend of the melt rate over a defined domain.

    Args:
        melt (xarray.DataArray): Melt rate data.
        domain (str): Domain name.
        output_path (str): Path to save the plot.
    """
    melt = melt - melt.mean(dim='Time')
    p = melt.polyfit(dim='Time', deg=1)
    trend = xr.polyval(melt.Time, p.polyfit_coefficients)

    fig, ax = plt.subplots(figsize=(8, 6))
    trend.plot(ax=ax)
    ax.set_title(f"Trend of Melt Rate ({domain})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    save_plot(fig, output_path)

# ----------------------------------------
# Time Series and Histograms
# ----------------------------------------

def plot_time_series(data, time, output_path, title="Time Series", ylabel="Value"):
    """
    Plot a time series.

    Args:
        data (array-like): Data to plot.
        time (array-like): Time values.
        output_path (str): Path to save the plot.
        title (str): Title of the plot. Defaults to "Time Series".
        ylabel (str): Label for the y-axis. Defaults to "Value".
    """
    ax = setup_plot(title, "Time", ylabel)
    ax.plot(time, data, label="Time Series")
    ax.legend()

    save_plot(ax.figure, output_path)

def plot_histogram(data, bins, output_path, title="Histogram", xlabel="Value", ylabel="Frequency"):
    """
    Plot a histogram.

    Args:
        data (array-like): Data to plot.
        bins (int): Number of bins.
        output_path (str): Path to save the plot.
        title (str): Title of the plot. Defaults to "Histogram".
        xlabel (str): Label for the x-axis. Defaults to "Value".
        ylabel (str): Label for the y-axis. Defaults to "Frequency".
    """
    ax = setup_plot(title, xlabel, ylabel)
    ax.hist(data, bins=bins, color="blue", alpha=0.7)
    save_plot(ax.figure, output_path)

def plot_variable_time_series(data, time, variable_name, output_path, title=None):
    """
    Plot a time series of a specific variable.

    Args:
        data (xarray.DataArray): Data to plot.
        time (xarray.DataArray): Time values.
        variable_name (str): Name of the variable being plotted.
        output_path (str): Path to save the plot.
        title (str, optional): Title of the plot. Defaults to None.
    """
    if title is None:
        title = f"Time Series of {variable_name}"

    ax = setup_plot(title, "Time", variable_name)
    ax.plot(time, data, label=variable_name)
    ax.legend()

    save_plot(ax.figure, output_path)

def plot_spatial_map(data, x, y, variable_name, output_path, cmap="RdBu_r", vmin=None, vmax=None):
    """
    Plot a spatial map of a variable.

    Args:
        data (xarray.DataArray): 2D data to plot.
        x (xarray.DataArray): X-coordinates (e.g., longitude).
        y (xarray.DataArray): Y-coordinates (e.g., latitude).
        variable_name (str): Name of the variable being plotted.
        output_path (str): Path to save the plot.
        cmap (str): Colormap. Defaults to "RdBu_r".
        vmin (float, optional): Minimum value for the color scale. Defaults to None.
        vmax (float, optional): Maximum value for the color scale. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.pcolormesh(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(c, ax=ax, label=variable_name)
    ax.set_title(f"Spatial Map of {variable_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    save_plot(fig, output_path)

def setup_plot(title, xlabel, ylabel, grid=True):
    """
    Set up a basic plot with title, labels, and optional grid.

    Args:
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        grid (bool): Whether to display a grid. Defaults to True.

    Returns:
        matplotlib.axes.Axes: The Axes object for the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid()
    return ax

def save_plot(fig, output_path):
    """
    Save a plot to a file.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        output_path (str): Path to save the plot.
    """
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

# ----------------------------------------
# General Plotting Utilities
# ----------------------------------------

def create_diverging_cmap(low_color, high_color):
    """
    Create a diverging colormap with white in the center.

    Args:
        low_color (str): Color for the low end.
        high_color (str): Color for the high end.

    Returns:
        LinearSegmentedColormap: Custom diverging colormap.
    """
    from matplotlib.colors import LinearSegmentedColormap
    colors = [low_color, 'white', high_color]
    return LinearSegmentedColormap.from_list("custom", colors, N=200)
