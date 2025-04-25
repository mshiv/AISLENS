import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from netCDF4 import Dataset
import xarray as xr
import seaborn as sns

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
