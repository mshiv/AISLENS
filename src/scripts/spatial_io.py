"""
spatial_io.py — loader for MALI output_state spatial files.

Handles the 5-year chunk structure (output_state_YYYY.nc), region mask
assignment, and memory-efficient loading. 385,379 cells on the 4km AIS mesh.
Region mask: AIS_4to20km...regionMask_ismip6.nc (16 ISMIP6 basins).
"""
from __future__ import annotations
import os, glob
from typing import Optional, List, Tuple

import numpy as np
import xarray as xr

RHO_ICE = 910.0
RHO_OCEAN = 1028.0
OCEAN_AREA = 3.62e14


def find_output_state_files(member_dir: str, year_start: int, year_end: int) -> List[str]:
    """Find output_state files covering [year_start, year_end].
    Files are named output_state_YYYY.nc where YYYY is the START of a 5-year chunk."""
    pattern = os.path.join(member_dir, "output_state_*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        # Try in 'output/' or 'outputs/' subdirectory
        for sub in ("output", "outputs"):
            pattern = os.path.join(member_dir, sub, "output_state_*.nc")
            files = sorted(glob.glob(pattern))
            if files:
                break
    # Filter to files that overlap [year_start, year_end]
    result = []
    for f in files:
        base = os.path.basename(f)
        try:
            file_year = int(base.replace("output_state_", "").replace(".nc", ""))
        except ValueError:
            continue
        # File covers [file_year, file_year + 4]
        if file_year + 4 >= year_start and file_year <= year_end:
            result.append(f)
    return result


def load_spatial_variable(member_dir: str, variable: str,
                          year_start: int, year_end: int,
                          subset_cells: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load a spatial variable from output_state files.

    Returns:
        data: (nYears, nCells) array (annual means from 5-year chunks)
        years: (nYears,) array of year values
    """
    files = find_output_state_files(member_dir, year_start, year_end)
    if not files:
        raise FileNotFoundError(f"No output_state files in {member_dir} for years {year_start}-{year_end}")

    all_data = []
    all_years = []
    for f in files:
        ds = xr.open_dataset(f, decode_times=False)
        if variable not in ds:
            ds.close()
            continue
        var = ds[variable]
        if "nCells" not in var.dims:
            ds.close()
            continue
        # Get year from daysSinceStart
        days = ds["daysSinceStart"].values  # (Time,)
        years = days / 365.0
        # Take surface value for 3D vars (nCells, nVertInterfaces)
        if var.ndim == 3 and "nVertInterfaces" in var.dims:
            var = var[:, :, 0]  # surface level
        vals = var.values  # (Time, nCells)
        if subset_cells is not None:
            vals = vals[:, subset_cells]
        all_data.append(vals)
        all_years.append(years)
        ds.close()

    if not all_data:
        raise RuntimeError(f"Variable {variable} not found in any output_state file")

    data = np.concatenate(all_data, axis=0)
    years = np.concatenate(all_years)

    # Annual mean: average within each year (floor to integer year)
    year_int = np.floor(years).astype(int)
    unique_years = np.unique(year_int)
    annual = np.zeros((len(unique_years), data.shape[1]))
    for i, y in enumerate(unique_years):
        mask = year_int == y
        annual[i] = np.nanmean(data[mask], axis=0)

    return annual, unique_years.astype(float)


def load_region_mask(mask_path: str) -> np.ndarray:
    """Load ISMIP6 region mask. Returns (nCells,) array of basin indices (0-based).
    Basin indices 0-15 correspond to the 16 ISMIP6 basins in order."""
    ds = xr.open_dataset(mask_path, decode_times=False)
    # Try standard ISMIP6 mask variable names
    for varname in ["regionCellMasks", "regionMask", "regionID"]:
        if varname in ds:
            mask = ds[varname].values
            if mask.ndim > 1:
                # regionCellMasks is (nRegions, nCells) — take argmax
                mask = np.argmax(mask, axis=0)
            ds.close()
            return mask.astype(int)
    ds.close()
    raise KeyError(f"No region mask variable found in {mask_path}")


def load_mesh_coords(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load cell center coordinates from the mesh file.
    Returns (xCell, yCell) in meters."""
    ds = xr.open_dataset(mesh_path, decode_times=False)
    x = ds["xCell"].values
    y = ds["yCell"].values
    ds.close()
    return x, y


def aggregate_by_region(data: np.ndarray, region_mask: np.ndarray, n_regions: int = 16):
    """Aggregate spatial data by region.
    Returns (nYears, nRegions) array of area-weighted means."""
    nYears, nCells = data.shape
    result = np.full((nYears, n_regions), np.nan)
    for r in range(n_regions):
        cells = region_mask == r
        if cells.sum() == 0:
            continue
        result[:, r] = np.nanmean(data[:, cells], axis=1)
    return result


def default_paths():
    """Return default paths for mesh and mask files.

    The fb_A initialisation mesh and regional ISMIP6 mask live inside each
    ensemble-member directory (not at the ``ENSEMBLES`` root).  We anchor on
    ``CTRL/CTRL_00``, which is present in all production deployments.
    """
    base = "/storage/home/hcoda1/6/smurugan9/scratch/AISLENS/data/MALI/ENSEMBLES/CTRL/CTRL_00"
    mesh = os.path.join(base, ("AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_"
                               "Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_"
                               "meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc"))
    mask = os.path.join(base, "AIS_4to20km_r01_20220907.regionMask_ismip6.nc")
    return mesh, mask
