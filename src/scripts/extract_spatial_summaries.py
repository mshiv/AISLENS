"""
extract_spatial_summaries.py — lightweight summary NetCDFs from output_state files.

Run ONCE on the HPC cluster after all ensemble members have completed, then
scp/scp the 4 summary files per scenario (~100 KB each) to your laptop so
AISLENS-06-spatial-output-analysis.ipynb can be run without direct HPC access.

Usage:
    python src/scripts/extract_spatial_summaries.py \\
        --ensembles-dir /path/to/ENSEMBLES \\
        --out-dir /path/to/summaries

Generates per scenario:
    velocity_evolution_<scenario>.nc     (member, year, nRegions) surfaceSpeed
    basal_temp_beta_<scenario>.nc        (member, year, nRegions) temperature, beta
    thickness_regional_<scenario>.nc     (member, year, nRegions) thickness
    dhdt_summary_<scenario>.nc           (nRegions,) dhdt at yr100/200/300 ensemble stats
"""
from __future__ import annotations
import os, glob, argparse, sys
from typing import List, Tuple

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
from spatial_io import (find_output_state_files, load_spatial_variable,
                         load_region_mask, default_paths, load_mesh_coords,
                         aggregate_by_region)
from ensemble_io import discover_members


SCENARIOS = {
    "SSP585":            dict(include=r"^SSP585_\d+$"),
    "SSP126":            dict(include=r"^SSP126_\d+$"),
    "SSP585_varScaled10x": dict(include=r"^SSP585_\d+$"),
    "CTRL":              dict(include=r"^CTRL_\d+$"),
}

VARS = {
    "surfaceSpeed":      dict(load_name="surfaceSpeed"),
    "temperature":       dict(load_name="temperature"),
    "basalFrictionFlowing": dict(load_name="basalFrictionFlowing"),
    "thickness":         dict(load_name="thickness"),
    "floatingBasalMassBalApplied": dict(load_name="floatingBasalMassBalApplied"),
}

YEAR_RANGE = (0, 300)


def extract_member_summaries(
    member_dir: str,
    var_list: List[str],
    y0: int,
    y1: int,
    region_mask: np.ndarray,
    n_regions: int = 16,
) -> dict:
    """Load spatial variables for one member over [y0, y1] and aggregate by region.

    Returns {varname: (nYears, nRegions) array}.
    """
    results = {}
    for v in var_list:
        try:
            data, years = load_spatial_variable(member_dir, v, y0, y1)
        except (FileNotFoundError, RuntimeError):
            continue
        regional = aggregate_by_region(data, region_mask, n_regions)
        results[v] = regional
        results[f"{v}_years"] = years
    return results


def build_summary_nc(
    out_path: str,
    scenario_name: str,
    member_data: dict,
    n_regions: int = 16,
):
    """Write a NetCDF with dims (member, year, nRegions) for each variable."""
    member_names = sorted(member_data.keys())
    all_years = None
    arrays = {}
    for v in ["surfaceSpeed", "temperature", "basalFrictionFlowing",
              "thickness"]:
        collected = []
        for mn in member_names:
            if v in member_data[mn]:
                arr = member_data[mn][v]
                collected.append(arr)
                if all_years is None:
                    all_years = member_data[mn].get(f"{v}_years",
                                                    np.arange(arr.shape[0]))
        if not collected:
            continue
        stack = np.stack(collected, axis=0)
        arrays[v] = (("member", "year", "nRegions"), stack)

    if not arrays:
        print(f"  No data for {scenario_name}")
        return

    coords = {
        "member": member_names,
        "year": all_years,
        "nRegions": np.arange(n_regions),
    }
    ds = xr.Dataset(arrays, coords=coords)
    ds.to_netcdf(out_path)
    print(f"  Wrote {out_path}  ({ds.nbytes / 1024:.0f} KB)")


def build_dhdt_summary(
    out_path: str,
    scenario_name: str,
    member_data: dict,
    years: np.ndarray,
    region_mask: np.ndarray,
    horizons: Tuple[int, ...] = (100, 200, 300),
):
    """Compute dH/dt ensemble stats at selected horizons from thickness."""
    n_regions = region_mask.max() + 1
    dhdt_maps = {}
    for h in horizons:
        hi = np.argmin(np.abs(years - h))
        if hi < 2:
            continue
        t0, t1 = years[hi - 1], years[hi]
        dt = t1 - t0
        dhdt_samples = []
        for mn in member_data:
            if "thickness" not in member_data[mn]:
                continue
            thick = member_data[mn]["thickness"]
            if thick.shape[0] <= hi or thick.shape[0] <= hi - 1:
                continue
            dh = (thick[hi] - thick[hi - 1]) / dt
            dhdt_samples.append(dh)
        if not dhdt_samples:
            continue
        dhdt_stack = np.stack(dhdt_samples, axis=0)
        dhdt_maps[f"dhdt_mean_yr{h}"] = (("nRegions",), dhdt_stack.mean(axis=0))
        dhdt_maps[f"dhdt_std_yr{h}"] = (("nRegions",), dhdt_stack.std(axis=0))

    if not dhdt_maps:
        print(f"  No dH/dt data for {scenario_name}")
        return

    ds = xr.Dataset(dhdt_maps, coords={"nRegions": np.arange(n_regions)})
    ds.to_netcdf(out_path)
    print(f"  Wrote {out_path}  ({ds.nbytes / 1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser(description="Extract spatial summaries from output_state files")
    ap.add_argument("--ensembles-dir", default=None,
                    help="Root ENSEMBLES directory. Defaults to spatial_io.default_paths base.")
    ap.add_argument("--out-dir", default="data/processed/spatial_summaries",
                    help="Output directory for summary NetCDFs")
    ap.add_argument("--scenarios", nargs="*",
                    default=list(SCENARIOS.keys()),
                    help=f"Scenarios to process (default: all: {list(SCENARIOS.keys())})")
    ap.add_argument("--y0", type=int, default=YEAR_RANGE[0])
    ap.add_argument("--y1", type=int, default=YEAR_RANGE[1])
    ap.add_argument("--n-regions", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mesh_path, mask_path = default_paths()
    if args.ensembles_dir:
        base_dir = args.ensembles_dir
    else:
        base_dir = os.path.dirname(os.path.dirname(mask_path))

    mask_path_local = mask_path
    if not os.path.isfile(mask_path_local):
        print(f"Warning: region mask not found at {mask_path_local}")
        return

    region_mask = load_region_mask(mask_path_local)
    print(f"Region mask: {len(region_mask)} cells, {args.n_regions} regions")

    for scen in args.scenarios:
        if scen not in SCENARIOS:
            print(f"Unknown scenario: {scen}, skipping")
            continue
        kw = SCENARIOS[scen]
        ensemble_dir = os.path.join(base_dir, scen)
        print(f"\nProcessing {scen} ({ensemble_dir}) ...")

        if not os.path.isdir(ensemble_dir):
            print(f"  Directory not found, skipping")
            continue

        # Second glob pattern to match any subdir pattern
        members = discover_members(ensemble_dir,
                                   stats_filename="output_state_*.nc",
                                   include=kw.get("include"))
        if not members:
            print(f"  No members with output_state files found")
            continue
        print(f"  Found {len(members)} members")

        member_data = {}
        for name, path in members:
            member_dir = os.path.dirname(path) if path else os.path.join(ensemble_dir, name)
            try:
                results = extract_member_summaries(
                    member_dir, list(VARS.keys()),
                    args.y0, args.y1, region_mask, args.n_regions)
                member_data[name] = results
            except (FileNotFoundError, RuntimeError) as e:
                print(f"    {name}: error ({e})")

        if not member_data:
            print(f"  No data extracted for {scen}")
            continue

        # Write per-variable-group summaries
        # 1. velocity/basal temperature/basal beta/thickness
        build_summary_nc(
            os.path.join(args.out_dir, f"spatial_evolution_{scen}.nc"),
            scen, member_data, args.n_regions)

        # 2. dH/dt at horizons
        # Find a common year axis from first member with thickness data
        yr_axis = None
        for mn in member_data:
            if "thickness_years" in member_data[mn]:
                yr_axis = member_data[mn]["thickness_years"]
                break
        if yr_axis is not None:
            build_dhdt_summary(
                os.path.join(args.out_dir, f"dhdt_summary_{scen}.nc"),
                scen, member_data, yr_axis, region_mask,
                horizons=(100, 200, 300))

    print("\nDone. Summary files in", args.out_dir)
    print("To copy to laptop:")
    print(f"  rsync -avP <hpc-host>:{args.out_dir}/ <local-path>/data/processed/spatial_summaries/")


if __name__ == "__main__":
    main()
