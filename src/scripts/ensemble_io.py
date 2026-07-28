"""
ensemble_io.py — shared loader/statistics layer for AISLENS ensemble analysis.

Discovers ensemble members, loads globalStats.nc/regionalStats.nc into xarray
with a 'year' coordinate and 'member' dimension. Converts VAF (m^3) to sea-level
equivalent (mm). Maps ISMIP6 basins to friendly names (FRIS = J-K, Ross = E-F, ...).

Directory convention: <ENSEMBLES_ROOT>/<ensemble>/<member>/{global,regional}Stats.nc

Author: Shivaprakash Muruganandham
"""
from __future__ import annotations

import os
import re
import glob
from typing import Optional

import numpy as np
import xarray as xr

try:
    from aislens.config import config
    RHO_ICE = float(config.RHO_ICE)          # 910 kg m^-3
    _DIR_MALI = config.DIR_MALI
except Exception:
    RHO_ICE = 910.0
    _DIR_MALI = None

# Sea-level-equivalent conversion constants (matched to the MPAS-Tools plotters:
# rhoi=910, rhosw=1028, ocean area 3.62e14 -- see plot_ensembleGlobalStats.py).
RHO_OCEAN = 1028.0          # kg m^-3, seawater
OCEAN_AREA = 3.62e14        # m^2, global ocean area (matches MPAS-Tools plotters)

# SIGN CONVENTION: vaf_to_sle_mm() returns sea-level RISE as POSITIVE (ice/VAF loss
# -> +mm). NOTE this is the NEGATIVE of the MPAS-Tools plotters, which plot VAF change
# directly (loss -> negative) and rescale to a sea-level axis with a positive factor.

def default_ensembles_root() -> str:
    if _DIR_MALI is not None:
        return str(_DIR_MALI / "diagnostics" / "ENSEMBLES")
    return os.path.join("data", "MALI", "diagnostics", "ENSEMBLES")


# ----------------------------------------------------------------------------
# ISMIP6 basin -> friendly name (mirrors plot_ensembleRegionalStats.py, Paolo23).
# Keys are the region name with all non-alphanumeric chars stripped, e.g.
# "ISMIP6 Basin J-K" -> "ISMIP6BasinJK".
# ----------------------------------------------------------------------------
ISMIP6_BASIN_NAMES = {
    "ISMIP6BasinAAp": "Dronning Maud Land",
    "ISMIP6BasinApB": "Enderby Land",
    "ISMIP6BasinBC": "Amery-Lambert",
    "ISMIP6BasinCCp": "Phillipi, Denman",
    "ISMIP6BasinCpD": "Totten",
    "ISMIP6BasinDDp": "Mertz",
    "ISMIP6BasinDpE": "Victoria Land",
    "ISMIP6BasinEF": "Ross",
    "ISMIP6BasinFG": "Getz",
    "ISMIP6BasinGH": "Thwaites/PIG",
    "ISMIP6BasinHHp": "Bellingshausen",
    "ISMIP6BasinHpI": "George VI",
    "ISMIP6BasinIIpp": "Larsen A-C",
    "ISMIP6BasinIppJ": "Larsen E",
    "ISMIP6BasinJK": "FRIS",
    "ISMIP6BasinKA": "Brunt-Stancomb",
}


# ----------------------------------------------------------------------------
# Member discovery & sorting
# ----------------------------------------------------------------------------
def natural_sort_key(s: str):
    """Sort so that member_2 < member_10 (numbers compared numerically)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def discover_members(ensemble_dir: str, stats_filename: str = "globalStats.nc",
                     include: Optional[str] = None):
    """Return [(member_name, stats_path), ...] for every subdir that contains
    `stats_filename`. Handles the repo's inconsistent member naming by globbing
    rather than assuming a pattern.

    `include` is an optional regex (matched with re.search against the member name)
    so you can select a clean subset, e.g. include=r'^SSP585_\\d+$' to take the
    0..300 yr batch and skip short -EM runs, _V2 negative-year runs, and CHANGEPOINT
    test runs that would otherwise truncate the ensemble.
    """
    members = []
    if not os.path.isdir(ensemble_dir):
        raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")
    pat = re.compile(include) if include else None
    for name in sorted(os.listdir(ensemble_dir), key=natural_sort_key):
        if pat and not pat.search(name):
            continue
        p = os.path.join(ensemble_dir, name, stats_filename)
        if os.path.isfile(p):
            members.append((name, p))
    return members


# ----------------------------------------------------------------------------
# Loading globalStats
# ----------------------------------------------------------------------------
def _add_year_coord(ds: xr.Dataset) -> xr.Dataset:
    """Attach a 'year' coordinate (float years since run start) from daysSinceStart."""
    if "daysSinceStart" in ds:
        yr = ds["daysSinceStart"].values / 365.0
        ds = ds.assign_coords(year=("Time", yr))
    return ds


def to_year_dim(ds: xr.Dataset) -> xr.Dataset:
    """Swap the Time dimension for a clean 'year' dimension: MALI globalStats often
    has duplicate / non-monotonic daysSinceStart (from run restarts), which breaks
    xarray interp/align. We drop duplicate years (keep first) and sort ascending."""
    if "year" not in ds.coords:
        ds = _add_year_coord(ds)
    ds = ds.swap_dims({"Time": "year"})
    yr = ds["year"].values
    _, first_idx = np.unique(yr, return_index=True)
    ds = ds.isel(year=np.sort(first_idx)).sortby("year")
    return ds


def load_member_globalstats(path: str) -> xr.Dataset:
    ds = xr.open_dataset(path, decode_times=False)
    return _add_year_coord(ds)


def load_ensemble_globalstats(
    ensemble_dir: str,
    variables=None,
    stats_filename: str = "globalStats.nc",
    include: Optional[str] = None,
    min_years: Optional[float] = None,
    start_at_zero: bool = True,
    align: str = "union",
    start_tol: float = 10.0,
) -> xr.Dataset:
    """Load every member's globalStats.nc and concatenate along a new 'member' dim.

    The ENSEMBLES dirs mix real members with short test/diagnostic runs (e.g.
    CHANGEPOINT, 5 timesteps) and alternative time conventions (_V2, negative years).
    Blindly aligning to the shortest member collapses the ensemble to a junk window,
    so:
      * `include`   : regex to keep only a clean member subset (see discover_members)
      * `min_years` : drop members whose record spans fewer than this many years
      * `start_at_zero`: shift each member's year axis to start at 0 (guards against
                         the negative-year _V2 convention); off if you need absolute yr

    After filtering, members are aligned onto the SHORTEST SURVIVING member's year axis
    (interpolation, no extrapolation). Returns dims (member, year).
    """
    members = discover_members(ensemble_dir, stats_filename, include=include)
    if not members:
        raise RuntimeError(f"No members with {stats_filename} under {ensemble_dir}"
                           + (f" matching /{include}/" if include else ""))

    per_member, names, dropped, offaxis = [], [], [], []
    for name, path in members:
        ds = load_member_globalstats(path)
        if variables is not None:
            keep = [v for v in variables if v in ds]
            ds = ds[keep]
        d = to_year_dim(ds)                     # clean 'year' dim (dedup+sort)
        raw_year0 = float(d["year"].values[0])
        # A member's record should begin at (near) year 0. If it starts far from 0 it
        # is a restart-only CONTINUATION segment (daysSinceStart continues from the
        # restart, e.g. yr ~200) or an off-convention run (e.g. _V2 negative years).
        # Shifting such a member to year 0 mislabels late states as early and corrupts
        # the ensemble statistics -- so drop it rather than shift.
        if abs(raw_year0) > start_tol:
            offaxis.append((name, round(raw_year0, 1)))
            continue
        if start_at_zero and raw_year0 != 0.0:
            d = d.assign_coords(year=d["year"] - raw_year0)  # tiny cleanup only
        span = float(d["year"].values[-1] - d["year"].values[0])
        if min_years is not None and span < min_years:
            dropped.append((name, round(span, 1)))
            continue
        per_member.append(d)
        names.append(name)

    if offaxis:
        print(f"  [ensemble_io] dropped {len(offaxis)} member(s) not starting near year 0 "
              f"(restart-continuation / off-convention segments): {offaxis}")
    if dropped:
        print(f"  [ensemble_io] dropped {len(dropped)} member(s) shorter than "
              f"{min_years} yr: {dropped}")
    if not per_member:
        raise RuntimeError("All members were dropped by min_years/include filters.")

    # Choose the common year grid.
    #   intersection (default): shortest member's axis -> every year has ALL members
    #                           (rectangular; conservative; truncates to the shortest run)
    #   union ("full period")  : longest member's axis -> shows the full available span;
    #                           members are NaN-padded beyond their own end, so late-year
    #                           statistics are over FEWER members (see 'n_members' below).
    end_year = lambda d: float(d["year"].values[-1])
    if align == "union":
        ref = max(per_member, key=end_year)
    elif align == "intersection":
        ref = min(per_member, key=end_year)
    else:
        raise ValueError("align must be 'intersection' or 'union'")
    ref_year = ref["year"].values
    span = ref_year[-1] - ref_year[0]
    longest = max(end_year(d) - float(d["year"].values[0]) for d in per_member)
    if align == "intersection" and span < 0.5 * longest:
        print(f"  [ensemble_io] NOTE: intersection span is {span:.1f} yr but the longest "
              f"member reaches {longest:.1f} yr. Use --full-period to show the full span "
              f"(or --min-years to drop short runs).")
    # interp onto ref_year; outside each member's own range -> NaN (no extrapolation)
    aligned = [d.interp(year=ref_year) for d in per_member]

    out = xr.concat(aligned, dim="member")
    out = out.assign_coords(member=("member", names))
    return out


def load_ensemble_regionalstats(
    ensemble_dir: str,
    variables=("regionalVolumeAboveFloatation",),
    stats_filename: str = "regionalStats.nc",
    include: Optional[str] = None,
    min_years: Optional[float] = None,
    start_at_zero: bool = True,
    align: str = "union",
    start_tol: float = 10.0,
) -> xr.Dataset:
    """Load every member's regionalStats.nc and concatenate along a new 'member' dim.

    Same guardrails as load_ensemble_globalstats — DROPS restart-continuation segments that
    start far from year 0 (|year0|>start_tol), drops runs shorter than min_years, and aligns
    onto a common year grid — but preserves the (year, nRegions) structure. Returns dims
    (member, year, nRegions). Use this instead of a hand-rolled discover_members +
    load_member_regionalstats loop so the restart-segment guard is applied consistently.
    """
    members = discover_members(ensemble_dir, stats_filename, include=include)
    if not members:
        raise RuntimeError(f"No members with {stats_filename} under {ensemble_dir}"
                           + (f" matching /{include}/" if include else ""))
    per_member, names, dropped, offaxis = [], [], [], []
    for name, path in members:
        ds = load_member_regionalstats(path)
        if variables is not None:
            keep = [v for v in variables if v in ds]
            ds = ds[keep]
        d = to_year_dim(ds)
        raw_year0 = float(d["year"].values[0])
        if abs(raw_year0) > start_tol:            # restart-continuation segment -> drop, don't relabel
            offaxis.append((name, round(raw_year0, 1)))
            continue
        if start_at_zero and raw_year0 != 0.0:
            d = d.assign_coords(year=d["year"] - raw_year0)
        span = float(d["year"].values[-1] - d["year"].values[0])
        if min_years is not None and span < min_years:
            dropped.append((name, round(span, 1)))
            continue
        per_member.append(d); names.append(name)
    if offaxis:
        print(f"  [ensemble_io] dropped {len(offaxis)} member(s) not starting near year 0 "
              f"(restart-continuation / off-convention segments): {offaxis}")
    if dropped:
        print(f"  [ensemble_io] dropped {len(dropped)} member(s) shorter than {min_years} yr: {dropped}")
    if not per_member:
        raise RuntimeError("All members were dropped by min_years/include filters.")
    end_year = lambda d: float(d["year"].values[-1])
    if align == "union":
        ref = max(per_member, key=end_year)
    elif align == "intersection":
        ref = min(per_member, key=end_year)
    else:
        raise ValueError("align must be 'intersection' or 'union'")
    ref_year = ref["year"].values
    aligned = [d.interp(year=ref_year) for d in per_member]   # interp on year; nRegions preserved
    out = xr.concat(aligned, dim="member")
    out = out.assign_coords(member=("member", names))
    return out


# ----------------------------------------------------------------------------
# VAF -> sea-level equivalent
# ----------------------------------------------------------------------------
def vaf_to_sle_mm(vaf_m3, reference: str = "first"):
    """Convert a VAF time series (m^3 of ice) to sea-level-equivalent CHANGE in mm.

    SLE_rise = -(VAF - VAF_ref) * (rho_ice / rho_ocean) / A_ocean * 1000
    (VAF loss -> positive sea-level rise). `reference='first'` subtracts the initial
    value; pass reference='none' to get absolute SLE.
    """
    vaf = np.asarray(vaf_m3, dtype=float)
    if reference == "first":
        vaf0 = np.take(vaf, 0, axis=-1) if vaf.ndim > 1 else vaf[0]
        vaf = vaf - np.expand_dims(vaf0, axis=-1) if vaf.ndim > 1 else vaf - vaf0
    elif reference != "none":
        raise ValueError("reference must be 'first' or 'none'")
    return -vaf * (RHO_ICE / RHO_OCEAN) / OCEAN_AREA * 1000.0


def masked_cv(sig, mean, floor_mm: float = 5.0):
    """Coefficient of variation sigma/|mean| in PERCENT, masked where |mean| < floor_mm.

    Below the floor the ensemble-mean change is ~0, so sigma/|mean| is a division
    artifact (spikes to hundreds/thousands of %), NOT a variability peak. Masking to
    NaN there keeps only the physically meaningful CV. Use a floor of a few mm SLE.
    """
    am = np.abs(np.asarray(mean, dtype=float))
    sig = np.asarray(sig, dtype=float)
    return np.where(am >= floor_mm, 100.0 * sig / np.where(am > 0, am, np.nan), np.nan)


# ----------------------------------------------------------------------------
# Ensemble statistics over the 'member' dimension
# ----------------------------------------------------------------------------
def ensemble_stats(da: xr.DataArray) -> xr.Dataset:
    """Median / mean / std / IQR / 5-95 / skewness across members, as a Dataset."""
    from scipy.stats import skew
    stats = xr.Dataset(
        {
            "median": da.median("member"),
            "mean": da.mean("member"),
            "std": da.std("member"),
            "p05": da.quantile(0.05, "member").drop_vars("quantile"),
            "p25": da.quantile(0.25, "member").drop_vars("quantile"),
            "p75": da.quantile(0.75, "member").drop_vars("quantile"),
            "p95": da.quantile(0.95, "member").drop_vars("quantile"),
        }
    )
    stats["iqr"] = stats["p75"] - stats["p25"]
    stats["rel_uncertainty"] = stats["std"] / np.abs(stats["mean"])
    # skewness across members at each time
    sk = xr.apply_ufunc(
        lambda x: skew(x, axis=-1, nan_policy="omit"),
        da,
        input_core_dims=[["member"]],
        output_core_dims=[[]],
        vectorize=False,
    )
    stats["skewness"] = sk
    return stats


# ----------------------------------------------------------------------------
# Regional (regionalStats.nc) helpers
# ----------------------------------------------------------------------------
def read_region_names(ds: xr.Dataset):
    """Return friendly region names in file order, from a regionalStats.nc (or the
    regionMask file) that has a 'regionNames' char variable."""
    if "regionNames" not in ds:
        raise KeyError("No 'regionNames' variable in dataset")
    raw = ds["regionNames"].values  # (nRegions, StrLen) char/bytes
    names = []
    for row in raw:
        if isinstance(row, bytes):
            s = row.decode("utf-8", "ignore")
        else:
            s = b"".join([c if isinstance(c, bytes) else bytes(c) for c in row]).decode(
                "utf-8", "ignore"
            )
        key = "".join(filter(str.isalnum, s.strip()))
        names.append(ISMIP6_BASIN_NAMES.get(key, key))
    return names


def region_names_default(regionmask_path: Optional[str] = None):
    """Friendly region names from the ISMIP6 regionMask file (used when regionalStats.nc
    itself lacks a regionNames variable). Falls back to the repo's default mask."""
    if regionmask_path is None and _DIR_MALI is not None:
        regionmask_path = str(_DIR_MALI / "AIS_4to20km_r01_20220907.regionMask_ismip6.nc")
    if regionmask_path is None or not os.path.isfile(regionmask_path):
        return None
    ds = xr.open_dataset(regionmask_path, decode_times=False)
    return read_region_names(ds)


def region_index(names, target: str) -> int:
    """Index of a friendly region name (e.g. 'FRIS', 'Ross'). Case-insensitive."""
    low = [n.lower() for n in names]
    if target.lower() in low:
        return low.index(target.lower())
    raise ValueError(f"Region '{target}' not found. Available: {names}")


def load_member_regionalstats(path: str) -> xr.Dataset:
    ds = xr.open_dataset(path, decode_times=False)
    return _add_year_coord(ds)


if __name__ == "__main__":
    # Tiny self-test / summary when run directly.
    root = default_ensembles_root()
    print(f"ENSEMBLES root: {root}")
    if os.path.isdir(root):
        for e in sorted(os.listdir(root)):
            d = os.path.join(root, e)
            if os.path.isdir(d):
                try:
                    m = discover_members(d)
                except Exception:
                    m = []
                if m:
                    print(f"  {e:32s} {len(m):3d} members  (e.g. {m[0][0]})")
