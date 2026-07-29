#!/usr/bin/env python3
"""
Along-flow geometry profiles AND grounding-line migration from MPAS-LI output.

Two things this does:
  1. GEOMETRY CROSS-SECTION (default): samples model fields along a transect
     (GeoJSON LineString) and draws a side-view of the glacier -- upper surface,
     ice base (lowerSurface), and bed -- with the GROUNDING LINE marked. Overlay
     several years / members, or animate over time (--animate) to watch the GL
     migrate up/down the bed.
  2. GL-vs-TIME MIGRATION (--gl-vs-time): extracts the grounding-line position
     (distance inland along the transect) at every timestep and plots it against
     time. With several members it shows each member plus the ENSEMBLE MEAN and
     the min-max envelope -- the ensemble grounding-line retreat figure.

Grounding line is detected by flotation along the transect: ice is grounded where
its base sits on the bed (lowerSurface - bedTopography <= --gl-tol) and floating
where the base is above the bed. The GL is the floating->grounded transition going
from the terminus inland. (Set --gl-tol to taste; 10 m is a reasonable default.)

Input fields required: xCell, yCell, thickness, lowerSurface, bedTopography
(upperSurface optional). Use the full-field output, e.g. output_state_YEAR.nc.

Examples:
  # geometry cross-section, two years overlaid, GL marked
  python plot_profile_along_flowline.py -f flowlines.geojson \
    -i .../SSP585_00/output/output_state_2000.nc,.../SSP585_00/output/output_state_2100.nc \
    --transect Thwaites --time-indexes 0,-1 --save --output-dir reports/figures/flowline

  # animate one member's cross-section over its 5-yearly snapshots
  python plot_profile_along_flowline.py -f flowlines.geojson \
    -i '.../SSP585_00/output/output_state_*.nc' --transect Thwaites --animate --save

  # ensemble GL-vs-time: one file per member, GL migration + ensemble mean
  python plot_profile_along_flowline.py -f flowlines.geojson \
    -i '.../SSP585_*/output/output_state_2100.nc' --transect Thwaites --gl-vs-time --save
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree

try:
    import geopandas as gpd
    from shapely.geometry import LineString
except Exception as exc:  # pragma: no cover
    raise ImportError("This script requires geopandas and shapely for transect GeoJSON.") from exc

RHO_ICE = 910.0
RHO_SW = 1028.0
DEFAULT_SPACING_M = 1000.0
GEOM_FIELDS = ["thickness", "upperSurface", "lowerSurface", "bedTopography"]


@dataclass(frozen=True)
class TransectSpec:
    name: str
    line: LineString
    reverse: bool = False


# ---------------------------------------------------------------- transects / io
def parse_comma_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def expand_input_files(patterns: Sequence[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            files.extend(sorted(p for p in glob.glob(pattern) if os.path.isfile(p)))
        elif os.path.isfile(pattern):
            files.append(pattern)
    return files


def load_transects(geojson: str, name: Optional[str], reverse_flag: bool) -> List[TransectSpec]:
    gdf = gpd.read_file(geojson)
    if gdf.empty:
        raise ValueError(f"No transects in {geojson}")
    if "name" not in gdf.columns and "shelf_name" not in gdf.columns:
        raise ValueError("Transect GeoJSON needs a 'name' or 'shelf_name' per feature.")
    specs = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.geom_type != "LineString":
            continue
        tname = str(row.get("name", row.get("shelf_name", f"transect_{idx}")))
        if name is not None and tname != name:
            continue
        specs.append(TransectSpec(tname, geom, bool(row.get("reverse", False)) or reverse_flag))
    if name is not None and not specs:
        raise ValueError(f"Transect '{name}' not found in {geojson}")
    return specs


def get_cell_xy(ds: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(ds.variables["xCell"][:], float)
    y = np.asarray(ds.variables["yCell"][:], float)
    return (x[0] if x.ndim > 1 else x), (y[0] if y.ndim > 1 else y)


def get_time_axis(ds: Dataset) -> np.ndarray:
    if "daysSinceStart" in ds.variables:
        return np.asarray(ds.variables["daysSinceStart"][:], float) / 365.0
    if "thickness" in ds.variables and ds.variables["thickness"].ndim > 1:
        return np.arange(ds.variables["thickness"].shape[0], float)
    return np.array([0.0])


def sample_points(line: LineString, spacing_m: float):
    length = float(line.length)
    if length == 0:
        raise ValueError("Transect has zero length")
    n = max(2, int(np.ceil(length / spacing_m)) + 1)
    d = np.linspace(0.0, length, n)
    coords = np.array([line.interpolate(dd).coords[0] for dd in d], float)
    return d, coords[:, 0], coords[:, 1]


def nearest_cells(xc, yc, sx, sy):
    return cKDTree(np.column_stack((xc, yc))).query(np.column_stack((sx, sy)), k=1)[1].astype(int)


def read_profile(ds: Dataset, idx: np.ndarray, t: int) -> Dict[str, np.ndarray]:
    prof = {}
    for v in GEOM_FIELDS + ["cellMask"]:
        if v not in ds.variables:
            continue
        arr = np.asarray(ds.variables[v][:])
        row = arr if arr.ndim == 1 else arr[t]
        if row.ndim == 1 and row.shape[0] > np.max(idx):
            prof[v] = row[idx]
    return prof


# ---------------------------------------------------------------- grounding line
def grounding_line_distance(distances: np.ndarray, prof: Dict[str, np.ndarray],
                            tol: float) -> float:
    """Distance (m) of the floating->grounded transition along the transect (which
    runs terminus -> inland). Grounded where ice base is within `tol` of the bed.
    Returns NaN if no clear transition (all grounded / all floating / no ice)."""
    if "lowerSurface" not in prof or "bedTopography" not in prof:
        return np.nan
    base = np.asarray(prof["lowerSurface"], float)
    bed = np.asarray(prof["bedTopography"], float)
    ice = np.asarray(prof.get("thickness", np.ones_like(base)), float) > 1.0
    grounded = (base - bed <= tol) & ice          # True where grounded
    if grounded.all() or (~grounded & ice).sum() == 0:
        return np.nan                              # fully grounded (no shelf on transect)
    if not grounded.any():
        return np.nan                              # nothing grounded on transect
    # walk terminus(0)->inland; GL = first grounded index following a floating one
    for i in range(1, len(grounded)):
        if grounded[i] and not grounded[i - 1] and ice[i]:
            # linear interpolate the (base-bed) zero-tol crossing for sub-cell precision
            g0, g1 = (base - bed)[i - 1], (base - bed)[i]
            if g0 != g1:
                frac = (g0 - tol) / (g0 - g1)
                return float(distances[i - 1] + frac * (distances[i] - distances[i - 1]))
            return float(distances[i])
    # fallback: terminus-side edge of the inland grounded block
    return float(distances[np.argmax(grounded)])


# ---------------------------------------------------------------- geometry x-sect
def draw_geometry(ax, distances_km, prof, label=None, gl_tol=10.0, color=None, alpha=0.9):
    up = prof.get("upperSurface"); base = prof.get("lowerSurface"); bed = prof.get("bedTopography")
    if bed is not None:
        ax.plot(distances_km, bed, color="0.35", lw=1.4, alpha=0.9,
                label="bed" if label is None else None)
        ax.fill_between(distances_km, bed, np.minimum(bed, 0.0), color="0.85", alpha=0.4)
    if up is not None:
        ax.plot(distances_km, up, color=color or "C0", lw=1.6, alpha=alpha, label=label)
    if base is not None:
        ax.plot(distances_km, base, color=color or "C0", lw=1.2, ls="--", alpha=alpha)
    ax.axhline(0.0, color="tab:blue", lw=0.8, alpha=0.5)  # sea level
    gl = grounding_line_distance(distances_km * 1000.0, prof, gl_tol)
    if np.isfinite(gl):
        ax.axvline(gl / 1000.0, color=color or "k", lw=1.0, ls=":", alpha=0.7)
    ax.set_ylabel("elevation (m)")
    return gl


def cmd_geometry(files, transect, tindex, spacing_m, gl_tol, out_dir, save):
    with Dataset(files[0]) as ds:
        xc, yc = get_cell_xy(ds)
    dist, sx, sy = sample_points(transect.line, spacing_m)
    idx = nearest_cells(xc, yc, sx, sy)
    if transect.reverse:
        dist = dist[::-1]; idx = idx[::-1]
    dist_km = dist / 1000.0

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(1, len(files) * len(tindex))))
    k = 0
    for fp in files:
        with Dataset(fp) as ds:
            ta = get_time_axis(ds)
            for t in tindex:
                prof = read_profile(ds, idx, t)
                yr = ta[t] if abs(t) < len(ta) else t
                lbl = f"{os.path.basename(fp)} | {yr:.0f} yr"
                draw_geometry(ax, dist_km, prof, label=lbl, gl_tol=gl_tol, color=colors[k])
                k += 1
    ax.set_xlabel("distance from terminus (km)")
    ax.set_title(f"Along-flow geometry & grounding line: {transect.name}")
    ax.legend(fontsize=7, loc="best"); fig.tight_layout()
    if save:
        os.makedirs(out_dir or ".", exist_ok=True)
        out = os.path.join(out_dir or ".", f"{transect.name}_geometry_GL.png")
        fig.savefig(out, dpi=200); print(f"Saved {out}")
    return fig


def cmd_animate(files, transect, spacing_m, gl_tol, out_dir, save):
    """Animate the geometry cross-section over all timesteps in the given file(s)."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    with Dataset(files[0]) as ds:
        xc, yc = get_cell_xy(ds)
    dist, sx, sy = sample_points(transect.line, spacing_m)
    idx = nearest_cells(xc, yc, sx, sy)
    if transect.reverse:
        dist = dist[::-1]; idx = idx[::-1]
    dist_km = dist / 1000.0
    # gather frames (file, time_index, year)
    frames = []
    for fp in files:
        with Dataset(fp) as ds:
            ta = get_time_axis(ds)
            for t in range(len(ta)):
                frames.append((fp, t, ta[t]))
    fig, ax = plt.subplots(figsize=(11, 5))

    def render(i):
        ax.clear()
        fp, t, yr = frames[i]
        with Dataset(fp) as ds:
            prof = read_profile(ds, idx, t)
        gl = draw_geometry(ax, dist_km, prof, gl_tol=gl_tol, color="C0")
        ax.set_xlabel("distance from terminus (km)"); ax.set_ylabel("elevation (m)")
        gltxt = f"GL = {gl/1000.0:.0f} km" if np.isfinite(gl) else "GL n/a"
        ax.set_title(f"{transect.name}   year {yr:.0f}   {gltxt}")

    anim = FuncAnimation(fig, render, frames=len(frames), interval=400)
    if save:
        os.makedirs(out_dir or ".", exist_ok=True)
        out = os.path.join(out_dir or ".", f"{transect.name}_GL_migration.gif")
        anim.save(out, writer=PillowWriter(fps=3)); print(f"Saved {out}")
    return anim


# ---------------------------------------------------------------- GL vs time
def cmd_gl_vs_time(files, transect, spacing_m, gl_tol, out_dir, save):
    with Dataset(files[0]) as ds:
        xc, yc = get_cell_xy(ds)
    dist, sx, sy = sample_points(transect.line, spacing_m)
    idx = nearest_cells(xc, yc, sx, sy)
    if transect.reverse:
        dist = dist[::-1]; idx = idx[::-1]

    # per-file GL(t): each file may itself have a time axis, OR be one snapshot.
    series = []  # list of (label, years[], gl_km[])
    for fp in files:
        with Dataset(fp) as ds:
            ta = get_time_axis(ds)
            gls = []
            for t in range(len(ta)):
                gls.append(grounding_line_distance(dist, read_profile(ds, idx, t), gl_tol) / 1000.0)
            series.append((os.path.basename(fp), ta, np.array(gls)))

    fig, ax = plt.subplots(figsize=(9, 5))
    # If every file is a single-snapshot member, treat files as ENSEMBLE at one time.
    single = all(len(yrs) == 1 for _, yrs, _ in series)
    if single and len(series) > 1:
        gl_vals = np.array([g[0] for _, _, g in series], float)
        gl_vals = gl_vals[np.isfinite(gl_vals)]
        ax.bar(range(len(series)), [g[0] for _, _, g in series], color="C0", alpha=0.7)
        ax.axhline(np.nanmean([g[0] for _, _, g in series]), color="k", lw=1.5,
                   label=f"ensemble mean = {np.nanmean(gl_vals):.0f} km")
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([s[0] for s in series], rotation=90, fontsize=6)
        ax.set_ylabel("GL distance from terminus (km)")
        ax.set_title(f"{transect.name}: grounding-line position across members")
        ax.legend()
    else:
        # time series per member + ensemble mean/envelope on a common year grid
        allyr = sorted(set(np.concatenate([y for _, y, _ in series])))
        allyr = np.array(allyr)
        stacked = []
        for lbl, yrs, gls in series:
            ax.plot(yrs, gls, color="0.7", lw=0.8)
            stacked.append(np.interp(allyr, yrs, gls, left=np.nan, right=np.nan))
        stacked = np.array(stacked)
        mean = np.nanmean(stacked, axis=0)
        ax.plot(allyr, mean, "k", lw=2, label="ensemble mean")
        ax.fill_between(allyr, np.nanmin(stacked, 0), np.nanmax(stacked, 0),
                        color="C0", alpha=0.2, label="member min-max")
        ax.set_xlabel("year"); ax.set_ylabel("GL distance from terminus (km)")
        ax.set_title(f"{transect.name}: grounding-line migration")
        ax.legend()
    fig.tight_layout()
    if save:
        os.makedirs(out_dir or ".", exist_ok=True)
        out = os.path.join(out_dir or ".", f"{transect.name}_GL_vs_time.png")
        fig.savefig(out, dpi=200); print(f"Saved {out}")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-f", "--flowlines", required=True)
    p.add_argument("-i", "--input", required=True, help="Comma-separated files or globs")
    p.add_argument("--transect", default=None)
    p.add_argument("--time-indexes", default="0,-1")
    p.add_argument("--sample-spacing-km", type=float, default=1.0)
    p.add_argument("--gl-tol", type=float, default=10.0, help="base-bed gap (m) below which ice is grounded")
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--gl-vs-time", action="store_true", help="plot GL position vs time (ensemble mean/envelope)")
    p.add_argument("--animate", action="store_true", help="animate the geometry cross-section over time -> GIF")
    p.add_argument("--save", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--list-transects", action="store_true")
    args = p.parse_args()

    if args.animate or args.save:
        matplotlib.use("Agg")

    transects = load_transects(args.flowlines, args.transect, args.reverse)
    if args.list_transects:
        for t in transects:
            print(t.name)
        return
    files = expand_input_files(parse_comma_list(args.input))
    if not files:
        raise FileNotFoundError("No input files matched")
    tindex = [int(x) for x in parse_comma_list(args.time_indexes)]
    spacing_m = args.sample_spacing_km * 1000.0

    for tr in transects:
        if args.gl_vs_time:
            cmd_gl_vs_time(files, tr, spacing_m, args.gl_tol, args.output_dir, args.save)
        elif args.animate:
            cmd_animate(files, tr, spacing_m, args.gl_tol, args.output_dir, args.save)
        else:
            cmd_geometry(files, tr, tindex, spacing_m, args.gl_tol, args.output_dir, args.save)
    if not args.save and not args.animate:
        plt.show()


if __name__ == "__main__":
    main()
