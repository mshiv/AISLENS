#!/usr/bin/env python3
"""
plot_ensemble_forcing_timeseries.py — per-scenario, per-member melt time series.

Global (1 panel) + regional (4x4 ISMIP6 basins) subshelf melt rate plots for
CTRL, SSP585, SSP126, and their -EXT extensions. Supports raw monthly, annual,
and deseasonalized views.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from ismip6_regions import BASIN_NAMES, SHORT_LABELS

# ── Scenario config ────────────────────────────────────────────────────────
SCENARIOS = {
    "SSP585-EXT": {
        "ensemble_dir": "SSP585-EXT",
        "include": r"^SSP585-EXT_\d+$",
        "color": "#C62828",       # red
        "alpha": 0.7,
        "label": "SSP585-EXT",
    },
    "SSP126-EXT": {
        "ensemble_dir": "SSP126-EXT",
        "include": r"^SSP126-EXT_\d+$",
        "color": "#1565C0",       # blue
        "alpha": 0.7,
        "label": "SSP126-EXT",
    },
    "SSP585": {
        "ensemble_dir": "SSP585",
        "include": r"^SSP585_0[0-9]$",
        "color": "#E53935",       # lighter red (distinguished from EXT)
        "alpha": 0.65,
        "label": "SSP585",
    },
    "SSP126": {
        "ensemble_dir": "SSP126",
        "include": r"^SSP126_0[0-9]$",
        "color": "#1E88E5",       # lighter blue (distinguished from EXT)
        "alpha": 0.65,
        "label": "SSP126",
    },
    "CTRL": {
        "ensemble_dir": "CTRL",
        "include": r"^CTRL_\d+$",
        "color": "#424242",       # dark gray
        "alpha": 0.6,
        "label": "CTRL",
    },
}

DEFAULT_GLOBAL_VAR = "avgSubshelfMelt"
DEFAULT_REGIONAL_VAR = "regionalAvgSubshelfMelt"

# Human-readable labels for common variables
VAR_LABELS = {
    "avgSubshelfMelt": ("Subshelf melt rate", "m/yr"),
    "totalFloatingBasalMassBal": ("Total floating basal mass balance", "kg/s"),
    "regionalAvgSubshelfMelt": ("Regional subshelf melt rate", "m/yr"),
    "regionalSumFloatingBasalMassBal": ("Regional floating basal mass balance", "kg/s"),
}


# ── Helpers ────────────────────────────────────────────────────────────────
def _calendar_year(xtime_values):
    """Extract float calendar year from byte xtime array ('2000-01-01_00:00:00')."""
    years = np.full(len(xtime_values), np.nan)
    for i, xt in enumerate(xtime_values):
        if isinstance(xt, bytes):
            s = xt.decode("utf-8", "ignore").strip()
        else:
            s = str(xt).strip()
        if not s or s.startswith("\x00"):
            continue
        try:
            years[i] = float(s[:4]) + (float(s[5:7]) - 1) / 12.0
        except (ValueError, IndexError):
            continue
    return years


def _is_bad_mask(years, values):
    """Boolean mask: True where xtime is NaN/empty OR value is NaN/≈0."""
    return np.isnan(years) | np.isnan(values) | (np.abs(values) < 1e-30)


def load_global(root, cfg, var_name):
    """Load globalStats for all members. Returns list of (year, values, name)."""
    members = eio.discover_members(
        os.path.join(root, cfg["ensemble_dir"]),
        stats_filename="globalStats.nc",
        include=cfg["include"],
    )
    result = []
    for name, path in members:
        ds = xr.open_dataset(path, decode_times=False)
        years = _calendar_year(ds.xtime.values)
        vals = ds[var_name].values
        bad = _is_bad_mask(years, vals)
        ds.close()
        years, vals = years[~bad], vals[~bad]
        if len(years) < 2:
            continue
        order = np.argsort(years)
        result.append((years[order], vals[order], name))
    return result


_WARNED_NO_AREA = False  # module-level flag so the missing-area warning prints once


def _is_floating_shelf_var(var_name):
    """True for regional vars that are only physically meaningful where the
    basin still has floating ice attached (subshelf melt, floating mass
    balance, face-melt, ...). A basin whose ice shelf has fully collapsed
    reports 0 for these quantities, which reads as "no melt" rather than the
    correct "no ice to melt" -- that 0 pollutes ensemble statistics."""
    name = var_name.lower()
    return (
        name == DEFAULT_REGIONAL_VAR.lower()
        or "floating" in name
        or "subshelf" in name
        or "facemelt" in name
    )


def load_regional(root, cfg, var_name):
    """Load regionalStats for all members.
    Returns list of (year, values[nT, 16], name).

    For floating-shelf-dependent variables (see `_is_floating_shelf_var`), any
    (timestep, basin) where `regionalFloatingIceArea` has collapsed to ~0 is
    set to NaN instead of being left at its raw 0 value -- a vanished shelf
    means the melt/flux quantity is undefined there, not zero. The threshold
    is basin-relative (1e-6 * that basin's own max floating area over the
    run) so a genuinely tiny sliver of shelf isn't treated as "collapsed".
    """
    global _WARNED_NO_AREA
    members = eio.discover_members(
        os.path.join(root, cfg["ensemble_dir"]),
        stats_filename="regionalStats.nc",
        include=cfg["include"],
    )
    apply_mask = _is_floating_shelf_var(var_name)
    result = []
    for name, path in members:
        ds = xr.open_dataset(path, decode_times=False)
        years = _calendar_year(ds.xtime.values)
        vals = np.array(ds[var_name].values, dtype=float)  # (Time, nRegions)
        if apply_mask:
            if "regionalFloatingIceArea" in ds:
                area = np.array(ds["regionalFloatingIceArea"].values, dtype=float)
                basin_max = np.nanmax(area, axis=0)  # (nRegions,)
                thresh = np.where(basin_max > 0, 1e-6 * basin_max, 0.0)
                collapsed = area <= thresh
                vals = np.where(collapsed, np.nan, vals)
            elif not _WARNED_NO_AREA:
                print(f"  [WARN] regionalFloatingIceArea not found in "
                      f"regionalStats.nc; skipping collapsed-shelf masking "
                      f"for '{var_name}'")
                _WARNED_NO_AREA = True
        # Filter row-wise: bad if xtime bad OR entire row is zero/NaN
        # (np.nanmean ignores the newly-masked NaN cells in other basins).
        row_bad = _is_bad_mask(years, np.nanmean(vals, axis=1))
        years, vals = years[~row_bad], vals[~row_bad]
        ds.close()
        if len(years) < 2:
            continue
        order = np.argsort(years)
        result.append((years[order], vals[order], name))
    return result


def _varying_shades(base_hex, n):
    """Generate n visually distinct shades from a base hex colour."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(base_hex)
    shades = []
    for i in range(n):
        # Spread lightness from 0.45 to 0.85 (relative luminance)
        frac = i / max(n - 1, 1)
        light = 0.45 + 0.40 * frac
        shades.append((r * light + (1 - light), g * light + (1 - light),
                       b * light + (1 - light)))
    return shades


# ── View transforms ────────────────────────────────────────────────────────
# The monthly `year` axis is float calendar year: year + (month-1)/12, so the
# integer part is the calendar year and the fractional part encodes the month.
def _annual_means(years, vals):
    """Collapse a monthly series to annual means (removes the seasonal cycle).

    `vals` may be 1D (nT,) or 2D (nT, k). Returns (annual_years, annual_vals)
    where annual_years are the integer calendar years present in the record.
    """
    cal = np.floor(years + 1e-6).astype(int)
    uniq = np.unique(cal)
    out_years = uniq.astype(float)
    if vals.ndim == 1:
        out = np.array([np.nanmean(vals[cal == y]) for y in uniq])
    else:
        out = np.array([np.nanmean(vals[cal == y], axis=0) for y in uniq])
    return out_years, out


def _deseasonalized_anomaly(years, vals):
    """Subtract the per-calendar-month climatology (computed per member over its
    own record), leaving only interannual+ variability. Keeps the monthly axis.

    `vals` may be 1D (nT,) or 2D (nT, k)."""
    month = np.round((years - np.floor(years + 1e-6)) * 12).astype(int)
    month = np.clip(month, 0, 11)
    anom = np.array(vals, dtype=float)
    if vals.ndim == 1:
        clim = np.array([np.nanmean(vals[month == m]) if np.any(month == m)
                         else np.nan for m in range(12)])
        anom = vals - clim[month]
    else:
        for m in range(12):
            sel = month == m
            if np.any(sel):
                anom[sel] = vals[sel] - np.nanmean(vals[sel], axis=0)
    return years, anom


def _transform_series(years, vals, view):
    """Apply the requested view transform to a single member's series."""
    if view == "annual":
        return _annual_means(years, vals)
    if view == "anomaly":
        return _deseasonalized_anomaly(years, vals)
    return years, vals  # raw — unchanged


def _ensemble_mean_std(plot_list, basin_idx=None, min_members=3):
    """Interpolate every member onto a common ANNUAL grid spanning the UNION of
    member time ranges, and return (grid, mean, std) across members at each
    time -- computed only from the members that actually have data there.

    Grid points where fewer than `min_members` members have data are NaN
    (mean/std not drawn), so a short member doesn't get artificially
    extrapolated, but the band still extends as far as the longer members
    reach once enough of them remain (e.g. CTRL members ending 2228-2591:
    the band now reaches into the 2500s rather than stopping at 2228, the
    shortest member's end).

    `plot_list` items are (years, vals, name) already in the target view.
    `basin_idx` selects a column when vals is 2D. Returns None if fewer than
    two members have usable data, or if no grid point reaches `min_members`.
    """
    valid = []
    for yr, vals, _name in plot_list:
        series = vals if basin_idx is None else vals[:, basin_idx]
        yr = np.asarray(yr, dtype=float)
        series = np.asarray(series, dtype=float)
        m = np.isfinite(yr) & np.isfinite(series)
        if m.sum() < 2:
            continue
        ys, vs = yr[m], series[m]
        order = np.argsort(ys)
        valid.append((ys[order], vs[order]))
    if len(valid) < 2:
        return None
    lo = min(v[0][0] for v in valid)   # union: earliest member start
    hi = max(v[0][-1] for v in valid)  # union: latest member end
    if hi <= lo:
        return None
    grid = np.arange(np.ceil(lo), np.floor(hi) + 1.0)
    if grid.size < 2:
        grid = np.linspace(lo, hi, 50)
    # left/right=nan: a member must NOT be extrapolated past its own record,
    # or every member would appear to "have data" across the whole union grid.
    stacked = np.column_stack([
        np.interp(grid, ys, vs, left=np.nan, right=np.nan) for ys, vs in valid
    ])
    count = np.sum(~np.isnan(stacked), axis=1)
    enough = count >= min_members
    if not np.any(enough):
        return None
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stacked, axis=1)
        std = np.nanstd(stacked, axis=1)
    mean = np.where(enough, mean, np.nan)
    std = np.where(enough, std, np.nan)
    return grid, mean, std


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_global(ax, data_list, cfg, ylabel="Value", view="raw", min_members=3):
    """Overlay per-member global values on *ax* for the requested view."""
    plot_list = [(*_transform_series(yr, vals, view), name)
                 for yr, vals, name in data_list]
    shades = _varying_shades(cfg["color"], len(plot_list))
    for (yr, vals, name), c in zip(plot_list, shades):
        label = name if len(plot_list) <= 3 else None
        ax.plot(yr, vals, color=c, alpha=cfg["alpha"], linewidth=0.8, label=label)
    if view == "raw":
        # Ensemble envelope (min/max at each time)
        if plot_list:
            common_yr = np.linspace(
                max(d[0][0] for d in plot_list),
                min(d[0][-1] for d in plot_list),
                500,
            )
            interpolated = []
            for yr, vals, _ in plot_list:
                interpolated.append(np.interp(common_yr, yr, vals))
            stacked = np.column_stack(interpolated)
            ax.fill_between(
                common_yr, stacked.min(axis=1), stacked.max(axis=1),
                color=cfg["color"], alpha=0.10, linewidth=0,
            )
    else:
        # Ensemble MEAN line + ±1σ inter-member spread on a common annual grid
        res = _ensemble_mean_std(plot_list, min_members=min_members)
        if res is not None:
            grid, mean, std = res
            ax.fill_between(grid, mean - std, mean + std, color="0.4",
                            alpha=0.25, linewidth=0, zorder=4,
                            label=r"$\pm1\sigma$ members")
            ax.plot(grid, mean, color="black", linewidth=2.0, zorder=5,
                    label="Ensemble mean")
        if view == "anomaly":
            ax.axhline(0.0, color="0.6", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{cfg['label']} — global")
    ax.grid(True, alpha=0.3)


def plot_regional_grid(axes_flat, data_list, cfg, ylabel="Value", view="raw",
                        min_members=3):
    """Fill a 4×4 grid of axes with per-basin per-member time series."""
    plot_list = [(*_transform_series(yr, vals, view), name)
                 for yr, vals, name in data_list]
    shades = _varying_shades(cfg["color"], len(plot_list))
    for basin_idx in range(16):
        ax = axes_flat[basin_idx]
        for (yr, vals_2d, name), c in zip(plot_list, shades):
            ax.plot(yr, vals_2d[:, basin_idx], color=c, alpha=cfg["alpha"],
                    linewidth=0.6)
        if view != "raw":
            res = _ensemble_mean_std(plot_list, basin_idx=basin_idx,
                                     min_members=min_members)
            if res is not None:
                grid, mean, std = res
                ax.fill_between(grid, mean - std, mean + std, color="0.4",
                                alpha=0.25, linewidth=0, zorder=4)
                ax.plot(grid, mean, color="black", linewidth=1.2, zorder=5)
            if view == "anomaly":
                ax.axhline(0.0, color="0.6", linewidth=0.5, linestyle="--",
                           zorder=1)
        short = SHORT_LABELS.get(BASIN_NAMES[basin_idx], BASIN_NAMES[basin_idx])
        ax.set_title(short, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        if basin_idx >= 12:
            ax.set_xlabel("Year", fontsize=7)
        if basin_idx % 4 == 0:
            ax.set_ylabel(ylabel, fontsize=7)


VIEW_TITLES = {
    "raw": "monthly",
    "annual": "annual mean",
    "anomaly": "deseasonalized anomaly",
}


def make_figure(scen_key, cfg, root, out_dir, global_var, regional_var,
                view="raw", min_members=3):
    """Build and save one scenario figure (global + 4×4 regional)."""
    g_data = load_global(root, cfg, global_var)
    r_data = load_regional(root, cfg, regional_var)
    if not g_data and not r_data:
        print(f"  [SKIP] {scen_key}: no data found")
        return

    g_label, g_unit = VAR_LABELS.get(global_var, (global_var, ""))
    r_label, r_unit = VAR_LABELS.get(regional_var, (regional_var, ""))
    view_lbl = VIEW_TITLES.get(view, view)
    if view == "anomaly":
        g_ylabel = f"{g_label} anomaly ({g_unit})" if g_unit else f"{g_label} anomaly"
    else:
        g_ylabel = f"{g_label} ({g_unit})" if g_unit else g_label
    r_ylabel = r_unit if r_unit else r_label

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(
        6, 4,
        height_ratios=[1.2, 0.15, 1, 1, 1, 1],
        hspace=0.35, wspace=0.30,
    )

    # Global panel (top row, spans all cols)
    ax_global = fig.add_subplot(gs[0, :])
    plot_global(ax_global, g_data, cfg, ylabel=g_ylabel, view=view,
                min_members=min_members)

    # Regional grid (rows 2–5, 4×4 = 16 basins)
    axes_regional = []
    for row in range(2, 6):
        for col in range(4):
            axes_regional.append(fig.add_subplot(gs[row, col]))
    plot_regional_grid(axes_regional, r_data, cfg, ylabel=r_ylabel, view=view,
                       min_members=min_members)

    # Legend (use line handles for clarity)
    n_members = max(len(g_data), len(r_data))
    handles = [
        Line2D([0], [0], color=cfg["color"], alpha=0.3 + 0.5 * (i / max(n_members - 1, 1)),
               linewidth=1.2, label=f"Member {i:02d}")
        for i in range(n_members)
    ]
    if view != "raw":
        # Make the ensemble-mean line and ±1σ member-spread band explicit.
        handles += [
            Line2D([0], [0], color="black", linewidth=2.0, label="Ensemble mean"),
            Patch(facecolor="0.4", alpha=0.25, label=r"$\pm1\sigma$ members"),
        ]
    ax_global.legend(handles=handles, loc="best", fontsize=7,
                     ncol=min(n_members, 5))

    fig.suptitle(f"{cfg['label']} — {g_label} ({view_lbl})",
                 fontsize=14, fontweight="bold", y=0.995)

    tag = scen_key.lower().replace(" ", "-")
    var_tag = global_var.lower().replace("total", "").replace("floating", "float")
    view_suffix = "" if view == "raw" else f"_{view}"
    out_path = os.path.join(out_dir, f"{var_tag}_{tag}{view_suffix}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root(),
                    help="ENSEMBLES root directory")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__), "..", "..",
                                         "reports", "figures", "forcing-timeseries"),
                    help="Output directory for figures")
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS.keys()),
                    help="Scenarios to plot (default: all)")
    ap.add_argument("--global-var", default=DEFAULT_GLOBAL_VAR,
                    help=f"Global variable to plot (default: {DEFAULT_GLOBAL_VAR})")
    ap.add_argument("--regional-var", default=DEFAULT_REGIONAL_VAR,
                    help=f"Regional variable to plot (default: {DEFAULT_REGIONAL_VAR})")
    ap.add_argument("--view", choices=["raw", "annual", "anomaly"], default="raw",
                    help="raw: monthly series (default); annual: per-member annual "
                         "means; anomaly: per-member deseasonalized monthly anomaly. "
                         "annual/anomaly add a black ensemble-mean line and ±1σ "
                         "inter-member spread band.")
    ap.add_argument("--min-members", type=int, default=3,
                    help="Minimum number of members with data at a given year "
                         "for the ensemble mean/±1σ band to be drawn there "
                         "(default: 3). The band is built over the UNION of "
                         "member time ranges and extends as far as this many "
                         "members reach, rather than stopping at the shortest "
                         "member.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for scen_key in args.scenarios:
        if scen_key not in SCENARIOS:
            print(f"  [WARN] Unknown scenario '{scen_key}', skipping")
            continue
        print(f"Plotting {scen_key} ({args.global_var}, view={args.view})...")
        make_figure(scen_key, SCENARIOS[scen_key], args.root, args.out_dir,
                    args.global_var, args.regional_var, view=args.view,
                    min_members=args.min_members)

    print("Done.")


if __name__ == "__main__":
    main()
