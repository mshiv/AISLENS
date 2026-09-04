#!/usr/bin/env python3
"""Plot the observational basis and production fb_A background-melt curves.

The figure reads time-mean Paolo et al. observations from the original gridded
product and the five parameter fields from the production MALI initial-condition
file.  It deliberately plots basal mass balance (negative for melting), matching
the sign convention used in Chapter 3 and in MALI.

Raw observations are rasterized in the vector PDF.  Black circles are descriptive
bin means shown to make the data cloud legible; the coloured curve is evaluated
from the parameters stored on the MALI mesh and is not refitted by this script.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import netCDF4 as nc
import numpy as np
from shapely import contains_xy


REPO = Path(__file__).resolve().parents[2]
OBS = REPO / "data/external/ANT_G1920V01_IceShelfMeltDraft_Time.nc"
SHELF_GEOMETRY = REPO / "data/external/iceShelves.geojson"
MESH = REPO / "data/MALI/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc"
MASK = REPO / "data/MALI/aislens_draftDepen_regionMasks.nc"
REGION_TABLE = REPO / "docs/region_mapping_133_to_ismip6.csv"

RHO_ICE = 910.0
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SI_TO_M_PER_YEAR = SECONDS_PER_YEAR / RHO_ICE

SHELVES = ["Sulzberger", "Stange", "Amery", "Riiser-Larsen", "Ninnis"]
DISPLAY = {
    "Sulzberger": "Sulzberger",
    "Stange": "Stange",
    "Amery": "Amery",
    "Riiser-Larsen": "Riiser-Larsen",
    "Ninnis": "Ninnis",
}
COLORS = {
    "Sulzberger": "#0072B2",
    "Stange": "#D55E00",
    "Amery": "#009E73",
    "Riiser-Larsen": "#CC79A7",
    "Ninnis": "#E69F00",
}
MAP_LABEL_OFFSETS = {
    "Sulzberger": (3, 3),
    "Stange": (3, 3),
    "Amery": (3, 3),
    "Riiser-Larsen": (3, 3),
    "Ninnis": (3, 3),
}


def publication_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })


def region_names() -> list[str]:
    names = [f"r{i}" for i in range(133)]
    with REGION_TABLE.open() as handle:
        for row in csv.DictReader(handle):
            names[int(row["idx_133"])] = row["name_133"]
    return names


def read_parameters(shelves: list[str] | None = None) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    shelves = SHELVES if shelves is None else shelves
    names = region_names()
    with nc.Dataset(MESH) as ds, nc.Dataset(MASK) as mask_ds:
        masks = np.asarray(mask_ds["regionCellMasks"][:])
        if masks.shape[0] > masks.shape[1]:
            masks = masks.T
        fields = {
            "p": np.asarray(ds["draftDepenBasalMelt_paramType"][0, :], float),
            "dmin": np.asarray(ds["draftDepenBasalMelt_minDraft"][0, :], float),
            "c": np.asarray(ds["draftDepenBasalMelt_constantMeltValue"][0, :], float),
            "a0": np.asarray(ds["draftDepenBasalMeltAlpha0"][0, :], float),
            "a1": np.asarray(ds["draftDepenBasalMeltAlpha1"][0, :], float),
        }
        xcell = np.asarray(ds["xCell"][:], float) / 1000.0
        ycell = np.asarray(ds["yCell"][:], float) / 1000.0
        thickness = np.asarray(ds["thickness"][0, :], float)

    params: dict[str, dict[str, float]] = {}
    for shelf in shelves:
        selected = masks[names.index(shelf)].astype(bool)
        p_values, counts = np.unique(fields["p"][selected], return_counts=True)
        p_mode = float(p_values[np.argmax(counts)])
        same_type = selected & (fields["p"] == p_mode)
        params[shelf] = {
            "p": p_mode,
            "dmin": abs(float(np.nanmedian(fields["dmin"][same_type]))),
            "c": float(np.nanmedian(fields["c"][same_type])) * SI_TO_M_PER_YEAR,
            "a0": float(np.nanmedian(fields["a0"][same_type])) * SI_TO_M_PER_YEAR,
            "a1": float(np.nanmedian(fields["a1"][same_type])) * SI_TO_M_PER_YEAR,
        }
    return params, xcell, ycell, thickness


def read_observations(geometry: gpd.GeoDataFrame,
                      shelves: list[str] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    shelves = SHELVES if shelves is None else shelves
    observations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with nc.Dataset(OBS) as ds:
        x = np.asarray(ds["x"][:], float)
        y = np.asarray(ds["y"][:], float)
        for shelf in shelves:
            polygon = geometry.loc[geometry.name == shelf].geometry.iloc[0]
            xmin, ymin, xmax, ymax = polygon.bounds
            ix = np.flatnonzero((x >= xmin) & (x <= xmax))
            iy = np.flatnonzero((y >= ymin) & (y <= ymax))
            if ix.size == 0 or iy.size == 0:
                raise RuntimeError(f"No observational grid cells intersect {shelf}")

            xs = x[ix]
            ys = y[iy]
            x_slice = slice(ix.min(), ix.max() + 1)
            y_slice = slice(iy.min(), iy.max() + 1)
            melt = np.ma.mean(ds["melt"][:, y_slice, x_slice], axis=0).filled(np.nan)
            draft = np.ma.mean(ds["draft"][:, y_slice, x_slice], axis=0).filled(np.nan)
            xx, yy = np.meshgrid(xs, ys)
            valid = (
                contains_xy(polygon, xx, yy)
                & np.isfinite(melt)
                & np.isfinite(draft)
                & (draft > 0.0)
            )
            observations[shelf] = (draft[valid], melt[valid])
    return observations


def evaluate_curve(draft: np.ndarray, pars: dict[str, float]) -> np.ndarray:
    if int(round(pars["p"])) == 1:
        return np.full_like(draft, pars["c"], dtype=float)
    threshold_value = pars["a0"] + pars["a1"] * pars["dmin"]
    return np.where(
        draft < pars["dmin"],
        threshold_value,
        pars["a0"] + pars["a1"] * draft,
    )


def descriptive_bins(draft: np.ndarray, melt: np.ndarray, n_bins: int = 100,
                     min_count: int = 20) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(float(np.nanmin(draft)), float(np.nanmax(draft)), n_bins + 1)
    index = np.digitize(draft, edges) - 1
    bx, by = [], []
    for i in range(n_bins):
        take = index == i
        if np.count_nonzero(take) >= min_count:
            bx.append(float(np.nanmean(draft[take])))
            by.append(float(np.nanmean(melt[take])))
    return np.asarray(bx), np.asarray(by)


def robust_limits(values: np.ndarray, curve: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(values, [3.0, 97.0])
    lo = min(lo, float(np.nanmin(curve)))
    hi = max(hi, float(np.nanmax(curve)))
    span = max(hi - lo, 1.0)
    return lo - 0.07 * span, hi + 0.07 * span


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO / "reports/dissertation/figures/tierA/F1_methods_piecewise_background_melt.png"),
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    publication_style()
    geometry = gpd.read_file(SHELF_GEOMETRY).to_crs(3031)
    geometry_km = geometry.copy()
    geometry_km.geometry = geometry_km.geometry.scale(
        xfact=0.001, yfact=0.001, origin=(0.0, 0.0)
    )
    params, xcell, ycell, thickness = read_parameters()
    observations = read_observations(geometry)

    fig = plt.figure(figsize=(7.5, 5.55), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.04, 1, 1])
    axes = [fig.add_subplot(grid[i // 3, i % 3]) for i in range(6)]

    # Panel (a): one shared locator map is more legible than repeated miniature maps.
    ax = axes[0]
    ice = thickness > 1.0
    stride = 6
    idx = np.flatnonzero(ice)[::stride]
    ax.scatter(xcell[idx], ycell[idx], s=0.16, color="#D9DEE3", linewidths=0,
               rasterized=True, zorder=0)
    geometry_km.boundary.plot(ax=ax, color="#AEB6BE", linewidth=0.25, zorder=1)
    panel_letters = ["b", "c", "d", "e", "f"]
    for shelf, letter in zip(SHELVES, panel_letters):
        row = geometry_km.loc[geometry_km.name == shelf]
        row.plot(ax=ax, facecolor=COLORS[shelf], edgecolor="#303030", linewidth=0.35,
                 alpha=0.9, zorder=3)
        point = row.geometry.representative_point().iloc[0]
        ax.annotate(letter, (point.x, point.y),
                    xytext=MAP_LABEL_OFFSETS[shelf], textcoords="offset points", fontsize=8,
                    fontweight="bold", color="#202020", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-2850, 2850)
    ax.set_ylim(-2850, 2850)
    ax.axis("off")
    ax.set_title("(a) selected ice-shelf regions", loc="left", pad=4)

    rng = np.random.default_rng(2026)
    for ax, shelf, letter in zip(axes[1:], SHELVES, panel_letters):
        draft, melt = observations[shelf]
        pars = params[shelf]
        # Deterministic subsampling controls PDF size; bin means use every valid cell.
        if draft.size > 15000:
            shown = np.sort(rng.choice(draft.size, 15000, replace=False))
        else:
            shown = np.arange(draft.size)
        ax.scatter(melt[shown], -draft[shown], s=2.0, color="#8A8F94", alpha=0.13,
                   edgecolors="none", rasterized=True, zorder=1)

        bx, by = descriptive_bins(draft, melt)
        ax.scatter(by, -bx, s=12, facecolor="white", edgecolor="#242424",
                   linewidth=0.65, zorder=3)

        d_lo = max(0.0, float(np.nanpercentile(draft, 1.0)))
        d_hi = float(np.nanpercentile(draft, 99.0))
        dd = np.linspace(d_lo, d_hi, 500)
        curve = evaluate_curve(dd, pars)
        ax.plot(curve, -dd, color=COLORS[shelf], linewidth=2.2, zorder=4)

        form = "constant" if int(round(pars["p"])) == 1 else "capped piecewise linear"
        if int(round(pars["p"])) == 0:
            ax.axhline(-pars["dmin"], color=COLORS[shelf], linestyle=(0, (3, 2)),
                       linewidth=1.05, zorder=2)
            ax.annotate(
                rf"$d_{{\min}}={pars['dmin']:.0f}$ m",
                xy=(0.03, -pars["dmin"]), xycoords=("axes fraction", "data"),
                xytext=(3, 4), textcoords="offset points", ha="left", va="bottom",
                color=COLORS[shelf], fontsize=7.6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
                      "pad": 0.8},
            )

        ax.axvline(0.0, color="#555555", linewidth=0.65, linestyle=":", zorder=0)
        ax.set_xlim(*robust_limits(melt, curve))
        ax.set_ylim(-d_hi * 1.02, 0.0)
        ax.grid(color="#C9CDD1", alpha=0.38, linewidth=0.55)
        ax.set_title(f"({letter}) {DISPLAY[shelf]}\n{form}", loc="left", pad=3)
        if ax in axes[3:]:
            ax.set_xlabel("basal mass balance, $B$ (m ice yr$^{-1}$)")
        if ax in (axes[1], axes[3]):
            ax.set_ylabel("ice-shelf draft (m)")

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=3.5,
               markerfacecolor="#8A8F94", markeredgewidth=0, alpha=0.5,
               label="cellwise time mean"),
        Line2D([], [], linestyle="none", marker="o", markersize=4.5,
               markerfacecolor="white", markeredgecolor="#242424",
               label="descriptive bin mean"),
        Line2D([], [], color="#303030", linewidth=2.2,
               label="production $B_r(d)$"),
        Line2D([], [], color="#303030", linewidth=1.05, linestyle=(0, (3, 2)),
               label="$d_{\min,r}$"),
    ]
    fig.legend(handles=handles, loc="outside lower center",
               ncol=4, frameon=False, columnspacing=1.25, handletextpad=0.5)

    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))

    table = out.with_name(out.stem + "_parameters.csv")
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shelf", "parameterization", "dmin_m", "c_m_ice_yr-1",
                         "alpha0_m_ice_yr-1", "alpha1_m_ice_yr-1_per_m",
                         "n_observational_cells"])
        for shelf in SHELVES:
            pars = params[shelf]
            writer.writerow([
                DISPLAY[shelf],
                "constant" if int(round(pars["p"])) == 1 else "capped_piecewise_linear",
                pars["dmin"], pars["c"], pars["a0"], pars["a1"],
                observations[shelf][0].size,
            ])
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.pdf')}")
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
