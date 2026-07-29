#!/usr/bin/env python3
"""
fig_jourdain_model_universe.py — model-universe comparison of AISLENS to Jourdain et al. (2020).

Generates dumbbell, violin, faceted, bubble, scatter, and timeseries views of
sub-shelf melt std (2000–2100). Data: digitized Jourdain CSVs + AISLENS SSP585 ensemble.

Author: Shivaprakash Muruganandham
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd
from netCDF4 import Dataset, chartostring
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASK = os.path.join(REPO, "data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc")
JOURDAIN_DIR = os.path.expanduser("~/Downloads")

Y0, Y1 = 2000, 2100
DETREND = False   # set via --detrend: linearly detrend each series before std (isolates variability from the trend)
YSCALE = "log"    # set via --yscale {log,linear} for the scatter (std) figure
NORMALIZE = False  # set via --normalize: divide each spectrum by its total variance (unit area) -> shape-only
VARPRESERVE = False  # set via --varpreserve: plot f*PSD vs log-period (linear y) -> area = variance, peak = dominant timescale


def _std(vals, detrend=None):
    """std of a 1-D series; if detrend, remove a linear fit first (isolates fluctuation variability)."""
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if v.size < 5:
        return np.nan
    if (DETREND if detrend is None else detrend):
        t = np.arange(v.size); v = v - np.polyval(np.polyfit(t, v, 1), t)
    return float(np.std(v))


def region_agg_std(mali_members, detrend, member=None):
    """AISLENS std computed the SAME way as the Jourdain sector series: std of the
    REGION-MEAN series (average basins first, then std) — apples-to-apples with the
    digitized sector series — averaged over members (or a single --member).
    Returns {basin_index: region_std} so fig_scatter's per-basin averaging returns it."""
    out = {}
    members = [member] if member is not None else range(mali_members.shape[0])
    for rname, info in REGIONS.items():
        vals = [_std(np.mean(np.stack([mali_members[m, :, b] for b in info["basins"]]), axis=0),
                     detrend=detrend) for m in members]
        rstd = float(np.nanmean(vals))
        for b in info["basins"]:
            out[b] = rstd
    return out


def region_agg_mean(mali_members, mali_ts, y0, y1, member=None):
    """AISLENS MEAN melt over [y0,y1] of the REGION-MEAN series (avg basins first),
    averaged over members (or a single --member). Returns {basin_index: region_mean}."""
    out = {}
    members = [member] if member is not None else range(mali_members.shape[0])
    for rname, info in REGIONS.items():
        yrs = np.asarray(mali_ts[info["basins"][0]][0])
        mask = (yrs >= y0) & (yrs <= y1)
        vals = []
        for m in members:
            reg = np.mean(np.stack([mali_members[m, :, b] for b in info["basins"]]), axis=0)
            vals.append(np.nanmean(reg[mask]))
        rmean = float(np.nanmean(vals))
        for b in info["basins"]:
            out[b] = rmean
    return out

# Region definitions
REGIONS = {
    "Ronne-Filchner":       {"csv": "panel_a_Ronne-Filchner.csv",        "basins": [14]},
    "Pine Island-Thwaites":  {"csv": "panel_b_Pine-Island_Thwaites.csv",  "basins": [9]},
    "Cook-Ninnis":           {"csv": "panel_c_Cook_Ninnis.csv",           "basins": [5, 6]},
    "Totten-Moscow U.":      {"csv": "panel_d_Totten_MoscowUniv.csv",     "basins": [4]},
}

# ─── Jourdain paper colors (IPSL = light blue) ─────────────────────────────
# Same color for a GCM whether MeanAnt or PIGL. Marker differs by forcing type.
JOURDAIN_COLORS = {
    "ccsm4":            "black",
    "csiro-mk3-6-0":    "grey",
    "hadgem2-es":       "orange",
    "ipsl-cm5a-mr":     "lightskyblue",   # light blue per request
    "miroc-esm-chem":   "green",
    "noresm1-m":        "#CCCC00",        # yellow
}
FESOM_COLORS = {
    "mmm":              "#1f77b4",        # blue
    "access1.0":        "#d62728",        # red
    "hadcm3":           "#9467bd",        # purple
}

# Keys must exactly match CSV column names: (label, forcing, gcm_key, marker, color)
MODEL_STYLE = {
    # MeanAnt (circles)
    "MeanAnt_ccsm4_rcp85":            ("MeanAnt CCSM4",         "MeanAnt", "ccsm4",          "o", JOURDAIN_COLORS["ccsm4"]),
    "MeanAnt_csiro-mk3-6-0_rcp85":    ("MeanAnt CSIRO-MK3.6",   "MeanAnt", "csiro-mk3-6-0",  "o", JOURDAIN_COLORS["csiro-mk3-6-0"]),
    "MeanAnt_hadgem2-es_rcp85":       ("MeanAnt HadGEM2-ES",     "MeanAnt", "hadgem2-es",     "o", JOURDAIN_COLORS["hadgem2-es"]),
    "MeanAnt_ipsl-cm5a-mr_rcp85":     ("MeanAnt IPSL-CM5A-MR",   "MeanAnt", "ipsl-cm5a-mr",   "o", JOURDAIN_COLORS["ipsl-cm5a-mr"]),
    "MeanAnt_miroc-esm-chem_rcp85":   ("MeanAnt MIROC-ESM-CHEM", "MeanAnt", "miroc-esm-chem", "o", JOURDAIN_COLORS["miroc-esm-chem"]),
    "MeanAnt_noresm1-m_rcp85":        ("MeanAnt NorESM1-M",      "MeanAnt", "noresm1-m",      "o", JOURDAIN_COLORS["noresm1-m"]),
    # PIGL (squares)
    "PIGL_ccsm4_rcp85":               ("PIGL CCSM4",             "PIGL", "ccsm4",          "s", JOURDAIN_COLORS["ccsm4"]),
    "PIGL_csiro-mk3-6-0_rcp85":       ("PIGL CSIRO-MK3.6",       "PIGL", "csiro-mk3-6-0",  "s", JOURDAIN_COLORS["csiro-mk3-6-0"]),
    "PIGL_hadgem2-es_rcp85":          ("PIGL HadGEM2-ES",         "PIGL", "hadgem2-es",     "s", JOURDAIN_COLORS["hadgem2-es"]),
    "PIGL_ipsl-cm5a-mr_rcp85":        ("PIGL IPSL-CM5A-MR",       "PIGL", "ipsl-cm5a-mr",   "s", JOURDAIN_COLORS["ipsl-cm5a-mr"]),
    "PIGL_miroc-esm-chem_rcp85":      ("PIGL MIROC-ESM-CHEM",     "PIGL", "miroc-esm-chem", "s", JOURDAIN_COLORS["miroc-esm-chem"]),
    "PIGL_noresm1-m_rcp85":           ("PIGL NorESM1-M",          "PIGL", "noresm1-m",      "s", JOURDAIN_COLORS["noresm1-m"]),
    # FESOM (triangles)
    "FESOM_mmm_rcp85":                ("FESOM MMM",               "FESOM", "mmm",           "^", FESOM_COLORS["mmm"]),
    "FESOM_access1.0_rcp85":          ("FESOM ACCESS1.0",         "FESOM", "access1.0",     "^", FESOM_COLORS["access1.0"]),
    "FESOM_hadcm3_A1B":               ("FESOM HadCM3",            "FESOM", "hadcm3",        "^", FESOM_COLORS["hadcm3"]),
}

# GCMs shared between MeanAnt and PIGL (for dumbbell connecting lines)
SHARED_GCMS = ["ccsm4", "csiro-mk3-6-0", "hadgem2-es", "ipsl-cm5a-mr", "miroc-esm-chem", "noresm1-m"]

# For timeseries legend: GCM key -> display name
GCM_DISPLAY = {
    "ccsm4": "CCSM4", "csiro-mk3-6-0": "CSIRO-MK3.6", "hadgem2-es": "HadGEM2-ES",
    "ipsl-cm5a-mr": "IPSL-CM5A-MR", "miroc-esm-chem": "MIROC-ESM-CHEM", "noresm1-m": "NorESM1-M",
}


# ─── Data loading ───────────────────────────────────────────────────────────

def load_jourdain(csv_path: str, y0: int, y1: int) -> pd.DataFrame:
    """Return DataFrame with columns: model, gcm, forcing, std, mean_melt, color, marker."""
    df = pd.read_csv(csv_path)
    df = df[(df["year"] >= y0) & (df["year"] <= y1)]
    rows = []
    for col in df.columns:
        if col == "year":
            continue
        vals = df[col].values.astype(float)
        valid = vals[~np.isnan(vals)]
        if len(valid) < 10:
            continue
        info = MODEL_STYLE.get(col, (col, "Other", col, "D", "gray"))
        rows.append({
            "model": info[0], "forcing": info[1], "gcm": info[2],
            "marker": info[3], "color": info[4],
            "std": _std(valid), "mean_melt": np.mean(valid),
        })
    return pd.DataFrame(rows)


def load_jourdain_timeseries(csv_path: str, y0: int, y1: int) -> list[dict]:
    """Return list of dicts with raw time series for plotting lines."""
    df = pd.read_csv(csv_path)
    df = df[(df["year"] >= y0) & (df["year"] <= y1)]
    years = df["year"].values
    series = []
    for col in df.columns:
        if col == "year":
            continue
        vals = df[col].values.astype(float)
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < 10:
            continue
        info = MODEL_STYLE.get(col, (col, "Other", col, "D", "gray"))
        # Determine line style from column prefix
        if col.startswith("MeanAnt_"):
            mtype, lw, ls = "meanant", 0.8, "-"
        elif col.startswith("PIGL_"):
            mtype, lw, ls = "pigl", 2.0, "-"
        elif col.startswith("FESOM_"):
            mtype, lw, ls = "fesom", 1.5, "--"
        else:
            mtype, lw, ls = "other", 1.0, "-"
        series.append({
            "col": col,
            "label": info[0],
            "color": info[4],
            "mtype": mtype,
            "lw": lw,
            "ls": ls,
            "years": years,
            "values": vals,
        })
    return series


def load_mali_ensemble(y0: int, y1: int, min_members: int = 3):
    """Load SSP585_00–09 ensemble.

    Returns (std_per_basin, mean_per_basin, full_timeseries, member_aligned)
    - full_timeseries: basin_index -> (years, mean_ts, min_ts, max_ts)
    - member_aligned: list of (years, melt_array_per_member) with shape (n_members, n_years, 16)
    """
    pattern = os.path.join(
        REPO, "data/MALI/diagnostics/ENSEMBLES/SSP585/SSP585_0[0-9]/regionalStats.nc"
    )
    files = sorted(glob.glob(pattern))
    all_stds, all_means = [], []
    member_ts = []  # list of (years, melt_array) per member

    for fpath in files:
        d = Dataset(fpath)
        melt = np.asarray(d.variables["regionalAvgSubshelfMelt"][:], float)
        xtime = chartostring(d.variables["xtime"][:])
        years = np.array([int(str(s).strip()[:4]) if str(s).strip()[:4].isdigit() else -1
                          for s in xtime])
        mask = (years >= y0) & (years <= y1)
        if mask.sum() < 10:
            continue
        mp = melt[mask]
        yr_masked = years[mask]
        all_stds.append(np.array([_std(mp[:, b]) for b in range(16)]))
        all_means.append(np.nanmean(mp, axis=0))
        member_ts.append((yr_masked, mp))
        d.close()

    if len(all_stds) < min_members:
        raise RuntimeError(f"Only {len(all_stds)} valid members")

    mean_std = {i: np.nanmean(np.stack(all_stds), axis=0)[i] for i in range(16)}
    mean_mean = {i: np.nanmean(np.stack(all_means), axis=0)[i] for i in range(16)}

    # Build per-basin time series summary + per-member aligned arrays
    all_years = np.unique(np.concatenate([mt[0] for mt in member_ts]))
    full_ts = {}
    member_aligned = []
    for yr_arr, mp in member_ts:
        ts_grid = np.full((len(all_years), 16), np.nan)
        for k, y in enumerate(yr_arr):
            idx = np.searchsorted(all_years, y)
            if idx < len(all_years) and all_years[idx] == y:
                ts_grid[idx] = mp[k]
        member_aligned.append(ts_grid)

    member_stack = np.stack(member_aligned)  # (n_members, n_years, 16)

    for b in range(16):
        full_ts[b] = (all_years,
                       np.nanmean(member_stack[:, :, b], axis=0),
                       np.nanmin(member_stack[:, :, b], axis=0),
                       np.nanmax(member_stack[:, :, b], axis=0))

    return mean_std, mean_mean, full_ts, member_stack


# ─── Shared legend helper ───────────────────────────────────────────────────

def _gcm_legend_handles():
    """Build legend handles for the Jourdain GCM color scheme."""
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markersize=8, label="MeanAnt (circle)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey", markersize=8, label="PIGL (square)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey", markersize=8, label="FESOM (triangle)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=12, label="AISLENS"),
        Line2D([0], [0], color="w", lw=0, label=""),  # spacer
    ]
    for gcm_key in ["ccsm4", "csiro-mk3-6-0", "hadgem2-es", "ipsl-cm5a-mr",
                     "miroc-esm-chem", "noresm1-m"]:
        handles.append(Line2D([0], [0], marker="s", color="w",
                              markerfacecolor=JOURDAIN_COLORS[gcm_key],
                              markersize=7, label=GCM_DISPLAY[gcm_key]))
    for fesom_key, label in [("mmm", "FESOM MMM"), ("access1.0", "FESOM ACCESS1.0"),
                              ("hadcm3", "FESOM HadCM3")]:
        handles.append(Line2D([0], [0], marker="^", color="w",
                              markerfacecolor=FESOM_COLORS[fesom_key],
                              markersize=7, label=label))
    return handles


# ─── Figure 1: Sorted dumbbell ──────────────────────────────────────────────

def fig_dumbbell(all_data: dict[str, pd.DataFrame], mali_std: dict, outpath: str):
    """For each region: sort models by std, connect same-GCM MeanAnt↔PIGL pairs."""
    region_labels = list(REGIONS.keys())
    n = len(region_labels)
    fig, axes = plt.subplots(1, n, figsize=(16, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (rname, info) in zip(axes, REGIONS.items()):
        df = all_data[rname].copy()
        df = df.sort_values("std").reset_index(drop=True)

        # Draw connecting lines for shared GCMs
        for gcm in SHARED_GCMS:
            subset = df[df["gcm"] == gcm]
            if len(subset) == 2:
                vals = subset["std"].values
                yy = subset.index.values
                ax.plot(vals, yy, color=subset["color"].values[0],
                        lw=1.5, alpha=0.5, zorder=1)

        # Draw points (all at same x = std, y = rank — no x-jitter needed)
        for idx, row in df.iterrows():
            ax.scatter(row["std"], idx, c=row["color"], marker=row["marker"],
                       s=70, edgecolors="white", linewidths=0.5, zorder=3)

        # AISLENS star
        mali_val = np.mean([mali_std[b] for b in info["basins"]])
        ax.axvline(mali_val, color="black", ls="--", lw=1.2, alpha=0.6, zorder=2)
        ax.scatter(mali_val, len(df) / 2, c="black", marker="*", s=200, zorder=5,
                   edgecolors="white", linewidths=0.8)

        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(df["model"], fontsize=7)
        ax.set_xlabel("Std (m/yr)", fontsize=9)
        ax.set_title(rname, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

    fig.legend(handles=_gcm_legend_handles(), loc="lower center", ncol=5, fontsize=8, framealpha=0.9)
    fig.suptitle(f"Sorted dumbbell: std of sub-shelf melt ({Y0}–{Y1})\n"
                 "Lines connect MeanAnt ↔ PIGL for the same GCM",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


# ─── Figure 2: Violin / box ────────────────────────────────────────────────

def fig_violin(all_data: dict[str, pd.DataFrame], mali_std: dict, outpath: str):
    """Violin + strip of model std per region, AISLENS star overlaid. No x-jitter."""
    region_labels = list(REGIONS.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = np.arange(len(region_labels))
    all_stds_by_region = [all_data[r]["std"].values for r in region_labels]

    # Violin
    parts = ax.violinplot(all_stds_by_region, positions=positions, showmeans=True,
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#B0C4DE")
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("#333")
    parts["cmedians"].set_color("#D32F2F")

    # Strip — all points at the same x, no jitter
    for i, (rname, df) in enumerate(all_data.items()):
        for _, row in df.iterrows():
            ax.scatter(positions[i], row["std"],
                       c=row["color"], marker=row["marker"], s=40, alpha=0.85,
                       edgecolors="white", linewidths=0.4, zorder=3)

    # AISLENS stars
    for i, (rname, info) in enumerate(REGIONS.items()):
        mali_val = np.mean([mali_std[b] for b in info["basins"]])
        ax.scatter(positions[i], mali_val, c="black", marker="*", s=250, zorder=5,
                   edgecolors="white", linewidths=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(region_labels, fontsize=11)
    ax.set_ylabel("Std of sub-shelf melt rate (m yr⁻¹)", fontsize=12)
    ax.set_title(f"Distribution of model stds ({Y0}–{Y1})\n"
                 "Violin = model distribution, star = AISLENS",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=12, label="AISLENS"),
        mpatches.Patch(facecolor="#B0C4DE", alpha=0.6, label="Model distribution"),
        Line2D([0], [0], color="#333", lw=1.5, label="Mean"),
        Line2D([0], [0], color="#D32F2F", lw=1.5, label="Median"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


# ─── Figure 3: Faceted by forcing type ──────────────────────────────────────

def fig_forcing_facets(all_data: dict[str, pd.DataFrame], mali_std: dict, outpath: str):
    """Three panels: MeanAnt / PIGL / FESOM. All points at same x, no jitter."""
    forcing_types = ["MeanAnt", "PIGL", "FESOM"]
    region_labels = list(REGIONS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    for ax, ftype in zip(axes, forcing_types):
        for i, rname in enumerate(region_labels):
            df = all_data[rname][all_data[rname]["forcing"] == ftype]
            for _, row in df.iterrows():
                ax.scatter(i, row["std"], c=row["color"], marker=row["marker"],
                           s=60, alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)

        # AISLENS stars
        for i, (rname, info) in enumerate(REGIONS.items()):
            mali_val = np.mean([mali_std[b] for b in info["basins"]])
            ax.scatter(i, mali_val, c="black", marker="*", s=200, zorder=5,
                       edgecolors="white", linewidths=0.8,
                       label="AISLENS" if i == 0 else None)

        ax.set_xticks(range(len(region_labels)))
        ax.set_xticklabels(region_labels, fontsize=9, rotation=15, ha="right")
        ax.set_title(ftype, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Std of sub-shelf melt rate (m yr⁻¹)", fontsize=11)
    fig.suptitle(f"Forcing-type facets: std of sub-shelf melt ({Y0}–{Y1})\n"
                 "Does forcing choice or GCM choice drive the spread?",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.legend(handles=_gcm_legend_handles(), loc="lower center", ncol=5, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


# ─── Figure 4: Bubble plot (mean vs std) ────────────────────────────────────

def fig_bubble(all_data: dict[str, pd.DataFrame], mali_mean: dict, mali_std: dict, outpath: str):
    """x = mean melt rate, y = std. Color = GCM, marker = forcing type. AISLENS as star."""
    fig, ax = plt.subplots(figsize=(11, 7))

    region_colors_bubble = {
        "Ronne-Filchner":      "#0072B2",
        "Pine Island-Thwaites": "#D55E00",
        "Cook-Ninnis":          "#009E73",
        "Totten-Moscow U.":     "#CC79A7",
    }

    for rname, df in all_data.items():
        for _, row in df.iterrows():
            ax.scatter(row["mean_melt"], row["std"], c=row["color"], marker=row["marker"],
                       s=80, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=3)

    # AISLENS stars per region
    for rname, info in REGIONS.items():
        basins = info["basins"]
        m_s = np.mean([mali_std[b] for b in basins])
        m_m = np.mean([mali_mean[b] for b in basins])
        ax.scatter(m_m, m_s, c="black", marker="*", s=300, zorder=5,
                   edgecolors="white", linewidths=1.0)
        ax.annotate(rname, (m_m, m_s), textcoords="offset points", xytext=(8, 8),
                    fontsize=8, fontweight="bold", color="black")

    ax.set_xlabel("Mean sub-shelf melt rate (m yr⁻¹)", fontsize=12)
    ax.set_ylabel("Std of sub-shelf melt rate (m yr⁻¹)", fontsize=12)
    ax.set_title(f"Mean vs variability of sub-shelf melt ({Y0}–{Y1})\n"
                 "Color = GCM, marker = forcing type, star = AISLENS",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    handles = _gcm_legend_handles()
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


# ─── Figure 4b: Scatter plot (no jitter, full legend) ───────────────────────

def fig_scatter(all_data: dict[str, pd.DataFrame], mali_vals: dict, outpath: str,
                metric: str = "std", ylabel: str | None = None):
    """Scatter of a per-model metric ('std' or 'mean_melt'), 4 regions on x-axis.
    mali_vals: {basin_index: value}; AISLENS star = mean over the region's basins."""
    regions = list(REGIONS.keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    for rname in regions:
        ridx = regions.index(rname)
        for _, row in all_data[rname].iterrows():
            ax.scatter(ridx, row[metric], c=row["color"], marker=row["marker"],
                       s=90, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)

    # AISLENS stars
    for rname in regions:
        ridx = regions.index(rname)
        m_s = np.mean([mali_vals[b] for b in REGIONS[rname]["basins"]])
        ax.scatter(ridx, m_s, c="black", marker="*", s=350, zorder=5,
                   edgecolors="white", linewidths=1.0)

    ax.set_yscale(YSCALE)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, fontsize=11)
    default_lab = "std of 2000–2100 melt rate (m/yr)" + (", detrended" if DETREND else "")
    ax.set_ylabel(ylabel or default_lab, fontsize=12)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)

    handles = _gcm_legend_handles()
    handles.append(Line2D([0], [0], color="black", marker="*", markersize=12,
                          linestyle="None", label="AISLENS (MALI SSP585)"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, ncol=3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)

    # summary CSV (all models + AISLENS, per region)
    recs = [(r, row["model"], row["forcing"], row[metric]) for r in regions for _, row in all_data[r].iterrows()]
    recs += [(r, "AISLENS SSP585", "AISLENS", float(np.mean([mali_vals[b] for b in REGIONS[r]["basins"]])))
             for r in regions]
    csvp = os.path.join(REPO, "reports", os.path.basename(outpath).replace(".png", f"_{metric}.csv"))
    pd.DataFrame(recs, columns=["region", "series", "family", f"{metric}_m_per_yr"]).to_csv(csvp, index=False)
    print(f"Saved: {csvp}")


# ─── Figure 5: Time series (Jourdain Fig 10 style) ─────────────────────────

def fig_timeseries(all_ts, mali_ts, mali_members, outpath: str):
    """4-panel time series matching original Jourdain figure styling:
    MeanAnt = thin solid, PIGL = thick solid, FESOM = dashed.
    MALI ensemble mean + spread band overlaid."""
    region_items = list(REGIONS.items())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.ravel()

    for ax, (rname, info) in zip(axes, region_items):
        # --- Jourdain model lines from raw timeseries ---
        for entry in all_ts[rname]:
            ax.plot(entry["years"], entry["values"],
                    color=entry["color"], lw=entry["lw"], ls=entry["ls"], alpha=0.75)

        # --- MALI ensemble mean + spread for this region's basins ---
        # Compute regional mean PER MEMBER first, then take min/max across members
        basins = info["basins"]
        mali_yr = mali_ts[basins[0]][0]  # common year grid
        regional_member_means = []
        for b in basins:
            regional_member_means.append(mali_members[:, :, b])  # (n_members, n_years)
        regional_stack = np.stack(regional_member_means, axis=0)  # (n_basins, n_members, n_years)
        regional_mean_per_member = np.mean(regional_stack, axis=0)  # (n_members, n_years)
        mali_mean = np.mean(regional_mean_per_member, axis=0)
        mali_min = np.min(regional_mean_per_member, axis=0)
        mali_max = np.max(regional_mean_per_member, axis=0)

        ax.plot(mali_yr, mali_mean, color="black", lw=2.2, ls=":", zorder=10,
                label="AISLENS ensemble mean")
        ax.fill_between(mali_yr, mali_min, mali_max, color="black", alpha=0.12, zorder=9,
                        label="AISLENS ensemble spread")

        ax.set_title(rname, fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel("Sub-shelf melt (m/yr)", fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    # Set x-axis: use max year from MALI data (extends to 2300 if available)
    max_year = int(mali_yr.max())
    for ax in axes:
        ax.set_xlim(Y0, max_year)

    axes[2].set_xlabel("Year", fontsize=11)
    axes[3].set_xlabel("Year", fontsize=11)

    legend_elements = [
        Line2D([0], [0], color="black", lw=0.8, ls="-",  label="MeanAnt (thin)"),
        Line2D([0], [0], color="black", lw=2.0, ls="-",  label="PIGL (thick)"),
        Line2D([0], [0], color="black", lw=1.5, ls="--", label="FESOM (dashed)"),
        Line2D([0], [0], color="black", lw=0, marker="", label=""),
        Line2D([0], [0], color="black", lw=2.2, ls=":", label="AISLENS mean"),
        mpatches.Patch(facecolor="black", alpha=0.12, label="AISLENS spread"),
        Line2D([0], [0], color="black", lw=0, marker="", label=""),
        Line2D([0], [0], color="black", lw=1, label="CCSM4"),
        Line2D([0], [0], color="grey",  lw=1, label="CSIRO-MK3.6"),
        Line2D([0], [0], color="orange", lw=1, label="HadGEM2-ES"),
        Line2D([0], [0], color="lightskyblue", lw=1, label="IPSL-CM5A-MR"),
        Line2D([0], [0], color="green", lw=1, label="MIROC-ESM-CHEM"),
        Line2D([0], [0], color="#CCCC00", lw=1, label="NorESM1-M"),
        Line2D([0], [0], color="#1f77b4", lw=1, ls="--", label="FESOM MMM"),
        Line2D([0], [0], color="#d62728", lw=1, ls="--", label="FESOM ACCESS1.0"),
        Line2D([0], [0], color="#9467bd", lw=1, ls="--", label="FESOM HadCM3"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Digitized Jourdain et al. (2020) — sub-shelf melt time series\n"
                 "with AISLENS (MALI SSP585) ensemble mean and spread",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


# ─── Figure 5b: DETRENDED time series (variability only) ───────────────────

def fig_timeseries_detrended(all_ts, mali_ts, mali_members, outpath: str, member=None):
    """4-panel time series like fig_timeseries but with each series LINEARLY DETRENDED
    (its own warming / SSP585 trend removed) -> the VARIABILITY that underlies the
    detrended-std model-universe scatter. Fluctuations about zero."""
    def _dt(years, vals):
        v = np.asarray(vals, float); t = np.asarray(years, float)
        m = np.isfinite(v) & np.isfinite(t)
        if m.sum() < 3:
            return v - np.nanmean(v)
        return v - np.polyval(np.polyfit(t[m], v[m], 1), t)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.ravel()
    for ax, (rname, info) in zip(axes, REGIONS.items()):
        for entry in all_ts[rname]:
            ax.plot(entry["years"], _dt(entry["years"], entry["values"]),
                    color=entry["color"], lw=entry["lw"], ls=entry["ls"], alpha=0.75)

        basins = info["basins"]
        mali_yr = mali_ts[basins[0]][0]
        regional_stack = np.stack([mali_members[:, :, b] for b in basins], axis=0)  # (nb, nm, ny)
        regional_mean_per_member = np.mean(regional_stack, axis=0)                   # (nm, ny)
        dt_members = np.stack([_dt(mali_yr, regional_mean_per_member[mi])
                               for mi in range(regional_mean_per_member.shape[0])], axis=0)
        ax.fill_between(mali_yr, np.min(dt_members, axis=0), np.max(dt_members, axis=0),
                        color="black", alpha=0.10, zorder=9)
        if member is not None:
            ax.plot(mali_yr, dt_members[member], color="black", lw=1.9, ls="-", zorder=11)
        else:
            ax.plot(mali_yr, np.mean(dt_members, axis=0), color="black", lw=2.2, ls=":", zorder=10)
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_title(rname, fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel("Detrended sub-shelf melt (m/yr)", fontsize=10)
        ax.grid(alpha=0.2); ax.set_axisbelow(True)

    for ax in axes:
        ax.set_xlim(Y0, int(mali_yr.max()))
    axes[2].set_xlabel("Year", fontsize=11); axes[3].set_xlabel("Year", fontsize=11)

    legend_elements = [
        Line2D([0], [0], color="black", lw=0.8, ls="-",  label="MeanAnt (thin)"),
        Line2D([0], [0], color="black", lw=2.0, ls="-",  label="PIGL (thick)"),
        Line2D([0], [0], color="black", lw=1.5, ls="--", label="FESOM (dashed)"),
        Line2D([0], [0], color="black", lw=0, marker="", label=""),
        (Line2D([0], [0], color="black", lw=1.9, ls="-", label=f"AISLENS member {member}")
         if member is not None else
         Line2D([0], [0], color="black", lw=2.2, ls=":", label="AISLENS mean")),
        mpatches.Patch(facecolor="black", alpha=0.10, label="AISLENS spread"),
        Line2D([0], [0], color="black", lw=0, marker="", label=""),
        Line2D([0], [0], color="black", lw=1, label="CCSM4"),
        Line2D([0], [0], color="grey",  lw=1, label="CSIRO-MK3.6"),
        Line2D([0], [0], color="orange", lw=1, label="HadGEM2-ES"),
        Line2D([0], [0], color="lightskyblue", lw=1, label="IPSL-CM5A-MR"),
        Line2D([0], [0], color="green", lw=1, label="MIROC-ESM-CHEM"),
        Line2D([0], [0], color="#CCCC00", lw=1, label="NorESM1-M"),
        Line2D([0], [0], color="#1f77b4", lw=1, ls="--", label="FESOM MMM"),
        Line2D([0], [0], color="#d62728", lw=1, ls="--", label="FESOM ACCESS1.0"),
        Line2D([0], [0], color="#9467bd", lw=1, ls="--", label="FESOM HadCM3"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Digitized Jourdain et al. (2020) — DETRENDED sub-shelf melt (variability)\n"
                 "each series' linear trend removed; AISLENS (MALI SSP585) ensemble mean and spread",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


# ─── Figure: variability power spectra ──────────────────────────────────────

def fig_spectra(all_ts, mali_members, outpath):
    """Power spectrum of the DETRENDED 2000-2100 melt series per region: all Jourdain models + AISLENS
    (mean of per-member spectra). Colours / line-styles match the time-series figure. Detrending removes
    the forced ramp so what is plotted is the VARIABILITY spectrum (interannual->multidecadal)."""
    from scipy.signal import welch

    def psd(series):
        v = np.asarray(series, float); v = v[np.isfinite(v)]
        if v.size < 20:
            return None, None
        t = np.arange(v.size); v = v - np.polyval(np.polyfit(t, v, 1), t)   # linear detrend
        f, P = welch(v, fs=1.0, nperseg=min(v.size, 50))                    # fs = 1/yr
        var = np.trapz(P, f)                                                # total variance = area
        g = f > 0
        # normalize (÷ total variance) for shape-only or varpreserve so models are comparable
        Pg = P[g] / var if ((NORMALIZE or VARPRESERVE) and var > 0) else P[g]
        per = 1.0 / f[g]
        y = Pg / per if VARPRESERVE else Pg     # f*PSD (= P/period) for variance-preserving form
        return per, y                           # period (yr), plotted value

    def draw(ax, per, y, **kw):                 # linear-y semilog-x for varpreserve, else log-log
        (ax.semilogx if VARPRESERVE else ax.loglog)(per, y, **kw)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for ax, (rname, info) in zip(axes, REGIONS.items()):
        for entry in all_ts[rname]:
            per, y = psd(entry["values"])
            if per is not None:
                draw(ax, per, y, color=entry["color"], lw=entry["lw"] * 0.8, ls=entry["ls"], alpha=0.55)
        Ps = []
        for mi in range(mali_members.shape[0]):
            reg = np.nanmean(np.stack([mali_members[mi, :, b] for b in info["basins"]]), axis=0)
            per, y = psd(reg)
            if per is not None:
                Ps.append((per, y))
        if Ps:
            draw(ax, Ps[0][0], np.mean([p[1] for p in Ps], axis=0), color="black", lw=2.6, ls=":", zorder=10)
        ax.set_title(rname, fontsize=12, fontweight="bold", loc="left")
        ax.set_xlabel("period (years)")
        ax.set_ylabel("f·PSD, area=variance (variance-preserving)" if VARPRESERVE
                      else ("normalized power (shape only)" if NORMALIZE else "power (detrended melt)"))
        ax.grid(alpha=0.2, which="both")

    handles = _gcm_legend_handles() + [Line2D([0], [0], color="black", lw=2.6, ls=":", label="AISLENS")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


MODE_INFO = {  # mode key -> (y-axis label, column title)
    "raw":  ("power (detrended melt)",        "raw — amplitude"),
    "norm": ("normalized power (shape only)", "normalized — shape"),
    "vp":   ("f·PSD, area=variance",          "variance-preserving — timescale"),
}


def fig_spectra_composite(all_ts, mali_members, outpath, modes=("raw", "norm", "vp")):
    """Single plate: rows = the 4 Jourdain sectors, columns = the spectral views of the DETRENDED
    2000-2100 melt variability — raw PSD (amplitude), normalized PSD (shape only), variance-preserving
    f*PSD (where variance concentrates). `modes` selects/orders the columns (drop 'raw' for a tighter
    shape‖timescale plate, since its amplitude duplicates the detrended-std scatterplot)."""
    from scipy.signal import welch

    def base_psd(series):
        v = np.asarray(series, float); v = v[np.isfinite(v)]
        if v.size < 20:
            return None, None, None
        t = np.arange(v.size); v = v - np.polyval(np.polyfit(t, v, 1), t)   # linear detrend
        f, P = welch(v, fs=1.0, nperseg=min(v.size, 50))                    # fs = 1/yr
        var = np.trapz(P, f)
        g = f > 0
        return 1.0 / f[g], P[g], var

    def transform(per, P, var, mode):                                       # -> plotted y
        if mode == "raw":
            return P
        Pn = P / var if var > 0 else P
        return Pn / per if mode == "vp" else Pn

    def draw(ax, per, y, mode, **kw):
        (ax.semilogx if mode == "vp" else ax.loglog)(per, y, **kw)

    regions = list(REGIONS.items())
    ncol = len(modes)
    fig, axes = plt.subplots(len(regions), ncol, figsize=(5 * ncol, 15.5), squeeze=False)
    for ri, (rname, info) in enumerate(regions):
        for ci, mode in enumerate(modes):
            ylab, title = MODE_INFO[mode]
            ax = axes[ri, ci]
            for entry in all_ts[rname]:
                per, P, var = base_psd(entry["values"])
                if per is not None:
                    draw(ax, per, transform(per, P, var, mode), mode,
                         color=entry["color"], lw=entry["lw"] * 0.8, ls=entry["ls"], alpha=0.55)
            ys, per0 = [], None
            for mi in range(mali_members.shape[0]):
                reg = np.nanmean(np.stack([mali_members[mi, :, b] for b in info["basins"]]), axis=0)
                per, P, var = base_psd(reg)
                if per is not None:
                    per0 = per; ys.append(transform(per, P, var, mode))
            if ys:
                draw(ax, per0, np.mean(ys, axis=0), mode, color="black", lw=2.6, ls=":", zorder=10)
            ax.grid(alpha=0.2, which="both")
            if ri == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if ri == len(regions) - 1:
                ax.set_xlabel("period (years)")
            ax.set_ylabel(ylab, fontsize=9)
        axes[ri, 0].annotate(rname, xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90,
                             ha="center", va="center", fontsize=13, fontweight="bold")

    handles = _gcm_legend_handles() + [Line2D([0], [0], color="black", lw=2.6, ls=":", label="AISLENS")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=[0, 0.035, 1, 0.99])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


# ─── Figure 6: AISLENS all-basins time series grid ──────────────────────────

BASIN_NAMES_16 = [
    "Dronning Maud Land", "Enderby Land", "Amery-Lambert", "Phillipi/Denman",
    "Totten", "Mertz", "Victoria Land", "Ross",
    "Getz", "Thwaites/PIG", "Bellingshausen", "George VI",
    "Larsen A-C", "Larsen E", "FRIS", "Brunt-Stancomb",
]


def fig_aislens_all_basins(mali_ts: dict, outpath: str):
    """4x4 grid of AISLENS ensemble mean + spread for all 16 ISMIP6 basins."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=True, sharey=True)
    axes = axes.ravel()

    for b in range(16):
        ax = axes[b]
        yr, mean, mn, mx = mali_ts[b]
        ax.plot(yr, mean, color="black", lw=1.8)
        ax.fill_between(yr, mn, mx, color="grey", alpha=0.3)
        ax.set_title(BASIN_NAMES_16[b], fontsize=9, fontweight="bold", loc="left")
        ax.set_xlim(Y0, 2300)
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)

    for ax in axes:
        ax.tick_params(labelsize=7)

    axes[12].set_xlabel("Year", fontsize=10)
    axes[13].set_xlabel("Year", fontsize=10)
    axes[14].set_xlabel("Year", fontsize=10)
    axes[15].set_xlabel("Year", fontsize=10)
    for b in [0, 4, 8, 12]:
        axes[b].set_ylabel("Melt (m/yr)", fontsize=9)

    fig.suptitle("AISLENS (MALI SSP585) — sub-shelf melt rate, all 16 ISMIP6 basins\n"
                 "Ensemble mean (black) and spread (grey, 10 members)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mali-y1", type=int, default=Y1)
    ap.add_argument("--detrend", action="store_true",
                    help="linearly detrend each series before std -> isolates fluctuation variability from the trend")
    ap.add_argument("--member", type=int, default=None,
                    help="show a single AISLENS ensemble member (solid line in timeseries; that member's "
                         "std in the scatter) instead of the ensemble mean/aggregate")
    ap.add_argument("--region-agg", action="store_true",
                    help="scatter: also emit a version where AISLENS std = std of the REGION-MEAN series "
                         "(apples-to-apples with the Jourdain sector series), not mean of per-basin stds")
    ap.add_argument("--mean-window", default="2080,2100",
                    help="year window 'Y0,Y1' for the MEAN-melt scatter (--which scatter_mean)")
    ap.add_argument("--yscale", choices=["log", "linear"], default="log", help="y-axis scale for the scatter (std) figure")
    ap.add_argument("--normalize", action="store_true", help="spectra: divide each PSD by its total variance (unit area) -> compare shape only")
    ap.add_argument("--varpreserve", action="store_true", help="spectra: variance-preserving form (f*PSD vs log-period, linear y; area=variance, peak=dominant timescale)")
    ap.add_argument("--which", nargs="+",
                    default=["dumbbell", "violin", "forcing", "bubble", "scatter", "timeseries", "aislens_all"],
                    choices=["dumbbell", "violin", "forcing", "bubble", "scatter", "scatter_mean", "timeseries", "timeseries_detrended", "spectra", "spectra_composite", "spectra_composite2", "aislens_all"],
                    help="Which figures to generate")
    a = ap.parse_args()
    global DETREND, YSCALE, NORMALIZE, VARPRESERVE
    DETREND = a.detrend
    YSCALE = a.yscale
    NORMALIZE = a.normalize
    VARPRESERVE = a.varpreserve
    mali_y0, mali_y1 = Y0, a.mali_y1

    # Always load full 2000-2300 for timeseries extension
    print(f"Loading MALI ensemble (2000–2300)...")
    mali_std_full, mali_mean_full, mali_ts_full, mali_members_full = load_mali_ensemble(Y0, 2300)
    # Load 2000-{mali_y1} for scatter/dumbbell/violin/forcing/bubble figures
    if mali_y1 < 2300:
        print(f"Loading MALI ensemble ({mali_y0}–{mali_y1}) for model-universe figs...")
        mali_std, mali_mean, mali_ts, mali_members = load_mali_ensemble(mali_y0, mali_y1)
    else:
        mali_std, mali_mean, mali_ts, mali_members = mali_std_full, mali_mean_full, mali_ts_full, mali_members_full
    print(f"  {len(mali_std)} basins")

    all_data = {}
    all_ts = {}
    for rname, info in REGIONS.items():
        csv_path = os.path.join(JOURDAIN_DIR, info["csv"])
        print(f"Loading {rname}...")
        all_data[rname] = load_jourdain(csv_path, Y0, Y1)
        all_ts[rname] = load_jourdain_timeseries(csv_path, Y0, Y1)
        print(f"  {len(all_data[rname])} models")

    figdir = os.path.join(REPO, "reports", "figures")

    if "dumbbell" in a.which:
        fig_dumbbell(all_data, mali_std,
                     os.path.join(figdir, "jourdain_model_universe_dumbbell.png"))
    if "violin" in a.which:
        fig_violin(all_data, mali_std,
                   os.path.join(figdir, "jourdain_model_universe_violin.png"))
    if "forcing" in a.which:
        fig_forcing_facets(all_data, mali_std,
                           os.path.join(figdir, "jourdain_model_universe_forcing.png"))
    if "bubble" in a.which:
        fig_bubble(all_data, mali_mean, mali_std,
                   os.path.join(figdir, "jourdain_model_universe_bubble.png"))
    if "scatter" in a.which:
        base = f"jourdain_model_universe_{'variability' if DETREND else 'raw'}_{YSCALE}"
        fig_scatter(all_data, mali_std, os.path.join(figdir, base + ".png"))
        if a.member is not None:
            std_mem = {b: _std(mali_members[a.member, :, b], detrend=DETREND) for b in mali_std}
            fig_scatter(all_data, std_mem, os.path.join(figdir, f"{base}_mem{a.member}.png"))
        if a.region_agg:
            std_ra = region_agg_std(mali_members, DETREND, member=a.member)
            suf = f"_mem{a.member}" if a.member is not None else ""
            fig_scatter(all_data, std_ra, os.path.join(figdir, f"{base}_regionagg{suf}.png"))
    if "timeseries" in a.which:
        fig_timeseries(all_ts, mali_ts, mali_members,
                       os.path.join(figdir, "jourdain_timeseries.png"))
        fig_timeseries(all_ts, mali_ts_full, mali_members_full,
                       os.path.join(figdir, "jourdain_timeseries_2300.png"))
    if "scatter_mean" in a.which:
        mw0, mw1 = (int(x) for x in a.mean_window.split(","))
        all_mw = {rn: load_jourdain(os.path.join(JOURDAIN_DIR, info["csv"]), mw0, mw1)
                  for rn, info in REGIONS.items()}
        mm, mts = ((mali_members, mali_ts) if mw1 <= mali_y1 else (mali_members_full, mali_ts_full))
        ais_mean = region_agg_mean(mm, mts, mw0, mw1, member=a.member)
        suf = f"_mem{a.member}" if a.member is not None else ""
        fig_scatter(all_mw, ais_mean,
                    os.path.join(figdir, f"jourdain_model_universe_mean_{mw0}_{mw1}_{YSCALE}{suf}.png"),
                    metric="mean_melt", ylabel=f"mean {mw0}–{mw1} sub-shelf melt rate (m/yr)")
    if "timeseries_detrended" in a.which:
        suf = f"_mem{a.member}" if a.member is not None else ""
        fig_timeseries_detrended(all_ts, mali_ts, mali_members,
                                 os.path.join(figdir, f"jourdain_timeseries_detrended{suf}.png"),
                                 member=a.member)
    if "spectra" in a.which:
        suffix = "_varpreserve" if VARPRESERVE else ("_normalized" if NORMALIZE else "")
        fn = "jourdain_variability_spectra" + suffix + ".png"
        fig_spectra(all_ts, mali_members, os.path.join(figdir, fn))
    if "spectra_composite" in a.which:
        fig_spectra_composite(all_ts, mali_members,
                              os.path.join(figdir, "jourdain_variability_spectra_composite.png"))
    if "spectra_composite2" in a.which:
        fig_spectra_composite(all_ts, mali_members,
                              os.path.join(figdir, "jourdain_variability_spectra_composite2.png"),
                              modes=("norm", "vp"))
    if "aislens_all" in a.which:
        fig_aislens_all_basins(mali_ts_full,
                               os.path.join(figdir, "aislens_all_basins_2300.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
