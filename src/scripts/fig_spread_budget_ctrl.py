#!/usr/bin/env python3
"""
fig_spread_budget_ctrl.py — Spread budget for CTRL (variability-only, no forced trend).

Figure (A): stacked area of per-shelf variance fraction + per-shelf σ + top-10 shelves.
Figure (B): map-based variance budget on MALI mesh. Uses 100 ice shelves (133-region mask).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from ismip6_regions import BASIN_NAMES

REGION_MASK_133 = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "projects",
    "MALI_AISLENS_DIR_TEMPLATE_4KM", "DRAFT-DEPEN-TEMPLATE-DIR",
    "aislens_draftDepen_regionMasks.nc")

MESH = ("data/MALI/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_"
        "Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_"
        "meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
MASK_ISMIP6 = "data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"


def load_133_region_names():
    ds = xr.open_dataset(REGION_MASK_133, decode_times=False)
    raw = ds["regionNames"].values
    names = []
    for row in raw:
        if isinstance(row, bytes):
            s = row.decode("utf-8", "ignore")
        else:
            s = b"".join([c if isinstance(c, bytes) else bytes(c) for c in row]).decode(
                "utf-8", "ignore")
        names.append(s.strip())
    ds.close()
    return names


def load_ctrl_shelf_data(root, include=r"^CTRL_\d+$", shelf_start=33, min_years=50):
    """min_years drops short members so the intersection span is not capped by them."""
    """Load CTRL regional data for the 100 shelf basins (regions 33-132)."""
    ens_dir = os.path.join(root, "CTRL")
    members = eio.discover_members(ens_dir, stats_filename="regionalStats.nc", include=include)
    stacks, nmin = [], None
    for name, path in members:
        try:
            ds = eio.to_year_dim(eio.load_member_regionalstats(path))
        except Exception:
            continue
        if "regionalVolumeAboveFloatation" not in ds:
            continue
        yr = ds["year"].values
        # NOTE: compare SPAN IN YEARS, not sample count. Output cadence varies, so a
        # member can have many samples over a short record (CTRL_09: 297 samples, 91.8 yr).
        if yr[0] > 5.0 or (yr[-1] - yr[0]) < min_years:
            continue
        vaf = ds["regionalVolumeAboveFloatation"].values  # (year, 133)
        sle_shelves = np.column_stack([
            eio.vaf_to_sle_mm(vaf[:, r], reference="first")
            for r in range(shelf_start, vaf.shape[1])
        ])
        stacks.append((yr, sle_shelves))
        nmin = len(yr) if nmin is None else min(nmin, len(yr))
    if len(stacks) < 3:
        return None, None, None
    # Common span in YEARS (not index): interpolate every member onto one annual grid,
    # so the horizon label means the same thing for all members regardless of cadence.
    y_end = min(s[0][-1] for s in stacks)
    years = np.arange(0.0, y_end + 1e-9, 1.0)
    arr = np.stack([
        np.column_stack([np.interp(years, yr_m, sh[:, c]) for c in range(sh.shape[1])])
        for yr_m, sh in stacks
    ], axis=0)  # (member, year, 100)
    return years, arr, len(stacks)


def paint_mesh(xC, yC, masks_133, in_any, values, shelf_start=33):
    """Paint mesh cells by their shelf basin's value.
    
    The 133-region mask has aggregate regions (0-32) that overlap with shelf regions (33-132).
    We find cells that belong to shelf regions and assign values accordingly.
    """
    n_cells = len(xC)
    cell_val = np.full(n_cells, np.nan)
    
    # Shelf masks: (n_cells, 100)
    shelf_masks = masks_133[:, shelf_start:]
    
    # For each cell, find which shelf region it belongs to
    # argmax gives the index of the first True, but we need to handle overlaps
    # Use: for each cell, find all shelf regions it belongs to, pick the first one
    has_shelf = shelf_masks.any(axis=1)  # cells in at least one shelf region
    
    # Vectorized: find the shelf index for each cell
    # Create a matrix of shelf indices repeated for each cell
    shelf_indices = np.arange(shelf_masks.shape[1])  # 0-99
    
    # For cells with shelf membership, find the first shelf index
    # Use argmax on the shelf masks to get the first shelf region
    first_shelf = np.argmax(shelf_masks, axis=1)  # first shelf index (0-99)
    
    # Only assign where cell has shelf membership
    cell_val[has_shelf] = values[first_shelf[has_shelf]]
    
    return cell_val


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--mesh", default=MESH)
    ap.add_argument("--start-year", type=float, default=2000.0)
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--min-years", type=float, default=50.0,
                    help="drop members shorter than this (yr). CTRL has 3 short members "
                         "(~92/109/145 yr) that otherwise cap the intersection span; "
                         "--min-years 150 keeps 7 members over ~215 yr.")
    ap.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    all_names = load_133_region_names()
    shelf_names = all_names[33:]  # 100 shelf names
    nshelves = len(shelf_names)

    years, arr, n_mem = load_ctrl_shelf_data(a.root, min_years=a.min_years)
    if arr is None:
        sys.exit("CTRL: no usable members")
    cal = a.start_year + years
    n_yr = arr.shape[1]
    print(f"CTRL: {n_mem} members, {n_yr} years, {nshelves} shelves")

    idx = int(np.argmin(np.abs(years - a.horizon)))
    # argmin CLAMPS to the nearest available year: if the ensemble is shorter than the
    # requested horizon (e.g. an in-progress CTRL reaching only ~77 yr), the figure would
    # otherwise be labelled "yr300" while showing yr77. Use the ACTUAL year everywhere.
    horizon_actual = float(years[idx])
    if abs(horizon_actual - a.horizon) > 1.0:
        print(f"[WARN] requested horizon yr{a.horizon:.0f} exceeds the ensemble's span "
              f"(max yr{years.max():.1f}); using yr{horizon_actual:.1f} and labelling it as such.")
    arr_h = arr[:, idx, :]  # (member, shelf) at horizon

    # ---- Per-shelf sigma and variance fraction ----
    sig = np.nanstd(arr_h, axis=0, ddof=1)  # (nshelves,)
    sig2 = sig ** 2
    total_sig2 = sig2.sum()
    fraction = sig2 / total_sig2
    sorted_idx = np.argsort(fraction)[::-1]

    # Per-shelf sigma over time
    sig_t = np.nanstd(arr, axis=0, ddof=1)  # (year, shelf)
    total_sig2_t = (sig_t ** 2).sum(axis=1)
    fraction_t = sig_t ** 2 / np.maximum(total_sig2_t[:, None], 1e-9)

    print(f"\nPer-shelf variance at yr{horizon_actual:.0f} (cal {a.start_year + horizon_actual:.0f}):")
    cumsum = 0
    for r in sorted_idx[:15]:
        cumsum += fraction[r]
        print(f"  {shelf_names[r]:30s}  σ={sig[r]:6.3f} mm  σ²={100*fraction[r]:5.2f}%  (cum: {100*cumsum:5.1f}%)")

    # ---- Figure A: 3-panel time series ----
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(18, 7),
                                            gridspec_kw={"width_ratios": [2, 1.2, 1.2]})

    # Panel (a): stacked area — top 10 shelves
    top10 = sorted_idx[:10]
    other_frac = 1.0 - fraction_t[:, top10].sum(axis=1)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    labels = [shelf_names[r] for r in top10]
    stack_data = np.column_stack([fraction_t[:, r] for r in top10])
    ax_a.stackplot(cal[:n_yr], stack_data[:n_yr].T, labels=labels,
                   colors=colors, alpha=0.85)
    ax_a.stackplot(cal[:n_yr], other_frac[:n_yr], labels=["other 90 shelves"],
                   colors=["0.7"], alpha=0.6)
    ax_a.set_xlabel("year")
    ax_a.set_ylabel("fraction of total σ²")
    ax_a.set_title("(a) CTRL variance budget: which shelves dominate?")
    ax_a.set_ylim(0, 1)
    ax_a.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, ncol=1)
    ax_a.grid(alpha=0.2)

    # Panel (b): per-shelf σ at horizons
    horizons_yr = [100, 200, min(300, n_yr-1), n_yr-1]
    bar_height = 0.18
    y_pos = np.arange(10)
    for j, h_yr in enumerate(horizons_yr):
        i = min(h_yr, n_yr-1)
        ax_b.barh(y_pos + j*bar_height, sig_t[i, top10], height=bar_height,
                  label=f"yr{years[i]:.0f}", alpha=0.8)
    ax_b.set_yticks(y_pos + 1.5*bar_height)
    ax_b.set_yticklabels([shelf_names[r] for r in top10], fontsize=7)
    ax_b.set_xlabel("ensemble spread σ (mm SLE)")
    ax_b.set_title(f"(b) Per-shelf σ at horizons")
    ax_b.legend(fontsize=8)
    ax_b.grid(axis="x", alpha=0.2)

    # Panel (c): top 10 at final year
    y_pos_c = np.arange(10)
    ax_c.barh(y_pos_c, sig[sorted_idx[:10]], height=0.6, color="C0", alpha=0.8)
    ax_c.set_yticks(y_pos_c)
    ax_c.set_yticklabels([shelf_names[r] for r in sorted_idx[:10]], fontsize=7)
    ax_c.set_xlabel("ensemble spread σ (mm SLE)")
    ax_c.set_title(f"(c) Top 10 shelves @ yr{horizon_actual:.0f}")
    ax_c.grid(axis="x", alpha=0.2)

    fig.suptitle(f"CTRL spread budget: spatial decomposition of ensemble uncertainty "
                 f"({n_mem} members, yr{a.horizon:.0f})",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_a = os.path.join(a.out_dir, "spread_budget_CTRL.png")
    fig.savefig(out_a, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure A -> {out_a}")

    # ---- Figure B: Map-based variance budget ----
    masks_133 = np.asarray(xr.open_dataset(REGION_MASK_133, decode_times=False)["regionCellMasks"].values)
    dmesh = xr.open_dataset(a.mesh, decode_times=False)
    xC = np.asarray(dmesh["xCell"].values) / 1e3
    yC = np.asarray(dmesh["yCell"].values) / 1e3
    
    # Only cells that are in shelf regions (33-132)
    shelf_masks = masks_133[:, 33:]
    in_shelf = shelf_masks.any(axis=1)
    # Also include cells in aggregate regions for context
    in_any = masks_133.any(axis=1)

    cell_frac = paint_mesh(xC, yC, masks_133, in_shelf, fraction, shelf_start=33)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Background: all cells in any region (light grey)
    ax.scatter(xC[in_any & ~in_shelf], yC[in_any & ~in_shelf], s=0.5, c="0.88", marker=".", linewidths=0)
    # Shelf cells colored by variance fraction
    shelf_cells = in_shelf & np.isfinite(cell_frac)
    vmax_f = max(0.1, float(np.nanpercentile(cell_frac[shelf_cells], 99)))
    sc = ax.scatter(xC[shelf_cells], yC[shelf_cells], s=1.2, c=cell_frac[shelf_cells],
                    cmap="YlOrRd", vmin=0, vmax=vmax_f, marker=".", linewidths=0)
    fig.colorbar(sc, ax=ax, fraction=0.046, label="fraction of total σ²")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"CTRL variance budget on mesh — yr{a.horizon:.0f}\n"
                 f"({n_mem} members, variability only, no forced trend)")

    # Label top 5 shelves at centroids
    for r in sorted_idx[:5]:
        # Find cells in this shelf region
        shelf_col = r  # 0-99 index in shelf_masks
        sel = shelf_masks[:, shelf_col] > 0
        if sel.any():
            ax.annotate(shelf_names[r], (xC[sel].mean(), yC[sel].mean()), fontsize=8,
                        fontweight="bold", color="k", ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    fig.tight_layout()
    out_b = os.path.join(a.out_dir, "map_spread_budget_CTRL.png")
    fig.savefig(out_b, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Figure B -> {out_b}")


if __name__ == "__main__":
    main()
