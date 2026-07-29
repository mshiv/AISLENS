#!/usr/bin/env python3
"""
fig_misi_onset_timing.py — Map of when MISI retreat begins per basin per member.

MISI onset = first year grounding line migration flux exceeds 2× baseline std (first 50 yr).
Left: heatmap (basin × member), right: histogram. Uses regionalStats.nc.
CTRL excluded (133-region mask).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from ismip6_regions import BASIN_NAMES, SHORT_LABELS

SCENARIO_DIRS = {"SSP585": "SSP585", "varScaled10x": "SSP585_varScaled10x", "SSP126": "SSP126"}
SCENARIO_INCLUDE = {"SSP585": r"^SSP585_\d+$", "varScaled10x": r"^SSP585_\d+$", "SSP126": r"^SSP126_\d+$"}
SCENARIO_COLORS = {"SSP585": "#C62828", "varScaled10x": "#E65100", "SSP126": "#1565C0"}


def load_regional_gl_migration(root, ensemble, include):
    """Load GL migration flux for all members.
    Returns (years, (member, year, nRegions)) in mm/yr SLE."""
    members = eio.discover_members(
        os.path.join(root, ensemble), stats_filename="regionalStats.nc", include=include
    )
    stacks, nmin = [], None
    for name, path in members:
        try:
            ds = eio.to_year_dim(eio.load_member_regionalstats(path))
        except Exception:
            continue
        var = "regionalSumGroundingLineMigrationFlux"
        if var not in ds:
            continue
        yr = ds["year"].values
        if yr[0] > 5.0 or len(yr) < 10:
            continue
        nreg = ds.dims["nRegions"]
        RHO_ICE = eio.RHO_ICE
        OCEAN_AREA = eio.OCEAN_AREA
        vals = np.column_stack([
            ds[var].isel(nRegions=r).values * (1.0 / (RHO_ICE * OCEAN_AREA)) * 1000.0
            for r in range(nreg)
        ])
        stacks.append((yr, vals))
        nmin = len(yr) if nmin is None else min(nmin, len(yr))
    if len(stacks) < 3:
        return None, None, None
    years = stacks[0][0][:nmin]
    arr = np.stack([s[:nmin] for _, s in stacks], axis=0)
    names = [n for n, _ in members[:len(stacks)]]
    return years, arr, names


def detect_onset(yrs, arr, baseline_years=50, threshold_sigma=2.0):
    """Detect MISI onset: first year where GL migration flux > baseline_mean + threshold * baseline_std.
    Uses POSITIVE flux (grounded → floating) as the retreat signal.
    Returns (nMembers, nRegions) array of onset years, NaN where no onset detected."""
    nMem, nYr, nReg = arr.shape
    onset = np.full((nMem, nReg), np.nan)

    baseline_mask = yrs < baseline_years
    for m in range(nMem):
        for r in range(nReg):
            ts = arr[m, :, r]
            bl = ts[baseline_mask]
            if len(bl) < 5:
                continue
            bl_mean = np.nanmean(bl)
            bl_std = np.nanstd(bl)
            if bl_std < 1e-10:
                continue
            threshold = bl_mean + threshold_sigma * bl_std
            # Find first year where flux stays above threshold for 5+ consecutive years
            above = ts > threshold
            # Smooth: require 5-year running mean above threshold
            kernel = np.ones(5) / 5
            smooth = np.convolve(above.astype(float), kernel, mode="valid")
            sustained = smooth >= 0.8  # 80% of 5-year window above threshold
            idx = np.argmax(sustained)
            if sustained[idx]:
                onset[m, r] = yrs[idx]
    return onset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--out-dir", default="/Users/smurugan9/research/aislens/AISLENS/reports/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scenarios = ["SSP585", "varScaled10x", "SSP126"]
    nsc = len(scenarios)

    fig, axes = plt.subplots(1, nsc + 1, figsize=(5 * nsc + 4, 6),
                             gridspec_kw={"width_ratios": [1] * nsc + [0.6]})

    onset_all = {}
    for col, sc in enumerate(scenarios):
        ax = axes[col]
        yrs, arr, mem_names = load_regional_gl_migration(
            args.root, SCENARIO_DIRS[sc], SCENARIO_INCLUDE[sc]
        )
        if arr is None:
            ax.set_visible(False)
            continue
        onset = detect_onset(yrs, arr)
        onset_all[sc] = onset
        nreg = onset.shape[1]

        # Heatmap: basins (y) × members (x)
        im = ax.imshow(onset.T, aspect="auto", cmap="viridis_r", vmin=0, vmax=300,
                       interpolation="nearest")
        ax.set_yticks(range(nreg))
        ax.set_yticklabels([SHORT_LABELS.get(BASIN_NAMES[r], BASIN_NAMES[r])
                            for r in range(nreg)], fontsize=7)
        ax.set_xticks(range(onset.shape[0]))
        ax.set_xticklabels([f"m{i}" for i in range(onset.shape[0])], fontsize=6, rotation=45)
        ax.set_title(sc, fontsize=11, fontweight="bold", color=SCENARIO_COLORS.get(sc, "k"))
        if col == 0:
            ax.set_ylabel("Basin")
        fig.colorbar(im, ax=ax, fraction=0.046, label="Onset year" if col == nsc - 1 else "")

    # Right panel: histogram of onset years for SSP585
    ax_hist = axes[-1]
    if "SSP585" in onset_all:
        onset_flat = onset_all["SSP585"].flatten()
        onset_valid = onset_flat[np.isfinite(onset_flat)]
        if len(onset_valid) > 0:
            ax_hist.hist(onset_valid, bins=range(0, 310, 20), color=SCENARIO_COLORS["SSP585"],
                         alpha=0.7, edgecolor="k", lw=0.5)
            ax_hist.axvline(np.median(onset_valid), color="k", ls="--", lw=1.5,
                            label=f"median = {np.median(onset_valid):.0f} yr")
            ax_hist.legend(fontsize=8)
    ax_hist.set_xlabel("Onset year")
    ax_hist.set_ylabel("Count (basin × member)")
    ax_hist.set_title("SSP585\nonset distribution", fontsize=9)
    ax_hist.tick_params(labelsize=8)

    fig.suptitle("MISI Onset Timing — When GL Retreat Begins",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "misi_onset_timing.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out}")

    for sc in scenarios:
        if sc not in onset_all:
            continue
        onset = onset_all[sc]
        valid = onset[np.isfinite(onset)]
        if len(valid) == 0:
            print(f"\n  {sc}: no MISI onset detected")
            continue
        print(f"\n  {sc}: {len(valid)}/{onset.size} basin-member pairs show onset")
        print(f"    median = {np.median(valid):.0f} yr, range = {valid.min():.0f}–{valid.max():.0f} yr")
        # Per-basin stats
        for r in range(onset.shape[1]):
            basin_onset = onset[:, r]
            v = basin_onset[np.isfinite(basin_onset)]
            if len(v) > 0:
                name = BASIN_NAMES[r]
                print(f"    {SHORT_LABELS.get(name, name):14s}: {np.median(v):6.0f} yr "
                      f"({len(v)}/{onset.shape[0]} members)")


if __name__ == "__main__":
    main()
