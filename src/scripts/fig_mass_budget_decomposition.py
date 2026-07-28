#!/usr/bin/env python3
"""
fig_mass_budget_decomposition.py — Decompose regional mass loss into physical components.

Two outputs: (A) stacked bars per basin at yr100/200/300, (B) 4×4 grid of budget term
evolution. Uses regionalStats.nc. CTRL excluded (133-region mask incompatible).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from ismip6_regions import BASIN_NAMES, SHORT_LABELS

RHO_ICE = eio.RHO_ICE
OCEAN_AREA = eio.OCEAN_AREA

SCENARIO_DIRS = {
    "SSP585":       "SSP585",
    "varScaled10x": "SSP585_varScaled10x",
    "SSP126":       "SSP126",
}
SCENARIO_INCLUDE = {
    "SSP585":       r"^SSP585_\d+$",
    "varScaled10x": r"^SSP585_\d+$",
    "SSP126":       r"^SSP126_\d+$",
}
SCENARIO_COLORS = {
    "SSP585":       "#C62828",
    "varScaled10x": "#E65100",
    "SSP126":       "#1565C0",
}

# Budget terms: (variable_in_netcdf, sign_multiplier, label, color)
BUDGET_TERMS = [
    ("sfcMassBal",           -1, "SMB loss",     "#43A047"),  # SMB: + = gain, negate → + = loss
    ("basalMelt",            +1, "Basal melt",    "#E65100"),  # grounded + floating BMB: + = gain, negate
    ("calvingFlux",          +1, "Calving",       "#C62828"),
    ("faceMeltFlux",         +1, "Face melt",     "#6A1B9A"),
]

# Variables to load per member
LOAD_VARS = [
    "regionalSumSfcMassBal",
    "regionalSumGroundedBasalMassBal",
    "regionalSumFloatingBasalMassBal",
    "regionalSumCalvingFlux",
    "regionalSumFaceMeltingFlux",
]


def kg_yr_to_mm_sle(kg_yr):
    """Convert area-integrated mass flux (kg/yr) to SLE rate (mm/yr)."""
    return kg_yr * (1.0 / (RHO_ICE * OCEAN_AREA)) * 1000.0


def load_budget_data(root, ensemble, include):
    """Load all budget variables for all members.
    Returns dict: varname -> (years, (member, year, nRegions)) in mm/yr SLE."""
    members = eio.discover_members(
        os.path.join(root, ensemble), stats_filename="regionalStats.nc", include=include
    )
    # First pass: find valid members
    valid = []
    for name, path in members:
        try:
            ds = eio.to_year_dim(eio.load_member_regionalstats(path))
        except Exception:
            continue
        yr = ds["year"].values
        if yr[0] > 5.0 or len(yr) < 10:
            continue
        has_all = all(v in ds for v in LOAD_VARS)
        if not has_all:
            continue
        valid.append((name, path, ds, yr))

    if len(valid) < 3:
        return None, None

    # Find common year range
    nmin = min(len(yr) for _, _, _, yr in valid)
    ref_yrs = valid[0][3][:nmin]

    result = {}
    for varname in LOAD_VARS:
        arrs = []
        for _, _, ds, yr in valid:
            nreg = ds.dims["nRegions"]
            vals = np.column_stack([
                kg_yr_to_mm_sle(ds[varname].isel(nRegions=r).values[:nmin])
                for r in range(nreg)
            ])
            arrs.append(vals)
        result[varname] = (ref_yrs, np.stack(arrs, axis=0))

    return ref_yrs, result


def compute_budget(budget_data):
    """Compute the 4 budget terms from raw variables.
    Returns dict: term_label -> (years, (member, year, nRegions)) in mm/yr SLE."""
    yrs = budget_data[list(budget_data.keys())[0]][0]
    terms = {}
    for varkey, sign, label, color in BUDGET_TERMS:
        if varkey == "sfcMassBal":
            vals = -budget_data["regionalSumSfcMassBal"][1]  # negate: SMB gain → loss
        elif varkey == "basalMelt":
            # groundedBMB + floatingBMB: both + = gain, negate to get loss
            vals = -(budget_data["regionalSumGroundedBasalMassBal"][1] +
                     budget_data["regionalSumFloatingBasalMassBal"][1])
        elif varkey == "calvingFlux":
            vals = budget_data["regionalSumCalvingFlux"][1]
        elif varkey == "faceMeltFlux":
            vals = budget_data["regionalSumFaceMeltingFlux"][1]
        terms[label] = (yrs, vals, color)
    return terms


def plot_stacked_bars(terms_dict, scenarios, nreg, outpath):
    """Stacked bar chart: per basin per scenario at key horizons."""
    horizons = [100, 200, 300]
    nsc = len(scenarios)
    nhr = len(horizons)

    fig, axes = plt.subplots(nhr, nsc, figsize=(4.2 * nsc, 3.8 * nhr), sharex=True)
    basin_labels = [SHORT_LABELS.get(BASIN_NAMES[r], BASIN_NAMES[r]) for r in range(nreg)]
    x = np.arange(nreg)
    width = 0.7

    for col, sc in enumerate(scenarios):
        if sc not in terms_dict:
            for ax in axes[:, col]:
                ax.set_visible(False)
            continue
        yrs, term_data = terms_dict[sc]  # term_data: {label: (yrs, (member, year, nReg), color)}

        for row, yr_target in enumerate(horizons):
            ax = axes[row, col]
            idx = np.argmin(np.abs(yrs - yr_target))

            bottoms_pos = np.zeros(nreg)
            bottoms_neg = np.zeros(nreg)
            for label, (_, vals, color) in term_data.items():
                mean_vals = np.nanmean(vals[:, idx, :], axis=0)
                std_vals = np.nanstd(vals[:, idx, :], axis=0)

                for r in range(nreg):
                    v = mean_vals[r]
                    if v >= 0:
                        ax.bar(r, v, width, bottom=bottoms_pos[r], color=color,
                               edgecolor="white", lw=0.3, label=label if row == 0 else "")
                        bottoms_pos[r] += v
                    else:
                        ax.bar(r, v, width, bottom=bottoms_neg[r], color=color,
                               edgecolor="white", lw=0.3, alpha=0.7,
                               label=label if row == 0 else "")
                        bottoms_neg[r] += v

            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(basin_labels, rotation=70, ha="right", fontsize=6)
            ax.set_ylabel("mm/yr SLE" if col == 0 else "")
            ax.set_title(f"{sc} yr{yr_target}", fontsize=9, fontweight="bold",
                         color=SCENARIO_COLORS.get(sc, "k"))
            ax.tick_params(labelsize=7)
            ax.set_xlim(-0.5, nreg - 0.5)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Regional Mass Budget Decomposition — AISLENS",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {outpath}")


def plot_timeseries_grid(terms_dict, scenarios, nreg, outpath):
    """4×4 grid (basins × scenarios) showing budget term evolution."""
    # Pick top 4 basins by total variance
    # Use SSP585 as reference
    if "SSP585" not in terms_dict:
        print("  skipping timeseries grid: SSP585 not available")
        return
    yrs_ref, ref_terms = terms_dict["SSP585"]
    total_var = np.zeros(nreg)
    for label, (_, vals, _) in ref_terms.items():
        total_var += np.nanvar(vals, axis=(0, 1))  # var over (member, year) per basin
    top_basins = np.argsort(total_var)[::-1][:4]

    nsc = len(scenarios)
    fig, axes = plt.subplots(len(top_basins), nsc, figsize=(4.5 * nsc, 3.5 * len(top_basins)),
                             sharex=True)
    if len(top_basins) == 1:
        axes = axes.reshape(-1, 1)

    for col, sc in enumerate(scenarios):
        if sc not in terms_dict:
            for row in range(len(top_basins)):
                axes[row, col].set_visible(False)
            continue
        yrs, term_data = terms_dict[sc]
        for row, r in enumerate(top_basins):
            ax = axes[row, col]
            for label, (_, vals, color) in term_data.items():
                ens_mean = np.nanmean(vals[:, :, r], axis=0)
                ens_std = np.nanstd(vals[:, :, r], axis=0)
                ax.plot(yrs, ens_mean, color=color, lw=1.2, label=label)
                ax.fill_between(yrs, ens_mean - ens_std, ens_mean + ens_std, color=color, alpha=0.15)
            ax.axhline(0, color="k", lw=0.5, ls="--")
            if col == 0:
                name = BASIN_NAMES[r]
                ax.set_ylabel(SHORT_LABELS.get(name, name), fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(sc, fontsize=10, fontweight="bold", color=SCENARIO_COLORS.get(sc, "k"))
            if row == len(top_basins) - 1:
                ax.set_xlabel("Year")
            ax.set_xlim(yrs[0], yrs[-1])
            ax.tick_params(labelsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Mass Budget Term Evolution — Top 4 Basins by Variance",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--out-dir", default="/Users/smurugan9/research/aislens/AISLENS/reports/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_terms = {}
    nreg = 16
    for sc, d in SCENARIO_DIRS.items():
        print(f"\nloading {sc}...")
        yrs, raw = load_budget_data(args.root, d, SCENARIO_INCLUDE[sc])
        if raw is None:
            print(f"  skipped {sc}: insufficient members")
            continue
        all_terms[sc] = (yrs, compute_budget(raw))
        print(f"  loaded {sc}: {raw[list(raw.keys())[0]][1].shape[0]} members, "
              f"{raw[list(raw.keys())[0]][1].shape[1]} years")

    if not all_terms:
        print("No data loaded — exiting")
        return

    plot_stacked_bars(all_terms, list(SCENARIO_DIRS.keys()), nreg,
                      os.path.join(args.out_dir, "mass_budget_stacked_bars.png"))

    plot_timeseries_grid(all_terms, list(SCENARIO_DIRS.keys()), nreg,
                         os.path.join(args.out_dir, "mass_budget_timeseries.png"))


if __name__ == "__main__":
    main()
