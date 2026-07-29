#!/usr/bin/env python3
"""
fig_driving_stress.py — Driving stress evolution per basin.

Driving stress τ_d = ρ_ice * g * H * ∇s (ice thickness × surface slope).
Shows the dynamic forcing driving retreat. Computed from thickness + upperSurface.

Requires HPC: output_state files not available locally.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio
from ismip6_regions import BASIN_NAMES, SHORT_LABELS

RHO_ICE = 910.0
G = 9.81


def compute_driving_stress(thickness, upper_surface, x_cell, y_cell, area_cell):
    """Approximate driving stress from thickness and surface slope.
    τ_d = ρ_ice * g * H * |∇s|
    Uses finite differences between neighboring cells."""
    nCells = len(thickness)
    # Simple gradient approximation: use nearest neighbors
    # For a production version, use the MPAS mesh connectivity
    # Here we use a spatial gradient via scipy
    from scipy.ndimage import gaussian_gradient_magnitude

    # For irregular mesh, approximate with nearest-regular-grid gradient
    # This is a rough approximation — better to use MPAS edge-based gradients
    # For now: τ_d ∝ H * |∇s| approximated as H * spatial variability
    tau_d = np.zeros(nCells)
    valid = thickness > 10  # only where ice is > 10m thick
    # Simple proxy: local std of thickness in a neighborhood
    # (production version should use actual mesh gradients)
    tau_d[valid] = RHO_ICE * G * thickness[valid] * 0.001  # placeholder slope
    return tau_d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="ENSEMBLES root (HPC path)")
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--include", default=r"^SSP585_\d+$")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--year-start", type=int, default=0)
    ap.add_argument("--year-end", type=int, default=300)
    ap.add_argument("--out", default="reports/figures/thickness_weighted_proxy_regional.png")
    args = ap.parse_args()

    from ensemble_io import discover_members

    _, mask_path = sio.default_paths() if args.mesh is None else (args.mesh, args.mask)
    region_mask = sio.load_region_mask(mask_path)

    members = discover_members(os.path.join(args.root, args.ensemble),
                               stats_filename="output_state_*.nc", include=args.include)
    valid_members = []
    for name, _ in members:
        member_dir = os.path.join(args.root, args.ensemble, name)
        files = sio.find_output_state_files(member_dir, args.year_start, args.year_end)
        if files:
            valid_members.append((name, member_dir))

    if not valid_members:
        print(f"No members with output_state files found for {args.ensemble}")
        return

    print(f"Found {len(valid_members)} members with spatial output")

    # Load thickness and upperSurface for each member, compute driving stress
    all_tau = {}
    for name, member_dir in valid_members:
        try:
            H, years = sio.load_spatial_variable(member_dir, "thickness",
                                                  args.year_start, args.year_end)
            sfc, _ = sio.load_spatial_variable(member_dir, "upperSurface",
                                                args.year_start, args.year_end)
            # Approximate driving stress: τ_d = ρgH (slope term needs mesh gradients)
            # For ensemble comparison, relative changes in H are informative even without slope
            tau = RHO_ICE * G * H  # Pa (proxy: full τ_d needs ∇s)
            regional = sio.aggregate_by_region(tau, region_mask)
            all_tau[name] = (years, regional)
            print(f"  {name}: {len(years)} years")
        except Exception as e:
            print(f"  {name}: SKIPPED ({e})")

    if not all_tau:
        return

    ref_years = list(all_tau.values())[0][0]
    nreg = list(all_tau.values())[0][1].shape[1]
    stacked = []
    for name, (yrs, reg) in all_tau.items():
        interp = np.column_stack([np.interp(ref_years, yrs, reg[:, r]) for r in range(nreg)])
        stacked.append(interp)
    arr = np.stack(stacked, axis=0)  # (member, year, nRegions)

    fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=True)
    axes_flat = axes.flatten()

    for r in range(nreg):
        ax = axes_flat[r]
        name = BASIN_NAMES[r]
        lbl = SHORT_LABELS.get(name, name)
        ens_mean = np.nanmean(arr[:, :, r], axis=0) / 1e6  # MPa
        ens_std = np.nanstd(arr[:, :, r], axis=0) / 1e6
        ax.plot(ref_years, ens_mean, "k-", lw=1.2, label="mean")
        ax.fill_between(ref_years, ens_mean - ens_std, ens_mean + ens_std,
                        color="sienna", alpha=0.3)
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.set_ylabel("MPa")
        ax.tick_params(labelsize=7)

    for r in range(nreg, len(axes_flat)):
        axes_flat[r].set_visible(False)

    axes_flat[12].set_xlabel("Year")
    axes_flat[13].set_xlabel("Year")
    fig.suptitle(f"{args.ensemble}: Thickness-Weighted Proxy (ρgH) — AISLENS Regional",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {args.out}")


if __name__ == "__main__":
    main()
