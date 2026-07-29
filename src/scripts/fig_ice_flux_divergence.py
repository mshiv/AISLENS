#!/usr/bin/env python3
"""
fig_ice_flux_divergence.py — ice flux divergence (dH/dt proxy) per basin.

Shows where ice is dynamically thinning. Requires HPC: output_state files.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio
from ismip6_regions import BASIN_NAMES, SHORT_LABELS

START_TOL = 5.0  # yr; drop members whose series starts later (restart segments, e.g. SSP585_10/_11 at ~yr200)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="ENSEMBLES root (HPC path)")
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--include", default=r"^SSP585_\d+$")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--year-start", type=int, default=0)
    ap.add_argument("--year-end", type=int, default=300)
    ap.add_argument("--out", default="reports/figures/ice_flux_divergence_regional.png")
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

    # For each member: load thickness, xvelmean, yvelmean → compute flux divergence proxy
    # Full divergence needs mesh edge connectivity; we use a thickness-change proxy:
    # ∇·(uH) ≈ -dH/dt + SMB - BMB - calving (from continuity equation)
    # Since we have thickness time series, compute dH/dt as a proxy for dynamic thinning

    all_div = {}
    for name, member_dir in valid_members:
        try:
            H, years = sio.load_spatial_variable(member_dir, "thickness",
                                                  args.year_start, args.year_end)
            if years[0] > START_TOL:  # restart segment (SSP585_10/_11 ~yr200); dropping avoids
                # np.interp back-extrapolating a flat fake value into yr0-200.
                print(f"  {name}: DROPPED restart segment (starts at yr {years[0]:.0f})")
                continue
            # dH/dt as proxy for flux divergence (ignoring surface/basal mass balance)
            nYears = H.shape[0]
            dhdt = np.zeros_like(H)
            for t in range(1, nYears):
                dt = years[t] - years[t - 1]
                dhdt[t] = (H[t] - H[t - 1]) / dt
            dhdt[0] = dhdt[1]

            regional = sio.aggregate_by_region(dhdt, region_mask)
            all_div[name] = (years, regional)
            print(f"  {name}: {len(years)} years")
        except Exception as e:
            print(f"  {name}: SKIPPED ({e})")

    if not all_div:
        return

    ref_years = list(all_div.values())[0][0]
    nreg = list(all_div.values())[0][1].shape[1]
    stacked = []
    for name, (yrs, reg) in all_div.items():
        interp = np.column_stack([np.interp(ref_years, yrs, reg[:, r]) for r in range(nreg)])
        stacked.append(interp)
    arr = np.stack(stacked, axis=0)

    fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=True)
    axes_flat = axes.flatten()

    for r in range(nreg):
        ax = axes_flat[r]
        name = BASIN_NAMES[r]
        lbl = SHORT_LABELS.get(name, name)
        ens_mean = np.nanmean(arr[:, :, r], axis=0)
        ens_std = np.nanstd(arr[:, :, r], axis=0)
        ax.plot(ref_years, ens_mean, "k-", lw=1.2, label="mean dH/dt")
        ax.fill_between(ref_years, ens_mean - ens_std, ens_mean + ens_std,
                        color="teal", alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.set_ylabel("m/yr")
        ax.tick_params(labelsize=7)

    for r in range(nreg, len(axes_flat)):
        axes_flat[r].set_visible(False)

    axes_flat[12].set_xlabel("Year")
    axes_flat[13].set_xlabel("Year")
    fig.suptitle(f"{args.ensemble}: Thickness Change Rate (dH/dt proxy for flux divergence)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {args.out}")


if __name__ == "__main__":
    main()
