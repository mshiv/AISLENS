#!/usr/bin/env python3
"""
fig_thickness_dhdt_maps.py — Spatial maps of thickness change rate at horizons.

(A) 3×3 grid: yr100/200/300 × (mean, spread, member overlay).
(B) Difference maps: varScaled10x − SSP585. Requires HPC (output_state files).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="ENSEMBLES root (HPC path)")
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--include", default=r"^SSP585_\d+$")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--year-start", type=int, default=0)
    ap.add_argument("--year-end", type=int, default=300)
    ap.add_argument("--out", default="reports/figures/dhdt_maps.png")
    args = ap.parse_args()

    from ensemble_io import discover_members

    mesh_path, mask_path = sio.default_paths() if args.mesh is None else (args.mesh, args.mask)
    x, y = sio.load_mesh_coords(mesh_path)
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

    horizons = [100, 200, 300]
    all_thickness = {}
    for name, member_dir in valid_members:
        try:
            data, years = sio.load_spatial_variable(member_dir, "thickness",
                                                     args.year_start, args.year_end)
            all_thickness[name] = (years, data)
            print(f"  {name}: {len(years)} years, {data.shape[1]} cells")
        except Exception as e:
            print(f"  {name}: SKIPPED ({e})")

    if not all_thickness:
        return

    ref_years = list(all_thickness.values())[0][0]
    nCells = list(all_thickness.values())[0][1].shape[1]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    for col, hr in enumerate(horizons):
        idx = np.argmin(np.abs(ref_years - hr))
        if idx == 0:
            continue
        # dH/dt = (H[t] - H[t-1]) / dt
        dt = ref_years[idx] - ref_years[idx - 1]

        dhdt_members = []
        for name, (yrs, data) in all_thickness.items():
            dhdt = (data[idx] - data[idx - 1]) / dt  # m/yr
            dhdt_members.append(dhdt)
        dhdt_arr = np.stack(dhdt_members, axis=0)  # (member, nCells)

        ens_mean = np.nanmean(dhdt_arr, axis=0)
        ens_std = np.nanstd(dhdt_arr, axis=0)

        # Downsample for plotting (every 4th cell for speed)
        step = 4
        x_ds = x[::step] / 1e3  # km
        y_ds = y[::step] / 1e3
        mean_ds = ens_mean[::step]
        std_ds = ens_std[::step]

        # Panel 1: ensemble mean dH/dt
        ax = axes[0, col]
        norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
        sc = ax.scatter(x_ds, y_ds, c=mean_ds, cmap="RdBu_r", norm=norm,
                        s=0.5, edgecolors="none")
        ax.set_title(f"yr{hr} — Ensemble mean dH/dt", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="m/yr")

        # Panel 2: ensemble spread
        ax = axes[1, col]
        sc = ax.scatter(x_ds, y_ds, c=std_ds, cmap="YlOrRd",
                        s=0.5, edgecolors="none", vmin=0)
        ax.set_title(f"yr{hr} — Ensemble σ(dH/dt)", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="m/yr")

        # Panel 3: member overlay (first 3 members)
        ax = axes[2, col]
        colors = ["#C62828", "#1565C0", "#2E7D32"]
        for m in range(min(3, dhdt_arr.shape[0])):
            ax.scatter(x_ds, y_ds, c=dhdt_arr[m, ::step], cmap="RdBu_r", norm=norm,
                       s=0.3, edgecolors="none", alpha=0.5, label=f"m{m}")
        ax.set_title(f"yr{hr} — Member overlay", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, markerscale=10)

    for ax in axes.flat:
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

    fig.suptitle(f"{args.ensemble}: Thickness Change Rate (dH/dt) Maps",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {args.out}")


if __name__ == "__main__":
    main()
