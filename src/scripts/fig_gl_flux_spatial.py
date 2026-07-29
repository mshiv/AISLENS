#!/usr/bin/env python3
"""
fig_gl_flux_spatial.py — Spatial pattern of grounding line flux.

Maps where ice is discharging across the grounding line, showing which
GL sectors are most variable across ensemble members.

Requires HPC: output_state files not available locally.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

START_TOL = 5.0  # yr; drop members whose series starts later (restart segments, e.g. SSP585_10/_11 at ~yr200)
HORIZON_TOL = 3.0  # yr; require the nearest available year to be within this of the target horizon


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="ENSEMBLES root (HPC path)")
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--include", default=r"^SSP585_\d+$")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--year-start", type=int, default=0)
    ap.add_argument("--year-end", type=int, default=300)
    ap.add_argument("--out", default="reports/figures/gl_flux_spatial.png")
    args = ap.parse_args()

    from ensemble_io import discover_members

    mesh_path, mask_path = sio.default_paths() if args.mesh is None else (args.mesh, args.mask)
    x, y = sio.load_mesh_coords(mesh_path)

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
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, hr in enumerate(horizons):
        speed_members = []
        for name, member_dir in valid_members:
            try:
                data, years = sio.load_spatial_variable(member_dir, "surfaceSpeed",
                                                         args.year_start, args.year_end)
                if years[0] > START_TOL:  # restart segment (SSP585_10/_11 ~yr200) — exclude entirely
                    continue                # so every horizon's spread uses the same member set
                idx = np.argmin(np.abs(years - hr))
                if abs(years[idx] - hr) > HORIZON_TOL:  # member doesn't actually reach this horizon
                    continue                            # don't substitute a far-off year
                speed_members.append(data[idx])
            except Exception:
                continue

        if not speed_members:
            continue
        speed_arr = np.stack(speed_members, axis=0)

        step = 4
        x_ds = x[::step] / 1e3
        y_ds = y[::step] / 1e3

        # Top: ensemble mean speed (log scale)
        ax = axes[0, col]
        mean_speed = np.nanmean(speed_arr, axis=0)[::step]
        mean_speed = np.maximum(mean_speed, 0.01)  # avoid log(0)
        sc = ax.scatter(x_ds, y_ds, c=np.log10(mean_speed), cmap="viridis",
                        s=0.5, edgecolors="none")
        ax.set_title(f"yr{hr} — Mean log₁₀(speed)", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="log₁₀(m/yr)")

        # Bottom: ensemble spread in speed
        ax = axes[1, col]
        std_speed = np.nanstd(speed_arr, axis=0)[::step]
        sc = ax.scatter(x_ds, y_ds, c=std_speed, cmap="YlOrRd",
                        s=0.5, edgecolors="none", vmin=0)
        ax.set_title(f"yr{hr} — σ(speed)", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="m/yr")

    for ax in axes.flat:
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

    fig.suptitle(f"{args.ensemble}: Ice Velocity & Discharge Pattern",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {args.out}")


if __name__ == "__main__":
    main()
