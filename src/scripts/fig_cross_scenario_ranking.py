#!/usr/bin/env python3
"""
fig_cross_scenario_ranking.py — rank members by VAF loss in SSP585, then check if
the same members rank similarly in SSP126 and SSP585_varScaled10x.  If rankings are
correlated - initial conditions dominate (deterministic).  If uncorrelated -
internal variability determines which member is "worst."

Panel (a): Heatmap of member ranks across scenarios.
Panel (b): SSP585 rank vs SSP126 rank scatter + Spearman r.
Panel (c): SSP585 rank vs varScaled10x rank scatter + Spearman r.

Author: Shivaprakash Muruganandham (2026-07-22)
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=eio.default_ensembles_root())
    p.add_argument("--min-years", type=float, default=50.0)
    p.add_argument("--out-dir", default="reports/figures/presentations/20260722-IceT")
    return p.parse_args()


def load_vaf_data(ensemble_dir, include, min_years=50):
    """Load ensemble VAF and return (sle_values, member_indices, member_names)."""
    ds = eio.load_ensemble_globalstats(
        ensemble_dir,
        variables=["volumeAboveFloatation", "daysSinceStart"],
        include=include, min_years=min_years, align="union",
    )
    vaf = ds["volumeAboveFloatation"]
    sle = xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, reference="first"), vaf)
    names = list(ds["member"].values)
    indices = [int(n.split("_")[-1]) for n in names]
    return sle.isel(year=-1).values, indices, names


def rank_members(sle_values):
    return np.argsort(np.argsort(sle_values))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfgs = [
        ("SSP585",            r"^SSP585_\d+$"),
        ("SSP126",            r"^SSP126_\d+$"),
        ("varScaled10x",      r"^SSP585_\d+$"),
    ]
    ensemble_dirs = {
        "SSP585":       os.path.join(args.root, "SSP585"),
        "SSP126":       os.path.join(args.root, "SSP126"),
        "varScaled10x": os.path.join(args.root, "SSP585_varScaled10x"),
    }

    # ---- collect final-year SLE and ranks ----
    all_sle = {}
    all_ranks = {}
    member_indices = {}
    member_names = {}
    for label, inc in cfgs:
        sle, indices, names = load_vaf_data(ensemble_dirs[label], inc, args.min_years)
        all_sle[label] = sle
        all_ranks[label] = rank_members(sle)
        member_indices[label] = indices
        member_names[label] = names
        print(f"  {label:16s}  n={len(sle):2d}  final-SLE range [{sle.min():+.2f}, {sle.max():+.2f}] mm")

    # ---- align members by numeric index (e.g. SSP585_03 ↔ SSP126_03) ----
    common_idx = set(member_indices["SSP585"])
    for label in ["SSP126", "varScaled10x"]:
        common_idx &= set(member_indices[label])
    common_idx = sorted(common_idx)
    print(f"\n  Common member indices across all 3 scenarios: {common_idx}")

    if len(common_idx) < 3:
        sys.exit(f"Fewer than 3 common member indices ({len(common_idx)} found).")

    # Build arrays aligned to common member indices
    labels = ["SSP585", "SSP126", "varScaled10x"]
    rank_matrix = np.full((len(common_idx), len(labels)), np.nan)
    for j, label in enumerate(labels):
        idx_list = member_indices[label]
        for i, ci in enumerate(common_idx):
            if ci in idx_list:
                pos = idx_list.index(ci)
                rank_matrix[i, j] = all_ranks[label][pos]

    n_members = len(common_idx)
    common_labels = [f"_{ci:02d}" for ci in common_idx]  # for y-axis display

    # ---- Panel (a): heatmap of ranks ----
    fig, axes = plt.subplots(1, 3, figsize=(13, max(4, 0.35 * n_members + 1)),
                             gridspec_kw={"width_ratios": [3, 1, 1], "wspace": 0.35})

    ax_hm = axes[0]
    im = ax_hm.imshow(rank_matrix, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=n_members - 1)
    ax_hm.set_xticks(range(len(labels)))
    ax_hm.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax_hm.set_yticks(range(n_members))
    ax_hm.set_yticklabels(common_labels, fontsize=8)
    for i in range(n_members):
        for j in range(len(labels)):
            r = rank_matrix[i, j]
            if np.isfinite(r):
                ax_hm.text(j, i, f"{int(r)}", ha="center", va="center",
                           fontsize=7, color="white" if r > n_members * 0.5 else "black")
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04, label="rank (0=smallest loss)")
    ax_hm.set_title("(a) Member ranks across scenarios\n(row = member, ranked by SSP585 VAF loss)")

    # ---- Panels (b) and (c): scatter of ranks ----
    pairs = [("SSP585", "SSP126", "(b)"),
             ("SSP585", "varScaled10x", "(c)")]
    for k, (xlab, ylab, tag) in enumerate(pairs):
        ax = axes[k + 1]
        jx = labels.index(xlab)
        jy = labels.index(ylab)
        rx = rank_matrix[:, jx]
        ry = rank_matrix[:, jy]
        ok = np.isfinite(rx) & np.isfinite(ry)
        r, p = spearmanr(rx[ok], ry[ok])
        ax.scatter(rx[ok], ry[ok], s=36, c="C3", edgecolors="k", linewidth=0.5, zorder=3)
        lims = [-0.5, n_members - 0.5]
        ax.plot(lims, lims, "k--", lw=0.7, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"{xlab} rank", fontsize=9)
        ax.set_ylabel(f"{ylab} rank", fontsize=9)
        ax.set_aspect("equal")
        ax.set_title(f"{tag} {xlab} vs {ylab}\nSpearman r={r:.2f} (p={p:.2g})", fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("Cross-scenario member ranking: do initial conditions determine which member loses the most?",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "cross_scenario_ranking.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure -> {out}")

    # ---- print summary ----
    for xlab, ylab, _tag in pairs:
        jx = labels.index(xlab); jy = labels.index(ylab)
        ok = np.isfinite(rank_matrix[:, jx]) & np.isfinite(rank_matrix[:, jy])
        r, p = spearmanr(rank_matrix[ok, jx], rank_matrix[ok, jy])
        verdict = "INITIAL CONDITIONS DOMINATE" if (r > 0.5 and p < 0.05) else "VARIABILITY DOMINATES"
        print(f"  {xlab:16s} vs {ylab:16s}  Spearman r={r:+.3f}  p={p:.2e}  -> {verdict}")


if __name__ == "__main__":
    main()
