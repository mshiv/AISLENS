#!/usr/bin/env python3
"""
fig_gating_multiband.py — per-basin spread vs forcing fraction per frequency band.

Extends Fig D: one scatter panel per band (seasonal/interannual/decadal/multidecadal)
and per ensemble. Tests if spread correlates with forcing in any single band.
"""
from __future__ import annotations
import os, sys, csv as _csv, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from connect_forcing_response import ensemble_region_sigma

BANDS = ["seasonal", "interannual", "decadal", "multidecadal"]


def load_bands(csv_path):
    names, frac = [], {b: [] for b in BANDS}
    with open(csv_path) as fh:
        for row in _csv.DictReader(fh):
            if not row["sector"].lower().startswith("ismip6 basin"):
                continue
            names.append(row["sector"].replace("ISMIP6 Basin ", "").strip())
            for b in BANDS:
                frac[b].append(float(row[b]))
    return names, {b: np.array(v) for b, v in frac.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--forcing-csv", default="reports/spectrum_percell_generated0.csv")
    ap.add_argument("--ensembles", default="SSP585,SSP585_varScaled10x")
    ap.add_argument("--members", default=r"^SSP585_\d+$")
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--out", default="reports/fig_gating_multiband.png")
    args = ap.parse_args()

    names, frac = load_bands(args.forcing_csv)
    ensembles = args.ensembles.split(",")
    fig, axs = plt.subplots(len(ensembles), len(BANDS),
                            figsize=(4*len(BANDS), 3.6*len(ensembles)), squeeze=False)
    for ei, ens in enumerate(ensembles):
        sig, years, used = ensemble_region_sigma(args.root, ens, args.members, [args.horizon])
        if sig is None:
            continue
        hy = list(sig)[-1]; s = sig[hy]
        for bi, b in enumerate(BANDS):
            ax = axs[ei][bi]
            x = 100 * frac[b]
            ok = np.isfinite(s) & np.isfinite(x)
            ax.scatter(x, s, c="C3", s=28, zorder=3)
            if ok.sum() >= 4 and np.std(s[ok]) > 0:
                pr, pp = pearsonr(x[ok], s[ok]); sr, _ = spearmanr(x[ok], s[ok])
                ax.set_title(f"{b}\nr={pr:+.2f} (p={pp:.2f}), ρ={sr:+.2f}", fontsize=9)
            for r, nm in enumerate(names):     # label only the big-spread basins
                if s[r] > 0.4 * np.nanmax(s):
                    ax.annotate(nm, (x[r], s[r]), fontsize=6, alpha=0.8)
            if bi == 0:
                ax.set_ylabel(f"{ens}\nσ ΔVAF (mm) @yr{int(hy)}", fontsize=8)
            ax.set_xlabel(f"forcing {b} fraction (%)", fontsize=8)
            ax.grid(alpha=0.2)
    fig.suptitle("Per-basin ΔVAF spread vs forcing fraction, by frequency band — seasonal null, "
                 "interannual/decadal +, multidecadal − (confounded with basin dynamics)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {args.out}")


if __name__ == "__main__":
    main()
