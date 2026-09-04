#!/usr/bin/env python3
"""
fig_drift_grid.py -- amplitude-dependent mean displacement, region x year.

WHAT THIS QUANTITY IS, PRECISELY
    Robel et al. (2024) define noise-induced drift for dx/dt = f(x) + g(x) eta as

        D(A) = E[x(t) | noise amplitude A] - x_det(t)

    i.e. the stochastic ensemble mean minus a DETERMINISTIC twin. This figure does
    NOT show that. It shows

        D(10A) - D(A) = mean(10x) - mean(1x)

    the difference between two noise amplitudes, which is the INCREMENT in drift
    rather than the drift itself. A non-zero value proves rectification exists and
    depends on amplitude -- with no rectification D would be zero at every amplitude
    and the difference would vanish -- but it does not recover D(A) at realistic
    amplitude, which still requires the matched deterministic run.

    Label it "amplitude-dependent mean displacement", not "noise-induced drift".

TWO VARIABLES
    vaf    only ice above flotation, so this is the sea-level relevant quantity and
           the one comparable with Robel's mass/GMSL numbers
    total  full thickness including floating ice; not a sea-level quantity, but it
           shows the MEDIATOR -- shelf thinning is the pathway by which melt
           variability reaches the grounding line

Usage
    python3 fig_drift_grid.py --variable vaf
    python3 fig_drift_grid.py --variable total --regions fris,amundsen,wilkes
"""
from __future__ import annotations
import os, csv, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "amaps", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_amundsen_maps.py"))
A = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A)

RHO_I, RHO_O = 910.0, 1028.0
PRETTY = {"amundsen": "Amundsen", "fris": "Filchner-Ronne", "wilkes": "Wilkes"}

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "savefig.dpi": 300,
})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", default="amundsen,fris,wilkes")
    ap.add_argument("--years", default="2100,2200,2300")
    ap.add_argument("--variable", default="vaf", choices=["vaf", "total"])
    ap.add_argument("--out-dir", default="reports/dissertation/figures/tierB")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    regions = [r.strip() for r in a.regions.split(",")]
    years = [int(y) for y in a.years.split(",")]

    bed = A.v(A.MESH, "bedTopography")
    area = A.v(A.MESH, "areaCell")
    hf = (RHO_O / RHO_I) * np.maximum(0.0, -bed)

    fig, axg = plt.subplots(len(regions), len(years),
                            figsize=(5.4 * len(years), 5.0 * len(regions)), squeeze=False)
    print(f"{'region':<14}{'year':>6}{'area-wtd mean':>15}{'net (Gt)':>12}{'p98 |val|':>11}")
    for ri, reg in enumerate(regions):
        shelves, labels = A.REGIONS[reg]
        x, y, inbox, box, centres = A.load_domain(shelves=shelves, labels=labels)
        sub = lambda arr: np.where(inbox, arr, np.nan)
        for ci, YR in enumerate(years):
            ax = axg[ri][ci]
            f5, f10 = A.stats_file("SSP585", YR), A.stats_file("SSP585_varScaled10x", YR)
            if not (f5 and f10):
                ax.set_visible(False)
                print(f"{reg:<14}{YR:>6}   no data"); continue
            h1 = sub(A.v(f5, "thickness_mean")); h2 = sub(A.v(f10, "thickness_mean"))
            if a.variable == "vaf":
                d = np.maximum(0.0, h2 - hf) - np.maximum(0.0, h1 - hf)
                lab = "$\\Delta$VAF (m)"
            else:
                d = h2 - h1
                lab = "$\\Delta$ thickness (m)"
            ok = np.isfinite(d)
            fin = d[ok]; w = area[ok]
            # area-weighted, and reported as an integrated total rather than a cell
            # fraction: the mesh spans 4-20 km, so a fraction of CELLS over-weights the
            # refined margin, and a fraction of AREA still says nothing about magnitude
            awm = float(np.sum(d[ok] * w) / np.sum(w))
            net = float(np.sum(d[ok] * w)) * RHO_I / 1e12      # Gt of ice
            lim = np.percentile(np.abs(fin), 98) or 1.0
            # grounding line from the 1x mean at THIS year, not a fixed reference
            g_ = np.zeros(x.size, bool)
            hm = A.v(f5, "thickness_mean")
            ok = np.isfinite(hm) & (hm > 1.0)
            g_[ok] = np.abs(hm[ok] - hf[ok]) < 40.0
            sc = A.paint(ax, x, y, d, box, "RdBu",
                         norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
                         centres=centres if ci == 0 else None, gl=g_)
            plt.colorbar(sc, ax=ax, shrink=.78, label=lab)
            letter = f"({'abcdefghijkl'[ri * len(years) + ci]})"
            ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha="left", va="top",
                    fontsize=14, fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=.75, pad=1.2))
            ax.text(0.98, 0.98, f"{PRETTY.get(reg, reg)}, yr {YR-2000}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=12,
                    bbox=dict(fc="white", ec="none", alpha=.75, pad=1.2))
            ax.text(0.02, 0.03, f"net {net:+,.0f} Gt; mean {awm:+.2f} m",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5,
                    bbox=dict(fc="white", ec="none", alpha=.80, pad=1.2))
            print(f"{reg:<14}{YR:>6}{awm:15.3f}{net:12.0f}{lim:11.2f}")

    fig.tight_layout()
    o = f"{a.out_dir}/F20_drift_grid_{a.variable}.png"
    fig.savefig(o, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(o)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
