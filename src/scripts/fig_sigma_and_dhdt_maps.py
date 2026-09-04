#!/usr/bin/env python3
"""
fig_sigma_and_dhdt_maps.py -- the two continental/sector map figures, with panel letters.

  --figure sigma   F17: ensemble standard deviation of ice thickness in one sector, for
                   every experiment at two years, on a shared logarithmic colour scale.
                   Cells below --mask-frac of the domain maximum are left blank; without
                   that mask the perimeter signal is invisible against the plateau, which
                   carries no measurable spread in any experiment.

  --figure dhdt    F23: ensemble-mean rate of thickness change over each integration,
                   one continental panel per experiment on a shared diverging scale.

Requested years are never replaced by an earlier diagnostic. Generate the matching
ensemble-statistics file first; this prevents a year-150 field from appearing in a
year-300 panel.

Usage
    python3 fig_sigma_and_dhdt_maps.py --figure sigma --region amundsen
    python3 fig_sigma_and_dhdt_maps.py --figure dhdt
"""
from __future__ import annotations
import os, argparse, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("amaps", os.path.join(_d, "fig_amundsen_maps.py"))
A = importlib.util.module_from_spec(_s); _s.loader.exec_module(A)

ENS = ["CTRL", "SSP126", "SSP585", "SSP585_varScaled10x", "SSP585-3X"]
PRETTY = {"SSP585_varScaled10x": "SSP5-8.5, 10$\\times$ variability",
          "SSP585-3X": "SSP5-8.5, 3$\\times$ melt trend",
          "SSP585": "SSP5-8.5", "SSP126": "SSP1-2.6", "CTRL": "control"}
SIGMA_COLUMN_LABELS = {
    "CTRL": "Control",
    "SSP126": "SSP1-2.6",
    "SSP585": "SSP5-8.5",
    "SSP585_varScaled10x": "10$\\times$ variability",
    "SSP585-3X": "3$\\times$ melt trend",
}
RHO_I, RHO_O = 910.0, 1028.0

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "savefig.dpi": 300,
})


def resolve(ens, year):
    """Return only the diagnostic for the requested year."""
    f = A.stats_file(ens, year)
    if f:
        return f, year
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=["sigma", "dhdt"], default="sigma")
    ap.add_argument("--region", default="amundsen")
    ap.add_argument("--years", default="2150,2300")
    ap.add_argument("--mask-frac", type=float, default=0.02)
    ap.add_argument("--out-dir", default="reports/dissertation/figures/tierA")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    years = [int(y) for y in a.years.split(",")]

    if a.figure == "sigma":
        shelves, labels = A.REGIONS[a.region]
        x, y, inbox, box, centres = A.load_domain(shelves=shelves, labels=labels)
        missing = [(e, Y) for e in ENS for Y in years if resolve(e, Y)[0] is None]
        if missing:
            details = ", ".join(f"{e}_{Y}.nc" for e, Y in missing)
            raise SystemExit(f"Missing requested ensemble statistics: {details}")
        # one shared colour scale across every panel, so panels are comparable
        vals = []
        for e in ENS:
            for Y in years:
                f, _ = resolve(e, Y)
                if f:
                    v = np.where(inbox, A.v(f, "thickness_std"), np.nan)
                    vals.append(v[np.isfinite(v)])
        allv = np.concatenate(vals)
        vmax = float(np.percentile(allv, 99.8))
        vmin = a.mask_frac * vmax

        # Two year rows by five experiment columns, intended for a landscape thesis page.
        # Column and row headings are shared rather than repeated inside every map.
        fig, axg = plt.subplots(len(years), len(ENS),
                                figsize=(16.0, 7.1), squeeze=False)
        k = 0
        for r, Y in enumerate(years):
            for c, e in enumerate(ENS):
                ax = axg[r][c]
                f, ay = resolve(e, Y)
                if not f:
                    ax.set_visible(False); k += 1; continue
                v = np.where(inbox, A.v(f, "thickness_std"), np.nan)
                # median over every ice-covered cell in the sector, not only the cells drawn:
                # conditioning on the mask would inflate it
                # median over cells that carry any spread at all; including the ocean
                # and the frozen interior would report the size of the mask, not the signal
                med = float(np.nanmedian(v[np.isfinite(v) & (v > 0)]))
                sc = A.paint(ax, x, y, np.where(v >= vmin, v, np.nan), box, "magma",
                             norm=LogNorm(vmin=vmin, vmax=vmax),
                             centres=centres if (r == 0 and c == 0) else None)
                ax.text(0.02, 0.98, f"({'abcdefghij'[k]})", transform=ax.transAxes,
                        ha="left", va="top", fontsize=14, fontweight="bold",
                        bbox=dict(fc="white", ec="none", alpha=.75, pad=1.2))
                ax.text(0.02, 0.03, f"median $\\sigma$ {med:.2f} m",
                        transform=ax.transAxes, ha="left", va="bottom", fontsize=10.0,
                        bbox=dict(fc="white", ec="none", alpha=.80, pad=1.2))
                if r == 0:
                    ax.set_title(SIGMA_COLUMN_LABELS[e], fontsize=13,
                                 fontweight="bold", pad=4)
                print(f"  {e:22s} yr {ay-2000}: median sigma {med:.3f} m")
                k += 1
        # Shared row labels keep the maps free of repeated headings.
        for r, Y in enumerate(years):
            ymid = 0.72 if r == 0 else 0.285
            fig.text(0.018, ymid, f"model year {Y-2000}", rotation=90,
                     ha="center", va="center", fontsize=13, fontweight="bold")
        fig.subplots_adjust(left=.045, right=.925, top=.925, bottom=.035,
                            wspace=.025, hspace=.10)
        cbar = fig.colorbar(sc, ax=axg, fraction=.020, pad=.012, aspect=34)
        cbar.set_label("$\\sigma$(ice thickness) (m)", fontsize=13)
        cbar.ax.tick_params(labelsize=11)
        o = f"{a.out_dir}/F17_{a.region}_sigma_by_ensemble.png"
    else:
        x = A.v(A.MESH, "xCell"); y = A.v(A.MESH, "yCell")
        box = (x.min(), x.max(), y.min(), y.max())
        Y = years[-1]
        fig, axg = plt.subplots(1, len(ENS), figsize=(4.0 * len(ENS), 4.6), squeeze=False)
        lim = 1.5
        for c, e in enumerate(ENS):
            ax = axg[0][c]
            f, ay = resolve(e, Y)
            if not f:
                ax.set_visible(False); continue
            d = A.v(f, "dhdt_mean")
            ok = np.isfinite(d)
            med = float(np.median(d[ok])); thin = float(np.mean(d[ok] < 0))
            sc = A.paint(ax, x, y, d, box, "RdBu",
                         norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
            ax.set_title(f"({'abcde'[c]}) {PRETTY.get(e, e)}\n"
                         f"to yr {ay-2000};  median {med:+.3f} m yr$^{{-1}}$;  "
                         f"{100*thin:.0f}% thinning", fontsize=8.8, loc="left")
            print(f"  {e:22s} to yr {ay-2000}: median {med:+.4f}, {100*thin:.0f}% thinning")
        fig.colorbar(sc, ax=axg, shrink=.7, label="d$h$/d$t$ (m yr$^{-1}$)")
        fig.suptitle("Ensemble-mean rate of ice-thickness change over each integration\n"
                     "blue denotes thickening, red thinning; shared scale saturating at "
                     "$\\pm$1.5 m yr$^{-1}$", fontsize=12.5)
        o = f"{a.out_dir}/F23_dhdt_all_ensembles.png"

    fig.savefig(o, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(o)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", o)


if __name__ == "__main__":
    main()
