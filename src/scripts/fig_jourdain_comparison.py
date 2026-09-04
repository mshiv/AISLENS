#!/usr/bin/env python3
"""
fig_jourdain_comparison.py -- where the AISLENS applied melt sits among the ISMIP6
ocean-forcing ensemble of Jourdain et al. (2020).

Jourdain et al. report sub-shelf melt for four merged sectors under RCP8.5, from six
CMIP5 models passed through two calibrations of the nonlocal quadratic parameterization
(MeanAnt and PIGL) plus three cavity-resolving FESOM simulations. Their sector
definitions are preserved here and mapped onto the 16-basin ISMIP6 mask.

Two axes are shown separately, because they answer different questions:

  MAGNITUDE    mean melt over 2080-2100. The raw series contains the scenario trend, so
               this measures how strongly the sector responds, not how variable it is.
  VARIABILITY  square root of the Welch power spectral density integrated over periods
               of 2--30 years. The same calculation is applied to every trajectory;
               AISLENS values are means of the memberwise estimates.

AISLENS is an applied, spatially resolved melt field rather than an output of the same
parameterization, so it is comparable to the FESOM series in kind and to the
parameterized cases only as an envelope.

Usage
    python3 fig_jourdain_comparison.py --ensemble SSP585 --out-dir reports/dissertation/figures/tierA
"""
from __future__ import annotations
import os, glob, argparse
import numpy as np
import pandas as pd
from netCDF4 import Dataset, chartostring
from scipy.signal import welch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JDIR = os.path.join(REPO, "data/jourdain2020")
Y0, Y1 = 2000, 2100
SNAP0, SNAP1 = 2080, 2100

# Jourdain sector -> ISMIP6 basin indices in the 16-basin mask
SECTORS = {
    "Ronne-Filchner":       ("panel_a_Ronne-Filchner.csv",       [14]),
    "Pine Island-Thwaites": ("panel_b_Pine-Island_Thwaites.csv", [9]),
    "Cook-Ninnis":          ("panel_c_Cook_Ninnis.csv",          [5, 6]),
    "Totten-Moscow U.":     ("panel_d_Totten_MoscowUniv.csv",    [4]),
}
C_MEAN, C_PIGL, C_FESOM = "#9E9E9E", "#B0A3C4", "#0072B2"
C_AIS, C_AIS10 = "#C1121F", "#E69F00"


def detrended_std(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size < 5:
        return np.nan
    t = np.arange(v.size)
    return float(np.std(v - np.polyval(np.polyfit(t, v, 1), t)))


def band_std(v, period_min=2.0, period_max=30.0):
    """Band-limited standard deviation from an annual series using Welch's PSD.

    A Hann-windowed 64-year segment and linear detrending limit leakage from the
    forced trajectory.  The same operation is applied to every Jourdain series and
    every AISLENS member.
    """
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 16:
        return np.nan
    filled = np.interp(np.arange(v.size), np.flatnonzero(ok), v[ok])
    nperseg = min(64, filled.size)
    freq, psd = welch(filled, fs=1.0, window="hann", nperseg=nperseg,
                      noverlap=nperseg // 2, detrend="linear", scaling="density")
    use = (freq >= 1.0 / period_max) & (freq <= 1.0 / period_min)
    if use.sum() < 2:
        return np.nan
    return float(np.sqrt(np.trapezoid(psd[use], freq[use])))


def jourdain(csv):
    """-> {column: (years, values)} restricted to Y0..Y1, split by family."""
    df = pd.read_csv(os.path.join(JDIR, csv))
    yr = df["year"].to_numpy(float)
    k = (yr >= Y0) & (yr <= Y1)
    fam = {"MeanAnt": [], "PIGL": [], "FESOM": []}
    for c in df.columns[1:]:
        v = df[c].to_numpy(float)[k]
        if np.isfinite(v).sum() < 20:
            continue
        fam[c.split("_")[0]].append((c, yr[k], v))
    return fam


def aislens(ens, basins, member_glob=None):
    """Annual sector-mean melt per member, m ice/yr. Basins are averaged first so the
    series matches the sector aggregation Jourdain et al. applied."""
    yrs = np.arange(Y0, Y1 + 1)
    out = []
    if member_glob is None:
        member_glob = f"{ens}_0[0-9]"
    for f in sorted(glob.glob(os.path.join(
            REPO, f"data/MALI/diagnostics/ENSEMBLES/{ens}/{member_glob}/regionalStats.nc"))):
        d = Dataset(f)
        m = np.asarray(d["regionalAvgSubshelfMelt"][:], float)
        yi = np.array([int(str(s).strip()[:4]) if str(s).strip()[:4].isdigit() else -1
                       for s in chartostring(d["xtime"][:])])
        d.close()
        reg = np.nanmean(m[:, basins], axis=1)
        # annual means, not point samples: globalStats/regionalStats are written every
        # ~118 days and the seasonal cycle would otherwise be sampled at one phase
        s = np.array([np.nanmean(reg[yi == y]) if (yi == y).any() else np.nan for y in yrs])
        if np.isfinite(s).sum() > 50:
            out.append(s)
    return yrs, np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", default="SSP585")
    ap.add_argument("--out-dir", default="reports/dissertation/figures/tierA")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    fig = plt.figure(figsize=(15.0, 8.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.26)

    snap, sig = {}, {}
    sensitivity_rows = []
    for j, (name, (csv, basins)) in enumerate(SECTORS.items()):
        ax = fig.add_subplot(gs[0, j])
        fam = jourdain(csv)
        for key, col, lw, ls in (("MeanAnt", C_MEAN, .9, "-"),
                                 ("PIGL", C_PIGL, .9, "--"),
                                 ("FESOM", C_FESOM, 1.3, "-")):
            for _, yy, vv in fam[key]:
                ax.plot(yy, vv, color=col, lw=lw, ls=ls, alpha=.85, zorder=2)
        yrs, M = aislens(a.ensemble, basins)
        yrs10, M10 = aislens("SSP585_varScaled10x", basins,
                             member_glob="SSP585_[0-9][0-9]")
        mu = np.nanmean(M, 0)
        ax.fill_between(yrs, np.nanmin(M, 0), np.nanmax(M, 0),
                        color=C_AIS, alpha=.22, lw=0, zorder=3)
        ax.plot(yrs, mu, color=C_AIS, lw=2.1, zorder=4)
        ax.axvspan(SNAP0, SNAP1, color="k", alpha=.055, lw=0, zorder=1)
        ax.set_title(f"({'abcd'[j]}) {name}", fontsize=10.5, loc="left")
        ax.set_xlim(Y0, Y1); ax.grid(alpha=.28)
        ax.set_xlabel("year")
        if j == 0:
            ax.set_ylabel("sector-mean melt (m ice yr$^{-1}$)")

        k = (yrs >= SNAP0) & (yrs <= SNAP1)
        k10 = (yrs10 >= SNAP0) & (yrs10 <= SNAP1)
        snap[name] = {"AISLENS": float(np.nanmean(mu[k])),
                      "AISLENS10": float(np.nanmean(np.nanmean(M10, axis=0)[k10])),
                      **{f: {c: float(np.nanmean(v[(y >= SNAP0) & (y <= SNAP1)]))
                             for c, y, v in fam[f]} for f in fam}}
        band_results = {}
        for band_name, pmin, pmax in (("2-30", 2.0, 30.0),
                                      ("2-8", 2.0, 8.0),
                                      ("8-30", 8.0, 30.0)):
            bd = {"AISLENS": float(np.nanmean([band_std(m, pmin, pmax) for m in M])),
                  "AISLENS10": float(np.nanmean([band_std(m, pmin, pmax) for m in M10])),
                  **{f: {c: band_std(v, pmin, pmax) for c, y, v in fam[f]} for f in fam}}
            comp = [x for f in ("MeanAnt", "PIGL", "FESOM") for x in bd[f].values()]
            sensitivity_rows.append({
                "sector": name, "period_band_years": band_name,
                "aislens_1x_sigma_m_per_yr": bd["AISLENS"],
                "aislens_10x_sigma_m_per_yr": bd["AISLENS10"],
                "realized_ratio": bd["AISLENS10"] / bd["AISLENS"],
                "aislens_1x_rank_among_16": 1 + int(np.sum(np.asarray(comp) < bd["AISLENS"])),
                "aislens_10x_rank_among_16": 1 + int(np.sum(np.asarray(comp) < bd["AISLENS10"])),
                "jourdain_min_sigma_m_per_yr": float(np.nanmin(comp)),
                "jourdain_max_sigma_m_per_yr": float(np.nanmax(comp)),
            })
            band_results[band_name] = bd
        sig[name] = band_results["2-30"]

    # ---- panel (e): late-century magnitude within the Jourdain ensemble
    for col, (D, lab, ttl) in enumerate(((
            snap, f"mean melt {SNAP0}-{SNAP1} (m ice yr$^{{-1}}$)",
            f"(e) magnitude: sector-mean melt averaged over {SNAP0}-{SNAP1}"),)):
        ax = fig.add_subplot(gs[1, :2])
        for i, name in enumerate(SECTORS):
            d = D[name]
            for key, c, mk in (("MeanAnt", C_MEAN, "o"), ("PIGL", C_PIGL, "s"),
                               ("FESOM", C_FESOM, "^")):
                v = list(d[key].values())
                ax.scatter(v, np.full(len(v), i) + {"MeanAnt": .16, "PIGL": 0,
                                                    "FESOM": -.16}[key],
                           s=27, c=c, marker=mk, alpha=.9, zorder=3, edgecolors="none")
            ax.scatter([d["AISLENS"]], [i - .055], s=155, marker="*", c=C_AIS, zorder=5,
                       edgecolors="k", linewidths=.5)
            ax.scatter([d["AISLENS10"]], [i + .055], s=72, marker="X", c=C_AIS10,
                       zorder=6, edgecolors="k", linewidths=.4)
            allv = [x for key in ("MeanAnt", "PIGL", "FESOM") for x in d[key].values()]
            rank = 1 + int(np.sum(np.array(allv) < d["AISLENS"]))
            rank10 = 1 + int(np.sum(np.array(allv) < d["AISLENS10"]))
            ax.plot([np.nanmin(allv), np.nanmax(allv)], [i, i], color="#CCCCCC",
                    lw=1.1, zorder=1)
            ax.annotate(f"{d['AISLENS']:.2f} ({rank}/{len(allv) + 1})",
                        (d["AISLENS"], i - .055),
                        textcoords="offset points", xytext=(0, 10), ha="center",
                        fontsize=8, color=C_AIS)
            ax.annotate(f"{d['AISLENS10']:.2f} ({rank10}/{len(allv) + 1})",
                        (d["AISLENS10"], i + .055),
                        textcoords="offset points", xytext=(0, -15), ha="center",
                        fontsize=8, color="#B26A00")
            print(f"{ttl[1]}  {name:<22} AISLENS {d['AISLENS']:7.3f} "
                  f"rank {rank}/{len(allv)}; 10x {d['AISLENS10']:7.3f} "
                  f"rank {rank10}/{len(allv)}   "
                  f"comparison {np.nanmin(allv):6.3f}..{np.nanmax(allv):6.3f}   "
                  )
        ax.set_xscale("symlog", linthresh=0.1, linscale=0.45)
        ax.set_yticks(range(len(SECTORS)))
        # (f) sits immediately right of (e) and shares its categories; repeating the sector
        # names there only overlaps the neighbouring axes
        ax.set_yticklabels(list(SECTORS) if col == 0 else [], fontsize=9.5)
        ax.set_xlabel(lab); ax.set_title(ttl, fontsize=10.5, loc="left")
        ax.grid(axis="x", alpha=.28); ax.set_ylim(-.6, len(SECTORS) - .4)
        ax.invert_yaxis()

    # ---- panel (f): common 2--30-year band-limited metric for every series
    ax = fig.add_subplot(gs[1, 2:])
    for i, name in enumerate(SECTORS):
        d = sig[name]
        for key, c, mk in (("MeanAnt", C_MEAN, "o"), ("PIGL", C_PIGL, "s"),
                           ("FESOM", C_FESOM, "^")):
            vals = list(d[key].values())
            ax.scatter(vals, np.full(len(vals), i) + {"MeanAnt": .16, "PIGL": 0,
                                                       "FESOM": -.16}[key],
                       s=27, c=c, marker=mk, alpha=.9, zorder=3, edgecolors="none")
        v1 = d["AISLENS"]
        v10 = d["AISLENS10"]
        ratio = v10 / v1
        allv = [x for key in ("MeanAnt", "PIGL", "FESOM") for x in d[key].values()]
        rank = 1 + int(np.sum(np.array(allv) < v1))
        rank10 = 1 + int(np.sum(np.array(allv) < v10))
        ax.plot([np.nanmin(allv), np.nanmax(allv)], [i, i], color="#CCCCCC",
                lw=1.1, zorder=1)
        ax.scatter(v1, i - .055, s=155, marker="*", c=C_AIS, zorder=4,
                   edgecolors="k", linewidths=.5)
        ax.scatter(v10, i + .055, s=72, marker="X", c=C_AIS10, zorder=5,
                   edgecolors="k", linewidths=.4)
        ax.annotate(f"{v1:.2f} ({rank}/16)", (v1, i - .055),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8, color=C_AIS)
        ax.annotate(f"{v10:.2f} ({rank10}/16)", (v10, i + .055),
                    textcoords="offset points", xytext=(0, -15), ha="center",
                    fontsize=8, color="#B26A00")
        ax.annotate(f"{ratio:.1f}$\\times$", (np.sqrt(v1 * v10), i),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8.2, fontweight="bold", color="#555555")
        print(f"2-30 yr variability {name:<22} AISLENS {v1:7.3f} "
              f"rank {rank}/16; 10x {v10:7.3f} rank {rank10}/16; ratio {ratio:5.2f}")
    ax.set_xscale("log")
    ax.set_yticks(range(len(SECTORS)))
    # Panel (e) supplies the shared sector labels for the aligned rows.
    ax.set_yticklabels([])
    ax.set_xlabel("band-limited $\\sigma$, 2--30-year periods (m ice yr$^{-1}$)")
    ax.set_title("(f) interannual-to-decadal variability", fontsize=10.5, loc="left")
    ax.grid(axis="x", alpha=.28)
    ax.set_ylim(-.6, len(SECTORS) - .4)
    ax.invert_yaxis()

    fig.legend(handles=[
        Line2D([], [], color=C_AIS, lw=2.1, marker="*", markerfacecolor=C_AIS,
               markeredgecolor="k", markersize=10, label="AISLENS SSP5-8.5"),
        Line2D([], [], color=C_AIS, lw=7, alpha=.22, label="AISLENS member range"),
        Line2D([], [], color="none", marker="X", markerfacecolor=C_AIS10,
               markeredgecolor="k", markersize=8, label="AISLENS 10$\\times$ variability"),
        Line2D([], [], color=C_MEAN, lw=1.2, marker="o", ms=5, label="CMIP5, MeanAnt calibration"),
        Line2D([], [], color=C_PIGL, lw=1.2, ls="--", marker="s", ms=5, label="CMIP5, PIGL calibration"),
        Line2D([], [], color=C_FESOM, lw=1.5, marker="^", ms=5, label="FESOM, cavity-resolving"),
    ], loc="upper center", ncol=3, fontsize=8.8, frameon=False,
       bbox_to_anchor=(.5, 1.01))
    o = os.path.join(a.out_dir, "F25_jourdain_comparison.png")
    fig.savefig(o, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(o)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    table = os.path.join(a.out_dir, "F25_jourdain_variability_band_sensitivity.csv")
    pd.DataFrame(sensitivity_rows).to_csv(table, index=False)
    print(f"\nwrote {o}")
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
