#!/usr/bin/env python3
"""
connect_forcing_response.py — does the ice-sheet ΔVAF spread concentrate in the sectors
whose FORCING carries the most low-frequency power? (connects task 1 <-> task 3)

For each ensemble it computes, per ISMIP6 basin:
  * ensemble spread sigma of the per-region VAF->SLE (mm) at a few horizons, from regionalStats.nc
and correlates that against the per-sector forcing low-frequency fraction from
`reports/spectrum_percell_generated0.csv` (task 1). The forcing spectrum is amplitude-independent,
so SSP126 / SSP585 / SSP585_varScaled10x all share the same per-sector low-freq fractions; only the
response spread differs (10x should amplify it and sharpen any relationship).

Region indexing: regionalStats nRegions (16) is index-aligned with the ISMIP6 region mask used to
build the forcing spectrum (same AIS_4to20km...regionMask_ismip6 ordering), so basin i <-> row i.

Author: Shivaprakash Muruganandham (2026-07-07)
"""
from __future__ import annotations
import os, sys, csv as _csv, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio

BANDS = ["seasonal", "interannual", "decadal", "multidecadal"]


def load_forcing_lowfreq(csv_path):
    """Return ordered (names, lowfreq_incl_interannual, lowfreq_deca_multi) for the 16 basins."""
    names, lf_all, lf_dm = [], [], []
    with open(csv_path) as fh:
        for row in _csv.DictReader(fh):
            if not row["sector"].lower().startswith("ismip6 basin"):
                continue
            names.append(row["sector"].replace("ISMIP6 Basin ", "").strip())
            inter, deca, multi = float(row["interannual"]), float(row["decadal"]), float(row["multidecadal"])
            lf_all.append(inter + deca + multi)     # >1.5 yr
            lf_dm.append(deca + multi)              # >8 yr (Robel tau_F band)
    return names, np.array(lf_all), np.array(lf_dm)


def ensemble_region_sigma(root, ensemble, include, horizons_yr):
    """sigma across members of per-region VAF->SLE (mm), at each horizon year. -> dict + years."""
    ens_dir = os.path.join(root, ensemble)
    members = [(n, p) for n, p in eio.discover_members(ens_dir, stats_filename="regionalStats.nc",
                                                       include=include)]
    stacks, nmin = [], None
    used = 0
    for name, path in members:
        try:
            ds = eio.to_year_dim(eio.load_member_regionalstats(path))
        except Exception:
            continue
        if "regionalVolumeAboveFloatation" not in ds:
            continue
        vaf = ds["regionalVolumeAboveFloatation"]           # (year, nRegions)
        yr = ds["year"].values
        if yr[0] > 5.0 or len(yr) < 10:                     # skip restart-segment / too short
            continue
        nreg = vaf.sizes["nRegions"]
        sle = np.column_stack([eio.vaf_to_sle_mm(vaf.isel(nRegions=r).values, reference="first")
                               for r in range(nreg)])        # (year, nRegions)
        stacks.append((yr, sle)); used += 1
        nmin = len(yr) if nmin is None else min(nmin, len(yr))
    if used < 3:
        return None, None, used
    years = stacks[0][0][:nmin]
    arr = np.stack([s[:nmin] for _, s in stacks], axis=0)    # (member, year, nRegions)
    out = {}
    for h in horizons_yr:
        i = int(np.argmin(np.abs(years - h)))
        out[float(years[i])] = np.nanstd(arr[:, i, :], axis=0, ddof=1)   # (nRegions,)
    return out, years, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--forcing-csv", default="reports/spectrum_percell_generated0.csv")
    ap.add_argument("--horizons", default="30,100,300")
    ap.add_argument("--band", default="lf_all", choices=["lf_all", "lf_dm"],
                    help="forcing metric: lf_all=>1.5yr, lf_dm=>8yr (decadal+multidecadal)")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    names, lf_all, lf_dm = load_forcing_lowfreq(args.forcing_csv)
    lf = lf_dm if args.band == "lf_dm" else lf_all
    horizons = [float(x) for x in args.horizons.split(",")]

    ensembles = [("SSP126", r"^SSP126_\d+$"), ("SSP585", r"^SSP585_\d+$"),
                 ("SSP585_varScaled10x", r"^SSP585_\d+$")]
    fig, axs = plt.subplots(1, len(ensembles), figsize=(5*len(ensembles), 4.6), squeeze=False)
    print(f"forcing metric: {args.band}  (per-basin low-freq fraction)\n")
    for k, (ens, inc) in enumerate(ensembles):
        sig, years, used = ensemble_region_sigma(args.root, ens, inc, horizons)
        ax = axs[0][k]
        if sig is None:
            print(f"{ens}: <3 usable members, skipped"); ax.set_title(f"{ens}: n/a"); continue
        print(f"=== {ens}  ({used} members, years {years[0]:.0f}..{years[-1]:.0f}) ===")
        print(f"  {'basin':7s}  lowfreq  " + "  ".join(f"sig@{int(h)}" for h in sig))
        for r, nm in enumerate(names):
            print(f"  {nm:7s}  {100*lf[r]:5.0f}%  " +
                  "  ".join(f"{sig[h][r]:7.3f}" for h in sig))
        # correlate at the LAST horizon (and print all)
        for h in sig:
            s = sig[h]
            ok = np.isfinite(s) & np.isfinite(lf)
            if ok.sum() >= 4 and np.std(s[ok]) > 0:
                pr, pp = pearsonr(lf[ok], s[ok]); sr, sp = spearmanr(lf[ok], s[ok])
                print(f"    corr(lowfreq, sigma@{int(h)}):  Pearson r={pr:+.2f} (p={pp:.2f})  "
                      f"Spearman={sr:+.2f} (p={sp:.2f})")
        hlast = list(sig)[-1]
        s = sig[hlast]
        ax.scatter(100*lf, s, c="C0")
        for r, nm in enumerate(names):
            ax.annotate(nm, (100*lf[r], s[r]), fontsize=6, alpha=0.7)
        ax.set_xlabel("forcing low-freq fraction (%)"); ax.set_ylabel(f"sigma SLE (mm) @yr{int(hlast)}")
        ax.set_title(f"{ens} ({used} mem)")
        print()
    fig.suptitle("Does ΔVAF spread concentrate where the forcing is low-frequency? "
                 f"(x = per-basin {args.band})")
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"forcing_vs_response_{args.band}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
