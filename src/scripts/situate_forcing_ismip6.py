#!/usr/bin/env python3
"""
situate_forcing_ismip6.py — overlay SORRM basal melt on the ISMIP6 CMIP model universe.

Loads SORRM melt flux, converts to m/yr, computes per-ISMIP6-sector mean and interannual
sigma. Optionally overlays on a Jourdain 2020 Fig 10/13 remake (CMIP5 + FESOM distribution)
via --jourdain-csv to show where SORRM sits in the model spread.

Author: Shivaprakash Muruganandham
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio  # RHO_ICE, region-name decoding conventions

SECONDS_PER_YEAR = 365.0 * 24 * 3600.0
RHO_ICE = getattr(eio, "RHO_ICE", 910.0)

# candidate melt-flux variable names (MPAS-O / regridded ISMF conventions), tried in order
MELT_VAR_CANDIDATES = [
    "timeMonthly_avg_landIceFreshwaterFlux",
    "landIceFreshwaterFlux",
    "timeMonthly_avg_landIceFreshwaterFluxTotal",
    "ismf", "melt", "meltRate", "basalMeltFlux", "freshwaterFlux",
]

# --- default data paths (prefer config; else repo-relative fallbacks) ---
try:
    from aislens.config import config
    DEF_SORRM = str(config.FILE_MPASO_MODEL)
    DEF_MASK = str(config.DIR_MALI / "AIS_4to20km_r01_20220907.regionMask_ismip6.nc")
except Exception:  # config not importable in this env
    DEF_SORRM = "data/external/Regridded_SORRMv2.1.ISMF.FULL.nc"
    DEF_MASK = "data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sorrm-file", default=DEF_SORRM,
                   help="SORRM ice-shelf melt-flux file (time x nCells)")
    p.add_argument("--melt-var", default=None,
                   help="melt-flux variable name (default: auto-detect from candidates)")
    p.add_argument("--flux-units", default="kg/m2/s",
                   choices=["kg/m2/s", "m/yr", "mm/yr"],
                   help="units of the melt-flux variable, for conversion to m/yr ice "
                        "(kg/m2/s -> /rho_ice*seconds_per_yr). Melt taken POSITIVE.")
    p.add_argument("--melt-sign", default="auto", choices=["auto", "pos", "neg"],
                   help="'neg' if the file stores melt as a negative mass flux "
                        "(freshwater INTO ocean); 'auto' flips so the sector mean is >0")
    p.add_argument("--region-mask", default=DEF_MASK,
                   help="ISMIP6 region mask (regionCellMasks[nCells,nRegions]+regionNames)")
    p.add_argument("--mesh-file", default=None,
                   help="optional MALI mesh file providing areaCell[nCells] for "
                        "area-weighted sector means (else simple cell mean)")
    p.add_argument("--detrend", action="store_true",
                   help="linearly detrend each sector's annual-mean series before taking "
                        "the interannual sigma (removes any embedded drift)")
    p.add_argument("--jourdain-csv", default=None,
                   help="CSV of per-model per-sector melt to overlay (schema in docstring). "
                        "If omitted, a SORRM-only sector-melt chart is produced.")
    p.add_argument("--out-dir", default=None, help="output dir (default: <repo>/reports)")
    p.add_argument("--list-vars", action="store_true",
                   help="just print the SORRM file's variables and exit")
    return p.parse_args()


# ----------------------------------------------------------------------------- loaders
def _open(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what} not found at:\n    {path}\n"
            f"(Is it a dangling symlink? `ls -laL {path}`. Point --"
            f"{'sorrm-file' if what=='SORRM melt file' else 'region-mask'} at the real file.)")
    return xr.open_dataset(path, decode_times=False)


def pick_melt_var(ds, requested):
    if requested:
        if requested not in ds.variables:
            raise KeyError(f"--melt-var '{requested}' not in file. Available: "
                           f"{[v for v in ds.variables if v not in ds.dims]}")
        return requested
    for c in MELT_VAR_CANDIDATES:
        if c in ds.variables:
            return c
    # fall back to the first 2-D (time x nCells) float var
    for v in ds.variables:
        if v in ds.dims:
            continue
        da = ds[v]
        if da.ndim == 2 and "nCells" in da.dims and np.issubdtype(da.dtype, np.floating):
            return v
    raise KeyError("Could not auto-detect a melt-flux variable; pass --melt-var. "
                   f"Vars: {[v for v in ds.variables if v not in ds.dims]}")


def to_m_per_yr(da, units, sign):
    if units == "kg/m2/s":
        out = da / RHO_ICE * SECONDS_PER_YEAR
    elif units == "mm/yr":
        out = da / 1000.0
    else:  # m/yr
        out = da
    if sign == "neg" or (sign == "auto" and float(np.nanmean(out)) < 0):
        out = -out
    return out


def decode_region_names(ds):
    """Return a list of str sector names from the mask's regionNames variable."""
    if "regionNames" not in ds.variables:
        n = ds.sizes.get("nRegions", 0)
        return [f"region{i}" for i in range(n)]
    rn = ds["regionNames"].values
    names = []
    for row in np.atleast_1d(rn):
        if isinstance(row, bytes):
            names.append(row.decode("utf-8", "ignore").strip())
        elif isinstance(row, np.ndarray):  # array of byte-chars (MPAS char convention)
            names.append(b"".join(x if isinstance(x, bytes) else bytes([x]) if isinstance(x, (int, np.integer)) else str(x).encode()
                                  for x in row.tolist()).decode("utf-8", "ignore").strip())
        else:
            names.append(str(row).strip())
    return names


# ----------------------------------------------------------------------------- core
def sector_series(melt, masks, area):
    """melt: (time,nCells) m/yr; masks: (nCells,nRegions) 0/1; area: (nCells,) or None.
    Returns (nRegions, time) area-weighted sector-mean melt."""
    w = masks.astype(float)
    if area is not None:
        w = w * area[:, None]
    denom = w.sum(axis=0)                       # (nRegions,)
    denom = np.where(denom > 0, denom, np.nan)
    # (time,nCells) @ (nCells,nRegions) -> (time,nRegions), NaN-safe
    m = np.where(np.isfinite(melt), melt, 0.0)
    num = m @ w                                 # (time,nRegions)
    return (num / denom).T                      # (nRegions,time)


def annual_means(series, n_per_year=12):
    """series: (time,) -> annual means; assumes contiguous monthly if n_per_year=12."""
    t = series.shape[0]
    nyr = t // n_per_year
    if nyr < 2:
        return series  # too short to bin; treat each sample as a 'year'
    return series[: nyr * n_per_year].reshape(nyr, n_per_year).mean(axis=1)


def interannual_sigma(series, detrend, n_per_year=12):
    ann = annual_means(series, n_per_year)
    ann = ann[np.isfinite(ann)]
    if ann.size < 2:
        return np.nan
    if detrend and ann.size >= 3:
        x = np.arange(ann.size)
        ann = ann - np.polyval(np.polyfit(x, ann, 1), x)
    return float(np.std(ann, ddof=1))


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    out_dir = args.out_dir or "reports"
    os.makedirs(out_dir, exist_ok=True)

    dss = _open(args.sorrm_file, "SORRM melt file")
    if args.list_vars:
        print("Variables:", [v for v in dss.variables if v not in dss.dims])
        print("Dims:", dict(dss.sizes)); return
    mvar = pick_melt_var(dss, args.melt_var)
    melt = to_m_per_yr(dss[mvar], args.flux_units, args.melt_sign)
    melt = melt.transpose(..., "nCells")  # ensure (time,nCells)
    melt_np = np.asarray(melt.values, dtype=float)
    if melt_np.ndim == 1:
        melt_np = melt_np[None, :]
    print(f"SORRM melt var '{mvar}': shape {melt_np.shape}, "
          f"mean {np.nanmean(melt_np):.3g} m/yr, max {np.nanmax(melt_np):.3g} m/yr "
          f"(sanity: per-shelf melt is O(0-30) m/yr)")

    dsm = _open(args.region_mask, "region mask")
    masks = np.asarray(dsm["regionCellMasks"].values)
    if masks.shape[0] != melt_np.shape[1]:          # (nCells,nRegions) expected
        if masks.shape[1] == melt_np.shape[1]:
            masks = masks.T
        else:
            raise ValueError(f"nCells mismatch: melt has {melt_np.shape[1]}, mask "
                             f"{masks.shape}. SORRM must be on the same grid as the mask.")
    names = decode_region_names(dsm)

    area = None
    if args.mesh_file:
        dmesh = _open(args.mesh_file, "mesh file")
        if "areaCell" in dmesh.variables:
            area = np.asarray(dmesh["areaCell"].values, dtype=float)
        else:
            print("  [warn] --mesh-file has no areaCell; using simple cell mean.")

    sect = sector_series(melt_np, masks, area)      # (nRegions,time)
    # circum-Antarctic = area/cell-weighted over ALL masked shelf cells
    allmask = (masks.sum(axis=1) > 0).astype(float)[:, None]
    circ = sector_series(melt_np, allmask, area)[0]

    rows = []
    print("\nSORRM per-sector melt (mean +/- interannual sigma), m/yr:")
    for r, nm in enumerate(names):
        mean_r = float(np.nanmean(sect[r]))
        sig_r = interannual_sigma(sect[r], args.detrend)
        rows.append((nm, mean_r, sig_r))
        print(f"  {nm:>14s}: {mean_r:8.3f}  +/- {sig_r:6.3f}")
    circ_mean, circ_sig = float(np.nanmean(circ)), interannual_sigma(circ, args.detrend)
    rows.append(("circumAntarctic", circ_mean, circ_sig))
    print(f"  {'circumAntarctic':>14s}: {circ_mean:8.3f}  +/- {circ_sig:6.3f}")

    # write a CSV concatenable with Jourdain's table
    csv_path = os.path.join(out_dir, "sorrm_sector_melt.csv")
    with open(csv_path, "w") as f:
        f.write("sector,model,melt_m_per_yr,sigma_interannual,parameterization,family\n")
        for nm, mean_r, sig_r in rows:
            f.write(f"{nm},SORRMv2.1,{mean_r:.4f},{sig_r:.4f},,SORRM\n")
    print(f"\nWrote {csv_path}")

    _plot(rows, args.jourdain_csv, out_dir)


def _plot(sorrm_rows, jourdain_csv, out_dir):
    sorrm = {nm: (m, s) for nm, m, s in sorrm_rows}
    sectors = [nm for nm, _, _ in sorrm_rows]

    if not jourdain_csv or not os.path.exists(jourdain_csv):
        # SORRM-only bar chart with variability whiskers
        fig, ax = plt.subplots(figsize=(10, 5))
        xs = np.arange(len(sectors))
        means = [sorrm[s][0] for s in sectors]
        sigs = [sorrm[s][1] for s in sectors]
        ax.bar(xs, means, yerr=sigs, capsize=3, color="C0", alpha=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(sectors, rotation=45, ha="right")
        ax.set_ylabel("SORRM basal melt (m/yr, ice-eq)")
        ax.set_title("SORRMv2.1 per-sector basal melt (mean +/- interannual sigma)\n"
                     "(pass --jourdain-csv to overlay on the CMIP5/FESOM distribution)")
        fig.tight_layout()
        out = os.path.join(out_dir, "sorrm_sector_melt.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"Figure -> {out}")
        if jourdain_csv:
            print(f"  [note] --jourdain-csv '{jourdain_csv}' not found; made SORRM-only chart.")
        return

    # ---- Jourdain Fig 10/13 remake: CMIP5 strip + FESOM + SORRM overlay ----
    import csv as _csv
    per_sector = {}  # sector -> list of (model, melt, family)
    with open(jourdain_csv) as f:
        for row in _csv.DictReader(f):
            per_sector.setdefault(row["sector"], []).append(
                (row.get("model", ""), float(row["melt_m_per_yr"]),
                 row.get("family", "CMIP5")))

    plot_sectors = [s for s in sectors if s in per_sector] or list(per_sector.keys())
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(plot_sectors) + 2), 6))
    rng = np.random.default_rng(0)
    for i, s in enumerate(plot_sectors):
        entries = per_sector[s]
        cmip = [m for _, m, fam in entries if fam.upper() == "CMIP5"]
        fesom = [m for _, m, fam in entries if fam.upper() == "FESOM"]
        if cmip:
            jitter = rng.uniform(-0.12, 0.12, size=len(cmip))
            ax.scatter(np.full(len(cmip), i) + jitter, cmip, s=22, color="0.5",
                       alpha=0.7, label="CMIP5 models" if i == 0 else None, zorder=2)
            ax.boxplot(cmip, positions=[i], widths=0.5, showfliers=False,
                       medianprops=dict(color="0.3"))
        if fesom:
            ax.scatter(np.full(len(fesom), i), fesom, marker="s", s=45, color="C2",
                       label="FESOM (cavity-resolving)" if i == 0 else None, zorder=3)
        if s in sorrm:
            m, sig = sorrm[s]
            ax.errorbar([i], [m], yerr=[sig], marker="*", ms=16, color="C3", capsize=4,
                        label="SORRM (this study)" if i == 0 else None, zorder=4)
    ax.set_xticks(range(len(plot_sectors)))
    ax.set_xticklabels(plot_sectors, rotation=45, ha="right")
    ax.set_ylabel("basal melt (m/yr)")
    ax.set_title("Antarctic basal melt across the ISMIP6 model universe\n"
                 "(Jourdain 2020 Fig 10/13 remake: CMIP5 vs FESOM, with SORRM overlaid)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(out_dir, "sorrm_vs_cmip_ismip6_melt.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
