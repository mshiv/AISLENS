#!/usr/bin/env python3
"""
forcing_spectrum_sectors.py — per-sector spectral audit of AISLENS ocean forcing.

For each ISMIP6 sector, computes area-weighted mean time series, Welch PSD, and
variance fractions in seasonal/interannual/decadal/multidecadal bands. Can compare
raw SORRM vs generated forcing with --compare to check low-frequency preservation.

Author: Shivaprakash Muruganandham
"""
from __future__ import annotations

import os
import sys
import csv as _csv
import argparse

import numpy as np
import xarray as xr
from scipy import signal as scipy_signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forcing_spectrum import band_variance, DEFAULT_BANDS   # reuse the tested band logic

BAND_ORDER = list(DEFAULT_BANDS.keys())  # seasonal, interannual, decadal, multidecadal

# variability/melt field variable candidates (time x nCells)
VAR_CANDIDATES = [
    "floatingBasalMassBalAdjustment",
    "timeMonthly_avg_landIceFreshwaterFlux", "landIceFreshwaterFlux",
    "basalMeltFlux", "melt", "variability", "anomaly",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", help="forcing field NetCDF (time x nCells)")
    p.add_argument("--var", default=None, help="field variable (default: auto-detect)")
    p.add_argument("--region-mask",
                   default="data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc",
                   help="ISMIP6 mask (regionCellMasks[nCells,nRegions] + regionNames)")
    p.add_argument("--mesh-file", default=None,
                   help="optional mesh providing areaCell[nCells] for area-weighting")
    p.add_argument("--dt-months", type=float, default=1.0,
                   help="sampling interval in months (default 1 = monthly)")
    p.add_argument("--detrend", default="constant", choices=["constant", "linear"],
                   help="Welch detrend (constant keeps the seasonal cycle; linear removes drift)")
    p.add_argument("--label", default="field", help="label for outputs/CSV")
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--out-csv", default=None, help="per-sector band-fraction CSV path")
    p.add_argument("--compare", nargs="+", default=None,
                   help="overlay mode: 2+ band-fraction CSVs to compare (skips --field)")
    p.add_argument("--list-vars", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------- helpers (self-contained)
def _open(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what} not found: {path}\n(dangling symlink? `ls -laL {path}`)")
    return xr.open_dataset(path, decode_times=False)


TIME_DIMS = ("Time", "time", "nTime", "t")


def pick_var(ds, requested):
    if requested:
        if requested not in ds.variables:
            raise KeyError(f"--var '{requested}' absent. Have: "
                           f"{[v for v in ds.variables if v not in ds.dims]}")
        return requested
    for c in VAR_CANDIDATES:
        if c in ds.variables:
            return c
    # first floating var with a time-like dim and >=1 spatial dim (nCells, or y/x)
    for v in ds.variables:
        if v in ds.dims or not np.issubdtype(ds[v].dtype, np.floating):
            continue
        if any(t in ds[v].dims for t in TIME_DIMS) and ds[v].ndim >= 2:
            return v
    raise KeyError("could not auto-detect a (time x space) field; pass --var. "
                   f"Have: {[v for v in ds.variables if v not in ds.dims]}")


def load_field_2d(da):
    """Return (field2d [time, ncells_flat], is_ncells_grid). Flattens (time,y,x)->
    (time, y*x); leaves (time,nCells) as is; (time,) -> (time,1)."""
    tdim = next((t for t in TIME_DIMS if t in da.dims), da.dims[0])
    da = da.transpose(tdim, ...)
    arr = np.asarray(da.values, dtype=float)
    is_ncells = "nCells" in da.dims
    if arr.ndim == 1:
        arr = arr[:, None]
    else:
        arr = arr.reshape(arr.shape[0], -1)   # (time, flattened space)
    return arr, is_ncells


def decode_region_names(ds):
    if "regionNames" not in ds.variables:
        return [f"region{i}" for i in range(ds.sizes.get("nRegions", 0))]
    names = []
    for row in np.atleast_1d(ds["regionNames"].values):
        if isinstance(row, bytes):
            names.append(row.decode("utf-8", "ignore").strip())
        elif isinstance(row, np.ndarray):
            names.append(b"".join(bytes([x]) if isinstance(x, (int, np.integer))
                                  else (x if isinstance(x, bytes) else str(x).encode())
                                  for x in row.tolist()).decode("utf-8", "ignore").strip())
        else:
            names.append(str(row).strip())
    return names


def sector_series(field, masks, area):
    """field (time,nCells); masks (nCells,nRegions) 0/1; area (nCells,) or None
    -> (nRegions,time) area-weighted sector-mean series."""
    w = masks.astype(float)
    if area is not None:
        w = w * area[:, None]
    denom = np.where(w.sum(axis=0) > 0, w.sum(axis=0), np.nan)
    m = np.where(np.isfinite(field), field, 0.0)
    return (m @ w / denom).T


def sector_band_fractions(series, dt_months, detrend):
    """series (time,) -> dict band->fraction of resolved (f>0) variance, + total."""
    x = series[np.isfinite(series)]
    if x.size < 64:
        return None
    fs = 12.0 / dt_months
    f, pxx = scipy_signal.welch(x, fs=fs, nperseg=min(len(x), 1024), detrend=detrend)
    nz = f > 0
    bands_var, total = band_variance(f[nz], pxx[nz], DEFAULT_BANDS)
    if total <= 0:
        return None
    fr = {b: bands_var[b] / total for b in BAND_ORDER}
    fr["_total"] = total
    return fr


# ---------------------------------------------------------------- compute mode
def run_field(args):
    os.makedirs(args.out_dir, exist_ok=True)
    dsf = _open(args.field, "field file")
    if args.list_vars:
        print("vars:", [v for v in dsf.variables if v not in dsf.dims],
              "\ndims:", dict(dsf.sizes)); return
    var = pick_var(dsf, args.var)
    da = dsf[var]
    tdim = next((t for t in TIME_DIMS if t in da.dims), da.dims[0])
    space_dims = [d for d in da.dims if d != tdim]
    n_space = int(np.prod([da.sizes[d] for d in space_dims]))
    print(f"field '{var}': time={da.sizes[tdim]}, space={n_space} "
          f"({'nCells' if 'nCells' in da.dims else 'x'.join(space_dims)})")

    # decide per-sector vs aggregate WITHOUT loading the (possibly huge) field
    per_sector, names, masks, area = False, [], None, None
    if os.path.exists(args.region_mask):
        dsm = _open(args.region_mask, "region mask")
        m = np.asarray(dsm["regionCellMasks"].values)
        if m.shape[0] != n_space and m.shape[1] == n_space:
            m = m.T
        if m.shape[0] == n_space:
            per_sector, masks, names = True, m, decode_region_names(dsm)
            if args.mesh_file:
                dmesh = _open(args.mesh_file, "mesh file")
                if "areaCell" in dmesh.variables:
                    area = np.asarray(dmesh["areaCell"].values, dtype=float)
                else:
                    print("  [warn] mesh has no areaCell; simple cell mean.")
        else:
            print(f"  [note] mask grid ({m.shape[0]} cells) != field ({n_space}) "
                  "-> AGGREGATE-only. Supply a matching --region-mask for per-sector.")
    else:
        print(f"  [note] no region mask at {args.region_mask} -> AGGREGATE-only.")

    if per_sector:
        field, _ = load_field_2d(da)                          # materialize (nCells case)
        sect = sector_series(field, masks, area)              # (nRegions,time)
        circ = sector_series(field, (masks.sum(axis=1) > 0).astype(float)[:, None], area)[0]
        sector_list = names + ["circumAntarctic"]
    else:
        # blocked spatial mean over all cells (skips NaN) -> 1D series; avoids a 10 GB load
        try:
            da = da.chunk({tdim: 120})           # dask-backed blockwise reduction
        except Exception:
            print("  [warn] dask chunking unavailable; spatial mean may load the full field.")
        circ = np.asarray(da.mean(dim=space_dims, skipna=True).values, dtype=float).squeeze()
        sect, sector_list = None, ["domainAggregate"]

    rows = []
    print(f"\nband fractions ({args.label}):  "
          + "  ".join(f"{b[:4]}" for b in BAND_ORDER))
    for r, nm in enumerate(sector_list):
        if nm in ("circumAntarctic", "domainAggregate"):
            series = circ
        else:
            series = sect[r]
        fr = sector_band_fractions(series, args.dt_months, args.detrend)
        if fr is None:
            print(f"  {nm:>14s}: (series too short / zero variance)")
            continue
        rows.append((nm, fr))
        print(f"  {nm:>14s}: " + "  ".join(f"{100*fr[b]:4.0f}%" for b in BAND_ORDER))

    csv_path = args.out_csv or os.path.join(
        args.out_dir, f"spectrum_sectors_{args.label}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["sector", "label"] + BAND_ORDER + ["total_var"])
        for nm, fr in rows:
            w.writerow([nm, args.label] + [f"{fr[b]:.5f}" for b in BAND_ORDER]
                       + [f"{fr['_total']:.6g}"])
    print(f"\nCSV -> {csv_path}")
    _plot_stacked(rows, args.label, args.out_dir)


def _plot_stacked(rows, label, out_dir):
    sectors = [nm for nm, _ in rows]
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(sectors) + 2), 5))
    bottom = np.zeros(len(sectors))
    colors = {"seasonal": "#f4a261", "interannual": "#2a9d8f",
              "decadal": "#264653", "multidecadal": "#7b2cbf"}
    for b in BAND_ORDER:
        vals = np.array([100 * fr[b] for _, fr in rows])
        ax.bar(sectors, vals, bottom=bottom, label=b, color=colors.get(b))
        bottom += vals
    ax.set_ylabel("% of resolved (f>0) variance")
    ax.set_title(f"Per-sector forcing variance by band — {label}\n"
                 "(Amundsen–Bellingshausen 'should' carry interannual–decadal power)")
    ax.set_xticklabels(sectors, rotation=45, ha="right")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    out = os.path.join(out_dir, f"spectrum_sectors_{label}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")


# ---------------------------------------------------------------- compare mode
def run_compare(args):
    os.makedirs(args.out_dir, exist_ok=True)
    datasets = {}   # label -> {sector -> {band->frac}}
    for path in args.compare:
        with open(path) as fh:
            for row in _csv.DictReader(fh):
                lab = row["label"]
                datasets.setdefault(lab, {})[row["sector"]] = {
                    b: float(row[b]) for b in BAND_ORDER}
    labels = list(datasets)
    if len(labels) < 2:
        print("[warn] --compare needs >=2 distinct labels; got:", labels)
    sectors = [s for s in next(iter(datasets.values()))]
    # focus metric: low-frequency fraction = decadal + multidecadal
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(sectors) + 2), 5))
    width = 0.8 / max(1, len(labels))
    x = np.arange(len(sectors))
    for i, lab in enumerate(labels):
        low = [100 * (datasets[lab].get(s, {}).get("decadal", 0)
                      + datasets[lab].get(s, {}).get("multidecadal", 0)) for s in sectors]
        ax.bar(x + i * width, low, width, label=lab)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(sectors, rotation=45, ha="right")
    ax.set_ylabel("% variance in decadal+multidecadal bands")
    ax.set_title("Low-frequency (decadal+multidecadal) forcing power per sector\n"
                 "raw SORRM vs generated → does the generator preserve low-freq?")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(args.out_dir, "spectrum_sectors_compare_lowfreq.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")
    # print the biggest generator losses
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        print(f"\nLow-freq fraction {a} -> {b} (Δ = {b} minus {a}):")
        for s in sectors:
            la = datasets[a].get(s, {}); lb = datasets[b].get(s, {})
            lowa = la.get("decadal", 0) + la.get("multidecadal", 0)
            lowb = lb.get("decadal", 0) + lb.get("multidecadal", 0)
            print(f"  {s:>14s}: {100*lowa:5.1f}% -> {100*lowb:5.1f}%  (Δ {100*(lowb-lowa):+5.1f})")


def main():
    args = parse_args()
    if args.compare:
        run_compare(args)
    elif args.field:
        run_field(args)
    else:
        sys.exit("Provide --field <NetCDF> (compute mode) or --compare <csv csv ...> (overlay).")


if __name__ == "__main__":
    main()
