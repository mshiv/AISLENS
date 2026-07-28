#!/usr/bin/env python3
"""
forcing_spectrum_percell.py — per-cell band-variance fractions of a MALI-grid forcing field.

Coherence-free: computes each cell's own Welch PSD and averages the band FRACTIONS
per ISMIP6 sector (avoids the spatial-aggregation sqrt(N) inflation). Input: a field
(Time, nCells), region mask with regionCellMasks, and mesh with areaCell.

Author: Shivaprakash Muruganandham
"""
from __future__ import annotations
import os, sys, csv as _csv, argparse
import numpy as np
import xarray as xr
from scipy import signal as scipy_signal
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forcing_spectrum import DEFAULT_BANDS
BANDS = list(DEFAULT_BANDS.keys())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", required=True, help="(Time,nCells) NetCDF")
    p.add_argument("--var", default="floatingBasalMassBalAdjustment")
    p.add_argument("--region-mask",
                   default="data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc")
    p.add_argument("--mesh-file", default=None, help="mesh with areaCell (area weighting)")
    p.add_argument("--dt-months", type=float, default=1.0)
    p.add_argument("--detrend", default="constant", choices=["constant", "linear"])
    p.add_argument("--cell-chunk", type=int, default=20000)
    p.add_argument("--label", default="field")
    p.add_argument("--out-dir", default="reports")
    return p.parse_args()


def band_fracs_vectorized(f, psd):
    """f:(nf,), psd:(nf,ncells) -> fracs dict band->(ncells,), total(ncells,). Partition-exact."""
    order = np.argsort(f); f = f[order]; psd = psd[order]
    df = np.diff(f)
    seg = df[:, None] * (psd[1:] + psd[:-1]) / 2.0            # (nf-1, ncells)
    cum = np.vstack([np.zeros((1, psd.shape[1])), np.cumsum(seg, axis=0)])  # (nf,ncells)
    total = cum[-1]
    def C(x):
        xi = min(max(x, f[0]), f[-1])
        j = int(np.searchsorted(f, xi))
        if j <= 0: return cum[0]
        if j >= len(f): return cum[-1]
        w = (xi - f[j-1]) / (f[j] - f[j-1])
        return cum[j-1] * (1 - w) + cum[j] * w
    out = {}
    for name, (pmin, pmax) in DEFAULT_BANDS.items():
        fmin = 1.0/pmax if np.isfinite(pmax) and pmax > 0 else 0.0
        fmax = 1.0/pmin if pmin > 0 else np.inf
        out[name] = np.maximum(0.0, C(fmax) - C(fmin))
    return out, total


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ds = xr.open_dataset(a.field, decode_times=False, chunks={"nCells": a.cell_chunk})
    da = ds[a.var] if a.var in ds else ds[[v for v in ds.data_vars if "nCells" in ds[v].dims][0]]
    da = da.transpose("Time", "nCells")
    nt, nc = da.sizes["Time"], da.sizes["nCells"]
    fs = 12.0 / a.dt_months
    nperseg = min(nt, 1024)
    print(f"field {a.label}: Time={nt}, nCells={nc}; per-cell Welch (nperseg={nperseg}) in "
          f"chunks of {a.cell_chunk}...")

    frac = {b: np.full(nc, np.nan) for b in BANDS}
    tot = np.full(nc, np.nan)
    for i0 in range(0, nc, a.cell_chunk):
        i1 = min(i0 + a.cell_chunk, nc)
        x = np.asarray(da.isel(nCells=slice(i0, i1)).values, dtype=float)   # (Time, m)
        # cells that are finite & non-constant
        good = np.isfinite(x).all(axis=0) & (np.nanstd(x, axis=0) > 0)
        if good.any():
            f, pxx = scipy_signal.welch(x[:, good], fs=fs, nperseg=nperseg,
                                        detrend=a.detrend, axis=0)
            nz = f > 0
            fr, tt = band_fracs_vectorized(f[nz], pxx[nz])
            idx = np.where(good)[0] + i0
            with np.errstate(invalid="ignore", divide="ignore"):
                for b in BANDS:
                    frac[b][idx] = fr[b] / tt
            tot[idx] = tt
        print(f"  cells {i0}-{i1}  ({int(good.sum())} usable)")

    # region mask + area
    dm = xr.open_dataset(a.region_mask, decode_times=False)
    masks = np.asarray(dm["regionCellMasks"].values)
    if masks.shape[0] != nc and masks.shape[1] == nc:
        masks = masks.T
    names = _region_names(dm)
    area = np.ones(nc)
    if a.mesh_file:
        dmesh = xr.open_dataset(a.mesh_file, decode_times=False)
        if "areaCell" in dmesh.variables:
            area = np.asarray(dmesh["areaCell"].values, dtype=float)

    rows = []
    print(f"\nAREA-WEIGHTED per-cell band fractions per sector ({a.label}):")
    print("  " + "sector".ljust(14) + "  " + "  ".join(b[:4] for b in BANDS) + "   ncells")
    order = list(range(masks.shape[1])) if masks.ndim == 2 else []
    for r in order + ["ALL"]:
        if r == "ALL":
            sel = (masks.sum(axis=1) > 0)
            nm = "ALL-shelf"
        else:
            sel = masks[:, r] > 0
            nm = names[r] if r < len(names) else f"region{r}"
        fr_row = {}
        for b in BANDS:
            v = frac[b]; w = area
            ok = sel & np.isfinite(v)
            fr_row[b] = float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan
        # area-weighted mean of the per-cell total (absolute) PSD variance over the sector's
        # usable cells (finite & non-zero) — the amplitude the band FRACTIONS were normalized by.
        okv = sel & np.isfinite(tot) & (tot != 0)
        total_var = float(np.sum(tot[okv] * area[okv]) / np.sum(area[okv])) if okv.any() else np.nan
        n_ok = int((sel & np.isfinite(frac["seasonal"])).sum())
        rows.append((nm, fr_row, n_ok, total_var))
        print(f"  {nm:14s}  " + "  ".join(f"{100*fr_row[b]:4.0f}%" for b in BANDS)
              + f"   {n_ok}")

    csv_path = os.path.join(a.out_dir, f"spectrum_percell_{a.label}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["sector", "label"] + BANDS + ["ncells", "total_var"])
        for nm, fr_row, n_ok, total_var in rows:
            w.writerow([nm, a.label] + [f"{fr_row[b]:.5f}" for b in BANDS]
                       + [n_ok, f"{total_var:.6e}"])
    print(f"\nCSV -> {csv_path}")

    # stacked bar
    sect = [nm for nm, _, _, _ in rows]
    fig, ax = plt.subplots(figsize=(max(9, 0.55*len(sect)+2), 5))
    bottom = np.zeros(len(sect))
    colors = {"seasonal": "#f4a261", "interannual": "#2a9d8f",
              "decadal": "#264653", "multidecadal": "#7b2cbf"}
    for b in BANDS:
        vals = np.array([100*fr[b] for _, fr, _, _ in rows])
        ax.bar(sect, vals, bottom=bottom, label=b, color=colors[b]); bottom += vals
    ax.set_ylabel("% of per-cell variance (area-weighted)")
    ax.set_title(f"Per-cell (coherence-free) forcing variance by band — {a.label}")
    ax.set_xticklabels(sect, rotation=45, ha="right")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    out = os.path.join(a.out_dir, f"spectrum_percell_{a.label}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")


def _region_names(dm):
    if "regionNames" not in dm.variables:
        return [f"region{i}" for i in range(dm.sizes.get("nRegions", 0))]
    out = []
    for row in np.atleast_1d(dm["regionNames"].values):
        if isinstance(row, bytes):
            out.append(row.decode("utf-8", "ignore").strip())
        elif isinstance(row, np.ndarray):
            out.append(b"".join(bytes([c]) if isinstance(c, (int, np.integer))
                                else (c if isinstance(c, bytes) else str(c).encode())
                                for c in row.tolist()).decode("utf-8", "ignore").strip())
        else:
            out.append(str(row).strip())
    return out


if __name__ == "__main__":
    main()
