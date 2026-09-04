#!/usr/bin/env python3
"""
hpc_sorrm_variability.py — how SORRM's melt variance splits across frequency. RUN ON HPC.

The regridded SORRM melt field is far too large to work with on a laptop, so this reduces it
there and writes back three small products: the domain-mean anomaly series, the band split for
the domain and for four sectors, and a per-cell map of the seasonal fraction.

The question it answers is whether the ocean forcing the generator resamples is mostly the
seasonal cycle or mostly slower. That decides whether phase randomisation is doing anything an
ice sheet can feel, since a 4-20 km ice-sheet model integrating over centuries cannot respond
to a seasonal signal the way it responds to a multidecadal one.

Bands, in years: seasonal < 1.5, interannual 1.5-10, decadal 10-30, multidecadal > 30.
Run --list-vars first if you do not know what the file holds.
"""
from __future__ import annotations

import os, argparse
import numpy as np
import netCDF4

BANDS = [("seasonal", 0.0, 1.5), ("interannual", 1.5, 10.0),
         ("decadal", 10.0, 30.0), ("multidecadal", 30.0, np.inf)]

# longitude sectors, degrees east. Amundsen is the warm-cavity sector that carries the
# Part I result; Weddell holds Filchner-Ronne, which is cold now and loud later.
SECTORS = [("Amundsen", -140.0, -80.0), ("Weddell", -80.0, 0.0),
           ("East", 0.0, 160.0), ("Ross", 160.0, 220.0)]

VAR_CANDIDATES = ["timeMonthly_avg_landIceFreshwaterFlux", "landIceFreshwaterFlux",
                  "floatingBasalMassBal", "basalMeltFlux", "melt", "ismf", "variability"]


def pick_var(d, want=None):
    if want:
        return want
    for c in VAR_CANDIDATES:
        if c in d.variables:
            return c
    best = max((v for v in d.variables.values() if v.ndim >= 2),
               key=lambda v: np.prod(v.shape), default=None)
    if best is None:
        raise SystemExit("no multidimensional variable found; use --list-vars")
    return best.name


def band_fractions(psd, freq_per_yr):
    """Fraction of total power in each band. psd may be (nfreq,) or (nfreq, ncell).

    Rectangular sums, not trapezoid: welch returns uniform spacing so the df cancels,
    the bands tile the axis so the fractions sum to one, and a band holding a single
    bin -- the multidecadal one, at any realistic segment length -- still counts.
    """
    out, tot = [], psd.sum(axis=0)
    for _, lo, hi in BANDS:
        f_lo = 1.0 / hi if np.isfinite(hi) else 0.0
        f_hi = 1.0 / lo if lo > 0 else np.inf
        m = (freq_per_yr >= f_lo) & (freq_per_yr < f_hi)
        out.append(psd[m].sum(axis=0) if m.any() else np.zeros_like(tot))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, np.array(out) / tot, np.nan)


def main():
    from scipy.signal import welch

    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True, help="regridded SORRM melt NetCDF")
    ap.add_argument("--var", default=None)
    ap.add_argument("--dt-months", type=float, default=1.0)
    ap.add_argument("--nperseg-years", type=float, default=50.0,
                    help="Welch segment length; caps the longest period resolved")
    ap.add_argument("--chunk", type=int, default=20000, help="cells per FFT chunk")
    ap.add_argument("--out", default="reports/sorrm_variability")
    ap.add_argument("--list-vars", action="store_true")
    a = ap.parse_args()

    d = netCDF4.Dataset(a.field)
    if a.list_vars:
        print(f"dims: { {k: v.size for k, v in d.dimensions.items()} }")
        for k, v in d.variables.items():
            print(f"  {k:38s} {v.dimensions} {v.shape} {getattr(v, 'units', '')}")
        return

    var = pick_var(d, a.var)
    V = d[var]
    print(f"variable {var} {V.shape} {V.dimensions}")
    F = np.ma.filled(np.asarray(V[:], dtype=np.float32), np.nan)
    # coordinates, for the sector split
    lon = None
    for c in ("lon", "longitude", "LONGITUDE"):
        if c in d.variables:
            lon = np.asarray(d[c][:], dtype=np.float64)
            break
    if lon is None and {"x", "y"} <= set(d.variables):
        X, Y = np.asarray(d["x"][:]), np.asarray(d["y"][:])
        if X.ndim == 1:
            X, Y = np.meshgrid(X, Y)
        lon = np.degrees(np.arctan2(X, Y))     # polar stereographic, pole at origin
    d.close()

    nt = F.shape[0]
    F = F.reshape(nt, -1)
    if lon is not None:
        lon = np.asarray(lon).ravel()
        if lon.size != F.shape[1]:
            print(f"  ! lon has {lon.size} points against {F.shape[1]} cells; "
                  f"sector split skipped")
            lon = None

    dt_yr = a.dt_months / 12.0
    fs = 1.0 / dt_yr
    nps = int(min(nt, max(64, a.nperseg_years / dt_yr)))
    print(f"{nt} steps at {a.dt_months} month(s) = {nt * dt_yr:.0f} years; "
          f"Welch segment {nps} steps = {nps * dt_yr:.0f} years")

    ok = np.isfinite(F).all(axis=0) & (np.nanstd(F, axis=0) > 0)
    print(f"{ok.sum():,} of {F.shape[1]:,} cells usable")

    # ---- domain aggregate
    dom = np.nanmean(F[:, ok], axis=1)
    dom = dom - dom.mean()
    f, p = welch(dom, fs=fs, nperseg=nps, detrend="constant")
    frac = band_fractions(p[1:, None], f[1:])[:, 0]
    print("\ndomain aggregate:")
    for (nm, _, _), v in zip(BANDS, frac):
        print(f"  {nm:14s} {100 * v:5.1f}%")

    # ---- per cell, in chunks
    idx = np.flatnonzero(ok)
    cell_frac = np.full((len(BANDS), F.shape[1]), np.nan, np.float32)
    for i in range(0, idx.size, a.chunk):
        j = idx[i:i + a.chunk]
        f2, p2 = welch(F[:, j], fs=fs, nperseg=nps, detrend="constant", axis=0)
        cell_frac[:, j] = band_fractions(p2[1:], f2[1:]).astype(np.float32)
        print(f"  cells {i + j.size:,}/{idx.size:,}", end="\r")
    print()

    # ---- sectors
    rows = [("domain", frac, int(ok.sum()))]
    if lon is not None:
        L = ((lon + 180.0) % 360.0) - 180.0
        for nm, lo, hi in SECTORS:
            lo2, hi2 = ((lo + 180) % 360) - 180, ((hi + 180) % 360) - 180
            m = (L >= lo2) & (L < hi2) if lo2 < hi2 else ((L >= lo2) | (L < hi2))
            m &= ok
            if m.sum() < 20:
                print(f"  ! sector {nm}: only {m.sum()} cells, skipped"); continue
            s = np.nanmean(F[:, m], axis=1); s = s - s.mean()
            fs_, ps_ = welch(s, fs=fs, nperseg=nps, detrend="constant")
            rows.append((nm, band_fractions(ps_[1:, None], fs_[1:])[:, 0], int(m.sum())))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + "_bands.csv", "w") as fh:
        fh.write("sector,ncell," + ",".join(b[0] for b in BANDS) + "\n")
        for nm, v, n in rows:
            fh.write(f"{nm},{n}," + ",".join(f"{x:.5f}" for x in v) + "\n")
    np.savez_compressed(a.out + "_series.npz", domain=dom.astype(np.float32),
                        dt_years=dt_yr, freq=f, psd=p,
                        seasonal_fraction=cell_frac[0], shape=np.array(V.shape[1:]))
    print(f"\nwrote {a.out}_bands.csv and {a.out}_series.npz")
    for nm, v, n in rows:
        print(f"  {nm:10s} n={n:8,}  " +
              "  ".join(f"{b[0][:4]} {100 * x:4.1f}%" for b, x in zip(BANDS, v)))


if __name__ == "__main__":
    main()
