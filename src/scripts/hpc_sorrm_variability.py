#!/usr/bin/env python3
"""
hpc_sorrm_variability.py — how SORRM's melt variance splits across frequency. RUN ON HPC.

The regridded field is 12000 x 601 x 601, about 17 GB as float32, so it is streamed in row
blocks and never held whole. What comes back is small: the domain and per-sector band splits
as CSV, plus an npz with the domain anomaly series, its spectrum, and a per-cell map of the
seasonal fraction.

The question it answers is whether the ocean forcing the generator resamples is mostly the
seasonal cycle or mostly slower. That decides whether phase randomisation is doing anything an
ice sheet can feel: a 4-20 km model integrating over centuries responds to a multidecadal
signal quite differently than to a seasonal one.

Bands, in years: seasonal < 1.5, interannual 1.5-10, decadal 10-30, multidecadal > 30.
The timestep is read from the file; --dt-months overrides it. Run --list-vars first.
"""
from __future__ import annotations

import os, re, argparse
import numpy as np
import netCDF4

BANDS = [("seasonal", 0.0, 1.5), ("interannual", 1.5, 10.0),
         ("decadal", 10.0, 30.0), ("multidecadal", 30.0, np.inf)]

# longitude sectors, degrees east. Amundsen is the warm-cavity sector that carries the
# Part I result; Weddell holds Filchner-Ronne, cold now and loud later.
SECTORS = [("Amundsen", -140.0, -80.0), ("Weddell", -80.0, 0.0),
           ("East", 0.0, 160.0), ("Ross", 160.0, -140.0)]

VAR_CANDIDATES = ["timeMonthly_avg_landIceFreshwaterFlux", "landIceFreshwaterFlux",
                  "floatingBasalMassBal", "basalMeltFlux", "melt", "ismf"]

_TO_DAYS = {"microsecond": 1.0 / 86400e6, "millisecond": 1.0 / 86400e3, "second": 1.0 / 86400,
            "minute": 1.0 / 1440, "hour": 1.0 / 24, "day": 1.0, "month": 30.4375, "year": 365.25}


def timestep_years(d):
    """Median timestep in years, read from whichever time variable carries units."""
    for nm in ("Time", "time", "xtime"):
        if nm in d.variables and d[nm].ndim == 1 and d[nm].size > 2:
            u = getattr(d[nm], "units", "")
            m = re.match(r"\s*(\w+?)s?\s+since", u)
            if not m or m.group(1) not in _TO_DAYS:
                continue
            v = np.asarray(d[nm][:2000], dtype=np.float64)
            step = float(np.median(np.diff(v))) * _TO_DAYS[m.group(1)]
            return step / 365.25
    return None


def pick_var(d, want=None):
    if want:
        return want
    for c in VAR_CANDIDATES:
        if c in d.variables:
            return c
    best = max((v for v in d.variables.values() if v.ndim >= 2),
               key=lambda v: int(np.prod(v.shape)), default=None)
    if best is None:
        raise SystemExit("no multidimensional variable found; use --list-vars")
    return best.name


def band_fractions(psd, freq_per_yr):
    """Fraction of total power per band. psd is (nfreq,) or (nfreq, ncell).

    Rectangular sums, not trapezoid: welch spacing is uniform so df cancels, the bands
    tile the axis so the fractions sum to one, and the multidecadal band holds a single
    bin at any realistic segment length -- which a trapezoid rule drops entirely.
    """
    out, tot = [], psd.sum(axis=0)
    for _, lo, hi in BANDS:
        f_lo = 1.0 / hi if np.isfinite(hi) else 0.0
        f_hi = 1.0 / lo if lo > 0 else np.inf
        m = (freq_per_yr >= f_lo) & (freq_per_yr < f_hi)
        out.append(psd[m].sum(axis=0) if m.any() else np.zeros_like(tot))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, np.array(out) / tot, np.nan)


def sector_of(lon):
    """Index into SECTORS for each cell, -1 where none matches."""
    L = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    out = np.full(L.shape, -1, np.int8)
    for i, (_, lo, hi) in enumerate(SECTORS):
        m = (L >= lo) & (L < hi) if lo < hi else ((L >= lo) | (L < hi))
        out[m & (out < 0)] = i
    return out


def main():
    from scipy.signal import welch

    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--var", default=None)
    ap.add_argument("--dt-months", type=float, default=None,
                    help="override the timestep read from the file")
    ap.add_argument("--nperseg-years", type=float, default=100.0,
                    help="Welch segment length; sets the longest period resolved")
    ap.add_argument("--rows", type=int, default=20, help="grid rows per streamed block")
    ap.add_argument("--min-mean", type=float, default=0.0,
                    help="skip cells whose |mean| is below this (open ocean, land)")
    ap.add_argument("--out", default="reports/sorrm_variability")
    ap.add_argument("--list-vars", action="store_true")
    a = ap.parse_args()

    d = netCDF4.Dataset(a.field)
    if a.list_vars:
        print(f"dims: { {k: v.size for k, v in d.dimensions.items()} }")
        for k, v in d.variables.items():
            print(f"  {k:38s} {v.dimensions} {v.shape} {getattr(v, 'units', '')}")
        dt = timestep_years(d)
        if dt:
            print(f"\ninferred timestep {dt * 12:.3f} months ({dt:.4f} yr)")
        return

    var = pick_var(d, a.var)
    V = d[var]
    nt, ny, nx = V.shape
    dt_yr = (a.dt_months / 12.0) if a.dt_months else timestep_years(d)
    if not dt_yr:
        raise SystemExit("could not infer the timestep; pass --dt-months")
    fs = 1.0 / dt_yr
    nps = int(min(nt, max(64, a.nperseg_years / dt_yr)))
    print(f"{var} {V.shape}, {getattr(V, 'units', '')}")
    print(f"{nt} steps of {dt_yr * 12:.3f} months = {nt * dt_yr:.0f} years; "
          f"Welch segment {nps} steps = {nps * dt_yr:.0f} years")

    lon = np.asarray(d["lon"][:]).ravel() if "lon" in d.variables else None
    sec = sector_of(lon) if lon is not None else None

    # streamed pass: domain and sector sums, and the per-cell band split
    dom_sum = np.zeros(nt); dom_n = 0
    sec_sum = np.zeros((len(SECTORS), nt)); sec_n = np.zeros(len(SECTORS), int)
    cell_frac = np.full((len(BANDS), ny * nx), np.nan, np.float32)
    nok = 0
    for r0 in range(0, ny, a.rows):
        r1 = min(r0 + a.rows, ny)
        blk = np.ma.filled(np.asarray(V[:, r0:r1, :], dtype=np.float32), np.nan)
        blk = blk.reshape(nt, -1)
        base = r0 * nx
        ok = np.isfinite(blk).all(axis=0) & (np.nanstd(blk, axis=0) > 0)
        if a.min_mean > 0:
            ok &= np.abs(np.nanmean(blk, axis=0)) >= a.min_mean
        if ok.any():
            dom_sum += blk[:, ok].sum(axis=1); dom_n += int(ok.sum())
            if sec is not None:
                s = sec[base:base + blk.shape[1]]
                for i in range(len(SECTORS)):
                    m = ok & (s == i)
                    if m.any():
                        sec_sum[i] += blk[:, m].sum(axis=1); sec_n[i] += int(m.sum())
            f2, p2 = welch(blk[:, ok], fs=fs, nperseg=nps, detrend="constant", axis=0)
            cell_frac[:, base + np.flatnonzero(ok)] = band_fractions(p2[1:], f2[1:])
            nok += int(ok.sum())
        print(f"  rows {r1}/{ny}  cells kept {nok:,}", end="\r")
    d.close()
    print()
    if dom_n == 0:
        raise SystemExit("no usable cells -- check --var and --min-mean")

    dom = dom_sum / dom_n
    dom -= dom.mean()
    f, p = welch(dom, fs=fs, nperseg=nps, detrend="constant")
    rows = [("domain", band_fractions(p[1:, None], f[1:])[:, 0], dom_n)]
    for i, (nm, _, _) in enumerate(SECTORS):
        if sec_n[i] < 20:
            continue
        s = sec_sum[i] / sec_n[i]; s = s - s.mean()
        fi, pi = welch(s, fs=fs, nperseg=nps, detrend="constant")
        rows.append((nm, band_fractions(pi[1:, None], fi[1:])[:, 0], int(sec_n[i])))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + "_bands.csv", "w") as fh:
        fh.write("sector,ncell," + ",".join(b[0] for b in BANDS) + "\n")
        for nm, v, n in rows:
            fh.write(f"{nm},{n}," + ",".join(f"{x:.5f}" for x in v) + "\n")
    np.savez_compressed(a.out + "_series.npz", domain=dom.astype(np.float32),
                        dt_years=dt_yr, freq=f, psd=p, ny=ny, nx=nx,
                        seasonal_fraction=cell_frac[0].reshape(ny, nx))
    print(f"wrote {a.out}_bands.csv and {a.out}_series.npz\n")
    for nm, v, n in rows:
        print(f"  {nm:10s} n={n:9,}  " +
              "  ".join(f"{b[0][:4]} {100 * x:5.1f}%" for b, x in zip(BANDS, v)))


if __name__ == "__main__":
    main()
