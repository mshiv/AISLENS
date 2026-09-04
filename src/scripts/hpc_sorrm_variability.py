#!/usr/bin/env python3
"""
hpc_sorrm_variability.py -- frequency split of SORRM melt variance. RUN ON HPC.

Takes the variability component F_v; the full field carries the drift and the draft
dependence. Fields are several GB and are streamed in row blocks.

Writes a bands CSV and an npz holding the domain anomaly series, its spectrum, the
per-cell band fractions and the per-cell seasonal fraction. Passing --seasonality adds
var(F_s)/(var(F_s)+var(F_v)).

Bands, in years: seasonal < 1.5, interannual 1.5-10, decadal 10-30, multidecadal > 30.
Timestep is read from the file. Run --list-vars first.
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

    Rectangular sums: welch spacing is uniform so df cancels, the bands tile the axis
    so the fractions sum to one, and a band holding a single bin still counts.
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
    ap.add_argument("--field", required=True,
                    help="the variability component F_v")
    ap.add_argument("--seasonality", default=None,
                    help="optional F_s file; adds var(F_s)/(var(F_s)+var(F_v)) per cell "
                         "and for the domain. Variance needs no time alignment, so the "
                         "two files may differ in length.")
    ap.add_argument("--var", default=None)
    ap.add_argument("--dt-months", type=float, default=None,
                    help="override the timestep read from the file")
    ap.add_argument("--nperseg-years", type=float, default=100.0,
                    help="Welch segment length; sets the longest period resolved")
    ap.add_argument("--rows", type=int, default=20, help="grid rows per streamed block")
    ap.add_argument("--min-mean", type=float, default=0.0,
                    help="skip cells whose |mean| is below this")
    ap.add_argument("--var-percentile", type=float, default=0.0,
                    help="drop cells below this percentile of standard deviation; a first "
                         "pass measures the distribution. Cells that are flat apart from a "
                         "slow drift otherwise put all their power in the lowest band.")
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

    ds_s = Vs = None
    if a.seasonality:
        ds_s = netCDF4.Dataset(a.seasonality)
        Vs = ds_s[pick_var(ds_s, None)]
        if Vs.shape[1:] != V.shape[1:]:
            raise SystemExit(f"seasonality grid {Vs.shape[1:]} != field grid {V.shape[1:]}")
        print(f"seasonality {Vs.name} {Vs.shape}")

    # first pass for the variance threshold: std only, no spectra
    std_thresh = 0.0
    if a.var_percentile > 0:
        allsd = np.full(ny * nx, np.nan, np.float32)
        for r0 in range(0, ny, a.rows):
            r1 = min(r0 + a.rows, ny)
            b0 = np.ma.filled(np.asarray(V[:, r0:r1, :], dtype=np.float32), np.nan)
            b0 = b0.reshape(nt, -1)
            allsd[r0 * nx: r0 * nx + b0.shape[1]] = np.nanstd(b0, axis=0)
            print(f"  pass 1 rows {r1}/{ny}", end="\r")
        good = np.isfinite(allsd) & (allsd > 0)
        std_thresh = float(np.percentile(allsd[good], a.var_percentile))
        print(f"\n  std threshold at p{a.var_percentile:g} = {std_thresh:.4g}; "
              f"keeps {int((allsd[good] >= std_thresh).sum()):,} of {int(good.sum()):,} cells")

    # streamed pass: domain and sector sums, and the per-cell band split
    dom_sum = np.zeros(nt); dom_n = 0
    sec_sum = np.zeros((len(SECTORS), nt)); sec_n = np.zeros(len(SECTORS), int)
    cell_frac = np.full((len(BANDS), ny * nx), np.nan, np.float32)
    seas_frac = np.full(ny * nx, np.nan, np.float32)
    var_v_cell = np.full(ny * nx, np.nan, np.float32)
    var_s_cell = np.full(ny * nx, np.nan, np.float32)
    var_v_tot = var_s_tot = 0.0
    nok = 0
    for r0 in range(0, ny, a.rows):
        r1 = min(r0 + a.rows, ny)
        blk = np.ma.filled(np.asarray(V[:, r0:r1, :], dtype=np.float32), np.nan)
        blk = blk.reshape(nt, -1)
        base = r0 * nx
        ok = np.isfinite(blk).all(axis=0) & (np.nanstd(blk, axis=0) > 0)
        if a.min_mean > 0:
            ok &= np.abs(np.nanmean(blk, axis=0)) >= a.min_mean
        if std_thresh > 0:
            ok &= np.nanstd(blk, axis=0) >= std_thresh
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
            if Vs is not None:
                sb = np.ma.filled(np.asarray(Vs[:, r0:r1, :], dtype=np.float32), np.nan)
                sb = sb.reshape(sb.shape[0], -1)
                vv = np.nanvar(blk[:, ok], axis=0)
                vs = np.nanvar(sb[:, ok], axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    seas_frac[base + np.flatnonzero(ok)] = np.where(
                        (vv + vs) > 0, vs / (vv + vs), np.nan)
                var_v_tot += float(np.nansum(vv)); var_s_tot += float(np.nansum(vs))
                var_v_cell[base + np.flatnonzero(ok)] = vv
                var_s_cell[base + np.flatnonzero(ok)] = vs
            nok += int(ok.sum())
        print(f"  rows {r1}/{ny}  cells kept {nok:,}", end="\r")
    d.close()
    if ds_s is not None:
        ds_s.close()
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
                        band_fraction=cell_frac.reshape(len(BANDS), ny, nx),
                        seasonal_fraction=seas_frac.reshape(ny, nx),
                        var_v=var_v_cell.reshape(ny, nx),
                        var_s=var_s_cell.reshape(ny, nx),
                        nperseg=nps, std_threshold=std_thresh)
    print(f"wrote {a.out}_bands.csv and {a.out}_series.npz\n")
    for nm, v, n in rows:
        print(f"  {nm:10s} n={n:9,}  " +
              "  ".join(f"{b[0][:4]} {100 * x:5.1f}%" for b, x in zip(BANDS, v)))
    if var_s_tot + var_v_tot > 0:
        sf = var_s_tot / (var_s_tot + var_v_tot)
        print(f"\n  seasonal fraction var(F_s)/(var(F_s)+var(F_v)) = {100 * sf:.1f}% "
              f"summed over cells; per-cell map is in the npz")
        with open(a.out + "_bands.csv", "a") as fh:
            fh.write(f"# seasonal_fraction_var_ratio,{sf:.5f}\n")


if __name__ == "__main__":
    main()
