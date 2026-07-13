#!/usr/bin/env python3
"""


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!! DEPRECATED! DO NOT USE! Use generate_forcings_spectral_cutoff.py instead.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



make_band_limited_forcing.py — P1 band-decomposition: split a variability forcing (time x nCells)
into frequency bands (seasonal / interannual / decadal / multidecadal) via per-cell temporal FFT,
and rescale each band to MATCHED total variance, so the resulting ensembles differ only in the
*timescale* of the forcing, not its amplitude. This is the clean test the natural ensemble can't do
(see wiki `SORRM forcing spectral content`: frequency vs dynamics are entangled).

For each input realization it writes 4 output files:  <name>__seasonal.nc, __interannual.nc,
__decadal.nc, __multidecadal.nc  — each a valid MALI adjustment forcing (same dims + xtime).

Pass 1 streams cell-chunks to get each band's total variance (for the matched-variance scale factors);
passes 2..5 write each scaled band field incrementally (netCDF4). Memory-safe on ~10 GB fields.

Example:
  python make_band_limited_forcing.py \
      --field data/processed/vargen_realizations-ssn-50/AIS_..._AISLENS-Forcing_0.nc \
      --out-dir data/processed/vargen_band_limited --match total

Author: Shivaprakash Muruganandham (2026-07-07)
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forcing_spectrum import DEFAULT_BANDS

BANDS = list(DEFAULT_BANDS.keys())


def band_mask(freqs_per_yr, pmin, pmax):
    fmin = 1.0/pmax if np.isfinite(pmax) and pmax > 0 else 0.0
    fmax = 1.0/pmin if pmin > 0 else np.inf
    return (freqs_per_yr > fmin) & (freqs_per_yr <= fmax)


def filter_chunk(x, masks):
    """x: (time, m). Return {band: filtered (time, m)} via rfft; f=0 always dropped."""
    nt = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    out = {}
    for b, mk in masks.items():
        Xb = np.where(mk[:, None], X, 0.0)
        out[b] = np.fft.irfft(Xb, n=nt, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--var", default="floatingBasalMassBalAdjustment")
    ap.add_argument("--out-dir", default="data/processed/vargen_band_limited")
    ap.add_argument("--dt-months", type=float, default=1.0)
    ap.add_argument("--match", default="total", choices=["total", "unit"],
                    help="total: match each band to the ORIGINAL field's total variance (equal amplitude)")
    ap.add_argument("--cell-chunk", type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    src = Dataset(args.field, "r")
    var = src.variables[args.var]
    dims = var.dimensions                      # expect (Time, nCells)
    tdim = dims[0]; cdim = dims[1]
    nt = len(src.dimensions[tdim]); nc = len(src.dimensions[cdim])
    fs = 12.0/args.dt_months
    freqs = np.fft.rfftfreq(nt, d=1.0/fs)      # cycles/yr
    masks = {b: band_mask(freqs, *DEFAULT_BANDS[b]) for b in BANDS}
    print(f"{os.path.basename(args.field)}: time={nt}, nCells={nc}, fs={fs}/yr")

    # ---- pass 1: per-band total sum-of-squares + original ----
    ss = {b: 0.0 for b in BANDS}; ss_orig = 0.0
    for i0 in range(0, nc, args.cell_chunk):
        i1 = min(i0+args.cell_chunk, nc)
        x = np.asarray(var[:, i0:i1], dtype=float)
        x = x - x.mean(axis=0, keepdims=True)          # anomaly
        ss_orig += float(np.nansum(x**2))
        for b, xb in filter_chunk(x, masks).items():
            ss[b] += float(np.nansum(xb**2))
    target = ss_orig if args.match == "total" else float(nc*nt)
    scale = {b: (np.sqrt(target/ss[b]) if ss[b] > 0 else 0.0) for b in BANDS}
    print("  band total-variance shares + matched-variance scale factors:")
    for b in BANDS:
        print(f"    {b:13s}: var-share {100*ss[b]/ss_orig:5.1f}%   scale x{scale[b]:.3f}")

    # ---- passes 2..5: write each scaled band field ----
    base = os.path.splitext(os.path.basename(args.field))[0]
    for b in BANDS:
        out_path = os.path.join(args.out_dir, f"{base}__{b}.nc")
        dst = Dataset(out_path, "w")
        for dn, d in src.dimensions.items():
            dst.createDimension(dn, None if d.isunlimited() else len(d))
        # copy xtime + coord/aux vars verbatim; create the band variable
        for vn, v in src.variables.items():
            nv = dst.createVariable(vn, v.datatype, v.dimensions)
            nv.setncatts({k: v.getncattr(k) for k in v.ncattrs()})
            if vn != args.var:
                nv[:] = v[:]
        bv = dst.variables[args.var]
        for i0 in range(0, nc, args.cell_chunk):
            i1 = min(i0+args.cell_chunk, nc)
            x = np.asarray(var[:, i0:i1], dtype=float)
            x = x - x.mean(axis=0, keepdims=True)
            bv[:, i0:i1] = filter_chunk(x, masks)[b] * scale[b]
        dst.close()
        print(f"  wrote {out_path}")
    src.close()


if __name__ == "__main__":
    main()
