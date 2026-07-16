#!/usr/bin/env python3
"""
subtract_global_mean_forcing.py — center a MALI-grid forcing realization by subtracting ONE global scalar
(the mean over Time x nCells) from the field, in place or to a new file.

WHY: the 1000-yr realizations came out with a constant positive DC offset (~1.2e-5 in the nCells-reduced
series) that the 300-yr ensemble doesn't have - the meanAdjust (notebook 06's single-scalar subtraction)
either didn't take or the generator used the un-adjusted variability. The variability std is identical,
so it's purely a missing constant. This reproduces notebook 06 exactly: it subtracted the global scalar
from filled(0) cells too, so subtracting it from the whole realization (shelf + non-shelf) matches the
300-yr construction. Do this to the base realizations BEFORE adding xtime / scenario trend.

By default computes each file's OWN global scalar and subtracts it (each file ends up centered at 0). Pass
--scalar VALUE to subtract a fixed shared scalar instead (e.g. to reuse one member's value for all).

Memory-safe: two chunked passes; writes HIGH-INDEX-FIRST so an unlimited Time dim is never truncated.

Usage:
  # in place on a whole dir (array-friendly with --file):
  python subtract_global_mean_forcing.py --dir <dir> --in-place
  # one file to a new location:
  python subtract_global_mean_forcing.py --file in.nc --out out.nc
"""
from __future__ import annotations
import argparse, glob, os, shutil, sys
import numpy as np
from netCDF4 import Dataset


def _fill_value(v):
    for att in ("_FillValue", "missing_value"):
        if att in v.ncattrs():
            try:
                return float(v.getncattr(att))
            except Exception:
                pass
    return None


def _read_filled(v, t0, t1, fv):
    """Read raw (no masked array), replace NaN/inf/_FillValue with 0.0 -- reproduces fillna(0)."""
    b = np.asarray(v[t0:t1], dtype=float)
    bad = ~np.isfinite(b)
    if fv is not None:
        bad |= (b == fv)
    if bad.any():
        b[bad] = 0.0
    return b


def compute_means(v, chunk, fv):
    """Return (mean_all, mean_valid), chunked over Time:
       mean_all   = mean over ALL cells with NaN/fill -> 0 (notebook 06's native-grid recipe);
       mean_valid = mean over ONLY finite, non-fill cells (nanmean) -- the physical shelf-mean.
    On the MALI grid these DIFFER because many cells are _FillValue; mean_valid is the correct demean
    (it centers the forcing over the cells MALI actually applies)."""
    Nf = v.shape[0]
    tot_all = 0.0; cnt_all = 0; tot_val = 0.0; cnt_val = 0
    for t0 in range(0, Nf, chunk):
        t1 = min(t0 + chunk, Nf)
        b = np.asarray(v[t0:t1], dtype=float)
        good = np.isfinite(b)
        if fv is not None:
            good &= (b != fv)
        tot_val += float(b[good].sum()); cnt_val += int(good.sum())
        tot_all += float(np.where(good, b, 0.0).sum()); cnt_all += b.size
    mean_all = tot_all / cnt_all if cnt_all else float("nan")
    mean_val = tot_val / cnt_val if cnt_val else float("nan")
    return mean_all, mean_val


def process(path, var, out, scalar, chunk, mean_over="valid"):
    if out and out != path:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        shutil.copy2(path, out)
        target = out
    else:
        target = path
    with Dataset(target, "r+") as ds:
        v = ds.variables[var]
        v.set_auto_mask(False)                      # raw values -- don't rely on numpy stripping masks
        if v.dimensions[0].lower() not in ("time",):
            sys.exit(f"{var} first dim is {v.dimensions[0]}, expected Time")
        fv = _fill_value(v)
        Nf = v.shape[0]
        if scalar is not None:
            s = scalar; m_all = m_val = None
        else:
            m_all, m_val = compute_means(v, chunk, fv)
            s = m_val if mean_over == "valid" else m_all
        if not np.isfinite(s):
            sys.exit(f"FATAL: computed scalar is not finite ({s}) -- refusing to write (would NaN the field)")
        # HIGH-INDEX-FIRST: pins an unlimited Time at Nf so low-slice writes don't truncate the file.
        for t0 in reversed(range(0, Nf, chunk)):
            t1 = min(t0 + chunk, Nf)
            b = _read_filled(v, t0, t1, fv)         # fillna(0) then subtract -> never writes NaN
            v[t0:t1] = b - s
        if len(ds.dimensions[v.dimensions[0]]) != Nf:
            sys.exit(f"FATAL: Time truncated {Nf} -> {len(ds.dimensions[v.dimensions[0]])}")
    return s, m_all, m_val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--dir")
    ap.add_argument("--pattern", default="AIS_*Forcing_*.nc")
    ap.add_argument("--var", default="floatingBasalMassBalAdjustment")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--out", help="output path (single --file) ")
    ap.add_argument("--out-dir", help="output dir (with --dir)")
    ap.add_argument("--scalar", type=float, default=None, help="subtract this fixed scalar instead of each file's own mean")
    ap.add_argument("--mean-over", choices=["valid", "all"], default="valid",
                    help="valid = nanmean over finite/non-fill cells (physical shelf-mean, DEFAULT); "
                         "all = fillna(0) mean over all cells (notebook-06 native recipe, diluted on MALI grid)")
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args()

    files = [a.file] if a.file else sorted(glob.glob(os.path.join(a.dir, a.pattern)))
    if not files:
        sys.exit("no files matched")
    for f in files:
        if a.in_place:
            out = f
        elif a.file:
            out = a.out or sys.exit("need --out or --in-place")
        else:
            out = os.path.join(a.out_dir or sys.exit("need --out-dir or --in-place"), os.path.basename(f))
        s, m_all, m_val = process(f, a.var, out, a.scalar, a.chunk, a.mean_over)
        extra = "" if a.scalar is not None else f"  (valid={m_val:.6e}, all={m_all:.6e}; used '{a.mean_over}')"
        print(f"{os.path.basename(f)}: subtracted scalar={s:.6e}{extra} -> {out}")


if __name__ == "__main__":
    main()
