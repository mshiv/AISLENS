#!/usr/bin/env python3
"""
verify_forcing_seam_sign.py -- test whether the APPLIED ocean forcing (per-cell
floatingBasalMassBalAdjustment, as actually consumed by MALI) INVERTS SIGN across the
2300 base->extension join.

Background: the 300yr forcing generator (`ssn-trend-ssp585`, used by the base run,
2000-2300) and the 1000yr forcing generator (`ssn-2000-3000-ssp585`, used by the
2300-3000 extension) were found to produce `floatingBasalMassBalAdjustment` variability
fields that are near-perfect MIRROR IMAGES of each other (Pearson r = -0.997) when
compared on synthetic/matched samples. That comparison was on the GENERATOR output, not
on the forcing actually read by a run. If the base member's forcing file and the
extension member's forcing file disagree in sign the same way, then the ocean melt
forcing MALI applies flips sign right at the seam - which would show up as an abrupt,
spurious step in basal melt (and therefore in VAF/SLE) at year 2300 that has nothing to
do with ice dynamics.

This script verifies that, per ensemble member, by reading the actual
per-cell forcing files (not the generator scratch files) from each member's run
directory and comparing a window on each side of the seam:

  1. Domain-mean magnitude+sign on each side (area-weighted if --mesh given).
     Flags if the two sides have OPPOSITE sign.
  2. Per-cell spatial-pattern correlation across the seam (time-mean field each side).
     r < -0.5  -> SIGN-INVERTED (the spatial pattern itself is mirrored)
     r > +0.5  -> consistent (same spatial pattern on both sides)
     |r| < 0.5 -> decorrelated / investigate further
  3. Continuity: the raw jump in domain-mean between the base run's last valid month
     and the extension's first valid month, reported in absolute units and as a
     fraction of the base-side standard deviation (is the jump within normal
     month-to-month scatter, or a step?).

IMPORTANT -- ext-file alignment: the ext file's Time=0 is NOT assumed to be the seam
date. In practice an "extension" forcing file can be the FULL multi-century generator
series (e.g. Time=0 labeled calendar year 2000, spanning 2000-2973) even though it's
the file used to force the 2300+ extension run -- the model presumably starts reading
partway through it. This script auto-detects the correct starting index by reading the
ext file's xtime and locating the first timestep at/after (base's last valid month + 1),
i.e. it aligns by CALENDAR DATE, not by file position. Override with --seam-year or
--ext-start-index if auto-detection is unreliable (e.g. missing xtime).

This script doesn't loads the full (Time x nCells) field. It only reads a
`--n-months` slice (default 240 = 20 years monthly) from the tail of the base file and
the head of the extension file via `xr.open_dataset(...)[var].isel(Time=slice(...))`,
which is lazy. A small buffer beyond `--n-months` is read on each
side so that any trailing/leading garbage timesteps (whole field == 0, or blank xtime -
e.g. unused pre-allocated slots) can be skipped without truncating the requested window.

--------------------------------------------------------------------------------------
Example (run ON HPC, where the per-cell forcing files actually live under
the scratch ENSEMBLES root):

  SSP585:
    python3 verify_forcing_seam_sign.py \\
        --base-dir data/MALI/ENSEMBLES/SSP585/SSP585_00 \\
        --ext-dir  data/MALI/ENSEMBLES/SSP585-EXT/SSP585-EXT_00 \\
        --mesh     data/MALI/mesh/AIS_4to20km_r01_20220907.nc

  SSP126:
    python3 verify_forcing_seam_sign.py \\
        --base-dir data/MALI/ENSEMBLES/SSP126/SSP126_00 \\
        --ext-dir  data/MALI/ENSEMBLES/SSP126-EXT/SSP126-EXT_00 \\
        --mesh     data/MALI/mesh/AIS_4to20km_r01_20220907.nc

  Without a mesh file (unweighted domain means -- fine for a quick sign check):
    python3 verify_forcing_seam_sign.py --base-dir <base member dir> --ext-dir <ext member dir>

  Self-test of the correlation/verdict logic on synthetic arrays (no files needed):
    python3 verify_forcing_seam_sign.py --self-test

Assumptions (override via CLI if the tree differs):
  - Each member directory directly contains the per-cell forcing file, filename
    `AIS_4to20km_r01_20220907_AISLENS-Forcing_0.nc` (--forcing-name).
  - That file has variable `floatingBasalMassBalAdjustment` (--var) on dims
    (Time, nCells), monthly.
  - Optional `xtime` variable on dim (Time[, StrLen]) used to detect blank/garbage
    timesteps; if absent, only the all-zero/all-NaN check is used.
  - Base and extension files share the same nCells ordering/count (same mesh) --
    the script errors out loudly if nCells differs.
  - --mesh, if given, carries --area-var (default areaCell) with length nCells for
    area-weighted domain means; if omitted or not found, domain means are unweighted.

"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


# ----------------------------------------------------------------------------
# Core, file-independent logic (kept as plain functions so the self-test can
# exercise it without touching any I/O).
# ----------------------------------------------------------------------------

def parse_year_month(xtime_str: str) -> "float | None":
    """Parse 'YYYY-MM-DD_...' -> fractional year (YYYY + (MM-1)/12). None if unparseable."""
    s = xtime_str.strip()
    if not s:
        return None
    try:
        y = int(s[0:4])
        m = int(s[5:7])
        return y + (m - 1) / 12.0
    except (ValueError, IndexError):
        return None


def decode_xtime(values: np.ndarray) -> list[str]:
    """Decode an xtime array slice to a list of stripped strings.

    Handles the common MPAS layouts:
      - 2-D (Time, StrLen) fixed-width byte/char arrays
      - 1-D array of Python bytes or str
    Falls back to str() on anything unrecognized rather than raising, since xtime
    decoding is only used for garbage-timestep detection (best-effort).
    """
    out = []
    if values.ndim == 2:
        for row in values:
            if row.dtype.kind in ("S", "a") or (row.size and isinstance(row.flat[0], bytes)):
                s = b"".join(c if isinstance(c, bytes) else bytes(c, "utf-8") for c in row.tolist())
                out.append(s.decode("utf-8", "ignore"))
            else:
                out.append("".join(str(c) for c in row.tolist()))
    else:
        for v in values.tolist():
            if isinstance(v, bytes):
                out.append(v.decode("utf-8", "ignore"))
            else:
                out.append(str(v))
    return [s.strip() for s in out]


def compute_valid_mask(block: np.ndarray, xtime_strs: "list[str] | None") -> np.ndarray:
    """Return a boolean mask (nt,) of timesteps that are NOT garbage.

    A timestep is garbage if xtime is blank/empty, OR the whole spatial field at
    that timestep is all-zero or all-NaN (both indicate an unused/unwritten slot).
    """
    nt = block.shape[0]
    valid = np.ones(nt, dtype=bool)
    if xtime_strs is not None:
        valid &= np.array([bool(s) and s.strip() != "" for s in xtime_strs])
    all_nan = np.all(np.isnan(block), axis=1)
    finite_or_nan = np.where(np.isnan(block), 0.0, block)
    all_zero = np.all(finite_or_nan == 0.0, axis=1)
    valid &= ~(all_nan | all_zero)
    return valid


def domain_mean(block: np.ndarray, weights: "np.ndarray | None") -> np.ndarray:
    """Per-timestep domain mean over cells; (nt, nCells) -> (nt,).

    Area-weighted if `weights` (nCells,) is given, else a plain nanmean. NaN cells
    are excluded from both the numerator and the weight sum.
    """
    if weights is None:
        return np.nanmean(block, axis=1)
    nan_mask = np.isnan(block)
    w = np.broadcast_to(weights, block.shape)
    w_eff = np.where(nan_mask, 0.0, w)
    vals = np.where(nan_mask, 0.0, block)
    denom = w_eff.sum(axis=1)
    out = np.full(block.shape[0], np.nan)
    ok = denom > 0
    out[ok] = (vals[ok] * w_eff[ok]).sum(axis=1) / denom[ok]
    return out


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equal-length 1-D arrays, NaN-masked.

    Prefers scipy.stats.pearsonr if available; falls back to np.corrcoef so the
    script stays usable with a bare numpy+xarray environment.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    try:
        from scipy.stats import pearsonr
        r, _ = pearsonr(a, b)
        return float(r)
    except ImportError:
        return float(np.corrcoef(a, b)[0, 1])


def classify_verdict(base_mean: float, ext_mean: float, r: float, jump_frac: float) -> str:
    """Combine the three tests into a single verdict string."""
    opposite_sign = (
        np.isfinite(base_mean) and np.isfinite(ext_mean)
        and np.sign(base_mean) != 0 and np.sign(ext_mean) != 0
        and np.sign(base_mean) != np.sign(ext_mean)
    )
    if np.isnan(r):
        base_verdict = "UNDETERMINED (correlation could not be computed)"
    elif r < -0.5:
        base_verdict = "SIGN-INVERTED across seam"
    elif r > 0.5:
        if np.isfinite(jump_frac) and abs(jump_frac) > 1.0:
            base_verdict = "consistent, large offset"
        else:
            base_verdict = "consistent, small offset"
    else:
        base_verdict = "decorrelated / investigate"

    if opposite_sign and "SIGN-INVERTED" not in base_verdict:
        base_verdict += " [NOTE: domain-mean signs also disagree]"
    return base_verdict


# ----------------------------------------------------------------------------
# File I/O (memory-safe: only ever materializes a small edge window)
# ----------------------------------------------------------------------------

def find_start_index_after(path: str, seam_year_month: float) -> int:
    """Scan the FULL `xtime` of `path` (cheap: it's a small 1-D string array, not the
    (Time x nCells) field) and return the index of the first timestep whose calendar
    date is >= `seam_year_month` (fractional year). Raises if xtime is absent/unparseable
    or no such timestep exists -- the caller should fall back to a manual --ext-start-index
    in that case.

    This exists because an "extension" forcing file is not guaranteed to start its Time
    axis at the seam date -- it may be the FULL multi-century generator series (e.g.
    Time=0 labeled year 2000even for a file used to force the 2300+ extension run), in
    which case blindly taking Time=0 compares the wrong calendar period entirely.
    """
    import xarray as xr

    with xr.open_dataset(path, decode_times=False) as ds:
        if "xtime" not in ds.variables:
            raise RuntimeError(
                f"{path} has no 'xtime' variable; cannot locate the seam date automatically. "
                f"Pass --ext-start-index to specify the Time index manually.")
        xt_full = decode_xtime(ds["xtime"].values)  # full array, but xtime is tiny (Time,[StrLen])

    for i, s in enumerate(xt_full):
        y = parse_year_month(s)
        if y is not None and y >= seam_year_month - 1e-6:
            return i
    raise RuntimeError(
        f"No timestep in {path} with date >= {seam_year_month:.4f} found "
        f"(file spans up to {xt_full[-1] if xt_full else '?'}). "
        f"Pass --ext-start-index to specify the Time index manually.")


def load_edge_block(path: str, var: str, side: str, n_months: int, buffer_months: int,
                     start_at: "int | None" = None):
    """Load a `side` ('start' or 'end') window of up to `buffer_months` timesteps
    from `var` in `path`, then trim to the last/first `n_months` NON-GARBAGE
    timesteps within that window.

    For side='start', if `start_at` is given, the window begins at that absolute Time
    index instead of index 0 (use this to align to the seam date -- see
    `find_start_index_after` -- rather than assuming the file's Time=0 is the seam).

    Returns (block, xtime_strs_or_None, abs_first_idx, abs_last_idx, Nt) where
    `block` is (m, nCells) with m <= n_months, and abs_first/abs_last are the
    absolute Time indices (within the full file) of the taken window's edges.
    """
    import xarray as xr

    with xr.open_dataset(path, decode_times=False) as ds:
        if var not in ds.variables:
            cand = [n for n, v in ds.variables.items() if "Time" in v.dims]
            raise KeyError(f"'{var}' not found in {path}. Time-varying candidates: {cand}")
        da = ds[var]
        if "Time" not in da.dims:
            raise ValueError(f"'{var}' in {path} has dims {da.dims}, expected a 'Time' dim")
        if da.dims[0] != "Time":
            raise ValueError(f"expected Time as dim 0 of '{var}' in {path}, got {da.dims}")
        Nt = ds.sizes["Time"]
        buf = min(Nt, max(buffer_months, n_months))
        if side == "end":
            sl = slice(Nt - buf, Nt)
        elif side == "start":
            base_idx = start_at if start_at is not None else 0
            sl = slice(base_idx, min(Nt, base_idx + buf))
        else:
            raise ValueError(f"side must be 'start' or 'end', got {side!r}")

        block = np.asarray(da.isel(Time=sl).values, dtype=float)  # (buf, nCells)

        xtime_strs = None
        if "xtime" in ds.variables:
            try:
                xt_vals = ds["xtime"].isel(Time=sl).values
                xtime_strs = decode_xtime(xt_vals)
            except Exception as e:  # best-effort only
                print(f"[warn] failed to decode xtime in {path}: {e}")

        valid = compute_valid_mask(block, xtime_strs)
        valid_idx = np.nonzero(valid)[0]
        if valid_idx.size == 0:
            raise RuntimeError(
                f"No valid (non-garbage) timesteps found in the {side} window of {path} "
                f"(buffer={buf} months). Try a larger --n-months / check the file."
            )

        if side == "end":
            take = valid_idx[-n_months:] if valid_idx.size >= n_months else valid_idx
        else:
            take = valid_idx[:n_months] if valid_idx.size >= n_months else valid_idx

        if take.size < n_months:
            print(f"[warn] only {take.size}/{n_months} valid months available on the "
                  f"{side} side of {path} (rest were garbage/blank within the buffer window)")

        sub = block[take]
        xt_sub = [xtime_strs[i] for i in take] if xtime_strs is not None else None
        abs_first = int(sl.start + take[0])
        abs_last = int(sl.start + take[-1])
        return sub, xt_sub, abs_first, abs_last, Nt


def load_mesh_area(mesh_path: str, area_var: str, nCells_expected: int) -> "np.ndarray | None":
    import xarray as xr

    with xr.open_dataset(mesh_path, decode_times=False) as md:
        if area_var not in md.variables:
            print(f"[warn] '{area_var}' not found in --mesh {mesh_path}; using UNWEIGHTED domain means")
            return None
        area = np.asarray(md[area_var].values, dtype=float)
    if area.shape[0] != nCells_expected:
        print(f"[warn] --mesh {area_var} length {area.shape[0]} != forcing nCells {nCells_expected}; "
              f"using UNWEIGHTED domain means")
        return None
    return area


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------

def run_seam_check(base_dir, ext_dir, forcing_name, var, n_months, buffer_months,
                    mesh_path, area_var, seam_year=None, ext_start_index=None):
    base_path = os.path.join(base_dir, forcing_name)
    ext_path = os.path.join(ext_dir, forcing_name)
    for tag, p in (("--base-dir", base_path), ("--ext-dir", ext_path)):
        if not os.path.isfile(p):
            sys.exit(f"not found ({tag}): {p}")

    print(f"base forcing: {base_path}")
    print(f"ext  forcing: {ext_path}")
    print(f"var={var}  n_months={n_months}  buffer_months={buffer_months}\n")

    base_block, base_xt, base_first, base_last, base_Nt = load_edge_block(
        base_path, var, "end", n_months, buffer_months)

    # Align the ext window to the seam DATE, not to Time=0 of the ext file -- the ext
    # file may be the full multi-century generator series (Time=0 labeled year 2000
    # even though it's used to force the post-2300 extension), so blindly taking the
    # first n_months would compare the wrong calendar period entirely.
    if ext_start_index is not None:
        start_idx = ext_start_index
        print(f"ext start index: {start_idx} (manual --ext-start-index override)\n")
    else:
        if seam_year is not None:
            target = seam_year
            print(f"seam target: year {target:.4f} (manual --seam-year override)")
        elif base_xt:
            base_end_ym = parse_year_month(base_xt[-1])
            if base_end_ym is None:
                sys.exit(f"could not parse base's last xtime {base_xt[-1]!r} to auto-detect the "
                          f"seam date; pass --seam-year or --ext-start-index explicitly")
            target = base_end_ym + 1.0 / 12.0  # the month right after the base run ends
            print(f"seam target: year {target:.4f} (auto: base's last month {base_xt[-1]} + 1)")
        else:
            sys.exit("base file has no xtime, so the seam date can't be auto-detected; "
                      "pass --seam-year or --ext-start-index explicitly")
        start_idx = find_start_index_after(ext_path, target)
        print(f"ext start index: {start_idx} (first ext timestep >= year {target:.4f})\n")

    ext_block, ext_xt, ext_first, ext_last, ext_Nt = load_edge_block(
        ext_path, var, "start", n_months, buffer_months, start_at=start_idx)

    nCells_base, nCells_ext = base_block.shape[1], ext_block.shape[1]
    if nCells_base != nCells_ext:
        sys.exit(f"nCells mismatch: base file has {nCells_base}, ext file has {nCells_ext} "
                  f"-- are these the same mesh?")
    nCells = nCells_base

    print(f"base: took months [Time={base_first}..{base_last}] of {base_Nt} total "
          f"({base_block.shape[0]} valid months)"
          + (f", xtime {base_xt[0]}..{base_xt[-1]}" if base_xt else ""))
    print(f"ext:  took months [Time={ext_first}..{ext_last}] of {ext_Nt} total "
          f"({ext_block.shape[0]} valid months)"
          + (f", xtime {ext_xt[0]}..{ext_xt[-1]}" if ext_xt else ""))
    print(f"nCells = {nCells}\n")

    area = None
    if mesh_path:
        if not os.path.isfile(mesh_path):
            print(f"[warn] --mesh {mesh_path} not found; using UNWEIGHTED domain means")
        else:
            area = load_mesh_area(mesh_path, area_var, nCells)

    # ---- Test 1: domain-mean each side ----
    base_domain_ts = domain_mean(base_block, area)   # (m_base,)
    ext_domain_ts = domain_mean(ext_block, area)     # (m_ext,)
    base_mean = float(np.nanmean(base_domain_ts))
    ext_mean = float(np.nanmean(ext_domain_ts))
    base_std = float(np.nanstd(base_domain_ts))
    ext_std = float(np.nanstd(ext_domain_ts))
    opposite_sign = (
        np.sign(base_mean) != 0 and np.sign(ext_mean) != 0
        and np.sign(base_mean) != np.sign(ext_mean)
    )

    print("=== Test 1: domain-mean magnitude+sign each side ===")
    print(f"  base (last {base_block.shape[0]} mo):  mean={base_mean:+.6e}  std={base_std:.3e}")
    print(f"  ext  (first {ext_block.shape[0]} mo):  mean={ext_mean:+.6e}  std={ext_std:.3e}")
    print(f"  {'*** OPPOSITE SIGN ***' if opposite_sign else 'same sign' if base_mean and ext_mean else 'sign check inconclusive (near-zero mean)'}\n")

    # ---- Test 2: per-cell spatial-pattern correlation across the seam ----
    base_cellmean = np.nanmean(base_block, axis=0)   # (nCells,)
    ext_cellmean = np.nanmean(ext_block, axis=0)     # (nCells,)
    r = pearson_r(base_cellmean, ext_cellmean)

    print("=== Test 2: per-cell spatial-pattern correlation across the seam ===")
    print(f"  Pearson r (base time-mean field vs ext time-mean field, per cell) = {r:+.4f}")
    if np.isnan(r):
        print("  -> UNDETERMINED (insufficient variance/overlap)")
    elif r < -0.5:
        print("  -> SIGN-INVERTED (spatial pattern mirrored across the seam)")
    elif r > 0.5:
        print("  -> consistent (same spatial pattern both sides)")
    else:
        print("  -> decorrelated / investigate")
    print()

    # ---- Test 3: continuity jump ----
    last_base_val = float(base_domain_ts[-1])
    first_ext_val = float(ext_domain_ts[0])
    jump_abs = first_ext_val - last_base_val
    jump_frac = jump_abs / base_std if base_std > 0 else float("nan")

    print("=== Test 3: continuity (domain-mean jump at the seam) ===")
    print(f"  base last valid month value = {last_base_val:+.6e}"
          + (f"  (xtime {base_xt[-1]})" if base_xt else ""))
    print(f"  ext first valid month value = {first_ext_val:+.6e}"
          + (f"  (xtime {ext_xt[0]})" if ext_xt else ""))
    print(f"  jump = {jump_abs:+.6e}  ({jump_frac:+.2f} x base-side std)" if np.isfinite(jump_frac)
          else f"  jump = {jump_abs:+.6e}  (base-side std is 0; fraction undefined)")
    print()

    verdict = classify_verdict(base_mean, ext_mean, r, jump_frac)
    print(f"VERDICT: {verdict}")
    return dict(base_mean=base_mean, ext_mean=ext_mean, r=r, jump_abs=jump_abs,
                jump_frac=jump_frac, verdict=verdict)


# ----------------------------------------------------------------------------
# Self-test: exercises the correlation/verdict logic on synthetic arrays only,
# no forcing files required. Run with `--self-test`.
# ----------------------------------------------------------------------------

def _selftest() -> bool:
    rng = np.random.default_rng(0)
    n_months, nCells = 24, 500

    # Shared spatial pattern + independent monthly noise, so the time-mean per-cell
    # field is dominated by the pattern (as it would be for a real seasonal/variability
    # forcing field averaged over enough months).
    pattern = rng.normal(size=nCells)

    def make_block(sign, noise_scale=0.05):
        noise = rng.normal(scale=noise_scale, size=(n_months, nCells))
        return sign * pattern[None, :] + noise

    ok = True

    # Case 1: ext = -base pattern -> expect r ~ -1 (sign-inverted)
    base_block = make_block(+1.0)
    ext_block_inv = make_block(-1.0)
    r_inv = pearson_r(np.nanmean(base_block, axis=0), np.nanmean(ext_block_inv, axis=0))
    print(f"[self-test] inverted case: r = {r_inv:+.4f} (expect ~ -1)")
    if not (r_inv < -0.9):
        print("  FAIL: expected r < -0.9")
        ok = False
    verdict_inv = classify_verdict(1.0, -1.0, r_inv, 0.0)
    print(f"[self-test] inverted case verdict: {verdict_inv!r}")
    if "SIGN-INVERTED" not in verdict_inv:
        print("  FAIL: expected verdict to contain 'SIGN-INVERTED'")
        ok = False

    # Case 2: ext = +base pattern -> expect r ~ +1 (consistent)
    ext_block_same = make_block(+1.0)
    r_same = pearson_r(np.nanmean(base_block, axis=0), np.nanmean(ext_block_same, axis=0))
    print(f"[self-test] consistent case: r = {r_same:+.4f} (expect ~ +1)")
    if not (r_same > 0.9):
        print("  FAIL: expected r > 0.9")
        ok = False
    verdict_same = classify_verdict(1.0, 1.0, r_same, 0.1)
    print(f"[self-test] consistent case verdict: {verdict_same!r}")
    if "consistent" not in verdict_same:
        print("  FAIL: expected verdict to contain 'consistent'")
        ok = False

    # Case 3: garbage-timestep filtering -- an all-zero trailing row must be dropped.
    block = rng.normal(size=(5, 10))
    block[-1, :] = 0.0
    mask = compute_valid_mask(block, xtime_strs=None)
    print(f"[self-test] garbage-filter mask: {mask.tolist()} (expect last entry False)")
    if mask[-1] or not mask[:-1].all():
        print("  FAIL: expected all-zero trailing timestep to be masked out, others kept")
        ok = False

    print(f"\n[self-test] {'PASSED' if ok else 'FAILED'}")
    return ok


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir",
                     help="base-run member directory (e.g. data/MALI/ENSEMBLES/SSP585/SSP585_00)")
    ap.add_argument("--ext-dir",
                     help="extension-run member directory "
                          "(e.g. data/MALI/ENSEMBLES/SSP585-EXT/SSP585-EXT_00)")
    ap.add_argument("--var", default="floatingBasalMassBalAdjustment",
                     help="(Time, nCells) forcing variable to check (default: %(default)s)")
    ap.add_argument("--forcing-name", default="AIS_4to20km_r01_20220907_AISLENS-Forcing_0.nc",
                     help="forcing filename inside each member dir (default: %(default)s)")
    ap.add_argument("--n-months", type=int, default=240,
                     help="months on EACH side of the seam to compare (default: %(default)s "
                          "= 20 years monthly)")
    ap.add_argument("--buffer-months", type=int, default=None,
                     help="extra months to read beyond --n-months so trailing/leading garbage "
                          "timesteps can be skipped without shrinking the requested window "
                          "(default: --n-months + 24)")
    ap.add_argument("--mesh", default=None,
                     help="optional MPAS mesh/restart file carrying --area-var for "
                          "area-weighted domain means; if omitted, domain means are unweighted")
    ap.add_argument("--area-var", default="areaCell",
                     help="cell-area variable name in --mesh (default: %(default)s)")
    ap.add_argument("--seam-year", type=float, default=None,
                     help="fractional year (e.g. 2300.0) the ext window should start at or after; "
                          "default: auto-detected as (base's last valid month's year) + 1 month. "
                          "The ext file's own xtime is used to locate the matching index -- its "
                          "Time=0 is NOT assumed to be the seam (it may be a full multi-century "
                          "series whose Time=0 is calendar year 2000).")
    ap.add_argument("--ext-start-index", type=int, default=None,
                     help="manually specify the absolute Time index in the ext file to start "
                          "from, bypassing xtime-based seam-date lookup entirely (use if the ext "
                          "file lacks xtime or --seam-year auto-detection is unreliable)")
    ap.add_argument("--self-test", action="store_true",
                     help="run the built-in synthetic self-test of the correlation/verdict "
                          "logic and exit (no forcing files needed)")
    a = ap.parse_args()

    if a.self_test:
        ok = _selftest()
        sys.exit(0 if ok else 1)

    if not a.base_dir or not a.ext_dir:
        ap.error("--base-dir and --ext-dir are required (unless using --self-test)")

    buffer_months = a.buffer_months if a.buffer_months is not None else a.n_months + 24

    run_seam_check(
        base_dir=a.base_dir, ext_dir=a.ext_dir, forcing_name=a.forcing_name, var=a.var,
        n_months=a.n_months, buffer_months=buffer_months, mesh_path=a.mesh, area_var=a.area_var,
        seam_year=a.seam_year, ext_start_index=a.ext_start_index,
    )


if __name__ == "__main__":
    main()
