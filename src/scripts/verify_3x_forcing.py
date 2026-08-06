#!/usr/bin/env python3
"""
verify_3x_forcing.py — check before launching the SSP585_meltScaled3x
10-member ensemble (see src/pace-jobs/aislens/make_3x_melt_forcing.sbatch).

Confirms, for ONE member at a time, that the scaled forcing file actually
satisfies  F_3x_i = F_i + (TREND_FACTOR-1)*T  and nothing else changed:

  (a) TREND CHECK: the domain-mean trend growth (late-window minus early-window
      domain mean) in the scaled file is ~TREND_FACTOR x that of the original.
  (b) VARIABILITY-UNCHANGED CHECK (the key assertion): at a handful of sampled
      time steps, (scaled - orig) must equal (TREND_FACTOR-1)*T almost exactly,
      per cell. This is the direct algebraic consequence of ncflint's weighted
      sum and is the strongest test that V_i (the variability) was preserved.
  (c) Prints the domain-mean melt-adjustment anomaly at a few epochs, orig vs
      scaled, for a human sanity check.

MEMORY SAFETY: files are Time x nCells (~3600 x ~385000, 11-34 GB each). This
script NEVER reads a full Time x nCells array. It only ever reads:
  - single-time-step nCells slices (a few MB each), for a small, evenly spaced
    sample of time indices (--n-check-times, default 6), and
  - Dimension/attribute metadata (cheap).

CRITICAL CAVEATS (see make_3x_melt_forcing.sbatch header for full discussion --
repeated here because this is the gate people actually read before launching):
  1. TREND_FACTOR x the ADJUSTMENT is NOT TREND_FACTOR x TOTAL melt. Total melt
     also includes the (unscaled) background from the draft-dependent
     parameterization. A nominal 3x adjustment has previously been estimated
     to yield roughly ~1.9x TOTAL melt, not 3x -- MEASURE realized
     avgSubshelfMelt from a single calibration MALI run; do not assume.
  2. Melt saturation (truncation at available floating ice) will bite harder
     under a larger mean forcing than it did in the 10x-variability ensemble
     (which only realized ~6.3-7.6x, not 10x). Expect the realized mean-melt
     ratio to again fall short of the nominal TREND_FACTOR.
  3. Trend file and member file Time dimensions must match exactly (expected
     3600 monthly steps) -- this script hard-errors otherwise.

Run --self-test (no files needed) to sanity-check the checking logic itself
on synthetic in-memory arrays before trusting it on real HPC files.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


CAVEATS = """
================================================================================
CRITICAL CAVEATS -- read before trusting this verification as a "3x" claim:
  1. TREND_FACTOR scales the ADJUSTMENT only, not total melt. Total melt =
     background (draft-dependent parameterization, untouched) + adjustment
     (scaled here). Prior estimate: 3x adjustment ~= 1.9x TOTAL melt at yr300
     (B~2.1 m/yr, adjustment~2.9 m/yr of a ~5 m/yr total). CALIBRATE with one
     MALI run and measure the realized avgSubshelfMelt ratio.
  2. Melt saturation (floating-ice truncation) will be STRONGER here than in
     the 10x-variability ensemble, which only realized ~6.3-7.6x amplitude.
     Expect the realized mean-melt ratio to undershoot the nominal factor.
  3. Trend/member Time dimensions must match exactly -- checked below.
================================================================================
"""


# --------------------------------------------------------------------------
# Core, array-based check functions (used by BOTH real-file mode and
# --self-test, so the exact same logic is exercised in both).
# --------------------------------------------------------------------------

def check_variability_unchanged(orig_slices, scaled_slices, trend_slices, w,
                                 atol=1e-6, rtol=1e-6):
    """(b) The key assertion: scaled - orig == (w-1)*trend, per sampled time step.

    Each *_slices is a list of 1-D arrays (nCells,), one per sampled time index,
    already read from disk (or synthetic). NaNs (land / non-floating cells) are
    ignored in the same position across all three; a NaN-pattern mismatch is
    itself reported as a failure.

    Returns (ok: bool, report: str).
    """
    wminus1 = w - 1.0
    lines = []
    ok = True
    for k, (o, s, t) in enumerate(zip(orig_slices, scaled_slices, trend_slices)):
        o = np.asarray(o, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)

        nan_o, nan_s = np.isnan(o), np.isnan(s)
        if not np.array_equal(nan_o, nan_s):
            n_mismatch = int(np.sum(nan_o != nan_s))
            lines.append(f"  [t={k}] FAIL: NaN pattern differs between orig and "
                         f"scaled at {n_mismatch} cells")
            ok = False
            continue

        valid = ~nan_o
        expected = wminus1 * t[valid]
        actual = s[valid] - o[valid]
        diff = actual - expected
        max_abs = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
        # scale-aware tolerance: atol floor + rtol * typical magnitude of the trend term
        typical = float(np.nanmax(np.abs(expected))) if expected.size else 0.0
        tol = atol + rtol * max(typical, 1.0)
        step_ok = max_abs <= tol
        ok = ok and step_ok
        status = "OK  " if step_ok else "FAIL"
        lines.append(f"  [t={k}] {status} max|((scaled-orig) - (w-1)*T)| = "
                     f"{max_abs:.3e}  (tol={tol:.3e}, n_valid_cells={valid.sum()})")
    return ok, "\n".join(lines)


def domain_mean(slice_1d):
    return float(np.nanmean(np.asarray(slice_1d, dtype=np.float64)))


def check_trend_ratio(orig_early, orig_late, scaled_early, scaled_late, w,
                       rtol=0.15):
    """(a) domain-mean trend GROWTH (late - early) in scaled should be ~w x orig's.

    Uses a generous rtol by default: single-epoch domain means still carry
    residual variability (they are NOT annual averages), so this is a coarse
    sanity check, not a precision test -- (b) is the precision test.
    Returns (ok, ratio, report).
    """
    d_orig = domain_mean(orig_late) - domain_mean(orig_early)
    d_scaled = domain_mean(scaled_late) - domain_mean(scaled_early)
    if d_orig == 0:
        return False, float("nan"), ("  trend-ratio check: orig domain-mean growth "
                                      "is exactly 0 -- cannot form a ratio "
                                      "(degenerate/synthetic input?)")
    ratio = d_scaled / d_orig
    ok = abs(ratio - w) <= rtol * w
    status = "OK  " if ok else "FAIL"
    report = (f"  {status} domain-mean trend growth ratio (scaled/orig) = "
              f"{ratio:.3f}  (target w={w:.3f}, rtol={rtol:.2f})\n"
              f"        orig growth={d_orig:.6g}, scaled growth={d_scaled:.6g}")
    return ok, ratio, report


# --------------------------------------------------------------------------
# File I/O (memory-safe: single-time-step slices only)
# --------------------------------------------------------------------------

def open_dataset(path):
    import netCDF4
    return netCDF4.Dataset(path)


def time_length(ds):
    return len(ds.dimensions["Time"])


def read_slice(ds, var, tidx):
    """Read ONE time step, all nCells, as a 1-D numpy array. Never reads full Time."""
    v = ds.variables[var]
    arr = np.asarray(v[tidx, :])
    return arr


def check_time_dims_match(paths_and_labels):
    """paths_and_labels: list of (netCDF4.Dataset, label). Hard-errors (returns
    (False, msg)) if any Time lengths differ."""
    lengths = {}
    for ds, label in paths_and_labels:
        lengths[label] = time_length(ds)
    uniq = set(lengths.values())
    if len(uniq) > 1:
        msg = "Time dimension mismatch: " + ", ".join(f"{k}={v}" for k, v in lengths.items())
        return False, msg
    return True, f"Time dims match: {lengths}"


def pick_epoch_indices(ntime, n_check_times):
    """Evenly spaced indices across the record, avoiding the very first/last
    step (edge artifacts), and always including one near-start / near-end pair
    for the trend-growth check."""
    n_check_times = max(2, n_check_times)
    lo = max(1, int(0.05 * ntime))
    hi = min(ntime - 2, int(0.95 * ntime))
    idxs = np.linspace(lo, hi, n_check_times).astype(int)
    return sorted(set(idxs.tolist()))


# --------------------------------------------------------------------------
# Real-file verification
# --------------------------------------------------------------------------

def run_file_verification(args):
    print(CAVEATS)

    ds_orig = open_dataset(args.orig)
    ds_scaled = open_dataset(args.scaled)
    ds_trend = open_dataset(args.trend)

    ok_dims, msg_dims = check_time_dims_match([
        (ds_orig, "orig"), (ds_scaled, "scaled"), (ds_trend, "trend"),
    ])
    print(msg_dims)
    if not ok_dims:
        sys.exit("HARD ERROR: Time dimensions must match exactly (expected 3600 "
                 "monthly steps). Refusing to proceed with a mismatched trend/member "
                 "pair -- see CAVEAT 3.")

    ntime = time_length(ds_orig)
    if args.var not in ds_orig.variables or args.var not in ds_scaled.variables \
            or args.var not in ds_trend.variables:
        sys.exit(f"HARD ERROR: variable {args.var!r} not present in all three files "
                 f"(orig has it: {args.var in ds_orig.variables}, "
                 f"scaled: {args.var in ds_scaled.variables}, "
                 f"trend: {args.var in ds_trend.variables})")

    idxs = pick_epoch_indices(ntime, args.n_check_times)
    print(f"\nSampling {len(idxs)} time indices out of {ntime}: {idxs}")

    orig_slices = [read_slice(ds_orig, args.var, t) for t in idxs]
    scaled_slices = [read_slice(ds_scaled, args.var, t) for t in idxs]
    trend_slices = [read_slice(ds_trend, args.var, t) for t in idxs]

    # (b) key assertion: variability unchanged
    print(f"\n(b) VARIABILITY-UNCHANGED CHECK: scaled - orig ?= "
          f"(TREND_FACTOR-1)*T  [TREND_FACTOR={args.trend_factor}]")
    ok_b, report_b = check_variability_unchanged(
        orig_slices, scaled_slices, trend_slices, args.trend_factor,
        atol=args.atol, rtol=args.rtol)
    print(report_b)

    # (a) trend ratio, using first and last sampled indices as early/late
    print(f"\n(a) TREND-RATIO CHECK (coarse, single-epoch domain means):")
    ok_a, ratio, report_a = check_trend_ratio(
        orig_slices[0], orig_slices[-1], scaled_slices[0], scaled_slices[-1],
        args.trend_factor)
    print(report_a)

    # (c) print domain-mean melt anomaly at each sampled epoch
    print(f"\n(c) Domain-mean {args.var} at sampled epochs (orig vs scaled):")
    print(f"  {'t_idx':>6}  {'orig':>14}  {'scaled':>14}  {'scaled/orig':>12}")
    for t, o, s in zip(idxs, orig_slices, scaled_slices):
        mo, ms = domain_mean(o), domain_mean(s)
        ratio_str = f"{ms / mo:.3f}" if mo != 0 else "n/a"
        print(f"  {t:6d}  {mo:14.6g}  {ms:14.6g}  {ratio_str:>12}")

    print()
    if ok_a and ok_b:
        print("RESULT: PASS -- scaled file is consistent with F_i + (w-1)*T; "
              "variability preserved.")
    else:
        print("RESULT: FAIL -- see checks above. Do NOT launch the 10-member "
              "ensemble until this passes.")
        sys.exit(1)

    print("\nReminder: PASSING this check confirms the ARITHMETIC (scaled file is "
          "correctly F_i + (w-1)*T). It does NOT confirm the realized total-melt "
          "ratio in MALI output -- run the calibration step (CAVEAT 1) separately.")


# --------------------------------------------------------------------------
# Self-test on synthetic data (no files needed)
# --------------------------------------------------------------------------

def self_test():
    print("Running --self-test on synthetic in-memory arrays (no files needed)...\n")
    rng = np.random.default_rng(42)
    ntime, ncells = 40, 2000
    w = 3.0

    t_axis = np.linspace(0.0, 1.0, ntime)
    # per-cell trend slope varies spatially but is deterministic/shared across "members"
    cell_slope = rng.uniform(0.5, 2.0, size=ncells)
    T = np.outer(t_axis, cell_slope)                       # (ntime, ncells)

    V = rng.normal(scale=0.3, size=(ntime, ncells))         # zero-mean variability
    # sprinkle some NaNs (non-floating cells) consistently across orig/scaled
    land_mask = rng.random(ncells) < 0.05
    V[:, land_mask] = np.nan
    T_masked = T.copy()
    T_masked[:, land_mask] = np.nan

    F_orig = T + V                                           # F_i = T + V_i
    F_scaled_good = F_orig + (w - 1.0) * T                    # correct construction
    F_scaled_bad = F_orig * w                                 # WRONG: also scales V_i

    idxs = pick_epoch_indices(ntime, 6)
    orig_slices = [F_orig[t, :] for t in idxs]
    trend_slices = [T_masked[t, :] for t in idxs]

    print("--- Positive case: correctly-scaled file (F_orig + (w-1)*T) ---")
    scaled_good_slices = [F_scaled_good[t, :] for t in idxs]
    ok_b_good, report_b_good = check_variability_unchanged(
        orig_slices, scaled_good_slices, trend_slices, w)
    print(report_b_good)
    ok_a_good, ratio_good, report_a_good = check_trend_ratio(
        orig_slices[0], orig_slices[-1], scaled_good_slices[0], scaled_good_slices[-1], w)
    print(report_a_good)
    assert ok_b_good, "self-test FAILED: variability-unchanged check rejected a correctly-scaled file"
    assert ok_a_good, "self-test FAILED: trend-ratio check rejected a correctly-scaled file"
    print("PASS: correctly-scaled synthetic file accepted by both checks.\n")

    print("--- Negative case: incorrectly-scaled file (F_orig * w, scales V too) ---")
    scaled_bad_slices = [F_scaled_bad[t, :] for t in idxs]
    ok_b_bad, report_b_bad = check_variability_unchanged(
        orig_slices, scaled_bad_slices, trend_slices, w)
    print(report_b_bad)
    assert not ok_b_bad, ("self-test FAILED: variability-unchanged check did NOT "
                          "catch a file where variability was also scaled")
    print("PASS: incorrectly-scaled synthetic file correctly REJECTED by the "
          "variability-unchanged check.\n")

    print("--- Negative case: mismatched Time dims should hard-error path ---")
    ok_dims, msg = check_time_dims_match([
        (_FakeDims(40), "orig"), (_FakeDims(36), "trend"),
    ])
    print("  " + msg)
    assert not ok_dims, "self-test FAILED: Time-dim mismatch was not detected"
    print("PASS: Time-dimension mismatch correctly detected.\n")

    print("ALL SELF-TESTS PASSED.")


class _FakeDims:
    """Minimal stand-in for netCDF4.Dataset exposing just .dimensions[...] len(),
    used only to exercise check_time_dims_match without real files."""
    def __init__(self, n):
        self.dimensions = {"Time": range(n)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig", help="original (1x) member forcing file")
    ap.add_argument("--scaled", help="scaled (Nx-trend) member forcing file to verify")
    ap.add_argument("--trend", help="deterministic trend file used to build --scaled "
                                    "(Path A or Path B from make_3x_melt_forcing.sbatch)")
    ap.add_argument("--var", default="floatingBasalMassBalAdjustment")
    ap.add_argument("--trend-factor", type=float, default=3.0,
                    help="TREND_FACTOR (w) used to build --scaled (default 3.0)")
    ap.add_argument("--n-check-times", type=int, default=6,
                    help="number of time indices to sample across the record "
                         "(default 6; each read is a single nCells slice, memory-safe)")
    ap.add_argument("--atol", type=float, default=1e-6,
                    help="absolute tolerance floor for the variability-unchanged check")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="relative tolerance (vs. typical |trend term|) for the "
                         "variability-unchanged check")
    ap.add_argument("--self-test", action="store_true",
                    help="run checks on synthetic in-memory data; no files needed")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    missing = [f"--{name}" for name, val in
               (("orig", args.orig), ("scaled", args.scaled), ("trend", args.trend))
               if not val]
    if missing:
        ap.error(f"missing required arguments: {', '.join(missing)} "
                 f"(or run with --self-test)")

    run_file_verification(args)


if __name__ == "__main__":
    main()
