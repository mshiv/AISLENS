#!/usr/bin/env python3
"""
spatial_ensemble_stats.py -- SUPERSEDED. Prefer the existing NCO pipeline.

    aislens_mali_ensemble_processing_fast_<scen>.sbatch  builds ensembleStats_<year>.nc
    (mean/min/max/std/range per cell) with ncra, then plots via plot_ensemble_maps.py
    and plot_ensemble_maps_ratio.py. That pipeline streams through NCO, is already
    tested at scale, and does grounding-line overlays. Use it.

    Keep this only if a per-cell statistic is needed that NCO cannot express.
    NOTE: the NCO pipeline computes POPULATION sigma (ddof=0); this script uses
    SAMPLE sigma (ddof=1), so the two differ by sqrt(N/(N-1)) -- 5.4% at N=10.

Original description follows.

spatial_ensemble_stats.py -- HPC-side reduction: (member x cell) -> per-cell statistics.

Run ONCE per ensemble on the cluster, copy the small output NetCDF to the laptop,
then plot with fig_spatial_ensemble_maps.py. Mirrors extract_spatial_summaries.py:
the ensemble of output_state files is ~TB; this output is ~20 MB.

WHY MAPS AND NOT REGIONS
    CTRL and SSP585-3X carry a 133-region mask while SSP126/SSP585/varScaled10x carry
    16 ISMIP6 regions, which locks the mean-forcing axis out of every regional
    comparison. Per-cell fields live on the shared MALI mesh, so they sidestep the
    mask split entirely and let all five ensembles be compared spatially.

QUANTITIES (per cell c, over members i)

  1. Cumulative thickness change to the target year:
         dH_i(c) = H_i(c, t) - H_i(c, t0)
     Referenced to each member's own t0, so every member starts at exactly 0 and all
     spread is generated during the run rather than inherited from the initial state.

  2. Secular thinning rate over a centred window of half-width w:
         dHdt_i(c) = [H_i(c, t+w) - H_i(c, t-w)] / (2w)
     Deliberately NOT MALI's instantaneous `dHdt` field: satellite dh/dt is a secular
     rate fitted over years, so a windowed difference is the like-for-like quantity to
     put beside Smith et al. (2020). At the window edges the calculation falls back to
     a one-sided difference rather than returning NaN.

  3. Applied sub-shelf melt (optional, --field floatingBasalMassBalApplied from the
     flux stream). This is the PER-CELL prescribed melt, free of the area-averaging
     that contaminates globalStats `avgSubshelfMelt`. It is the correct diagnostic for
     both the deterministic-twin gate and for a defensible amplitude exponent.

For each quantity the reduction is mean, sample sigma (ddof=1), min, max and N, all
computed per cell with np.nan* functions so a member missing that year is simply absent
from that cell's statistics rather than poisoning them.

  range = max - min is reported alongside sigma because at N=10 the range is often the
  more honest summary; for a Gaussian, E[range] ~ 3.08 sigma at N=10.

Usage
    python3 spatial_ensemble_stats.py \
        --root /path/to/ENSEMBLES --ensemble SSP585 \
        --years 100 200 300 --window 10 --out stats_SSP585.nc
"""
from __future__ import annotations
import os, sys, glob, argparse
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

MEMBER_GLOB = "*_[0-9][0-9]"


def member_dirs(root, ensemble):
    """Member directories, restart fragments excluded by the caller's year check."""
    return sorted(glob.glob(os.path.join(root, ensemble, MEMBER_GLOB)))


def load_years(member_dir, field, want_years, flux=False):
    """Return {year: (nCells,)} for the requested years, nearest available within 3 yr.

    Output is 5-yearly in practice (the streams file says 1 yr, but the production runs
    write every 5), so an exact year match cannot be assumed.
    """
    lo, hi = min(want_years) - 25, max(want_years) + 25
    if flux:
        # flux stream uses a different filename template
        files = sorted(glob.glob(os.path.join(member_dir, "output", "output_flux_all_timesteps_*.nc")))
        if not files:
            files = sorted(glob.glob(os.path.join(member_dir, "output_flux_all_timesteps_*.nc")))
        data, years = [], []
        for f in files:
            ds = xr.open_dataset(f, decode_times=False)
            if field not in ds:
                ds.close(); continue
            yr = ds["daysSinceStart"].values / 365.0
            v = ds[field].values
            if v.ndim == 3:
                v = v[:, :, 0]
            data.append(v); years.append(yr)
            ds.close()
        if not data:
            raise FileNotFoundError(f"{field} not found under {member_dir}")
        arr = np.concatenate(data, 0); yrs = np.concatenate(years)
    else:
        arr, yrs = sio.load_spatial_variable(member_dir, field, int(lo), int(hi))

    out = {}
    for y in want_years:
        j = int(np.argmin(np.abs(yrs - y)))
        if abs(yrs[j] - y) <= 3.0:
            out[y] = np.asarray(arr[j], dtype=np.float32)
    # t0 reference: earliest available record
    out["_t0"] = np.asarray(arr[int(np.argmin(yrs))], dtype=np.float32)
    out["_t0_year"] = float(np.min(yrs))
    return out


def reduce_stack(stack):
    """(member, nCells) -> per-cell mean, sigma, min, max, N. NaN-safe."""
    with np.errstate(invalid="ignore", all="ignore"):
        n = np.sum(np.isfinite(stack), axis=0).astype(np.int16)
        mean = np.nanmean(stack, axis=0)
        # ddof=1 needs >=2 members; np.nanstd has no ddof-aware NaN count, so do it by hand
        dev = stack - mean[None, :]
        ss = np.nansum(dev * dev, axis=0)
        sigma = np.where(n >= 2, np.sqrt(ss / np.maximum(n - 1, 1)), np.nan)
        vmin = np.nanmin(stack, axis=0)
        vmax = np.nanmax(stack, axis=0)
    mean[n < 1] = np.nan
    return mean, sigma, vmin, vmax, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="ENSEMBLES root (HPC path)")
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--field", default="thickness",
                    help="thickness (default) or floatingBasalMassBalApplied (implies --flux)")
    ap.add_argument("--flux", action="store_true", help="read from the output_flux stream")
    ap.add_argument("--years", type=float, nargs="+", default=[100, 200, 300])
    ap.add_argument("--window", type=float, default=10.0,
                    help="half-width (yr) of the centred window for the secular rate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    flux = a.flux or a.field.endswith("Applied")
    w = a.window
    # years needed: the targets themselves plus the window shoulders for the rate
    want = sorted({float(y) for y in a.years}
                  | {float(y) - w for y in a.years} | {float(y) + w for y in a.years})

    dirs = member_dirs(a.root, a.ensemble)
    if not dirs:
        sys.exit(f"no member directories under {a.root}/{a.ensemble}")

    per_member, names = [], []
    for d in dirs:
        try:
            per_member.append(load_years(d, a.field, want, flux=flux))
            names.append(os.path.basename(d))
            print(f"  loaded {os.path.basename(d)}")
        except Exception as e:
            print(f"  SKIP {os.path.basename(d)}: {e}")
    if len(per_member) < 2:
        sys.exit("need >= 2 members")

    nCells = per_member[0]["_t0"].size
    ds = xr.Dataset()
    ds.attrs.update(ensemble=a.ensemble, field=a.field, window_yr=w,
                    members=", ".join(names), n_members=len(names),
                    note="dH referenced to each member's own first record; "
                         "secular rate is a centred difference over +/- window_yr")

    for y in a.years:
        y = float(y)
        # ---- cumulative change since t0
        stack = np.full((len(per_member), nCells), np.nan, dtype=np.float32)
        for i, m in enumerate(per_member):
            if y in m:
                stack[i] = m[y] - m["_t0"]
        mean, sig, vmin, vmax, n = reduce_stack(stack)
        tag = f"{int(y)}"
        ds[f"dH_mean_{tag}"] = ("nCells", mean)
        ds[f"dH_sigma_{tag}"] = ("nCells", sig)
        ds[f"dH_range_{tag}"] = ("nCells", vmax - vmin)
        ds[f"dH_n_{tag}"] = ("nCells", n)
        # signal-to-noise: forced signal relative to internal spread
        with np.errstate(invalid="ignore", divide="ignore"):
            ds[f"dH_snr_{tag}"] = ("nCells", np.where(sig > 0, np.abs(mean) / sig, np.nan))

        # ---- secular rate over the centred window
        rstack = np.full((len(per_member), nCells), np.nan, dtype=np.float32)
        for i, m in enumerate(per_member):
            lo, hi = y - w, y + w
            if lo in m and hi in m:
                rstack[i] = (m[hi] - m[lo]) / (2 * w)
            elif y in m and lo in m:                      # one-sided fallback
                rstack[i] = (m[y] - m[lo]) / w
            elif y in m and hi in m:
                rstack[i] = (m[hi] - m[y]) / w
        rmean, rsig, rmin, rmax, rn = reduce_stack(rstack)
        ds[f"rate_mean_{tag}"] = ("nCells", rmean)
        ds[f"rate_sigma_{tag}"] = ("nCells", rsig)
        ds[f"rate_range_{tag}"] = ("nCells", rmax - rmin)
        ds[f"rate_n_{tag}"] = ("nCells", rn)
        print(f"  yr{tag}: N={int(np.nanmax(n))} cells={nCells} "
              f"median sigma(dH)={np.nanmedian(sig):.3f}")

    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(a.out, encoding=enc)
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
