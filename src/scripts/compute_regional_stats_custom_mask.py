#!/usr/bin/env python3
"""
compute_regional_stats_custom_mask.py -- recompute regionalStats under ANY mask.

WHY
    CTRL and SSP585-3X were run with the 133-region draft-dependent mask; SSP126,
    SSP585 and varScaled10x with the 16 ISMIP6 basins. That makes region-level
    cross-ensemble comparison impossible FROM THE EXISTING regionalStats.nc -- but
    only from those files. The per-cell output is still on disk, so the regional
    sums can simply be recomputed under a common mask for every ensemble.

WHAT IS COMPUTED, per region r and output time t

    iceVolume(r,t) = sum_{c in r} h(c,t) * A(c)

    VAF(r,t)       = sum_{c in r} max(0, h(c,t) - h_f(c)) * A(c)
    with the flotation thickness   h_f(c) = (rho_o/rho_i) * max(0, -b(c))

    The bed b is taken as FIXED. That is exact here, not an approximation:
    config_uplift_method = 'none' in these runs, so there is no GIA and the bed
    never moves.

    A(c) = areaCell from the mesh file.

The 133-region mask contains OVERLAPPING regions (nested aggregates over IMBIE
basins over individual shelves). That is harmless here because each region is summed
independently -- but it does mean the regions must never be summed with each other.

VALIDATE FIRST. --validate recomputes under the ensemble's OWN mask and compares to
its existing regionalStats.nc. Agreement confirms the flotation treatment matches
MALI's. Expect small differences: MALI applies a sub-grid grounding-line scheme
(config_use_glp = YES) that partially grounds edge cells, which a pure cell-centre
threshold cannot reproduce. A few percent at the grounding line is expected; tens of
percent means something is wrong.

Usage
    # validate against MALI's own numbers first
    python3 compute_regional_stats_custom_mask.py --member <dir> --mesh <mesh.nc> \
        --mask <ismip6_mask.nc> --validate

    # then produce the common-mask series
    python3 compute_regional_stats_custom_mask.py --member <dir> --mesh <mesh.nc> \
        --mask aislens_draftDepen_regionMasks.nc --out <member>_shelfStats.nc
"""
from __future__ import annotations
import os, sys, glob, argparse
import numpy as np
import netCDF4

RHO_I, RHO_O = 910.0, 1028.0


def load_mask(path):
    d = netCDF4.Dataset(path)
    m = np.asarray(d["regionCellMasks"][:])
    names = []
    if "regionNames" in d.variables:
        names = [b"".join(r).decode(errors="ignore").strip()
                 for r in np.asarray(d["regionNames"][:]).astype("S1")]
    d.close()
    if m.ndim != 2:
        raise SystemExit("regionCellMasks must be 2-D")
    # orient as (nRegions, nCells): nCells is the much larger axis
    if m.shape[0] > m.shape[1]:
        m = m.T
    return m.astype(np.float32), names


def state_files(member_dir):
    for sub in ("output", "outputs", ""):
        f = sorted(glob.glob(os.path.join(member_dir, sub, "output_state_*.nc")))
        if f:
            return f
    return []


def compute(member_dir, mesh, mask, chunk_report=True):
    d = netCDF4.Dataset(mesh)
    area = np.asarray(d["areaCell"][:], dtype=np.float64).ravel()
    bed = np.asarray(d["bedTopography"][:], dtype=np.float64).ravel()
    d.close()
    if bed.ndim > 1:
        bed = bed[0]
    hf = (RHO_O / RHO_I) * np.maximum(0.0, -bed)          # flotation thickness

    files = state_files(member_dir)
    if not files:
        raise FileNotFoundError(f"no output_state files under {member_dir}")

    W = mask * area[None, :]                               # (nRegions, nCells) area-weighted
    yrs, VOL, VAF = [], [], []
    for f in files:
        ds = netCDF4.Dataset(f)
        if "thickness" not in ds.variables:
            ds.close(); continue
        t = np.asarray(ds["daysSinceStart"][:], dtype=np.float64) / 365.0
        h = np.asarray(ds["thickness"][:], dtype=np.float64)     # (Time, nCells)
        ds.close()
        if h.ndim == 1:
            h = h[None, :]
        # region sums as one matvec each: (Time,nCells) @ (nCells,nRegions)
        VOL.append(h @ W.T)
        VAF.append(np.maximum(0.0, h - hf[None, :]) @ W.T)
        yrs.append(t)
        if chunk_report:
            print(f"    {os.path.basename(f)}  {h.shape[0]} steps", flush=True)
    if not VOL:
        raise RuntimeError("no thickness data found")
    yrs = np.concatenate(yrs)
    VOL = np.concatenate(VOL, axis=0)
    VAF = np.concatenate(VAF, axis=0)
    o = np.argsort(yrs)
    return yrs[o], VOL[o], VAF[o]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", required=True, help="member directory")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate", action="store_true",
                    help="compare against the member's own regionalStats.nc")
    a = ap.parse_args()

    mask, names = load_mask(a.mask)
    print(f"mask: {mask.shape[0]} regions x {mask.shape[1]} cells")
    yr, vol, vaf = compute(a.member, a.mesh, mask)
    print(f"computed {vol.shape[0]} timesteps x {vol.shape[1]} regions")

    if a.validate:
        # On the cluster regionalStats.nc sits in <member>/output/; local copies
        # have it flattened to <member>/. Check both.
        f = next((c for c in (os.path.join(a.member, "output", "regionalStats.nc"),
                              os.path.join(a.member, "regionalStats.nc"))
                  if os.path.exists(c)), None)
        if f is None:
            sys.exit(f"no regionalStats.nc under {a.member} or {a.member}/output")
        print(f"validating against {f}")
        d = netCDF4.Dataset(f)
        if len(d.dimensions["nRegions"]) != mask.shape[0]:
            d.close(); sys.exit("mask nRegions differs from regionalStats -- "
                                "validate against the ensemble's OWN mask")
        ry = np.asarray(d["daysSinceStart"][:], float) / 365.0
        rv = np.asarray(d["regionalVolumeAboveFloatation"][:], float)
        ri = np.asarray(d["regionalIceVolume"][:], float)
        d.close()
        print(f"\n{'region':>7} {'MALI VAF':>13} {'recomputed':>13} {'rel diff':>9}"
              f" | {'MALI vol':>13} {'recomputed':>13} {'rel diff':>9}")
        j = min(len(ry) - 1, len(yr) - 1)
        k = int(np.argmin(np.abs(yr - ry[j])))
        dv, di = [], []
        for r in range(mask.shape[0]):
            a1, b1 = rv[j, r], vaf[k, r]
            a2, b2 = ri[j, r], vol[k, r]
            e1 = abs(b1 - a1) / max(abs(a1), 1e-9); e2 = abs(b2 - a2) / max(abs(a2), 1e-9)
            dv.append(e1); di.append(e2)
            if r < 16:
                print(f"{r:>7} {a1:13.5e} {b1:13.5e} {100*e1:8.2f}%"
                      f" | {a2:13.5e} {b2:13.5e} {100*e2:8.2f}%")
        print(f"\n  median rel diff  VAF {100*np.median(dv):.2f} %   "
              f"iceVolume {100*np.median(di):.2f} %   (at yr {ry[j]:.0f})")
        print("  iceVolume should agree to <0.1 % -- it is a plain sum with no")
        print("  grounding-line subtlety. VAF may differ by a few % because MALI")
        print("  uses a sub-grid grounding-line parameterisation (config_use_glp=YES).")
        return

    if not a.out:
        sys.exit("--out required unless --validate")
    ds = netCDF4.Dataset(a.out, "w")
    ds.createDimension("Time", None); ds.createDimension("nRegions", vol.shape[1])
    ds.createVariable("daysSinceStart", "f8", ("Time",))[:] = yr * 365.0
    ds.createVariable("regionalIceVolume", "f8", ("Time", "nRegions"), zlib=True)[:] = vol
    ds.createVariable("regionalVolumeAboveFloatation", "f8", ("Time", "nRegions"),
                      zlib=True)[:] = vaf
    ds.setncattr("source", "compute_regional_stats_custom_mask.py")
    ds.setncattr("mask_file", os.path.basename(a.mask))
    ds.setncattr("member", os.path.basename(os.path.normpath(a.member)))
    ds.setncattr("note", "VAF from fixed bed (config_uplift_method=none); no sub-grid GL scheme")
    if names:
        ds.setncattr("region_names", "|".join(names))
    ds.close()
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
