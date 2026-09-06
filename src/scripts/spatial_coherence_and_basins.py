#!/usr/bin/env python3
"""
spatial_coherence_and_basins.py -- two analyses of the per-cell sigma fields.

(A) SPATIAL COHERENCE -- does greater forcing produce a LESS spatially organised
    response? This tests the hypothesis raised by the per-cell sigma ratio: the
    LOCAL ratio sigma_10x/sigma_1x is ~8.7 at yr100, yet the GLOBAL ratio is only
    3.71. Something discards the difference on the way from cell to continent.

    The test does not need per-member fields. Write continental change as an
    area-weighted sum over cells, S = sum_c w_c dh_c. Then

        Var(S) = sum_c sum_c' w_c w_c' Cov(dh_c, dh_c')

    Two limits bracket it:
        perfectly coherent cells :  sigma_S = sum_c w_c sigma_c        (call it T)
        the actual ensemble      :  sigma_S = the measured global sigma

    Their ratio is a COHERENCE FACTOR

        phi = sigma_S(global, measured) / T,     0 < phi <= 1

    phi = 1 means every cell moves in lockstep; small phi means cells cancel.
    phi itself needs an absolute calibration (thickness -> VAF -> SLE) that is
    approximate, but the RATIO phi_10x/phi_1x does not -- the calibration divides
    out. That ratio is the answer:

        phi_10x / phi_1x = (global sigma ratio) / (area-weighted local sigma ratio)

    < 1 means the 10x response is less coherent, confirming the hypothesis.
    ~ 1 means coherence is unchanged and the sub-linearity is local damping after all.

(B) BASIN AGGREGATION -- sums per-cell fields into the 16 ISMIP6 basins using the
    mask file directly, independent of whatever region mask each RUN was configured
    with. This is what finally brings CTRL and SSP585-3X (133-region) into the same
    regional frame as the 16-region ensembles.

Usage
    python3 spatial_coherence_and_basins.py \
        --sigma-ratio-dir reports/dissertation/figures/spatial/sigma_ratio/SSP585_varScaled10x_over_SSP585 \
        --mesh data/MALI/AIS_4to20km_...fb_A.nc \
        --mask data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc
"""
from __future__ import annotations
import os, glob, argparse
import numpy as np
import netCDF4

# global sigma(SLE) ratio, 10x over 1x, from the frozen results table (yr = key-100)
GLOBAL_SIGMA_RATIO = {2100: 3.71, 2150: 3.58, 2200: 3.55, 2300: 3.11}


def rd(path, var):
    d = netCDF4.Dataset(path)
    a = np.ma.filled(np.asarray(d[var][:], dtype=float), np.nan)
    d.close()
    return np.ravel(a)


def load_mask(mask_path):
    """(nRegions, nCells) 0/1 masks -> per-cell basin index, and the basin names."""
    d = netCDF4.Dataset(mask_path)
    m = np.asarray(d["regionCellMasks"][:])
    names = [b"".join(r).decode(errors="ignore").strip()
             for r in np.asarray(d["regionNames"][:]).astype("S1")] \
        if "regionNames" in d.variables else []
    d.close()
    if m.shape[0] != 16 and m.shape[1] == 16:      # stored transposed
        m = m.T
    idx = np.argmax(m, axis=0)
    idx[m.sum(axis=0) == 0] = -1                   # cells in no basin
    return idx, names


def coherence(args, area):
    print("=" * 78)
    print("(A) SPATIAL COHERENCE  --  is the 10x response less organised?")
    print("=" * 78)
    print("    phi = sigma_global / sum_c w_c sigma_c      (1 = lockstep, small = cancelling)")
    print("    phi_10x/phi_1x = (global sigma ratio) / (area-weighted local sigma ratio)\n")
    print(f"    {'yr':>5} {'T_1x':>12} {'T_10x':>12} {'local ratio':>12} "
          f"{'global ratio':>13} {'phi_10x/phi_1x':>15}")
    rows = []
    for f in sorted(glob.glob(os.path.join(args.sigma_ratio_dir, "sigmaRatio_*.nc"))):
        yr = int(os.path.basename(f).split("_")[-1][:4])
        if yr not in GLOBAL_SIGMA_RATIO:
            continue
        s1 = rd(f, "thickness_std_den")     # 1x
        s2 = rd(f, "thickness_std_num")     # 10x
        m = np.isfinite(s1) & np.isfinite(s2) & (s1 >= 0) & (s2 >= 0)
        # T = area-weighted sum of marginal sigma = the perfectly-coherent limit
        T1 = float(np.sum(area[m] * s1[m]))
        T2 = float(np.sum(area[m] * s2[m]))
        loc = T2 / T1
        glob_r = GLOBAL_SIGMA_RATIO[yr]
        phi = glob_r / loc
        rows.append((yr, T1, T2, loc, glob_r, phi))
        print(f"    {yr:5d} {T1:12.4e} {T2:12.4e} {loc:12.2f} {glob_r:13.2f} {phi:15.3f}")
    print()
    if rows:
        med = np.median([r[5] for r in rows])
        print(f"    median phi_10x/phi_1x = {med:.3f}")
        if med < 0.75:
            print(f"    -> CONFIRMED: the 10x response is ~{1/med:.1f}x LESS spatially coherent.")
            print("       Local cells respond nearly proportionally; the continental integral")
            print("       discards the difference because the extra spread does not add up.")
        elif med > 1.33:
            print("    -> the 10x response is MORE coherent; sub-linearity is not an")
            print("       aggregation effect and must be local damping.")
        else:
            print("    -> coherence is essentially unchanged; the aggregation hypothesis")
            print("       is NOT supported and sub-linearity is local after all.")
    return rows


def basins(args, area, idx, names):
    print()
    print("=" * 78)
    print("(B) BASIN AGGREGATION  --  16 ISMIP6 basins, from the mask, not the run config")
    print("=" * 78)
    files = sorted(glob.glob(os.path.join(args.sigma_ratio_dir, "sigmaRatio_*.nc")))
    if not files:
        print("  no sigmaRatio files"); return
    for f in files:
        yr = int(os.path.basename(f).split("_")[-1][:4])
        if yr != args.basin_year:
            continue
        s1, s2 = rd(f, "thickness_std_den"), rd(f, "thickness_std_num")
        print(f"\n  year {yr}   T_r = area-weighted sum of sigma within basin r")
        print(f"  {'#':>3} {'basin':<22} {'T_1x':>11} {'T_10x':>11} {'ratio':>7} {'% of 1x total':>14}")
        tot1 = np.nansum(area * np.where(np.isfinite(s1), s1, 0))
        out = []
        for r in range(16):
            c = (idx == r) & np.isfinite(s1) & np.isfinite(s2)
            if c.sum() == 0:
                continue
            t1 = float(np.sum(area[c] * s1[c])); t2 = float(np.sum(area[c] * s2[c]))
            out.append((r, t1, t2))
        for r, t1, t2 in sorted(out, key=lambda z: -z[1]):
            nm = names[r] if r < len(names) and names[r] else f"region {r}"
            print(f"  {r:3d} {nm:<22} {t1:11.4e} {t2:11.4e} {t2/t1:7.2f} {100*t1/tot1:13.1f}%")
        print("\n  Ratios far from the continental median mark basins whose spread responds")
        print("  differently to amplitude -- the spatial beta, per basin, with no")
        print("  contaminated forcing denominator.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma-ratio-dir", required=True)
    ap.add_argument("--mesh", required=True, help="mesh file providing areaCell")
    ap.add_argument("--mask", required=True, help="ISMIP6 16-region mask")
    ap.add_argument("--basin-year", type=int, default=2150)
    a = ap.parse_args()

    area = rd(a.mesh, "areaCell")
    idx, names = load_mask(a.mask)
    print(f"mesh: {area.size} cells   mask: {idx.size} cells, "
          f"{(idx >= 0).sum()} assigned to a basin\n")
    if area.size != idx.size:
        raise SystemExit("mesh and mask disagree on nCells")

    coherence(a, area)
    basins(a, area, idx, names)


if __name__ == "__main__":
    main()
