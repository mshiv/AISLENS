#!/usr/bin/env python3
"""
drift_basin_horizons.py — per-basin noise-induced drift at every available horizon.

D_r = SLE contribution of mean(VAF_10x) - mean(VAF_1x) summed over basin r; positive means the
10x ensemble has less ice above flotation there. Extends the year-150 decomposition in the
spatial-results note to 2100/2200/2250/2300, which is the horizon Chapter 3 actually quotes.
"""
import os
import numpy as np
import netCDF4

ROOT = "/Users/smurugan9/research/aislens/AISLENS"
SPAT = f"{ROOT}/reports/dissertation/figures/spatial/stats_sample"
MESH = (f"{ROOT}/data/MALI/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m"
        "_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu"
        "_meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
MASK = f"{ROOT}/data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc"

RHO_I, RHO_O, A_O = 910.0, 1028.0, 3.625e14

LETTERS = ["A-Ap", "Ap-B", "B-C", "C-Cp", "Cp-D", "D-Dp", "Dp-E", "E-F",
           "F-G", "G-H", "H-Hp", "Hp-I", "I-Ipp", "Ipp-J", "J-K", "K-A"]
NAMES = {"A-Ap": "Dronning Maud Land", "Ap-B": "Enderby Land", "B-C": "Amery-Lambert",
         "C-Cp": "Philippi-Denman", "Cp-D": "Totten", "D-Dp": "Mertz",
         "Dp-E": "Victoria Land", "E-F": "Ross", "F-G": "Getz", "G-H": "Thwaites/PIG",
         "H-Hp": "Bellingshausen", "Hp-I": "George VI", "I-Ipp": "Larsen A-C",
         "Ipp-J": "Larsen E", "J-K": "FRIS", "K-A": "Brunt-Stancomb"}


def rd(path, var):
    d = netCDF4.Dataset(path)
    a = np.ma.filled(np.asarray(d[var][:], dtype=float), np.nan)
    d.close()
    return np.ravel(a) if a.ndim > 1 else a


def vaf(h, bed):
    """Ice above flotation, per cell (m)."""
    return np.maximum(0.0, h - (RHO_O / RHO_I) * np.maximum(0.0, -bed))


def main():
    bed = rd(MESH, "bedTopography")
    area = rd(MESH, "areaCell")
    m = netCDF4.Dataset(MASK)
    masks = np.asarray(m["regionCellMasks"][:])          # (nCells, 16)
    m.close()

    # mm SLE per m^3 of ice above flotation
    to_mm = RHO_I / (RHO_O * A_O) * 1000.0

    years = [2100, 2150, 2200, 2250, 2300]
    print(f"{'':22s}" + "".join(f"{y:>10d}" for y in years))
    rows = {}
    totals = {}
    for y in years:
        f1 = f"{SPAT}/SSP585_{y}.nc"
        f10 = f"{SPAT}/SSP585_varScaled10x_{y}.nc"
        if not (os.path.exists(f1) and os.path.exists(f10)):
            totals[y] = None
            continue
        h1 = rd(f1, "thickness_mean")
        h10 = rd(f10, "thickness_mean")
        dv = vaf(h10, bed) - vaf(h1, bed)                # m, per cell
        dv = np.where(np.isfinite(dv), dv, 0.0)
        # positive drift (more sea level) = LESS VAF under 10x  -> negate
        sle = -(dv * area) * to_mm                       # mm SLE per cell
        totals[y] = float(np.nansum(sle))
        for i, L in enumerate(LETTERS):
            rows.setdefault(L, {})[y] = float(np.nansum(sle[masks[:, i] > 0]))

    order = sorted(LETTERS, key=lambda L: -abs(rows[L].get(2300, 0.0)))
    for L in order:
        cells = "".join(f"{rows[L].get(y, float('nan')):>10.2f}" for y in years)
        print(f"{L:6s} {NAMES[L]:15.15s}" + cells)

    print("-" * 72)
    for y in years:
        if totals[y] is None:
            continue
        vals = np.array([rows[L][y] for L in LETTERS])
        pos, neg = vals[vals > 0].sum(), vals[vals < 0].sum()
        gross = np.abs(vals).sum()
        print(f"yr {y-2000:3d}   net {totals[y]:+8.2f} mm   "
              f"pos {pos:+7.2f}   neg {neg:+7.2f}   "
              f"gross {gross:7.2f}   surviving {100*abs(totals[y])/gross:5.1f}%")


if __name__ == "__main__":
    main()
