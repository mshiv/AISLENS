#!/usr/bin/env python3
"""
fig_sorrm_vs_dias_seasonality.py — Cross-model comparison of SORRM seasonal fraction vs Dias 2025 regimes.

Maps Dias shelf regime calls (seasonal vs steady) onto ISMIP6 basins via nearest centroid
and plots SORRM's seasonal fraction against them. These are different metrics — a regime-agreement
check, not like-for-like comparison.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset, chartostring

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASK = os.path.join(REPO, "data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc")
MESH = os.path.join(REPO, "data/MALI/AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m_"
                          "Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu_"
                          "meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
GEOJSON = os.path.join(REPO, "data/external/iceShelves.geojson")

# Dias et al. 2025 regime calls, by their named shelves (geojson feature names)
DIAS = {
    "steady":   ["Getz", "Pine_Island", "Thwaites"],          # West/Amundsen — low-freq expected
    "seasonal": ["Totten", "Ross", "Amery", "Fimbul", "Brunt_Stancomb"],  # East — seasonal expected
}
COL = {"steady": "#0072B2", "seasonal": "#E69F00"}


def unit_vec(lon, lat):   # lon,lat radians -> 3D unit vectors on the sphere
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _coords(geom):        # flatten all [lon,lat] vertices from a (Multi)Polygon
    out = []
    def walk(c):
        if isinstance(c[0], (int, float)):
            out.append(c[:2])
        else:
            for x in c:
                walk(x)
    walk(geom["coordinates"])
    return np.asarray(out, dtype=float)


def shelf_centroid_uv(geom):   # dateline-safe centroid as a mean 3D unit vector (deg -> rad)
    cc = _coords(geom)
    v = unit_vec(np.radians(cc[:, 0]), np.radians(cc[:, 1])).mean(axis=1)
    return v / np.linalg.norm(v)


def main():
    # --- basin centroids (unit-vector mean over each basin's cells) ---
    mk = Dataset(MASK); rcm = mk.variables["regionCellMasks"][:]  # (nCells, nRegions)
    names = [str(s).strip().replace("ISMIP6 Basin ", "") for s in chartostring(mk.variables["regionNames"][:])]
    me = Dataset(MESH); lon = np.asarray(me.variables["lonCell"][:]); lat = np.asarray(me.variables["latCell"][:])
    V = unit_vec(lon, lat)                                       # (3, nCells)
    basin_c = {}
    for j, nm in enumerate(names):
        sel = rcm[:, j] > 0
        v = V[:, sel].mean(axis=1); v /= np.linalg.norm(v)
        basin_c[nm] = v
    # basin centroid longitude (deg, -180..180) for circumpolar ordering
    basin_lon = {nm: (np.degrees(np.arctan2(v[1], v[0])) + 180) % 360 - 180 for nm, v in basin_c.items()}

    # --- Dias shelf centroids -> nearest basin ---
    gj = json.load(open(GEOJSON))
    feats = {f["properties"]["name"]: f for f in gj["features"]}
    shelf2basin = {}
    for regime, shelves in DIAS.items():
        for sh in shelves:
            if sh not in feats:
                raise SystemExit(f"shelf '{sh}' not in geojson")
            sv = shelf_centroid_uv(feats[sh]["geometry"])       # dateline-safe unit vector
            nm = max(basin_c, key=lambda b: float(basin_c[b] @ sv))   # nearest basin (max dot)
            shelf2basin[sh] = (nm, regime)

    # basin -> Dias regime (a basin may collect several shelves; keep the list)
    basin_regime = {}
    for sh, (nm, regime) in shelf2basin.items():
        basin_regime.setdefault(nm, {"steady": [], "seasonal": []})[regime].append(sh)

    # --- SORRM seasonal fraction per basin (avg of 2 members) ---
    d0 = pd.read_csv(os.path.join(REPO, "reports/spectrum_percell_generated0.csv"))
    d1 = pd.read_csv(os.path.join(REPO, "reports/spectrum_percell_generated1.csv"))
    d0["sector"] = d0["sector"].str.replace("ISMIP6 Basin ", "", regex=False)
    d1["sector"] = d1["sector"].str.replace("ISMIP6 Basin ", "", regex=False)
    seas = {r.sector: 0.5 * (r.seasonal + d1.loc[d1.sector == r.sector, "seasonal"].values[0])
            for _, r in d0.iterrows() if r.sector in names}

    # order basins circumpolar by centroid longitude
    order = sorted(names, key=lambda nm: basin_lon[nm])

    print(f"{'basin':6s} {'lon':>6s} {'SORRM seas':>10s}  Dias shelves (regime)")
    rows = []
    for nm in order:
        s = seas[nm]; reg = basin_regime.get(nm)
        tag = ""
        if reg:
            parts = [f"{r}:{'/'.join(v)}" for r, v in reg.items() if v]
            tag = "  " + "; ".join(parts)
        print(f"{nm:6s} {basin_lon[nm]:6.0f} {s:10.2f}{tag}")
        rows.append((nm, s, reg))

    # --- figure ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(order))
    seas_vals = [seas[nm] for nm in order]
    bars = ax.bar(x, seas_vals, color="0.8", edgecolor="0.5", width=0.72)
    ax.axhline(0.5, color="0.3", ls="--", lw=1)

    # mark Dias-classified basins
    for i, nm in enumerate(order):
        reg = basin_regime.get(nm)
        if not reg:
            continue
        for regime in ("steady", "seasonal"):
            if reg[regime]:
                bars[i].set_color(COL[regime]); bars[i].set_edgecolor("k"); bars[i].set_alpha(0.9)
                ax.text(i, seas[nm] + 0.02, "\n".join(reg[regime]), ha="center", va="bottom",
                        fontsize=7, color=COL[regime], fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("SORRM seasonal fraction of F$_v$", fontsize=11)
    ax.set_xlabel("ISMIP6 basin (W→E)", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COL["steady"], label="Dias: steady (West)"),
                       Patch(facecolor=COL["seasonal"], label="Dias: seasonal (East)"),
                       Patch(facecolor="0.8", label="not classified")],
              loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.10)
    out = os.path.join(REPO, "reports/figures/sorrm_vs_dias_seasonality.png")
    fig.savefig(out, dpi=150); print("\nwrote", out)


if __name__ == "__main__":
    main()
