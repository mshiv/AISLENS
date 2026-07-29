#!/usr/bin/env python3
"""
fig_forcing_melt_vs_jourdain.py — companion to Jourdain 2020 Fig 10.

Plots the AISLENS forcing's mean sub-shelf melt (m/yr) per Jourdain sector
from deterministic-control regionalStats. View side-by-side with Fig 10.
Uses iceShelves.geojson for sector-to-basin mapping.
"""
from __future__ import annotations
import os, json
import numpy as np
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
RSTATS = os.path.join(REPO, "data/MALI/diagnostics/deterministic/ctrlfb_A/regionalStats.nc")

JSECTORS = {
    "Ronne-Filchner":          ["Filchner-Ronne"],
    "Pine Island-Thwaites":    ["Pine_Island", "Thwaites"],
    "Cook-Ninnis":             ["Cook", "Ninnis"],
    "Totten-Moscow University":["Totten", "Moscow_University"],
}


def unit_vec(lon, lat):
    return np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])


def shelf_uv(geom):
    out = []
    def walk(c):
        if isinstance(c[0], (int, float)): out.append(c[:2])
        else:
            for x in c: walk(x)
    walk(geom["coordinates"])
    cc = np.asarray(out, float)
    v = unit_vec(np.radians(cc[:, 0]), np.radians(cc[:, 1])).mean(axis=1)
    return v/np.linalg.norm(v)


def main():
    # basin centroids (unit-vector mean over each basin's cells)
    mk = Dataset(MASK); rcm = mk.variables["regionCellMasks"][:]
    names = [str(s).strip().replace("ISMIP6 Basin ", "") for s in chartostring(mk.variables["regionNames"][:])]
    me = Dataset(MESH); lon = np.asarray(me.variables["lonCell"][:]); lat = np.asarray(me.variables["latCell"][:])
    V = unit_vec(lon, lat)
    basin_c = {}
    for j, nm in enumerate(names):
        v = V[:, rcm[:, j] > 0].mean(axis=1); basin_c[nm] = v/np.linalg.norm(v)

    # map each Jourdain shelf -> nearest basin index; collect basin indices per sector
    feats = {f["properties"]["name"]: f for f in json.load(open(GEOJSON))["features"]}
    sector_basins = {}
    for sec, shelves in JSECTORS.items():
        idxs = set()
        for sh in shelves:
            sv = shelf_uv(feats[sh]["geometry"])
            nm = max(basin_c, key=lambda b: float(basin_c[b] @ sv))
            idxs.add(names.index(nm))
        sector_basins[sec] = sorted(idxs)

    # control melt per basin (Time x 16), m/yr
    d = Dataset(RSTATS); melt = np.asarray(d.variables["regionalAvgSubshelfMelt"][:], float)  # (Time, 16)
    nt = melt.shape[0]; yr = np.arange(nt) / 12.0     # assume monthly output

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (sec, idxs) in zip(axes.ravel(), sector_basins.items()):
        series = melt[:, idxs].mean(axis=1)            # avg over the sector's basin(s)
        mu, sd = series.mean(), series.std()
        ax.plot(yr, series, color="#D55E00", lw=1.0)
        ax.axhline(mu, color="0.3", ls="--", lw=1)
        ax.fill_between(yr, mu - sd, mu + sd, color="#D55E00", alpha=0.12)
        ax.set_title(f"{sec}  ({', '.join(names[i] for i in idxs)})", fontsize=10, loc="left")
        ax.text(0.98, 0.92, f"{mu:.2f} m/yr", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, fontweight="bold")
        ax.set_xlabel("model year"); ax.set_ylabel("sub-shelf melt (m/yr)")
        ax.set_ylim(bottom=0)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.08, hspace=0.35, wspace=0.22)
    out = os.path.join(REPO, "reports/figures/forcing_melt_vs_jourdain.png")
    fig.savefig(out, dpi=150); print("wrote", out)
    print("\nsector -> basin mapping + control mean melt:")
    for sec, idxs in sector_basins.items():
        s = melt[:, idxs].mean(axis=1)
        print(f"  {sec:26s} basins={[names[i] for i in idxs]}  mean={s.mean():.2f} m/yr  std={s.std():.2f}")


if __name__ == "__main__":
    main()
