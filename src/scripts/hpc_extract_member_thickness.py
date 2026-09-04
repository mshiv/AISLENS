#!/usr/bin/env python3
"""
hpc_extract_member_thickness.py — per-member ice thickness, subset to shelf catchments. RUN ON HPC.

Writes one .npz per ensemble: cells (indices into nCells), years, members, h (M, T, n) float32.
Only thickness is pulled -- bed, coordinates and areas are static in the mesh file, and
grounded/floating is recomputed from flotation because cellMask is written as zeros in these runs.
Takes explicit years or a START:STOP:STEP range.
"""
from __future__ import annotations

import os, re, glob, argparse
import numpy as np
import netCDF4

try:
    from aislens.config import config
    _MALI = str(config.DIR_MALI)
except Exception:
    _MALI = os.path.join(os.environ.get("AISLENS_DATA_DIR", "."), "data", "MALI")

MESH = os.path.join(_MALI, "AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m"
                    "_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu"
                    "_meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
SHELF_MASK = os.path.join(_MALI, "aislens_draftDepen_regionMasks.nc")
ENS_ROOT = os.path.join(_MALI, "diagnostics", "ENSEMBLES")


def year_of(path):
    m = re.search(r"(\d{4})", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens-root", default=ENS_ROOT)
    ap.add_argument("--mesh", default=MESH)
    ap.add_argument("--shelf-mask", default=SHELF_MASK)
    ap.add_argument("--ensembles", nargs="+", required=True)
    ap.add_argument("--shelves", nargs="+", required=True)
    ap.add_argument("--years", nargs="+", default=["2000:2300:5"],
                    help='explicit years, or a START:STOP:STEP range like 2000:2300:5')
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--state-glob", default="output/output_state*.nc")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave ensembles that already have an .npz alone")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(os.environ.get("AISLENS_DATA_DIR", "."),
                             "reports/dissertation/figures/spatial/members")
    os.makedirs(a.out, exist_ok=True)
    for label, path in (("mesh", a.mesh), ("shelf mask", a.shelf_mask),
                        ("ensembles", a.ens_root)):
        if not os.path.exists(path):
            raise SystemExit(f"{label} not found: {path}\n"
                             "  set AISLENS_DATA_DIR, or pass --mesh/--shelf-mask/--ens-root")

    if len(a.years) == 1 and ":" in str(a.years[0]):
        lo, hi, st = (int(v) for v in str(a.years[0]).split(":"))
        a.years = list(range(lo, hi + 1, st))
    else:
        a.years = [int(v) for v in a.years]
    print(f"{len(a.years)} years: {a.years[0]}..{a.years[-1]}")

    # ---- cell subset: everything within radius of any requested shelf
    d = netCDF4.Dataset(a.mesh)
    x = np.asarray(d["xCell"][:]).ravel()
    y = np.asarray(d["yCell"][:]).ravel()
    d.close()

    d = netCDF4.Dataset(a.shelf_mask)
    raw = d["regionNames"][:]
    names = ["".join(c.decode() if isinstance(c, bytes) else str(c) for c in r)
             .replace("\x00", "").strip("-").strip() for r in raw]
    masks = np.asarray(d["regionCellMasks"][:])
    d.close()

    keep = np.zeros(x.size, bool)
    for sh in a.shelves:
        if sh not in names:
            print(f"  ! shelf not in mask: {sh}")
            continue
        cells = masks[:, names.index(sh)] > 0
        cx, cy = x[cells].mean(), y[cells].mean()
        keep |= np.hypot(x - cx, y - cy) < a.radius_km * 1e3
    cells = np.where(keep)[0].astype(np.int32)
    print(f"subset: {cells.size} of {x.size} cells "
          f"({100*cells.size/x.size:.1f}%)")

    for ens in a.ensembles:
        out = os.path.join(a.out, f"member_thickness_{ens}.npz")
        if a.skip_existing and os.path.exists(out):
            print(f"skip {ens} (exists)"); continue
        root = os.path.join(a.ens_root, ens)
        if not os.path.isdir(root):
            print(f"  ! no such ensemble dir: {root}"); continue
        members = sorted(m for m in os.listdir(root)
                         if os.path.isdir(os.path.join(root, m)))
        H = np.full((len(members), len(a.years), cells.size), np.nan, np.float32)
        got = 0
        for mi, mem in enumerate(members):
            files = sorted(glob.glob(os.path.join(root, mem, a.state_glob)))
            by_year = {}
            for f in files:
                yr = year_of(f)
                if yr in a.years:
                    by_year.setdefault(yr, f)
            for ti, yr in enumerate(a.years):
                f = by_year.get(yr)
                if f is None:
                    continue
                if mi == 0 and ti == 0:
                    print(f"  {ens}: {len(members)} members, first file {f}")
                dd = netCDF4.Dataset(f)
                th = np.asarray(dd["thickness"][:])
                dd.close()
                th = th[-1] if th.ndim == 2 else th          # last slice in the file
                H[mi, ti] = th.ravel()[cells].astype(np.float32)
                got += 1
        np.savez_compressed(out, cells=cells, years=np.array(a.years, np.int32),
                            members=np.array(members), h=H)
        print(f"wrote {out}   members={len(members)}  fields={got}  "
              f"{os.path.getsize(out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
