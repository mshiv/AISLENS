#!/usr/bin/env python3
"""
hpc_extract_member_thickness.py -- per-member ice thickness on a cell subset. RUN ON HPC.

Writes one .npz per ensemble: cells (indices into nCells), years, members, h (M, T, n)
float32, year_got (M, T) int16 giving the year each field was read at, -1 where missing.

Only thickness is pulled; bed, coordinates and areas are static in the mesh file, and
grounded/floating is recomputed from flotation. Each output_state file holds five annual
slices, so any year can be asked for and the slice is matched on xtime.

--along-flowline selects a corridor around each shelf's along-flow line, which covers
long shelves that no disc radius reaches.
"""
from __future__ import annotations

import os, re, glob, argparse, sys
import numpy as np
import netCDF4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flowline import build_flowline  # noqa: E402

def _mali_dir():
    """Where the MALI data lives.

    AISLENS_DATA_DIR is set to the repo root on some machines and to the data
    directory itself on others, so try the sensible spellings and take the first
    that is actually there rather than assuming one layout.
    """
    env = os.environ.get("AISLENS_DATA_DIR")
    roots = [env] if env else []
    roots.append(ROOT if "ROOT" in globals() else ".")
    for r in roots:
        for tail in (("data", "MALI"), ("MALI",), ()):
            p = os.path.join(r, *tail)
            if os.path.isdir(p):
                return p
    return os.path.join(roots[0], "data", "MALI")


try:
    from aislens.config import config
    _MALI = str(config.DIR_MALI)
except Exception:
    _MALI = _mali_dir()

MESH = os.path.join(_MALI, "AIS_4to20km_r01_20220907_m5_drop_bed_20m_bulldoze_troughs_75_to_400m"
                    "_Enderby_maxstiffness_0.8_TG_pinning_40maf_bedmap2_surface_ASE_05perc_seafloor_mu"
                    "_meanSatObsBMB_Paolo2023_draftDepenPiecewise_fb_A.nc")
SHELF_MASK = os.path.join(_MALI, "aislens_draftDepen_regionMasks.nc")
ENS_ROOT = os.path.join(_MALI, "ENSEMBLES")


def year_of(path):
    m = re.search(r"(\d{4})", os.path.basename(path))
    return int(m.group(1)) if m else None


def slice_of(ds, year, tol):
    """(index, year) of the slice nearest `year`, or (None, None) if none is within tol.

    The last file of a run holds one slice dated 2299-12-01, hence the tolerance.
    """
    if "xtime" not in ds.variables:
        return 0, year
    years = []
    for r in ds["xtime"][:]:
        s = "".join(c.decode() if isinstance(c, bytes) else str(c) for c in r)
        years.append(int(s[:4]) if s[:4].isdigit() else -9999)
    years = np.array(years)
    i = int(np.argmin(np.abs(years - year)))
    return (i, int(years[i])) if abs(years[i] - year) <= tol else (None, None)


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
    ap.add_argument("--along-flowline", action="store_true",
                    help="corridor around each shelf's along-flow line, rather than a disc "
                         "around its centroid")
    ap.add_argument("--corridor-km", type=float, default=25.0,
                    help="--along-flowline: half-width of the corridor")
    ap.add_argument("--year-tol", type=int, default=1,
                    help="how far a slice may sit from the requested year and still be used")
    ap.add_argument("--state-glob", default="output_state*.nc",
                    help="searched in the member dir and in its output/ subdir")
    ap.add_argument("--vars", nargs="+", default=["thickness"],
                    help="state variables to carry out; cellMask is written as int32 so "
                         "the bit flags survive, everything else as float32")
    ap.add_argument("--full-domain", action="store_true",
                    help="keep every cell instead of a shelf subset, for whole-sheet maps")
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
    if a.along_flowline:
        from scipy.spatial import cKDTree
        d = netCDF4.Dataset(a.mesh)
        bed = np.asarray(d["bedTopography"][:]).ravel()
        h0 = np.asarray(d["thickness"][:]).ravel()
        vx = np.asarray(d["observedSurfaceVelocityX"][:]).ravel()
        vy = np.asarray(d["observedSurfaceVelocityY"][:]).ravel()
        d.close()
        tree = cKDTree(np.column_stack([x, y]))
    for sh in a.shelves:
        if sh not in names:
            print(f"  ! shelf not in mask: {sh}")
            continue
        cells = masks[:, names.index(sh)] > 0
        if a.along_flowline:
            _, pts = build_flowline(x, y, cells, vx, vy, tree, bed, h0)
            near = cKDTree(pts).query(np.column_stack([x, y]))[0]
            keep |= near < a.corridor_km * 1e3
            print(f"  {sh}: flowline {np.hypot(*(pts[-1] - pts[0])) / 1e3:.0f} km end to end, "
                  f"{int((near < a.corridor_km * 1e3).sum())} cells in corridor")
        else:
            cx, cy = x[cells].mean(), y[cells].mean()
            keep |= np.hypot(x - cx, y - cy) < a.radius_km * 1e3
    if a.full_domain:
        keep[:] = True
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
        def states(mem):
            d = os.path.join(root, mem)
            return sorted(glob.glob(os.path.join(d, a.state_glob))
                          or glob.glob(os.path.join(d, "output", a.state_glob)))

        cand = sorted(m for m in os.listdir(root)
                      if os.path.isdir(os.path.join(root, m)))
        members = [m for m in cand if states(m)]
        if not members:
            print(f"  ! {ens}: no {a.state_glob} under any of {len(cand)} member dirs")
            continue
        if len(cand) > len(members):
            print(f"  {ens}: using {len(members)} of {len(cand)} dirs "
                  f"(skipped {', '.join(sorted(set(cand) - set(members))[:4])} ...)")
        # cellMask carries bit flags, so it must not be stored as float, and the
        # empty value has to be 0 rather than -1: every bit of -1 is set, so an
        # unread frame would test true for grounding line, floating, everything
        V = {v: np.full((len(members), len(a.years), cells.size),
                        0 if v == "cellMask" else np.nan,
                        np.int32 if v == "cellMask" else np.float32) for v in a.vars}
        got = np.full((len(members), len(a.years)), -1, np.int16)   # year actually read
        for mi, mem in enumerate(members):
            # each file starts at the year in its name and runs five annual slices,
            # so the file for a given year is the latest one starting at or before it
            starts = sorted((year_of(f), f) for f in states(mem) if year_of(f))
            # every file holds five annual slices, so gather the years that fall in
            # each one and open it once rather than once per year
            byfile = {}
            for ti, yr in enumerate(a.years):
                f = next((p for y0, p in reversed(starts) if y0 <= yr), None)
                if f is not None:
                    byfile.setdefault(f, []).append((ti, yr))
            for f, wanted in byfile.items():
                dd = netCDF4.Dataset(f)
                cache = {v: np.asarray(dd[v][:]) for v in a.vars if v in dd.variables}
                for ti, yr in wanted:
                    if a.vars[0] not in cache:
                        continue
                    si, ya = (slice_of(dd, yr, a.year_tol)
                              if cache[a.vars[0]].ndim == 2 else (0, yr))
                    if si is None:
                        continue
                    if not (got >= 0).any():
                        print(f"  {ens}: {len(members)} members, first read {yr} = "
                              f"slice {si} ({ya}) of {f}")
                    for v, arr in cache.items():
                        sl = arr[si] if arr.ndim == 2 else arr
                        V[v][mi, ti] = sl.ravel()[cells].astype(V[v].dtype)
                    got[mi, ti] = ya
                dd.close()
        H = V.get("thickness", next(iter(V.values())))
        icy = np.nansum(H > 1.0, axis=2)          # cells with ice, per member and year
        if (got >= 0).any() and not icy.any():
            print(f"  ! {ens}: every field read is empty -- the output_state files hold "
                  f"no thickness. Check one with ncdump -v thickness before trusting this.")
        np.savez_compressed(out, cells=cells, years=np.array(a.years, np.int32),
                            members=np.array(members), h=H, year_got=got,
                            **{v: V[v] for v in a.vars})
        span = [f"{m}:{a.years[np.flatnonzero(got[i] >= 0)[0]]}-{a.years[np.flatnonzero(got[i] >= 0)[-1]]}"
                for i, m in enumerate(members) if (got[i] >= 0).any()]
        print(f"wrote {out}   members={len(members)}  fields={int((got >= 0).sum())}  "
              f"median icy cells/field {int(np.median(icy[icy > 0])) if (icy > 0).any() else 0}  "
              f"{os.path.getsize(out)/1e6:.1f} MB")
        print(f"  coverage: {' '.join(span)}")
        off = {(a.years[ti], int(got[mi, ti])) for mi, ti in zip(*np.where(got >= 0))
               if got[mi, ti] != a.years[ti]}
        if off:
            print(f"  asked/read year differs: {sorted(off)}")


if __name__ == "__main__":
    main()
