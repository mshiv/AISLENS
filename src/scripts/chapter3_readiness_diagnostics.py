#!/usr/bin/env python3
"""
chapter3_readiness_diagnostics.py -- the checks that decide what Chapter 3 can claim.

Five diagnostics, all from data already on disk (globalStats.nc / regionalStats.nc):

  A  metric dependence   is "small spread" an artefact of choosing cumulative SLE?
  B  timing spread       spread in the YEAR each member crosses an SLE threshold
  C  covariance budget   Eq 3.13 -- do basins reinforce or cancel in the continental sum?
  D  drift vs twin       ensemble mean minus matched deterministic run (Eq 3.12)
  E  twin validation     IS the deterministic run actually matched? (gates D)

E gates D and must be read first. A deterministic run is only a valid twin if
(i) its MALI configuration is identical -- checkable from the netCDF global
attributes, which carry the full namelist -- and (ii) it was forced with the
ensemble-MEAN forcing. Test (ii) is the one that fails silently: compare the
DET run's realized melt against the ensemble mean and express the deviation as
a multiple of member-level variability. A ratio >> 1 means the twin is running
different forcing and any "drift" it produces is a forcing artefact.

Usage:  python3 chapter3_readiness_diagnostics.py [--root DIR] [--section ABCDE]
"""
from __future__ import annotations
import os, sys, glob, argparse
import numpy as np
import netCDF4

RHO_I, RHO_O, A_O = 910.0, 1028.0, 3.625e14   # kg/m3, kg/m3, m2

# Deterministic runs. Several vintages exist at different depths; the FIRST path that
# exists wins, newest first. DETERMINISTIC/DET-CTRL is the 2026-08 re-run (133-region,
# matching CTRL); DET-CTRL/DET-CTRL is the superseded 16-region version.
DET_PATHS = {
    "CTRL": ["DETERMINISTIC/DET-CTRL/globalStats.nc",
             "DET-CTRL/DET-CTRL/globalStats.nc"],
    "SSP585": ["DETERMINISTIC/DET-SSP585/globalStats.nc",
               "DET-SSP585/globalStats.nc"],
}


def resolve_det(root, ens):
    for rel in DET_PATHS.get(ens, []):
        if os.path.exists(os.path.join(root, rel)):
            return rel
    return None


def vaf_to_sle_mm(v):
    """VAF (m3) -> sea-level equivalent (mm), referenced to the member's own t=0."""
    return -(v - v[0]) * (RHO_I / RHO_O) / A_O * 1e3


def read_member(f, keys=("volumeAboveFloatation",)):
    d = netCDF4.Dataset(f)
    yr = np.asarray(d["daysSinceStart"][:], float) / 365.0
    out = {k: np.asarray(d[k][:], float) for k in keys if k in d.variables}
    d.close()
    v = out.get("volumeAboveFloatation")
    ok = np.isfinite(yr) & (np.isfinite(v) & (v > 0) if v is not None else True)
    return yr[ok], {k: x[ok] for k, x in out.items()}


def load(root, ens, keys, pat="*_[0-9][0-9]"):
    """Members with the restart-segment guard: a record starting after yr 5 is a
    restart fragment, not a member, and would bias every ensemble statistic."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, ens, pat, "globalStats.nc"))):
        try:
            yr, v = read_member(f, keys)
        except Exception:
            continue
        if len(yr) < 50 or yr[0] > 5.0:
            continue
        out.append((yr, v))
    return out


def regrid(members, key, grid, transform=None):
    M = np.full((len(members), grid.size), np.nan)
    for i, (yr, V) in enumerate(members):
        if key not in V:
            continue
        v = transform(V[key]) if transform else V[key]
        m = np.isfinite(v)
        if m.sum() > 2:
            M[i] = np.interp(grid, yr[m], v[m], left=np.nan, right=np.nan)
    return M


def common_grid(members):
    return np.arange(1.0, max(m[0][-1] for m in members) + 1e-9, 1.0)


def cv_at(M, i):
    x = M[:, i]
    x = x[np.isfinite(x)]
    if x.size < 3 or not np.isfinite(x.mean()) or x.mean() == 0:
        return np.nan, np.nan, x.size
    return x.std(ddof=1), abs(x.std(ddof=1) / x.mean()) * 100, x.size


# ----------------------------------------------------------------- A
def section_A(root):
    print("=" * 74)
    print("A. METRIC DEPENDENCE -- does 'small spread' survive a change of metric?")
    print("=" * 74)
    keys = ("volumeAboveFloatation", "groundingLineFlux", "floatingIceArea")
    for ens, H in [("SSP585", 300.0), ("SSP126", 300.0), ("CTRL", 300.0), ("SSP585-3X", 178.0)]:
        mem = load(root, ens, keys)
        if not mem:
            continue
        g = common_grid(mem)
        S = regrid(mem, "volumeAboveFloatation", g, vaf_to_sle_mm)
        # rate via centred difference; NEVER evaluate at the array edge (all-NaN)
        w = 10
        RATE = np.full_like(S, np.nan)
        RATE[:, w:-w] = (S[:, 2 * w:] - S[:, :-2 * w]) / (2 * w)
        i = int(np.argmin(abs(g - H)))
        print(f"\n  {ens} @ yr{int(H)}")
        for lab, M, u in [("cumulative SLE", S, "mm"),
                          ("SLE rate(10yr)", RATE, "mm/yr"),
                          ("GL flux", regrid(mem, "groundingLineFlux", g), "kg/s"),
                          ("floating area", regrid(mem, "floatingIceArea", g), "m2")]:
            s, c, n = cv_at(M, i)
            print(f"     {lab:15s} n={n:2d} sigma={s:11.4g} {u:6s} CV={c:8.3f} %")


# ----------------------------------------------------------------- B
def section_B(root):
    print("\n" + "=" * 74)
    print("B. TIMING SPREAD -- year at which each member crosses an SLE threshold")
    print("=" * 74)
    for ens, levels in [("SSP585", [100, 500, 1000, 1500]), ("SSP126", [50, 100, 150])]:
        mem = load(root, ens, ("volumeAboveFloatation",))
        if not mem:
            continue
        g = common_grid(mem)
        S = regrid(mem, "volumeAboveFloatation", g, vaf_to_sle_mm)
        print(f"\n  {ens}")
        for L in levels:
            ts = [np.interp(L, r[np.isfinite(r)], g[np.isfinite(r)])
                  for r in S if np.isfinite(r).any() and np.nanmax(r) >= L]
            if len(ts) >= 3:
                ts = np.array(ts)
                print(f"     SLE={L:5d} mm : mean yr {ts.mean():6.1f}  sd {ts.std(ddof=1):5.2f} yr"
                      f"  range {ts.ptp():5.2f} yr  (n={len(ts)})")


# ----------------------------------------------------------------- C
def section_C(root, ens="SSP585"):
    print("\n" + "=" * 74)
    print(f"C. REGIONAL VARIANCE BUDGET (Eq 3.13) -- {ens}")
    print("=" * 74)
    print("   Var(S) = SUM_r Var(S_r) + 2 SUM_{r<q} Cov(S_r,S_q)")
    print("   A basin ranking on Var(S_r)/SUM Var(S_r) is a MARGINAL budget, not a")
    print("   decomposition of continental variance. The cross term is the difference.")
    rec = []
    for f in sorted(glob.glob(os.path.join(root, ens, "*_[0-9][0-9]", "regionalStats.nc"))):
        d = netCDF4.Dataset(f)
        yr = np.asarray(d["daysSinceStart"][:], float) / 365.0
        v = np.asarray(d["regionalVolumeAboveFloatation"][:], float)
        d.close()
        if yr[0] > 5 or yr[-1] < 200:
            continue
        rec.append((yr, v))
    for H in (200.0, 300.0):
        A = []
        for yr, v in rec:
            if yr[-1] < H - 1:
                continue
            i = int(np.argmin(abs(yr - H)))
            A.append(-(v[i] - v[0]) * (RHO_I / RHO_O) / A_O * 1e3)
        A = np.array(A)
        if A.shape[0] < 3:
            continue
        C = np.cov(A, rowvar=False)
        marg, tot = np.trace(C), C.sum()
        print(f"\n  yr{int(H)}  n={A.shape[0]}")
        print(f"     SUM basin variances (marginal) = {marg:10.4f} mm^2  -> sigma {np.sqrt(marg):.3f} mm")
        print(f"     TRUE continental variance      = {tot:10.4f} mm^2  -> sigma {np.sqrt(tot):.3f} mm")
        print(f"     cross-covariance term          = {tot-marg:+10.4f} mm^2 "
              f"= {100*(tot-marg)/marg:+.1f} % of marginal")
        print("     -> " + ("basins REINFORCE (no cancellation)" if tot > marg
                            else "basins partially CANCEL in the continental sum"))
        sd = np.sqrt(np.diag(C))
        for j in np.argsort(-sd)[:4]:
            print(f"        region {j:3d}: sigma={sd[j]:.3f} mm ({100*C[j,j]/marg:5.1f} % of marginal)")


# ----------------------------------------------------------------- E (gates D)
def section_E(root):
    print("\n" + "=" * 74)
    print("E. TWIN VALIDATION -- is the deterministic run actually matched? (gates D)")
    print("=" * 74)
    verdict = {}
    for ens in DET_PATHS:
        rel = resolve_det(root, ens)
        if rel is None:
            print(f"\n  {ens}: no deterministic run found")
            continue
        det = os.path.join(root, rel)
        # (i) configuration match, straight from the namelist in the global attributes
        ref = sorted(glob.glob(os.path.join(root, ens, "*_[0-9][0-9]", "globalStats.nc")))[0]
        a, b = (netCDF4.Dataset(x) for x in (ref, det))
        A = {k: a.getncattr(k) for k in a.ncattrs()}
        B = {k: b.getncattr(k) for k in b.ncattrs()}
        a.close(); b.close()
        ignore = {"history", "parent_id", "file_id", "git_version", "config_stop_time"}
        cfg = [k for k in sorted(set(A) | set(B))
               if k not in ignore and str(A.get(k)) != str(B.get(k))]

        # (ii) forcing match -- EARLY-TIME test.
        #
        # The obvious test (does DET's realized melt track the ensemble mean?) is
        # WRONG on its own. avgSubshelfMelt is an area-average over floating cells,
        # so a perfectly matched twin STILL diverges once member geometries diverge
        # -- that divergence is precisely the nonlinear rectification we want to
        # measure (Hoffman et al. 2019). Testing over the whole run therefore
        # rejects a good twin for showing the very signal it exists to reveal.
        #
        # The discriminator is TIME. At yr ~2 no ice has moved: floating area is
        # identical to the ensemble to <0.01%, so realized melt ~ prescribed melt.
        # A deviation THERE cannot be rectification -- it can only be a different
        # input file. A twin that matches early and diverges later is exactly right.
        dy, dV = read_member(det, ("volumeAboveFloatation", "avgSubshelfMelt", "floatingIceArea"))
        mem = load(root, ens, ("volumeAboveFloatation", "avgSubshelfMelt", "floatingIceArea"))
        g = np.arange(1.0, min(dy[-1], 300.0) + 1e-9, 1.0)
        mu = np.nanmean(regrid(mem, "avgSubshelfMelt", g), axis=0)
        mua = np.nanmean(regrid(mem, "floatingIceArea", g), axis=0)
        dmelt = np.interp(g, dy, dV["avgSubshelfMelt"])
        darea = np.interp(g, dy, dV["floatingIceArea"])

        rows = []
        for H in (2, 5, 10, 25):
            if H > g[-1]:
                continue
            i = int(np.argmin(abs(g - H)))
            rows.append((H, 100 * (dmelt[i] - mu[i]) / mu[i], 100 * (darea[i] - mua[i]) / mua[i]))
        early = float(np.mean([abs(r[1]) for r in rows])) if rows else np.inf
        geom = float(np.mean([abs(r[2]) for r in rows])) if rows else np.inf

        ok = (len(cfg) == 0) and (early < 2.0)
        verdict[ens] = ok
        print(f"\n  {ens} vs {rel.split('/')[0]}")
        print(f"     (i)  config attrs differing : {len(cfg)}  {cfg if cfg else '-> IDENTICAL'}")
        print(f"     (ii) early-time forcing check (before geometry can diverge):")
        print(f"          {'yr':>4} {'melt dev':>10} {'area dev':>10}")
        for H, md, ad in rows:
            print(f"          {H:4d} {md:+9.1f}% {ad:+9.2f}%")
        print(f"          mean |melt dev| over yr2-25 = {early:.1f} %   "
              f"(geometry has moved only {geom:.2f} %)")
        print(f"          {'OK -- twin reproduces the prescribed input' if early < 2 else 'FAIL -- different input file; geometry cannot explain a yr-2 offset'}")
        print(f"     VERDICT: {'MATCHED -- Eq 3.12 usable' if ok else 'NOT MATCHED -- do not report drift'}")
    return verdict


# ----------------------------------------------------------------- D
def section_D(root, verdict):
    print("\n" + "=" * 74)
    print("D. DRIFT vs DETERMINISTIC TWIN (Eq 3.12)  D(t) = mean S_stoch - S_det")
    print("=" * 74)
    for ens in DET_PATHS:
        rel = resolve_det(root, ens)
        if rel is None:
            continue
        det = os.path.join(root, rel)
        if not verdict.get(ens, False):
            print(f"\n  {ens}: SUPPRESSED -- section E says this twin is not matched.")
            print("         Any number here would be a forcing artefact, not noise-induced drift.")
            continue
        dy, dV = read_member(det, ("volumeAboveFloatation",))
        dS = vaf_to_sle_mm(dV["volumeAboveFloatation"])
        mem = load(root, ens, ("volumeAboveFloatation",))
        g = common_grid(mem)
        S = regrid(mem, "volumeAboveFloatation", g, vaf_to_sle_mm)
        print(f"\n  {ens}  (twin spans yr {dy[0]:.0f}-{dy[-1]:.0f})")
        for H in (50, 100, 150, 200, 300):
            if H > dy[-1] or H > g[-1]:
                continue
            i = int(np.argmin(abs(g - H)))
            x = S[:, i]; x = x[np.isfinite(x)]
            if x.size < 3:
                continue
            D = x.mean() - float(np.interp(H, dy, dS))
            print(f"     yr{H:3d}  D={D:+8.3f} mm = {D/x.std(ddof=1):+6.2f} sigma_ens "
                  f"= {100*D/abs(x.mean()):+6.2f} % of signal  (n={x.size})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/MALI/diagnostics/ENSEMBLES")
    ap.add_argument("--section", default="ABCDE", help="subset of ABCDE to run")
    a = ap.parse_args()
    s = a.section.upper()
    if "A" in s: section_A(a.root)
    if "B" in s: section_B(a.root)
    if "C" in s: section_C(a.root)
    verdict = section_E(a.root) if ("E" in s or "D" in s) else {}
    if "D" in s: section_D(a.root, verdict)


if __name__ == "__main__":
    main()
