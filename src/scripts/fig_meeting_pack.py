#!/usr/bin/env python3
"""
fig_meeting_pack.py — three global-comparison figures across the full AISLENS design.

  1. meeting_overview.png        ensemble mean SLE + spread, all five ensembles
  2. meeting_forcing_response.png  realized melt vs SLE response; power-law fit on the
                                   mean-forcing axis (sub-linearity across the design)
  3. meeting_design_matrix.png   what has been sampled: mean-forcing axis vs
                                   variability axis, marker area = sigma(SLE)

GLOBAL ONLY. CTRL and SSP585-3X carry nRegions=133 while SSP126/SSP585/varScaled10x
carry 16, so no regional comparison is attempted here.

Guards (each adopted after a specific artefact):
  * drop members whose record starts after yr 5 (restart segments)
  * interpolate every member onto a COMMON ANNUAL GRID before any ensemble statistic --
    per-member output cadence is irregular AND differs between members, so statistics on
    a merged raw axis produce spurious spikes
  * never extrapolate past a member's own record; require >= --min-members at each step
"""
from __future__ import annotations
import os, sys, glob, argparse
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio

ENS = [
    ("CTRL",                "CTRL_[0-9][0-9]",                "#888888"),
    ("SSP126",              "SSP126_[0-9][0-9]",              "#0072B2"),
    ("SSP585",              "SSP585_[0-9][0-9]",              "#D55E00"),
    ("SSP585_varScaled10x", "SSP585_[0-9][0-9]",              "#7B3FA0"),
    ("SSP585-3X",           "SSP585-3X_[0-9][0-9]",           "#8B0000"),
]
# ensembles that differ in MEAN forcing at fixed 1x variability -> used for the power-law fit
MEAN_AXIS = {"CTRL", "SSP126", "SSP585", "SSP585-3X"}


def load(root, ens, pat, variables):
    """Per-member (year, {var: series}) with the restart-segment guard applied."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, ens, pat))):
        f = os.path.join(d, "globalStats.nc")
        if not os.path.exists(f):
            continue
        try:
            ds = netCDF4.Dataset(f)
            yr = np.asarray(ds["daysSinceStart"][:], dtype=float) / 365.0
            vals = {v: np.asarray(ds[v][:], dtype=float) for v in variables if v in ds.variables}
            ds.close()
        except Exception:
            continue
        if not vals or len(yr) < 50:
            continue
        ok = np.isfinite(yr) & np.isfinite(vals.get("volumeAboveFloatation", yr))
        ok &= vals.get("volumeAboveFloatation", np.ones_like(yr)) > 0
        if ok.sum() < 50 or yr[ok][0] > 5.0:       # restart segment -> drop
            continue
        out.append((yr[ok], {k: v[ok] for k, v in vals.items()}))
    return out


def regrid(members, key, grid):
    """(member, year) on a shared annual grid; NaN outside each member's own record."""
    M = np.full((len(members), grid.size), np.nan)
    for i, (yr, vals) in enumerate(members):
        if key not in vals:
            continue
        v = vals[key]
        m = np.isfinite(v)
        if m.sum() < 2:
            continue
        M[i] = np.interp(grid, yr[m], v[m], left=np.nan, right=np.nan)
    return M


def stats(M, min_members):
    n = np.sum(np.isfinite(M), axis=0)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0, ddof=1)
    mu[n < min_members] = np.nan
    sd[n < min_members] = np.nan
    return mu, sd, n


def runmed(x, w):
    if not w or w < 2:
        return x
    h, out = w // 2, np.full_like(x, np.nan)
    for i in range(x.size):
        seg = x[max(0, i - h): i + h + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--horizon", type=float, default=180.0,
                    help="common horizon (yr); SSP585-3X only reaches ~182")
    ap.add_argument("--min-members", type=int, default=3)
    ap.add_argument("--smooth", type=int, default=5, help="running median on sigma (cosmetic)")
    ap.add_argument("--out-dir", default="reports/figures/meeting")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    VARS = ["volumeAboveFloatation", "avgSubshelfMelt"]
    data = {}
    for ens, pat, col in ENS:
        mem = load(a.root, ens, pat, VARS)
        if not mem:
            print(f"  {ens}: no usable members -- skipped"); continue
        # Grid to the LONGEST member, not the shortest: regrid() leaves NaN outside each
        # member's own record and stats() then masks where fewer than --min-members remain.
        # Cutting at the shortest member truncated SSP126 at yr241 (SLE 84.6 mm), which is
        # not comparable to the yr300 numbers quoted elsewhere (188 mm).
        end = max(m[0][-1] for m in mem)
        grid = np.arange(1.0, end + 1e-9, 1.0)
        sle = regrid([(y, {"v": eio.vaf_to_sle_mm(v["volumeAboveFloatation"], reference="first")})
                      for y, v in mem], "v", grid)
        melt = regrid(mem, "avgSubshelfMelt", grid)
        mu_s, sd_s, n_s = stats(sle, a.min_members)
        mu_m, _, _ = stats(melt, a.min_members)
        # realized stochastic forcing amplitude: SD over time of (member - ensemble mean)
        with np.errstate(invalid="ignore"):
            resid = melt - np.nanmean(melt, axis=0)[None, :]
            amp = float(np.nanmean(np.nanstd(resid, axis=1, ddof=1)))
        data[ens] = dict(color=col, n=len(mem), grid=grid, sle=mu_s, sig=sd_s,
                         melt=mu_m, amp=amp, end=end)
        print(f"  {ens:22s} n={len(mem):2d}  span 1-{end:.0f} yr  realized amp={amp:.4f} m/yr")

    H = a.horizon
    def at(ens, key, h=H):
        d = data[ens]; i = int(np.argmin(np.abs(d["grid"] - h)))
        return float(d[key][i]) if np.isfinite(d[key][i]) else np.nan

    # ---------------- FIG 1: overview ----------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    for ens in data:
        d = data[ens]
        axL.plot(d["grid"], d["sle"], color=d["color"], lw=2, label=ens)
        axR.plot(d["grid"], runmed(d["sig"], a.smooth), color=d["color"], lw=2, label=ens)
    axL.set_xlabel("year"); axL.set_ylabel("ensemble-mean SLE (mm)")
    axL.set_title("(a) mean sea-level contribution"); axL.grid(alpha=.3); axL.legend(fontsize=8)
    axR.set_xlabel("year"); axR.set_ylabel("ensemble spread $\\sigma$ (mm SLE)")
    axR.set_title(f"(b) variability-driven spread ({a.smooth}-yr median)")
    axR.grid(alpha=.3); axR.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/meeting_overview.png", dpi=130); plt.close(fig)

    # Report at BOTH horizons. yr180 is the only horizon that includes SSP585-3X (it reaches
    # ~182); yr300 is where the headline CV numbers in the chapter are defined. Quoting only
    # one invites an apparent contradiction with the 0.12% CV figure.
    for hh in (H, 300.0):
        avail = [e for e in data if data[e]["grid"][-1] >= hh - 1]
        if not avail:
            continue
        print(f"\n=== horizon yr {int(hh)} "
              f"({'all five ensembles' if len(avail) == len(data) else 'ensembles reaching it: ' + ', '.join(avail)}) ===")
        print(f"{'ensemble':22s} {'n':>3s} {'meanSLE':>10s} {'sigma':>8s} {'CV %':>7s} {'melt':>8s}")
        for ens in avail:
            m, sg = at(ens, "sle", hh), at(ens, "sig", hh)
            cv = 100 * sg / abs(m) if np.isfinite(m) and abs(m) > 5 else np.nan
            print(f"{ens:22s} {data[ens]['n']:3d} {m:10.1f} {sg:8.2f} "
                  f"{cv:7.2f} {at(ens,'melt',hh):8.2f}")
        if len(avail) < len(data):
            print(f"  (excluded: {', '.join(e for e in data if e not in avail)} "
                  f"-- does not reach yr {int(hh)})")

    # ---------------- FIG 2: forcing -> response ----------------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    xs, ys, names = [], [], []
    for ens in data:
        # realized MEAN melt over 0..H  and  SLE loss at H
        d = data[ens]; k = d["grid"] <= H
        x = float(np.nanmean(d["melt"][k])); y = at(ens, "sle")
        if not (np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0):
            continue
        infit = ens in MEAN_AXIS
        ax.scatter(x, y, s=150, color=d["color"], marker="o" if infit else "D",
                   edgecolor="k", zorder=3,
                   label=ens + ("" if infit else "  (variability axis — excluded from fit)"))
        ax.annotate(ens, (x, y), fontsize=8, xytext=(6, -10), textcoords="offset points")
        if infit:
            xs.append(x); ys.append(y); names.append(ens)
    # A SINGLE power law across all four is NOT appropriate: the relation is concave in
    # log-log (steep at low forcing, flat at high forcing), i.e. saturating. Fitting one
    # line averages those regimes and yields a number describing neither. Report the
    # PAIRWISE exponents instead, and mark the one clean comparison.
    order = sorted([(x, y, n) for x, y, n in zip(xs, ys, names)], key=lambda t: t[0])
    print("\npairwise exponents  p = ln(SLE ratio)/ln(melt ratio)   [1.0 = linear]")
    for (x0, y0, n0), (x1, y1, n1) in zip(order[:-1], order[1:]):
        pw = np.log(y1 / y0) / np.log(x1 / x0)
        clean = (n0, n1) == ("SSP585", "SSP585-3X") or (n1, n0) == ("SSP585", "SSP585-3X")
        tag = "  <-- CLEAN (same scenario, trend scaled only)" if clean else \
              "      (differs in trend SHAPE too, not just amplitude)"
        print(f"  {n0:>20s} -> {n1:<20s} melt x{x1/x0:5.2f}  SLE x{y1/y0:5.2f}  p={pw:5.2f}{tag}")
        ax.plot([x0, x1], [y0, y1], "-" if clean else ":", lw=2.2 if clean else 1.2,
                color="k" if clean else "0.5", zorder=2,
                label=(f"{n0}$\\to${n1}: p={pw:.2f}  (clean)" if clean
                       else f"{n0}$\\to${n1}: p={pw:.2f}"))
    xf = np.array([min(xs) * .8, max(xs) * 1.2])
    ax.plot(xf, order[0][1] * (xf / order[0][0])**1.0, "k--", lw=1, alpha=.5,
            label="slope 1 (linear reference)")
    print("  NOTE: the relation is concave -- a single global exponent is not meaningful.")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"realized mean melt, yr 0-{int(H)} (m/yr)")
    ax.set_ylabel(f"ensemble-mean SLE at yr {int(H)} (mm)")
    ax.set_title("Melt $\\to$ sea-level response saturates with forcing")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/meeting_forcing_response.png", dpi=130); plt.close(fig)

    # ---------------- FIG 3: design matrix ----------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sig_all = [at(e, "sig") for e in data if np.isfinite(at(e, "sig"))]
    smax = max(sig_all) if sig_all else 1.0
    for ens in data:
        d = data[ens]; k = d["grid"] <= H
        x = float(np.nanmean(d["melt"][k])); y = d["amp"]; s = at(ens, "sig")
        if not np.isfinite(x):
            continue
        area = 120 + 1400 * (s / smax if np.isfinite(s) and smax > 0 else 0.2)
        ax.scatter(x, y, s=area, color=d["color"], alpha=.65, edgecolor="k", zorder=3)
        ax.annotate(f"{ens}\n$\\sigma$={s:.2f} mm", (x, y), fontsize=8,
                    xytext=(10, 6), textcoords="offset points")
    ax.set_xlabel(f"MEAN-forcing axis: realized mean melt, yr 0-{int(H)} (m/yr)")
    ax.set_ylabel("VARIABILITY axis: realized stochastic melt amplitude (m/yr)")
    ax.set_title("What has been sampled (marker area $\\propto$ ensemble spread)")
    ax.grid(alpha=.3)
    ax.text(0.02, 0.98, "CAVEAT: amplitude is an area-average over floating cells, so once\n"
                        "shelves shrink and member geometries diverge it mixes forcing with\n"
                        "geometry. SSP585-3X lost ~44% of floating area by yr160 — treat its\n"
                        "y-value as an upper bound.",
            transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=.9))
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/meeting_design_matrix.png", dpi=130); plt.close(fig)

    print(f"\nwrote 3 figures to {a.out_dir}/")


if __name__ == "__main__":
    main()
