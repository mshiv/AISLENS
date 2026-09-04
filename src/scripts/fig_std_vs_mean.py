#!/usr/bin/env python3
"""Two-panel: (a) ensemble std dev, (b) ensemble mean — both as time series.
Default variable: totalIceVolume (m^3 → 10^6 km^3 for readability).
Use --variable to switch (e.g. volumeAboveFloatation).
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio

SCEN = [("CTRL", r"^CTRL_\d+$", "#444444"),
        ("SSP126", r"^SSP126_\d+$", "#0072B2"),
        ("SSP585", r"^SSP585_\d+$", "#D55E00"),
        ("SSP585_varScaled10x", r"^SSP585_\d+$", "#7B3FA0"),
        ("SSP585-3X", r"^SSP585-3X_\d+$", "#8B0000")]


def _despike(vals_m, years, window=5, thresh=5):
    """Replace single-timestep outliers with NaN via a Hampel filter on dV.

    Work on **raw first-differences** (dV), not rates (dV/dt), so that
    sub-annual output spacing near restart boundaries does not create false
    positives: a normal ~10 km^3 change over 0.01 yr should not be flagged
    even though its rate is ~1000 km^3/yr.

    After identifying anomalous jumps, we flag the *value at the end* of the
    interval.  Consecutive flagged jumps (same value flagged from left and
    right) mark a single-point spike at that value.
    """
    n = len(vals_m)
    out = vals_m.copy().astype(float)
    dv = np.diff(vals_m)
    flagged = np.zeros(len(dv), dtype=bool)
    for i in range(len(dv)):
        lo = max(0, i - window // 2)
        hi = min(len(dv), i + window // 2 + 1)
        seg = dv[lo:hi]
        seg_ok = seg[np.isfinite(seg)]
        if len(seg_ok) < 3:
            continue
        med = np.median(seg_ok)
        mad = np.median(np.abs(seg_ok - med))
        if mad == 0 or not np.isfinite(mad):
            continue
        if abs(dv[i] - med) > thresh * max(mad, 1e-300):
            flagged[i] = True
    for i in range(len(flagged)):
        if flagged[i]:
            out[i + 1] = np.nan
    for i in range(1, len(flagged)):
        if flagged[i - 1] and flagged[i]:
            out[i] = np.nan
    return out


def load_var(root, ensemble, include, variable, min_years, align, regrid_dt=1.0,
             min_members=3, despike_thresh=0, mask_years=None):
    """Load an ensemble variable and (by default) regrid every member onto a common
    regular time axis before any ensemble statistic is taken.

    Why regridding matters: MALI writes globalStats at irregular intervals, and the
    interval differs between members (e.g. SSP585_00 has sub-day spacing near yr 270
    where other members are ~12-daily). With align="union" the members are merged onto
    the union of their timestamps, so at a timestamp belonging to only one or two
    members the ensemble std is computed over 1-2 samples -> spurious spikes (this is
    the origin of the apparent yr 270-272 jump in SSP585; the underlying VAF/volume is
    smooth there). Interpolating all members onto a shared grid removes the artefact
    without smoothing the physical signal.

    Set regrid_dt=0 to disable and reproduce the old raw-axis behaviour.
    """
    ens_dir = os.path.join(root, ensemble)
    ds = eio.load_ensemble_globalstats(
        ens_dir, variables=[variable, "daysSinceStart"],
        include=include, min_years=min_years, align=align,
    )
    x = ds[variable]
    years = ds["year"].values
    ds.close()

    vals = np.asarray(x.values, dtype=float)          # (member, time)

    if mask_years:
        for seg in mask_years:
            lo, hi = map(float, seg.split("-"))
            msk = (years >= lo) & (years <= hi)
            n = int(msk.sum())
            if n:
                vals[:, msk] = np.nan
                print(f"  mask {ensemble}: years {lo}-{hi} ({n} timesteps NaNed)")

    if despike_thresh > 0:
        n_spikes = 0
        for m in range(vals.shape[0]):
            clean = _despike(vals[m], years, thresh=despike_thresh)
            n_spikes += int(np.sum(np.isfinite(vals[m]) & ~np.isfinite(clean)))
            vals[m] = clean
        if n_spikes:
            print(f"  despike: flagged {n_spikes} point(s) across {vals.shape[0]} members")

    if not regrid_dt or regrid_dt <= 0:
        da = xr.DataArray(vals, dims=("member", "time"),
                          coords={"member": x["member"].values if "member" in x.coords
                                  else np.arange(vals.shape[0]),
                                  "time": years})
        return da, years

    grid = np.arange(np.floor(np.nanmin(years)), np.nanmax(years) + 1e-9, regrid_dt)
    out = np.full((vals.shape[0], grid.size), np.nan)
    for m in range(vals.shape[0]):
        ok = np.isfinite(vals[m]) & np.isfinite(years)
        if ok.sum() < 2:
            continue
        # never extrapolate past a member's own record
        out[m] = np.interp(grid, years[ok], vals[m][ok], left=np.nan, right=np.nan)

    # require enough members at each time for a meaningful std
    counts = np.sum(np.isfinite(out), axis=0)
    out[:, counts < min_members] = np.nan

    da = xr.DataArray(out, dims=("member", "time"),
                      coords={"member": x["member"].values if "member" in x.coords
                              else np.arange(vals.shape[0])})
    return da, grid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--variable", default="totalIceVolume",
                    help="globalStats variable (default: totalIceVolume)")
    ap.add_argument("--min-years", type=float, default=50.0)
    ap.add_argument("--align", default="union")
    ap.add_argument("--regrid-dt", type=float, default=1.0,
                    help="regrid all members onto a common time step (yr) before taking "
                         "ensemble statistics; removes the varying-member-count artefact from "
                         "irregular per-member output spacing (0 = disable, use raw axis)")
    ap.add_argument("--mask-years", action="append", default=None,
                    help="Range(s) to NaN out entirely (all members), e.g. "
                         "--mask-years 269-273. Repeat for multiple ranges. "
                         "Catches the SSP585 restart artifact at yr 269-273.")
    ap.add_argument("--despike-thresh", type=float, default=0.0,
                    help="Hampel filter threshold for per-member spike removal on dV. "
                         "A value's first-difference from its (window/2)-neighbourhood median "
                         "must exceed thresh×MAD to be flagged (0 = disable).  "
                         "Default 10 catches SSP585 yr-270 restart artifacts (dev/mad > 300) "
                         "while sparing genuine interannual dV variations (typically < 8×).")
    ap.add_argument("--min-members", type=int, default=3,
                    help="minimum members required at a time step for mean/std to be plotted")
    ap.add_argument("--smooth", type=int, default=0,
                    help="cosmetic: running-median window (yr) applied to the sigma curve only. "
                         "Residual sub-decadal wiggles in sigma are ~1e-5 of the mean and partly "
                         "reflect differing sub-annual output sampling between members; smoothing "
                         "suppresses them for presentation. DISCLOSE if used. 0 = off (raw).")
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--sigma-end", action="append", default=None,
                    metavar="ENS=YEAR",
                    help="stop the sigma curve for one ensemble at YEAR, e.g. "
                         "--sigma-end SSP585-3X=185. Use where a single member runs away "
                         "before the others and the ensemble standard deviation stops "
                         "describing the ensemble. Repeat for multiple ensembles.")
    ap.add_argument("--sigma-full-only", action="store_true",
                    help="plot the sigma curve only where every member of that ensemble is "
                         "still present. Ensemble spread is not comparable across a changing "
                         "sample: as members terminate, sigma jumps when a diverging member "
                         "leaves and again when the sample shrinks. Affects the sigma panel "
                         "only; the mean is plotted over whatever members exist.")
    ap.add_argument("--no-member-counts", action="store_true",
                    help="omit '(n=N)' from legend labels (publication style)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    RHO_ICE = eio.RHO_ICE

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ens, inc, col in SCEN:
        try:
            x, yr = load_var(a.root, ens, inc, a.variable, a.min_years, a.align,
                             regrid_dt=a.regrid_dt, min_members=a.min_members,
                             despike_thresh=a.despike_thresh,
                             mask_years=a.mask_years)
        except Exception as e:
            print(f"{ens}: skip ({e})")
            continue
        vals = np.asarray(x.values, dtype=float)  # (member, time)
        if a.variable == "volumeAboveFloatation":
            conv = eio.vaf_to_sle_mm(vals, reference="first")
            unit_label = "(mm SLE)"
        elif a.variable == "totalIceVolume":
            conv = vals * RHO_ICE / 1e12
            unit_label = "(Gt)"
        else:
            conv = vals * 1e-15
            unit_label = "($10^6$ km$^3$)"
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(conv, axis=0)
            sig = np.nanstd(conv, axis=0, ddof=1)
        cap = dict(kv.split("=") for kv in (a.sigma_end or []))
        if ens in cap:
            sig = np.where(yr <= float(cap[ens]), sig, np.nan)
            print(f"  {ens}: sigma stopped at yr {cap[ens]} by --sigma-end")
        if a.sigma_full_only:
            present = np.sum(np.isfinite(conv), axis=0)
            full = present.max()
            sig = np.where(present == full, sig, np.nan)
            if (present < full).any():
                lost = yr[np.argmax(present < full)]
                print(f"  {ens}: sigma truncated at yr {lost:.0f}; "
                      f"{full} members before, {present.min()} at the end")
        if a.smooth and a.smooth > 1:
            w = int(a.smooth); half = w // 2
            sm = np.full_like(sig, np.nan)
            for i in range(sig.size):
                seg = sig[max(0, i - half): i + half + 1]
                seg = seg[np.isfinite(seg)]
                if seg.size:
                    sm[i] = np.median(seg)
            sig = sm
        n = x.sizes["member"]
        lbl = ens if a.no_member_counts else f"{ens} (n={n})"

        axL.plot(yr, sig, col, lw=2, label=lbl)
        axR.plot(yr, mean, col, lw=2, label=lbl)

        sig_end = sig[-1] if np.isfinite(sig[-1]) else np.nan
        mean_end = mean[-1] if np.isfinite(mean[-1]) else np.nan
        print(f"{ens}: n={n}, {a.variable} σ@end={sig_end:.4g}, mean@end={mean_end:.4g}")

    axL.set_xlabel("year")
    axL.set_ylabel(f"ensemble σ {unit_label}")
    axL.set_title(f"(a) spread of {a.variable}")
    axL.legend(fontsize=8)
    axL.grid(alpha=0.3)

    axR.set_xlabel("year")
    axR.set_ylabel(f"ensemble mean {unit_label}")
    axR.set_title(f"(b) mean of {a.variable}")
    axR.legend(fontsize=8)
    axR.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(a.out_dir, f"std_vs_mean_{a.variable}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
