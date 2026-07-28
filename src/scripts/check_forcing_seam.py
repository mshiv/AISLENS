#!/usr/bin/env python3
"""
check_forcing_seam.py — verify forcing continuity across the 2300 trend-freeze seam.

Aggregates the forcing field to 16 ISMIP6 basins (area-weighted), computes annual means
to isolate the deterministic trend, and tests whether the 2300 jump exceeds normal
year-to-year scatter. Options: monthly zoom, cross-file comparison with the 300yr
generation, and trend-file continuity check.

Memory-safe: reads one year (12 steps x nCells) at a time.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
from netCDF4 import Dataset, chartostring

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def decode_names(arr):
    out = []
    for r in range(arr.shape[0]):
        s = chartostring(arr[r]).item() if hasattr(chartostring(arr[r]), "item") else "".join(
            c.decode() if isinstance(c, bytes) else c for c in arr[r])
        out.append("".join(ch for ch in str(s).strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forcing", required=True, help="34GB 2000-3000 forcing file (Time,nCells)")
    ap.add_argument("--region-mask", required=True,
                    help="regionMask_ismip6.nc with regionCellMasks(nCells,nRegions)+regionNames")
    ap.add_argument("--varname", default="floatingBasalMassBalAdjustment")
    ap.add_argument("--start-year", type=int, default=2000, help="calendar year of Time index 0")
    ap.add_argument("--seam-year", type=int, default=2300, help="year the trend is frozen from")
    ap.add_argument("--broad-years", type=int, default=100,
                    help="+/- years around the seam for the annual-mean trend figure")
    ap.add_argument("--fine-years", type=int, default=12,
                    help="+/- years around the seam for the monthly zoom figure")
    ap.add_argument("--area-var", default="areaCell",
                    help="cell-area var for area weighting (looked up in forcing, mask, then --mesh)")
    ap.add_argument("--mesh", default=None,
                    help="optional MPAS mesh/restart file carrying areaCell (variable-resolution "
                         "mesh -> area weighting; otherwise basin means are unweighted)")
    ap.add_argument("--trend", default=None,
                    help="optional: scenario trend file to cross-check the frozen 2300 slice")
    ap.add_argument("--forcing-300", default=None,
                    help="optional: the DIFFERENT-generation 300yr forcing file the base run "
                         "(<=2300) used. Overlaid on the annual-trend figure so the EXPECTED "
                         "variability-generation difference is visible next to any real trend step.")
    ap.add_argument("--inspect", action="store_true",
                    help="just print variables/dims/time coverage of the input file(s) and exit "
                         "(cheap: metadata only). Run this FIRST to confirm --varname before the big job.")
    ap.add_argument("--out-prefix", default="forcing_seam_check",
                    help="output path prefix for figures/table")
    a = ap.parse_args()

    if not os.path.isfile(a.forcing):
        sys.exit(f"not found: {a.forcing}")

    # ---- cheap inspect mode: confirm variable names before a 34GB job ----
    if a.inspect:
        for tag, p in (("forcing", a.forcing), ("region-mask", a.region_mask),
                       ("forcing-300", a.forcing_300)):
            if not p or not os.path.isfile(p):
                continue
            print(f"\n===== {tag}: {p} =====")
            d = Dataset(p)
            print("dims:", {k: len(v) for k, v in d.dimensions.items()})
            cand = []
            for name, var in d.variables.items():
                dims = ",".join(var.dimensions)
                print(f"  {name:32s} ({dims})  {var.shape}  {var.dtype}")
                if "Time" in var.dimensions and "nCells" in var.dimensions:
                    cand.append(name)
            if "xtime" in d.variables:
                xt = d.variables["xtime"]
                first = "".join(chartostring(xt[0]).item()) if xt.ndim == 2 else str(xt[0])
                last = "".join(chartostring(xt[-1]).item()) if xt.ndim == 2 else str(xt[-1])
                print(f"  xtime coverage: {str(first).strip()} .. {str(last).strip()}")
            if tag == "forcing":
                print(f"  --> (Time,nCells) candidates for --varname: {cand}")
            d.close()
        print("\ninspect done.")
        return

    if not os.path.isfile(a.region_mask):
        sys.exit(f"not found: {a.region_mask}")
    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)

    # ---- region masks + names ----
    rm = Dataset(a.region_mask)
    masks = np.asarray(rm.variables["regionCellMasks"][:])          # (nCells, nRegions)
    names = decode_names(rm.variables["regionNames"][:]) if "regionNames" in rm.variables \
        else [f"reg{r}" for r in range(masks.shape[1])]
    nRegions = masks.shape[1]

    fd = Dataset(a.forcing)
    if a.varname not in fd.variables:
        cand = [n for n, var in fd.variables.items()
                if "Time" in var.dimensions and "nCells" in var.dimensions]
        sys.exit(f"'{a.varname}' not in forcing file. (Time,nCells) candidates: {cand}\n"
                 f"Re-run with --varname <name> (or --inspect to see everything).")
    v = fd.variables[a.varname]
    if v.dimensions[0] != "Time":
        sys.exit(f"expected Time as dim 0 of {a.varname}, got {v.dimensions}")
    Nt = v.shape[0]
    nCells = v.shape[1]
    if masks.shape[0] != nCells:
        sys.exit(f"mask nCells {masks.shape[0]} != forcing nCells {nCells}")

    # ---- cell areas (area-weighted basin mean); fall back to unweighted ----
    area = None
    md = Dataset(a.mesh) if (a.mesh and os.path.isfile(a.mesh)) else None
    for src in (fd, rm, md):
        if src is not None and a.area_var in src.variables:
            area = np.asarray(src.variables[a.area_var][:], dtype=float)
            print(f"area weighting: using {a.area_var} from {src.filepath()}")
            break
    if md is not None:
        md.close()
    if area is None:
        print(f"[warn] {a.area_var} not found in forcing/mask/--mesh -> UNWEIGHTED basin means "
              f"(fine for step detection; pass --mesh <file with areaCell> for weighted magnitudes)")
        area = np.ones(nCells, dtype=float)
    elif area.shape[0] != nCells:
        sys.exit(f"{a.area_var} length {area.shape[0]} != nCells {nCells}")

    # precompute per-basin cell indices + weights
    basin_cells, basin_w, basin_wsum = [], [], []
    for r in range(nRegions):
        idx = np.nonzero(masks[:, r] > 0)[0]
        basin_cells.append(idx)
        w = area[idx]
        basin_w.append(w)
        basin_wsum.append(w.sum() if w.size else np.nan)

    def basin_mean_block(block):
        """block: (nt, nCells) -> (nt, nRegions) area-weighted mean per basin."""
        out = np.full((block.shape[0], nRegions), np.nan)
        for r in range(nRegions):
            idx = basin_cells[r]
            if idx.size:
                out[:, r] = (block[:, idx] * basin_w[r]).sum(axis=1) / basin_wsum[r]
        return out

    # ---- scan a year-by-year window around the seam (bounded RAM) ----
    y0 = max(a.start_year, a.seam_year - a.broad_years)
    y1 = min(a.start_year + Nt // 12 - 1, a.seam_year + a.broad_years)
    years = list(range(y0, y1 + 1))
    ann = np.full((len(years), nRegions), np.nan)         # annual-mean basin field
    monthly = {}                                          # year -> (12, nRegions), only near seam
    fine_lo, fine_hi = a.seam_year - a.fine_years, a.seam_year + a.fine_years
    print(f"Scanning {y0}..{y1} ({len(years)} yr) of {a.varname}  nCells={nCells} Nt={Nt}")
    for k, yr in enumerate(years):
        i0 = (yr - a.start_year) * 12
        i1 = min(i0 + 12, Nt)
        block = np.asarray(v[i0:i1, :], dtype=float)      # (<=12, nCells)
        bm = basin_mean_block(block)                      # (<=12, nRegions)
        ann[k] = np.nanmean(bm, axis=0)
        if fine_lo <= yr <= fine_hi:
            monthly[yr] = bm
    fd.close(); rm.close()

    # ---- optional: annual-mean trend from the DIFFERENT-generation 300yr file ----
    # (what the base run used <=2300). Trend+mean are supposed to match; only the
    # variability generation differs. So the ANNUAL means should overlap the 1000yr
    # file's; a gap here would mean the trend/mean themselves differ between builds.
    ann300 = np.full((len(years), nRegions), np.nan)
    if a.forcing_300 and os.path.isfile(a.forcing_300):
        gd = Dataset(a.forcing_300)
        if a.varname in gd.variables:
            gv = gd.variables[a.varname]
            if gv.shape[1] == nCells:
                Nt3 = gv.shape[0]
                last3 = a.start_year + Nt3 // 12 - 1
                for k, yr in enumerate(years):
                    if yr > min(a.seam_year - 1, last3):
                        continue
                    i0 = (yr - a.start_year) * 12
                    if i0 >= Nt3:
                        continue
                    blk = np.asarray(gv[i0:min(i0 + 12, Nt3), :], dtype=float)
                    ann300[k] = np.nanmean(basin_mean_block(blk), axis=0)
                print(f"300yr cross-file: read {a.forcing_300} (ends {last3})")
            else:
                print(f"[warn] --forcing-300 nCells {gv.shape[1]} != {nCells}; skipping overlay")
        else:
            print(f"[warn] '{a.varname}' not in --forcing-300; skipping overlay")
        gd.close()
    have300 = np.isfinite(ann300).any()

    yrs = np.array(years)
    seam = a.seam_year
    ks = np.where(yrs == seam)[0]
    if ks.size == 0:
        sys.exit(f"seam year {seam} not in scanned range")
    ks = ks[0]

    # ---- diagnostics ----
    pre = yrs < seam
    post = yrs >= seam
    # year-to-year annual-mean scatter BEFORE the seam (natural variability of the annual mean)
    d_pre = np.diff(ann[pre], axis=0)
    yy_std = np.nanstd(d_pre, axis=0)                     # (nRegions,)
    jump = ann[ks] - ann[ks - 1]                          # annual-mean(2300) - (2299)
    zscore = np.where(yy_std > 0, jump / yy_std, np.nan)
    # post-seam slope of the annual mean (should be ~0 if trend truly frozen)
    post_slope = np.full(nRegions, np.nan)
    yp = yrs[post].astype(float)
    if yp.size >= 3:
        for r in range(nRegions):
            post_slope[r] = np.polyfit(yp - yp[0], ann[post, r], 1)[0]
    # recent PRE-seam slope (context: user reports the trend declines into 2300)
    pre_slope = np.full(nRegions, np.nan)
    recent = (yrs >= seam - 20) & (yrs < seam)
    yr_ = yrs[recent].astype(float)
    if yr_.size >= 3:
        for r in range(nRegions):
            pre_slope[r] = np.polyfit(yr_ - yr_[0], ann[recent, r], 1)[0]

    # ---- optional trend-file cross-check (frozen slice continuity) ----
    trend_note = ""
    if a.trend:
        try:
            td = Dataset(a.trend)
            tvn = a.varname if a.varname in td.variables else a.varname + "_var"
            tv = td.variables[tvn]
            last = basin_mean_block(np.asarray(tv[-1:], dtype=float))[0]
            prev = basin_mean_block(np.asarray(tv[-2:-1], dtype=float))[0]
            td.close()
            trend_note = ("\nTrend-file frozen-slice check (last vs 2nd-last basin mean; "
                          "should be ~equal = continuous freeze):\n")
            for r in range(nRegions):
                trend_note += f"  {names[r]:20s} last={last[r]:+.4e} prev={prev[r]:+.4e} d={last[r]-prev[r]:+.2e}\n"
        except Exception as e:
            trend_note = f"\n[warn] trend cross-check failed: {e}\n"

    # ---- verdict table ----
    lines = []
    lines.append(f"# Forcing seam continuity check: {a.varname} at year {seam}")
    lines.append(f"# forcing: {a.forcing}")
    lines.append(f"# area-weighted basin means; annual mean isolates trend (seasonality removed)")
    lines.append(f"# seam Time index = {(seam - a.start_year)*12} (Jan {seam})")
    lines.append("")
    hdr = f"{'basin':20s} {'ann(2299)':>12s} {'ann(2300)':>12s} {'jump':>11s} " \
          f"{'yy_std':>9s} {'z':>7s} {'pre_slope/yr':>12s} {'post_slope/yr':>13s}  verdict"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    n_flag = 0
    for r in range(nRegions):
        flag = ""
        if abs(zscore[r]) > 3:                      # jump far outside normal annual scatter
            flag = "STEP?"; n_flag += 1
        elif abs(post_slope[r]) > 3 * (abs(pre_slope[r]) + 1e-30) and abs(post_slope[r]) > abs(yy_std[r]):
            flag = "post-trend-leak?"
        lines.append(f"{names[r]:20s} {ann[ks-1,r]:12.4e} {ann[ks,r]:12.4e} {jump[r]:+11.3e} "
                     f"{yy_std[r]:9.3e} {zscore[r]:+7.1f} {pre_slope[r]:+12.3e} {post_slope[r]:+13.3e}  {flag}")
    lines.append("")
    lines.append(f"SUMMARY: {n_flag}/{nRegions} basins have |z|>3 at the seam "
                 f"(annual-mean jump far outside normal year-to-year scatter).")
    lines.append("  z small everywhere  => forcing FILE is continuous at 2300 => investigate the MODEL")
    lines.append("     restart/forcing-index (xtime alignment, streams) as the source of the melt step.")
    lines.append("  z large             => the forcing FILE steps at 2300 => construction bug "
                 "(trend 2300 slice / off-by-one / negAdj).")
    if have300:
        lines.append("")
        lines.append("# CROSS-FILE (the actual run seam): 300yr file (<=2300) vs 1000yr file (>2300).")
        lines.append("# annual means SHOULD match (same trend+mean); a gap = trend/mean differ between builds.")
        lines.append(f"{'basin':20s} {'ann300(2299)':>13s} {'ann1000(2299)':>14s} "
                     f"{'ann1000(2300)':>14s} {'cross_jump':>11s}")
        for r in range(nRegions):
            a300 = ann300[ks - 1, r]
            cross = ann[ks, r] - a300
            lines.append(f"{names[r]:20s} {a300:13.4e} {ann[ks-1,r]:14.4e} "
                         f"{ann[ks,r]:14.4e} {cross:+11.3e}")
        lines.append("  NOTE: monthly variability WILL differ across the two generations by design;")
        lines.append("  that is expected and shows only in the monthly-zoom figure, not these annual means.")
    lines.append(trend_note)
    report = "\n".join(lines)
    print(report)
    with open(a.out_prefix + "_report.txt", "w") as f:
        f.write(report)

    # ---- figures ----
    ncol, nrow = 4, int(np.ceil(nRegions / 4))

    def grid(title, plot_fn, fname):
        fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), squeeze=False)
        fig.suptitle(title, fontsize=11)
        for r in range(nRegions):
            ax = axs[r // ncol][r % ncol]
            plot_fn(ax, r)
            ax.axvline(seam, color="k", lw=1.2, ls="--")
            ax.set_title(names[r], fontsize=9)
            ax.grid(alpha=0.3)
        for r in range(nRegions, nrow * ncol):
            axs[r // ncol][r % ncol].axis("off")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(fname, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {fname}")

    # (1) annual-mean trend across the seam (1000yr solid; 300yr overlay dashed if given)
    def _annual(ax, r):
        if have300:
            m = np.isfinite(ann300[:, r])
            if m.any():
                ax.plot(yrs[m], ann300[m, r], "--", color="C1", lw=1.3,
                        label="300yr (base run)")
        ax.plot(yrs, ann[:, r], "-", color="C0", lw=1.5, label="1000yr (extension)")
        if r == 0:
            ax.legend(fontsize=6, loc="best")
    grid(f"{a.varname}: annual-mean basin forcing across {seam} seam (trend; seasonality removed)",
         _annual, a.out_prefix + "_annual_trend.png")

    # (2) monthly zoom on the seam
    fine_years = sorted(monthly)
    if fine_years:
        tmonth, series = [], []
        for yr in fine_years:
            bm = monthly[yr]
            for m in range(bm.shape[0]):
                tmonth.append(yr + m / 12.0)
                series.append(bm[m])
        tmonth = np.array(tmonth); series = np.array(series)   # (T, nRegions)
        grid(f"{a.varname}: monthly basin forcing, seam zoom (continuity + step test)",
             lambda ax, r: ax.plot(tmonth, series[:, r], "-", color="C3", lw=1.0),
             a.out_prefix + "_monthly_zoom.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
