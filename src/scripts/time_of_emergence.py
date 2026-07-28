#!/usr/bin/env python3
"""
time_of_emergence.py — when does the forced signal emerge from the variability envelope?

Computes forced signal S(t) = ensemble-mean VAF->SLE change, variability sigma(t)
from the control ensemble, and time of emergence = first year |S| > k*sigma.
Optionally per ISMIP6 basin from regionalStats.nc.

Author: Shivaprakash Muruganandham
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=eio.default_ensembles_root())
    p.add_argument("--forced", required=True, help="Forced ensemble (e.g. SSP585)")
    p.add_argument("--control", required=True,
                   help="Control ensemble providing the variability envelope (e.g. CTRL)")
    p.add_argument("--k", type=float, default=1.0,
                   help="Emergence threshold in units of sigma (default 1)")
    p.add_argument("--forced-members", default=None,
                   help=r"Regex selecting the forced-ensemble member subset, e.g. '^SSP585_\d+$'")
    p.add_argument("--control-members", default=None,
                   help=r"Regex selecting the control-ensemble member subset, e.g. '^CTRL-EM\d+$'")
    p.add_argument("--min-years", type=float, default=50.0,
                   help="Drop members shorter than this many years (default 50; "
                        "removes CHANGEPOINT-type test runs)")
    p.add_argument("--regional", action="store_true",
                   help="Also compute per-ISMIP6-basin emergence from regionalStats.nc")
    p.add_argument("--out-fig-dir", default=None)
    return p.parse_args()


def signal_and_sigma_global(root, forced, control, f_inc=None, c_inc=None, min_years=None):
    """Return (year, signal_mm, sigma_mm) at whole-AIS level, in SLE mm."""
    fds = eio.load_ensemble_globalstats(os.path.join(root, forced),
                                        variables=["volumeAboveFloatation", "daysSinceStart"],
                                        include=f_inc, min_years=min_years)
    cds = eio.load_ensemble_globalstats(os.path.join(root, control),
                                        variables=["volumeAboveFloatation", "daysSinceStart"],
                                        include=c_inc, min_years=min_years)
    year = fds["year"].values
    # forced signal = ensemble-mean SLE change
    f_sle = xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, "first"),
                           fds["volumeAboveFloatation"])
    signal = f_sle.mean("member").values
    # variability envelope = control-ensemble SLE spread, interpolated to forced years
    c_sle = xr.apply_ufunc(lambda a: eio.vaf_to_sle_mm(a, "first"),
                           cds["volumeAboveFloatation"])
    c_sigma = c_sle.std("member")
    c_sigma = c_sigma.interp(year=year).values
    return year, signal, c_sigma


def emergence_year(year, signal, sigma, k=1.0):
    """First year where |signal| > k*sigma and it stays that way to the end."""
    ratio = np.abs(signal) / np.where(sigma > 0, sigma, np.nan)
    above = ratio > k
    if not np.any(above):
        return np.nan
    # require it to remain above from that point onward
    for i in range(len(above)):
        if above[i] and np.all(above[i:][np.isfinite(ratio[i:])]):
            return float(year[i])
    return float(year[np.argmax(above)])  # fallback: first crossing


def main():
    args = parse_args()
    fig_dir = args.out_fig_dir or os.path.join(args.root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    year, signal, sigma = signal_and_sigma_global(
        args.root, args.forced, args.control,
        f_inc=args.forced_members, c_inc=args.control_members, min_years=args.min_years)
    toe = emergence_year(year, signal, sigma, args.k)

    # Two panels: (top) signal vs envelope at full scale, and a ZOOMED inset over the
    # early/emergence window where the envelope is actually resolvable; (bottom) the
    # signal-to-noise ratio, where emergence = the year S/sigma crosses k. The S/N panel
    # is the readable view: the +/-sigma envelope (~1 mm) is invisible next to a forced
    # signal of ~1800 mm, so the overlay alone looks empty even when correct.
    ratio = np.abs(signal) / np.where(sigma > 0, sigma, np.nan)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax.fill_between(year, -args.k * sigma, args.k * sigma, color="C1", alpha=0.35,
                    label=f"+/-{args.k:g} sigma internal variability ({args.control})")
    ax.plot(year, signal, "C0", lw=2, label=f"forced signal ({args.forced} mean)")
    if np.isfinite(toe):
        ax.axvline(toe, color="k", ls=":", lw=1.5)
    ax.set_ylabel("sea-level equivalent (mm)")
    ax.set_title(f"Time of emergence: {args.forced} vs {args.control} "
                 f"(envelope ~+/-{np.nanmax(sigma):.2g} mm)")
    ax.legend(loc="upper left")
    # zoomed inset over the emergence window
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="45%", height="45%", loc="lower right")
        emax = (toe * 2 if np.isfinite(toe) else year[np.isfinite(sigma)][-1])
        zmask = year <= max(emax, 20)
        axins.fill_between(year[zmask], -args.k * sigma[zmask], args.k * sigma[zmask],
                           color="C1", alpha=0.35)
        axins.plot(year[zmask], signal[zmask], "C0", lw=1.5)
        if np.isfinite(toe):
            axins.axvline(toe, color="k", ls=":", lw=1)
        axins.set_title("zoom: emergence window", fontsize=8)
        axins.tick_params(labelsize=7)
    except Exception:
        pass

    axr.axhline(args.k, color="0.5", ls="--", lw=1, label=f"emergence (S/N = {args.k:g})")
    axr.plot(year, ratio, "C3", lw=1.8)
    if np.isfinite(toe):
        axr.axvline(toe, color="k", ls=":", lw=1.5)
        axr.text(toe, axr.get_ylim()[1] * 0.85, f" emergence ~{toe:.0f}", fontsize=9)
    axr.set_xlabel("year"); axr.set_ylabel("signal / sigma  (S/N)")
    axr.set_yscale("log"); axr.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"time_of_emergence_{args.forced}_vs_{args.control}.png"),
                dpi=150)
    plt.close(fig)
    print(f"[whole-AIS] time of emergence (k={args.k}): "
          f"{toe:.0f}" if np.isfinite(toe) else "[whole-AIS] never emerges in record")

    # ---- Regional ----
    if args.regional:
        fm = eio.discover_members(os.path.join(args.root, args.forced), "regionalStats.nc",
                                  include=args.forced_members)
        cm = eio.discover_members(os.path.join(args.root, args.control), "regionalStats.nc",
                                  include=args.control_members)
        if not fm or not cm:
            print("Regional: missing regionalStats.nc in forced or control; skipping.")
            return
        # stack members (start each at year 0; drop runs shorter than --min-years)
        def stack_regional(members):
            arr, names = [], None
            for _, path in members:
                d = eio.to_year_dim(eio.load_member_regionalstats(path))
                d = d.assign_coords(year=d["year"] - float(d["year"].values[0]))
                span = float(d["year"].values[-1] - d["year"].values[0])
                if args.min_years is not None and span < args.min_years:
                    continue
                arr.append(d["regionalVolumeAboveFloatation"])
                names = eio.read_region_names(d) if "regionNames" in d else None
            if not arr:
                return None, None, None
            ref = min(arr, key=lambda a: a.sizes["year"])
            arr = [a.interp(year=ref["year"].values) for a in arr]
            da = xr.concat(arr, dim="member")
            return da, ref["year"].values, names
        f_da, f_year, names = stack_regional(fm)
        c_da, _, _ = stack_regional(cm)
        if f_da is None or c_da is None:
            print(f"Regional: no members survived the --min-years={args.min_years} filter; "
                  "loosen it or pass --forced-members/--control-members.")
            return
        if names is None:
            names = eio.region_names_default()  # from the ISMIP6 regionMask file
        if names is None:
            names = [f"region{i}" for i in range(f_da.sizes["nRegions"])]

        nreg = f_da.sizes["nRegions"]
        rows = int(np.ceil(nreg / 4))
        fig, axs = plt.subplots(rows, 4, figsize=(16, 3 * rows), squeeze=False)
        print("\nPer-basin time of emergence:")
        for r in range(nreg):
            fv = f_da.isel(nRegions=r)
            cv = c_da.isel(nRegions=r)
            f_sig = eio.vaf_to_sle_mm(fv.values, "first").mean(axis=0)
            c_sig = eio.vaf_to_sle_mm(cv.values, "first").std(axis=0)
            c_sig = np.interp(f_year, c_da["year"].values, c_sig)
            toe_r = emergence_year(f_year, f_sig, c_sig, args.k)
            ax = axs.flat[r]
            ax.fill_between(f_year, -args.k * c_sig, args.k * c_sig, color="C1", alpha=0.25)
            ax.plot(f_year, f_sig, "C0", lw=1.2)
            if np.isfinite(toe_r):
                ax.axvline(toe_r, color="k", ls=":", lw=1)
            ax.set_title(f"{names[r]}  ToE={toe_r:.0f}" if np.isfinite(toe_r)
                         else f"{names[r]}  ToE=—", fontsize=8)
            print(f"  {names[r]:20s} {toe_r:.0f}" if np.isfinite(toe_r)
                  else f"  {names[r]:20s} never")
        for r in range(nreg, rows * 4):
            axs.flat[r].axis("off")
        fig.suptitle(f"Per-basin time of emergence: {args.forced} vs {args.control}")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir,
                    f"time_of_emergence_regional_{args.forced}.png"), dpi=150)
        plt.close(fig)
        print(f"Figures -> {fig_dir}")


if __name__ == "__main__":
    main()
