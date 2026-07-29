#!/usr/bin/env python3
"""
fig_cv_diagnostic.py — honest visualization of relative ensemble spread.

The naive CV = sigma / |mean change| spikes wherever the ensemble-MEAN change crosses ~0
(the denominator -> 0). Those spikes are a DIVISION ARTIFACT, not a variability peak.

This figure makes that explicit:
  Row 1 (twin axes): sigma(t) [left] and |mean change|(t) [right] — the two ingredients.
                     sigma is small and smooth; the spike comes from |mean| -> 0.
  Row 2: CV(t) raw (faint grey; the spikes) vs masked where |mean| >= floor (solid; honest).
Columns: CTRL, SSP126, SSP585.

Author: Shivaprakash Muruganandham (2026-07-22)
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from amplitude_response import load_sle

SCEN = [("CTRL", r"^CTRL_\d+$"), ("SSP126", r"^SSP126_\d+$"), ("SSP585", r"^SSP585_\d+$")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--min-years", type=float, default=50.0)
    ap.add_argument("--align", default="union")
    ap.add_argument("--floor-mm", type=float, default=5.0,
                    help="mask CV where |mean change| < this (mm); below it CV is a 1/0 artifact")
    ap.add_argument("--out-dir", default="reports")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    fig, axs = plt.subplots(2, len(SCEN), figsize=(4.3 * len(SCEN), 6.6), sharex=True)
    for k, (ens, inc) in enumerate(SCEN):
        try:
            sle, yr = load_sle(a.root, ens, inc, a.min_years, a.align)
        except Exception as e:
            print(f"{ens}: skip ({e})"); axs[0][k].set_title(f"{ens}: n/a"); continue
        mean = sle.mean("member").values
        sig = sle.std("member", ddof=1).values
        am = np.abs(mean)
        cv = 100.0 * sig / np.where(am > 0, am, np.nan)
        masked = np.where(am >= a.floor_mm, cv, np.nan)

        # --- Row 1: twin axes (the two ingredients) ---
        ax = axs[0][k]; ax2 = ax.twinx()
        l1, = ax.plot(yr, sig, "C0", lw=2, label="σ (ensemble spread)")
        l2, = ax2.plot(yr, am, "k", lw=1.5, label="|mean change|")
        # shade the artifact zone where |mean| < floor
        art = am < a.floor_mm
        if art.any():
            ax.axvspan(yr[art][0], yr[art][-1], color="red", alpha=0.08)
        ax.set_ylabel("σ  (mm SLE)", color="C0"); ax.tick_params(axis="y", colors="C0")
        ax2.set_ylabel("|mean change|  (mm)")
        ax.set_title(f"{ens}  (n={sle.sizes['member']})")
        ax.grid(alpha=0.2)
        if k == 0:
            ax.legend(handles=[l1, l2], fontsize=7, loc="upper left")

        # --- Row 2: CV raw vs masked ---
        bx = axs[1][k]
        bx.plot(yr, cv, color="0.75", lw=1.0, label="CV raw  (spike = |mean|→0 artifact)")
        bx.plot(yr, masked, "C3", lw=2.0, label=f"CV masked  (|mean|≥{a.floor_mm:g} mm)")
        if art.any():
            bx.axvspan(yr[art][0], yr[art][-1], color="red", alpha=0.08)
        cap = np.nanpercentile(masked, 95) * 1.4 if np.isfinite(masked).any() else 5.0
        bx.set_ylim(0, max(1.0, cap))
        bx.set_ylabel("CV = σ / |mean|  (%)"); bx.set_xlabel("year"); bx.grid(alpha=0.2)
        if k == 0:
            bx.legend(fontsize=7, loc="upper right")

        j = np.where(np.isfinite(masked))[0]
        if j.size:
            print(f"{ens}: CV@end={masked[j[-1]]:.2f}% (masked); raw peak={np.nanmax(cv):.0f}% "
                  f"(artifact); σ@end={sig[-1]:.2f} mm, |mean|@end={am[-1]:.0f} mm")

    fig.suptitle("Relative ensemble spread — the sharp CV spike is a |mean|→0 artifact, not a "
                 "variability peak\nTop: the two ingredients (σ small & smooth; |mean| grows).  "
                 "Bottom: masked CV is the honest signal (high at small signal, then decays).",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(a.out_dir, "cv_diagnostic.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
