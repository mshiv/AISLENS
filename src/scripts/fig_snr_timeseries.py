#!/usr/bin/env python3
"""
fig_snr_timeseries.py — signal-to-noise ratio |mean|/σ vs year.
SNR = 1 means the forced signal equals the internal-variability envelope.
The canonical emergence threshold is SNR = 2 (signal twice the noise).

CTRL is omitted because its mean ~ 0 makes SNR meaningless.
"""

from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from amplitude_response import load_sle

SCEN = [("SSP126", r"^SSP126_\d+$", "#0072B2"),
        ("SSP585", r"^SSP585_\d+$", "#D55E00")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--min-years", type=float, default=50.0)
    ap.add_argument("--align", default="union")
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--no-member-counts", action="store_true",
                    help="omit '(n=N)' from legend labels (publication style)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    snr_floor_mm = 1.0  # mask years where |mean| < 1 mm

    for ens, inc, col in SCEN:
        try:
            sle, yr = load_sle(a.root, ens, inc, a.min_years, a.align)
        except Exception as e:
            print(f"{ens}: skip ({e})"); continue

        mean = sle.mean("member").values
        sig = sle.std("member", ddof=1).values
        n = sle.sizes["member"]
        lbl = ens if a.no_member_counts else f"{ens} (n={n})"

        snr = np.full_like(mean, np.nan)
        ok = (sig > 0) & (np.abs(mean) >= snr_floor_mm)
        snr[ok] = np.abs(mean[ok]) / sig[ok]

        ax.plot(yr, snr, col, lw=2, label=lbl)
        good = np.where(np.isfinite(snr))[0]
        if good.size:
            t_emerge = None
            cross = np.where(snr[good] >= 2.0)[0]
            if cross.size:
                t_emerge = yr[good][cross[0]]
                ax.axvline(t_emerge, color=col, ls=":", lw=1, alpha=0.6)
                ax.text(t_emerge, ax.get_ylim()[1]*0.9, f" {t_emerge:.0f}", color=col, fontsize=8)
            print(f"{ens}: SNR@end={snr[good][-1]:.2f}, "
                  f"emergence (SNR≥2) @ yr {t_emerge}" if t_emerge else
                  f"{ens}: SNR@end={snr[good][-1]:.2f}, never emerges")

    ax.axhline(2, color="0.4", ls="--", lw=1, label="SNR = 2 (emergence)")
    ax.axhline(1, color="0.7", ls=":", lw=1, label="SNR = 1")
    ax.set_xlabel("year")
    ax.set_ylabel("signal-to-noise ratio |mean| / σ")
    ax.set_title("Signal-to-noise ratio of sea-level response")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    out = os.path.join(a.out_dir, "snr_timeseries.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
