#!/usr/bin/env python3
"""
fig_forcing_spectrum_real.py — data version of the F_s/F_v decomposition.

Computes domain-mean Welch power spectra of real forcing components (F_v, F_s)
to show total is F_s-dominated (1-yr spike) while F_v is low-frequency-rich.
Data: Orig_VAR_SSN files, variable timeMonthly_avg_landIceFreshwaterFlux.
"""
from __future__ import annotations
import argparse, os
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def domain_mean_ts(path, var, chunk=120):
    """Spatial-mean per timestep -> 1-D series (nan/fill-aware, chunked over Time)."""
    d = Dataset(path); v = d.variables[var]; v.set_auto_mask(False)
    fv = None
    for a in ("_FillValue", "missing_value"):
        if a in v.ncattrs():
            fv = float(v.getncattr(a)); break
    nt = v.shape[0]; out = np.empty(nt)
    for t0 in range(0, nt, chunk):
        t1 = min(t0 + chunk, nt)
        b = np.asarray(v[t0:t1], float).reshape(t1 - t0, -1)
        good = np.isfinite(b)
        if fv is not None:
            good &= (b != fv)
        b[~good] = np.nan
        out[t0:t1] = np.nanmean(b, axis=1)
    d.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fv", required=True, help="variability (F_v) file")
    ap.add_argument("--fs", help="seasonality (F_s) file; if given, total = F_s + F_v is also plotted")
    ap.add_argument("--var", default="timeMonthly_avg_landIceFreshwaterFlux")
    ap.add_argument("--dt-months", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(REPO, "reports/figures/forcing_spectrum_real.png"))
    a = ap.parse_args()

    print("computing domain-mean series (this reads the full files; a few minutes each)...")
    fv = domain_mean_ts(a.fv, a.var)
    fs = domain_mean_ts(a.fs, a.var) if a.fs else None
    fs2 = fs[:len(fv)] if fs is not None else None
    total = (fv + fs2) if fs2 is not None else None

    fsamp = 12.0 / a.dt_months
    def psd(x):
        f, P = welch(x - np.nanmean(x), fs=fsamp, nperseg=min(len(x), 2048))
        m = f > 0
        return 1.0 / f[m], P[m]                     # period (yr), power

    fig, ax = plt.subplots(figsize=(9, 5.2))
    if total is not None:
        pt, Pt = psd(total); ax.loglog(pt, Pt, color="0.35", lw=1.8, label="total forcing (F$_s$+F$_v$)")
    pv, Pv = psd(fv); ax.loglog(pv, Pv, color="#D55E00", lw=1.8, label="F$_v$ only (variability)")
    for x0, x1, c in [(0.1, 1.5, "#E69F00"), (1.5, 8, "#F0E442"), (8, 30, "#009E73"), (30, 1e4, "#0072B2")]:
        ax.axvspan(x0, x1, color=c, alpha=0.10)
    ax.axvline(1.0, color="#0072B2", ls=":", lw=1)
    ax.set_xlabel("period (years)  —  seasonal ← → multidecadal"); ax.set_ylabel("power")
    ax.set_title("SORRM forcing spectrum (domain-mean): total is F$_s$-dominated (1-yr spike); "
                 "F$_v$ is low-frequency-rich", fontsize=10, loc="left")
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150); print("wrote", a.out)


if __name__ == "__main__":
    main()
