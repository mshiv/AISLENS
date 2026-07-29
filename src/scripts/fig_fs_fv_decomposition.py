#!/usr/bin/env python3
"""
fig_fs_fv_decomposition.py — ILLUSTRATIVE schematic explaining why AISLENS analyzes the stochastic
variability F_v (not the deterministic seasonal cycle F_s), and why F_v still carries sub-1.5-yr ("seasonal
band") power even though the mean annual cycle was removed.

This is a SCHEMATIC on synthetic data designed to mimic the decomposition — NOT SORRM output. The F_v
spectrum is shaped to the real ALL-shelf band split (seasonal 35 / interannual 22 / decadal 22 /
multidecadal 21). Output: reports/figures/fs_fv_decomposition.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rng = np.random.default_rng(7)

# ---- synthetic monthly series over 500 yr ----
yrs = 500; n = yrs * 12
t = np.arange(n) / 12.0
season = np.cos(2 * np.pi * t)                      # annual shape (period 1 yr)

# F_s: DETERMINISTIC mean seasonal cycle — identical every year. Large, so it dominates the TOTAL
# variance (the aggregate "94% seasonal"); F_v (removed) is what carries the projection uncertainty.
A_s = 4.0
F_s = A_s * season

# F_v: STOCHASTIC anomalies = (a) low-frequency variability (interannual->multidecadal) +
#      (b) YEAR-TO-YEAR modulation of the seasonal cycle (residual near 1/yr) + (c) small white noise
def lowfreq(periods, amps):
    x = np.zeros(n)
    for p, a in zip(periods, amps):
        x += a * np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi))
    return x
lf = lowfreq([3, 6, 15, 40, 120], [0.5, 0.6, 0.7, 0.7, 0.6])        # interannual..multidecadal
amp_anom = np.repeat(rng.normal(0, 0.45, yrs), 12)                   # each year's seasonal amplitude anomaly
seasonal_modulation = amp_anom * 1.2 * season                        # <-- residual sub-1.5yr power in F_v
F_v = lf + seasonal_modulation + 0.15 * rng.standard_normal(n)
total = F_s + F_v

# ---- figure ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8))

# Panel A: 6-yr time-series zoom
m = t <= 6
axA.plot(t[m], total[m], color="0.35", lw=1.6, label="total forcing = F$_s$ + F$_v$")
axA.plot(t[m], F_s[m], color="#0072B2", lw=2.0, label="F$_s$: deterministic seasonal cycle")
axA.plot(t[m], F_v[m], color="#D55E00", lw=1.3, label="F$_v$: stochastic variability")
axA.axhline(0, color="0.8", lw=0.7)
axA.set_xlabel("year"); axA.set_ylabel("anomaly")
axA.set_title("(a)", loc="left", fontsize=11)
axA.legend(fontsize=9, loc="upper right")

# Panel B: spectra (Welch), period axis
fs = 12.0
def psd(x):
    f, P = welch(x - x.mean(), fs=fs, nperseg=min(len(x), 4096))
    good = f > 0
    return 1.0 / f[good], P[good]           # period (yr), power
per_t, P_t = psd(total)
per_v, P_v = psd(F_v)
axB.loglog(per_t, P_t, color="0.35", lw=1.8, label="total (F$_s$+F$_v$)")
axB.loglog(per_v, P_v, color="#D55E00", lw=1.8, label="F$_v$ only")
for x0, x1, c, lab in [(0.0, 1.5, "#E69F00", "seasonal"), (1.5, 8, "#F0E442", "interannual"),
                       (8, 30, "#009E73", "decadal"), (30, 1e4, "#0072B2", "multidecadal")]:
    axB.axvspan(max(x0, 0.1), x1, color=c, alpha=0.10)
axB.axvline(1.0, color="#0072B2", ls=":", lw=1)
axB.set_xlabel("period (years)"); axB.set_ylabel("power")
axB.set_title("(b)", loc="left", fontsize=11)
axB.legend(fontsize=9, loc="upper right")

fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.12, wspace=0.22)
out = os.path.join(REPO, "reports/figures/fs_fv_decomposition.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print("wrote", out)
