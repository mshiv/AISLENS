#!/usr/bin/env python3
"""
fig_generator_phase_randomization.py — what the forcing generator keeps and what it resamples.

Parent SORRM melt anomaly, one phase-randomised realisation, then the leading mode's time
series (visibly different) beside its power spectrum (identical to machine precision). The
pair is the claim; neither panel makes it alone. Plain numpy SVD stands in for xeofs.
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = (f"{ROOT}/data/processed/"
       "SORRMv21_flux_detrend_uniform_deseasonalize_uniform_dedraft_anm_annual_50.nc")

SHOW_YEAR = 12      # year of the record displayed in (a) and (b)
N_MODES = 20        # EOF truncation
N_REAL = 40         # realisations behind the spectrum envelope
N_TRACE = 3         # realisations drawn in the time-series panel
SEED = 11


def phase_randomize(pcs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomise the Fourier phase of each PC, preserving its amplitude spectrum.

    rfft/irfft keeps the result real by construction; the DC term keeps its phase so
    the series mean is unchanged, and the Nyquist term is forced real.
    """
    spec = np.fft.rfft(pcs, axis=0)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=spec.shape))
    phase[0] = 1.0
    if pcs.shape[0] % 2 == 0:
        sgn = np.sign(phase[-1].real)
        phase[-1] = np.where(sgn == 0, 1.0, sgn)
    return np.fft.irfft(np.abs(spec) * phase, n=pcs.shape[0], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/reports/dissertation/figures/slides/"
                                     "fig_generator_phase_randomization.png")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ds.apply()
    rng = np.random.default_rng(SEED)

    d = xr.open_dataset(SRC)
    field = d["__xarray_dataarray_variable__"].values
    nt = field.shape[0]
    valid = np.isfinite(field).all(axis=0)
    X = field[:, valid]
    Xm = X.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(X - Xm, full_matrices=False)
    k = min(N_MODES, S.size)
    pcs, eofs = U[:, :k] * S[:k], Vt[:k]
    varexp = (S**2 / np.sum(S**2))[:k]

    recon = (phase_randomize(pcs, rng) @ eofs) + Xm

    # crop to the cells that actually carry signal, so the maps are legible on a slide
    rows, cols = np.where(valid)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

    def to_grid(vec):
        g = np.full(valid.shape, np.nan)
        g[valid] = vec
        return g[r0:r1, c0:c1]

    parent_map, real_map = to_grid(X[SHOW_YEAR]), to_grid(recon[SHOW_YEAR])

    freqs = np.fft.rfftfreq(nt, d=1.0)[1:]

    def psd(x):
        return (np.abs(np.fft.rfft(x - x.mean()))**2)[1:]

    parent_psd = psd(pcs[:, 0])
    traces, real_psds = [], []
    for i in range(N_REAL):
        p = phase_randomize(pcs, rng)[:, 0]
        real_psds.append(psd(p))
        if i < N_TRACE:
            traces.append(p)
    real_psds = np.array(real_psds)

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(13.0, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.25, 1.25], wspace=0.30,
                          left=0.012, right=0.985, top=0.78, bottom=0.16)

    def header(ax, title, sub):
        ax.text(0.0, 1.20, title, transform=ax.transAxes, fontsize=13,
                color=ds.INK, ha="left", va="bottom")
        ax.text(0.0, 1.05, sub, transform=ax.transAxes, fontsize=9.5,
                color=ds.INK_SOFT, ha="left", va="bottom")

    # (a) (b) maps -- shared colour limit so they are directly comparable
    lim = float(np.nanpercentile(np.abs(parent_map), 92))
    map_axes = []
    for j, (g, title, sub) in enumerate((
            (parent_map, "the parent", f"SORRMv2.1 melt anomaly · year {SHOW_YEAR}"),
            (real_map, "a realisation", f"phases resampled · year {SHOW_YEAR}"))):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(g, origin="lower", cmap=ds.DIVERGING, vmin=-lim, vmax=lim,
                  interpolation="bilinear")
        ax.axis("off")
        header(ax, title, sub)
        map_axes.append(ax)

    # the transformation, anchored to the parent panel so it cannot drift on reflow
    ax0 = map_axes[0]
    ax0.annotate("", xy=(1.20, 0.55), xytext=(1.03, 0.55),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=ds.INK_SOFT, lw=1.3))
    ax0.text(1.115, 0.60, f"{k} EOF modes\nrandom phases", transform=ax0.transAxes,
             fontsize=9, color=ds.INK_SOFT, ha="center", va="bottom", linespacing=1.5)

    # (c) the mode's history: visibly different
    axc = fig.add_subplot(gs[0, 2])
    yrs = np.arange(nt)
    sc = 1.0 / np.std(pcs[:, 0])
    for t in traces:
        axc.plot(yrs, t * sc, color=ds.ICE, lw=1.1, alpha=0.75)
    axc.plot(yrs, pcs[:, 0] * sc, color=ds.INK, lw=2.2)
    axc.axhline(0, color=ds.RULE, lw=0.9, zorder=0)
    ds.strip(axc)
    axc.set_xlabel("year of the parent record", labelpad=6)
    axc.set_ylabel("leading mode  (σ)", labelpad=4)
    axc.tick_params(length=3)
    axc.set_xlim(0, nt - 1)
    header(axc, "different history", f"parent and {N_TRACE} realisations")

    # (d) the mode's spectrum: identical by construction
    axd = fig.add_subplot(gs[0, 3])
    for p in real_psds:
        axd.loglog(freqs, p, color=ds.ICE, lw=2.6, alpha=0.30, zorder=2)
    axd.loglog(freqs, parent_psd, color=ds.INK, lw=1.4, zorder=4)
    ds.strip(axd)
    axd.set_xlabel("frequency  (1 / yr)", labelpad=6)
    axd.set_ylabel("power", labelpad=4)
    axd.tick_params(length=3, which="major")
    axd.tick_params(length=0, which="minor")
    axd.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
    axd.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
    axd.minorticks_off()
    header(axd, "identical statistics", f"parent and {N_REAL} realisations, overlaid")

    err = np.abs(real_psds.mean(axis=0) / parent_psd - 1).max()
    axd.text(0.03, 0.05, f"agreement to {err:.0e}", transform=axd.transAxes,
             fontsize=10, color=ds.INK_SOFT, ha="left", va="bottom")

    fig.text(0.012, 0.005,
             f"EOF truncation at {k} modes · {100*varexp.sum():.0f}% of the variance retained · "
             "annual anomalies after trend, seasonality and draft dependence are removed",
             fontsize=9, color=ds.INK_SOFT, ha="left", va="bottom", style="italic")

    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.14)
    print(f"wrote {a.out}")
    print(f"  modes {k}   variance retained {100*varexp.sum():.1f}%   "
          f"mode 1 {100*varexp[0]:.1f}%")
    print(f"  max |realisation PSD / parent PSD - 1| = {err:.2e}")


if __name__ == "__main__":
    main()
