#!/usr/bin/env python3
"""
extrapolation_seasonality_check.py — does nearest-neighbor extrapolation inflate seasonal dominance?

Compares per-cell and aggregate seasonal fractions over original vs extrapolated fields
to isolate the spatial-coherence and extrapolation effects.

Author: Shivaprakash Muruganandham
"""
from __future__ import annotations
import sys
import numpy as np
import xarray as xr

BASE = "data/processed"
FILES = {
    "seas_orig": f"{BASE}/sorrm_seasonality.nc",
    "var_orig":  f"{BASE}/sorrm_variability.nc",
    "seas_ext":  f"{BASE}/sorrm_seasonality_extrapolated_fillNA.nc",
    "var_ext":   f"{BASE}/sorrm_variability_extrapolated_fillNA_meanAdjusted.nc",
}


def load(path):
    d = xr.open_dataset(path, decode_times=False, chunks={"Time": 120})
    v = list(d.data_vars)[0]
    return d[v]


def cell_and_agg_var(da):
    """Return (per-cell temporal variance [y,x], aggregate-mean temporal variance [scalar])."""
    cellvar = da.var("Time", skipna=True)                 # (y,x)
    agg = da.mean(["y", "x"], skipna=True)                # (Time,)  spatial mean
    aggvar = agg.var("Time", skipna=True)                 # scalar
    return np.asarray(cellvar.values, dtype=float), float(aggvar.values)


def frac_stats(vs, vv, mask):
    tot = vs + vv
    ok = mask & np.isfinite(vs) & np.isfinite(vv) & (tot > 0)
    fr = vs[ok] / tot[ok]
    return ok.sum(), float(np.mean(fr)), float(np.median(fr))


def main():
    print("Loading (lazy) + reducing 4 x ~10 GB fields; streaming, be patient...\n")
    vs_o, avs_o = cell_and_agg_var(load(FILES["seas_orig"]))
    vv_o, avv_o = cell_and_agg_var(load(FILES["var_orig"]))
    vs_e, avs_e = cell_and_agg_var(load(FILES["seas_ext"]))
    vv_e, avv_e = cell_and_agg_var(load(FILES["var_ext"]))

    orig_mask = np.isfinite(vs_o) & np.isfinite(vv_o)
    ext_mask = np.isfinite(vs_e) & np.isfinite(vv_e)
    n_o, n_e = int(orig_mask.sum()), int(ext_mask.sum())

    # PER-CELL seasonal fraction (physical, coherence-free)
    no, mo, medo = frac_stats(vs_o, vv_o, orig_mask)
    ne, me, mede = frac_stats(vs_e, vv_e, ext_mask)

    # AGGREGATE seasonal fraction (coherence-sensitive)
    agg_o = avs_o / (avs_o + avv_o) if (avs_o + avv_o) > 0 else np.nan
    agg_e = avs_e / (avs_e + avv_e) if (avs_e + avv_e) > 0 else np.nan

    print("=" * 66)
    print(f"cells: original (real shelf) = {n_o:,}   extrapolated (all) = {n_e:,}"
          f"   ratio = {n_e/max(1,n_o):.2f}x   sqrt(ratio) = {np.sqrt(n_e/max(1,n_o)):.2f}")
    print("=" * 66)
    print("\nPER-CELL seasonal fraction  var_s/(var_s+var_v)  [the PHYSICAL number]")
    print(f"  original cells   : mean {100*mo:5.1f}%   median {100*medo:5.1f}%   (n={no:,})")
    print(f"  extrapolated all : mean {100*me:5.1f}%   median {100*mede:5.1f}%   (n={ne:,})")
    print("\nAGGREGATE seasonal fraction  var(<s>)/(var(<s>)+var(<v>))  [coherence-sensitive]")
    print(f"  original cells   : {100*agg_o:5.1f}%")
    print(f"  extrapolated all : {100*agg_e:5.1f}%")
    print("\nINTERPRETATION")
    print(f"  spatial-coherence inflation  (aggregate/per-cell, original): "
          f"{agg_o/mo:.1f}x")
    print(f"  extrapolation effect on the AGGREGATE fraction: "
          f"{100*agg_o:.1f}% -> {100*agg_e:.1f}%  (Δ {100*(agg_e-agg_o):+.1f} pts)")
    print(f"  extrapolation effect on the PER-CELL fraction : "
          f"{100*mo:.1f}% -> {100*me:.1f}%  (Δ {100*(me-mo):+.1f} pts)")
    print("\n  If per-cell << aggregate: the '94% seasonal' is mostly a spatial-aggregation")
    print("  artifact (use per-cell / ice response, not the domain-aggregate spectrum).")
    print("  If extrapolated >> original: nearest-neighbor fill is inflating seasonality;")
    print("  revisit the extrapolation (taper/decay, or exclude interior from diagnostics).")

    # distribution of the per-cell fraction (original cells) + spatial maps
    tot_o = vs_o + vv_o
    ok_o = orig_mask & np.isfinite(tot_o) & (tot_o > 0)
    fr_o = np.full(vs_o.shape, np.nan); fr_o[ok_o] = vs_o[ok_o] / tot_o[ok_o]
    pcts = [10, 25, 50, 75, 90, 95]
    vals = np.nanpercentile(fr_o[ok_o], pcts)
    print("\nPER-CELL seasonal-fraction distribution (original cells):")
    print("  " + "  ".join(f"p{p}={100*v:.0f}%" for p, v in zip(pcts, vals)))
    print(f"  cells >50% seasonal: {100*np.mean(fr_o[ok_o] > 0.5):.1f}%   "
          f">10%: {100*np.mean(fr_o[ok_o] > 0.1):.1f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tot_e = vs_e + vv_e
        ok_e = ext_mask & np.isfinite(tot_e) & (tot_e > 0)
        fr_e = np.full(vs_e.shape, np.nan); fr_e[ok_e] = vs_e[ok_e] / tot_e[ok_e]
        fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
        for ax, fr, ttl in [(axs[0], fr_o, "original (real shelf cells)"),
                            (axs[1], fr_e, "extrapolated (interior filled)")]:
            im = ax.imshow(100 * fr, origin="lower", vmin=0, vmax=100, cmap="RdYlBu_r")
            ax.set_title(f"seasonal fraction var_s/(var_s+var_v)\n{ttl}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, label="% seasonal")
        fig.tight_layout()
        out = "reports/seasonal_fraction_map.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"\nMap -> {out}")
    except Exception as e:
        print(f"  [map skipped: {e}]")


if __name__ == "__main__":
    main()
