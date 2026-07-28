#!/usr/bin/env python3
"""
fig_regional_pdf_evolution.py — Per-region ensemble member spread over time.

Per-basin panel: member lines + ensemble mean + min–max band of VAF→SLE (mm).
CTRL: 133-region mask (100 shelves, pass --regions 33-132). Forced: 16 ISMIP6 regions (4×4 grid).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_io as eio
from ismip6_regions import BASIN_NAMES


def load_region_sle(root, ensemble, include):
    ens_dir = os.path.join(root, ensemble)
    members = eio.discover_members(ens_dir, stats_filename="regionalStats.nc", include=include)
    stacks, nmin = [], None
    for name, path in members:
        try:
            ds = eio.to_year_dim(eio.load_member_regionalstats(path))
        except Exception:
            continue
        if "regionalVolumeAboveFloatation" not in ds:
            continue
        yr = ds["year"].values
        if yr[0] > 5.0 or len(yr) < 10:                 # skip restart-only / too short
            continue
        vaf = np.asarray(ds["regionalVolumeAboveFloatation"].values)   # (year, nRegions)
        sle = np.column_stack([eio.vaf_to_sle_mm(vaf[:, r], reference="first")
                               for r in range(vaf.shape[1])])
        stacks.append((yr, sle))
        nmin = len(yr) if nmin is None else min(nmin, len(yr))
    if len(stacks) < 3:
        return None, None, 0
    years = stacks[0][0][:nmin]
    arr = np.stack([s[:nmin] for _, s in stacks], axis=0)   # (member, year, nRegions)
    return years, arr, len(stacks)


def region_names(path, nreg):
    """Load region names: use shared ISMIP6 mapping for 16-basin case, else read from file."""
    if nreg == 16:
        return BASIN_NAMES
    if path and os.path.isfile(path):
        try:
            d = xr.open_dataset(path, decode_times=False)
            if "regionNames" in d:
                rn = d["regionNames"].values
                def _dec(row):
                    s = []
                    for c in row:
                        if isinstance(c, bytes):
                            s.append(c.decode("ascii", "ignore"))
                        elif isinstance(c, (int, np.integer)):
                            s.append(chr(int(c)) if 0 < int(c) < 256 else "")
                        else:
                            s.append(str(c))
                    return "".join(s).strip()
                return [_dec(row) for row in rn]
        except Exception:
            pass
    return [f"reg{r}" for r in range(nreg)]


def plot_grid(years, arr, regions, names, title, out, nrow, ncol):
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.0 * nrow), sharex=True)
    axs = np.atleast_2d(axs)
    for k, r in enumerate(regions):
        ax = axs[k // ncol][k % ncol]
        m = arr[:, :, r]                                # (member, year)
        ax.plot(years, m.T, color="0.7", lw=0.4)
        ax.plot(years, np.nanmean(m, 0), "k", lw=1.2)
        ax.fill_between(years, np.nanmin(m, 0), np.nanmax(m, 0), color="C0", alpha=0.25)
        sig_end = np.nanstd(m[:, -1], ddof=1)
        nm = names[r] if r < len(names) else f"reg{r}"
        ax.set_title(f"{nm}  σ_end={sig_end:.1f}", fontsize=6)
        ax.tick_params(labelsize=5); ax.axhline(0, color="k", lw=0.3, alpha=0.4)
    for k in range(len(regions), nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.supxlabel("year", fontsize=9); fig.supylabel("VAF→SLE (mm, rise +)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=eio.default_ensembles_root())
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--members", required=True)
    ap.add_argument("--names-file", default=None, help="mask .nc with regionNames (for labels)")
    ap.add_argument("--regions", default=None, help="index range 'LO-HI' (default: all)")
    ap.add_argument("--per-fig", type=int, default=16)
    ap.add_argument("--grid", default="4,4", help="'nrow,ncol'")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    years, arr, nmem = load_region_sle(a.root, a.ensemble, a.members)
    if years is None:
        sys.exit("no usable members")
    nreg = arr.shape[2]
    names = region_names(a.names_file, nreg)
    if a.regions:
        lo, hi = a.regions.split("-"); regs = list(range(int(lo), int(hi) + 1))
    else:
        regs = list(range(nreg))
    nrow, ncol = (int(x) for x in a.grid.split(","))
    tag = a.tag or a.ensemble
    print(f"{a.ensemble}: {nmem} members, {nreg} regions, years {years[0]:.0f}..{years[-1]:.0f}, "
          f"plotting {len(regs)} region(s)")
    nfig = (len(regs) + a.per_fig - 1) // a.per_fig
    for i in range(0, len(regs), a.per_fig):
        chunk = regs[i:i + a.per_fig]
        suffix = f"_{i // a.per_fig}" if nfig > 1 else ""
        out = os.path.join(a.out_dir, f"{tag}_regional_pdf{suffix}.png")
        plot_grid(years, arr, chunk, names,
                  f"{a.ensemble} — per-region member spread (VAF→SLE); grey=members, black=mean, band=min–max",
                  out, nrow, ncol)


if __name__ == "__main__":
    main()
