#!/usr/bin/env python3
"""
figstyle_legible.py -- a less minimal figure style, applied without editing any figure script.

Two sources. Thyng et al. (2016) note that diverging colormaps lose their sign when printed
in greyscale, because the two halves have mirrored lightness, and recommend overlaying
contours with a different line style either side of the hinge. They also argue each variable
in a manuscript should carry its own colormap, the way each symbol carries one meaning.

Elavsky (2025) argues the data-to-ink ratio has no principled stopping point and that pursuing
it trades away legibility for an aesthetic. The parts of that critique which bite here are the
ones about size and redundancy: nothing sets a floor on text size, and a figure encoding a
distinction in colour alone fails a projector, a greyscale printer, and a colourblind viewer
all at once.

So this module raises a floor rather than imposing a look:

  * no text below MIN_PT, enforced wherever a size is set
  * spines and gridlines restored on axes that are read for values
  * diverging maps get contours, solid above the hinge and dashed below
  * colorbars get enough ticks to read a value off
  * heat and temperature stop sharing a colormap

Everything is a monkeypatch installed by apply(); the figure scripts are untouched.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

MIN_PT = 13.0        # smallest text anywhere; a projector at the back of a room
GRID_KW = dict(color="#C9CFD6", lw=0.7, alpha=0.55)
_installed = False


def _patch_text_floor():
    """No text smaller than MIN_PT, wherever the size is set from."""
    # set_size is an alias that dispatches to set_fontsize, so only the real
    # method may be wrapped -- patching both sends the alias into a loop
    orig = Text.set_fontsize

    def set_fontsize(self, size):
        try:
            v = float(mpl.font_manager.font_scalings.get(size, size))
            size = max(v, MIN_PT)
        except (TypeError, ValueError):
            pass
        return orig(self, size)

    Text.set_fontsize = set_fontsize


def _patch_strip():
    """slidestyle.strip removes spines; keep the value-bearing ones and add a grid."""
    import slidestyle as ds
    orig = ds.strip

    def strip(ax, keep=("left", "bottom")):
        keep = tuple(set(keep) | {"left", "bottom"})
        orig(ax, keep=keep)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#8C97A3")
            ax.spines[side].set_linewidth(1.0)
        # only where an axis is actually labelled -- maps clear their ticks
        if len(ax.get_xticks()) or len(ax.get_yticks()):
            ax.set_axisbelow(True)
            ax.grid(True, **GRID_KW)

    ds.strip = strip


def _patch_tripcolor():
    """Thyng's greyscale fix: contour a diverging field, dashed below the hinge."""
    orig = Axes.tripcolor

    def tripcolor(self, *args, **kw):
        out = orig(self, *args, **kw)
        norm = kw.get("norm")
        if not isinstance(norm, TwoSlopeNorm):
            return out
        try:
            tri, C = args[0], args[1]
            c = float(norm.vcenter)
            lo, hi = float(norm.vmin), float(norm.vmax)
            below = [c - (c - lo) * f for f in (0.66, 0.33)]
            above = [c + (hi - c) * f for f in (0.33, 0.66)]
            self.tricontour(tri, C, levels=sorted(below), colors="#33404D",
                            linewidths=0.6, linestyles="dashed", zorder=5)
            self.tricontour(tri, C, levels=sorted(above), colors="#33404D",
                            linewidths=0.6, linestyles="solid", zorder=5)
        except Exception:
            pass
        return out

    Axes.tripcolor = tripcolor


def _patch_colorbar():
    """Enough ticks that a reader can put a number on a colour."""
    orig = mpl.figure.Figure.colorbar

    def colorbar(self, mappable, *a, **kw):
        cb = orig(self, mappable, *a, **kw)
        if "ticks" not in kw:
            try:
                cb.locator = MaxNLocator(nbins=6)
                cb.update_ticks()
            except Exception:
                pass
        return cb

    mpl.figure.Figure.colorbar = colorbar


def _patch_palette():
    """Thyng: one variable, one colormap. heat and temperature were sharing thermal."""
    try:
        import oceancolors as oc
        oc._CMO = dict(oc._CMO)
        oc._CMO["heat"] = "solar"       # geothermal flux, distinct from air temperature
        oc._CMO["anomaly"] = "delta"    # distinct from melt, which keeps balance
    except Exception:
        pass


def apply():
    global _installed
    if _installed:
        return
    os.environ.setdefault("AISLENS_PALETTE", "cmocean")
    import slidestyle as ds
    ds.apply()
    mpl.rcParams.update({
        "font.size": MIN_PT,
        "axes.labelsize": MIN_PT + 1,
        "axes.titlesize": MIN_PT + 2,
        "xtick.labelsize": MIN_PT,
        "ytick.labelsize": MIN_PT,
        "legend.fontsize": MIN_PT,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#C9CFD6",
        "axes.axisbelow": True,
        "figure.dpi": 130,
        "savefig.dpi": 220,
    })
    _patch_text_floor()
    _patch_strip()
    _patch_tripcolor()
    _patch_colorbar()
    _patch_palette()
    _installed = True
