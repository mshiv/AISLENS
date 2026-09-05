#!/usr/bin/env python3
"""
oceancolors.py -- cmocean colormaps chosen by what the field physically is.

cmocean's maps are perceptually uniform: lightness rises or falls at a constant rate, so
a step in colour is a step in value everywhere on the scale. Two consequences drive the
choices below.

Sequential fields get a monotonic-lightness map, and the map is oriented so the values
that carry the argument end up dark. On a white slide a colormap that runs to L*~98 puts
its top end at the paper colour, which loses exactly the cells being talked about.

Anomalies get a diverging map only when zero is physically meaningful -- melt against
refreezing, thickening against thinning. Diverging maps peak in lightness at the hinge,
so the hinge must sit at the value that means "no change", never at the data midpoint.

Roles rather than field names, so a figure asks for what it is showing.

Measured lightness of the maps used here (CIELAB L*, start -> end):
  topo      11.8 -> 98.5  two-sided, hinge at sea level
  speed     98.3 -> 12.2  sequential
  deep      98.4 -> 11.8  sequential, built for depth
  ice        1.8 -> 98.0  sequential
  thermal   12.5 -> 94.5  sequential
  rain      94.1 -> 11.9  sequential
  amp       94.0 -> 10.6  sequential, built for magnitude
  matter    93.8 -> 11.3  sequential
  dense     94.3 -> 10.9  sequential
  balance / curl / delta / diff / tarn   diverging, hinge at 0.50
  phase     constant lightness, circular data only
"""
from __future__ import annotations

import os

_LEGACY = {
    "topography":   "terrain",
    "speed":        "viridis",
    "depth":        "Blues_r",
    "thickness":    "Blues",
    "temperature":  "RdYlBu_r",
    "heat":         "inferno",
    "precip":       "YlGnBu",
    "magnitude":    "magma_r",
    "friction":     "cividis",
    "resolution":   "cividis_r",
    "melt":         "RdBu_r",
    "tendency":     "RdBu",
    "difference":   "PuOr",
    "ratio":        "PRGn",
    "anomaly":      "RdBu_r",
    "basemap":      "Greys",
    "discharge":    "Blues",
}

# role -> cmocean name.  "_r" reverses, as in matplotlib.
_CMO = {
    # sequential ---------------------------------------------------------
    "speed":       "speed",      # built for speed; light slow interior, dark fast streams
    "depth":       "deep_r",     # built for depth; deepest is dark, since draft is negative
    "thickness":   "ice_r",      # built for ice; reversed so thick ice is dark
    "temperature": "thermal",    # built for temperature
    "heat":        "thermal",
    "precip":      "rain",       # built for precipitation
    "magnitude":   "amp",        # built for amplitude
    "friction":    "matter",
    "resolution":  "dense",
    "discharge":   "dense",     # river flow; distinct from rain, which is a different variable
    # a DEM drawn *under* an overlay is context, not the subject: keep it desaturated
    # so the data on top still reads.  A DEM that is the subject uses "topography".
    "basemap":     "gray_r",
    # two-sided ----------------------------------------------------------
    "topography":  "topo",       # built for topography, hinge at sea level
    "melt":        "balance",    # melt against refreezing, hinge at zero
    "tendency":    "balance_r",  # thinning reads warm, thickening cool
    "difference":  "diff",       # built for differences
    "ratio":       "curl",       # hinge at 1
    "anomaly":     "balance",
    # circular -----------------------------------------------------------
    "phase":       "phase",
}


def palette() -> str:
    """'cmocean' or 'legacy'; set AISLENS_PALETTE to switch."""
    return os.environ.get("AISLENS_PALETTE", "legacy").lower()


def cmap(role: str, palette_name: str | None = None):
    """Colormap for a semantic role. Falls back to the legacy choice if cmocean is absent."""
    p = (palette_name or palette()).lower()
    if p == "cmocean":
        try:
            import cmocean  # noqa: F401
            from matplotlib import colormaps
            name = _CMO[role]
            base, rev = (name[:-2], True) if name.endswith("_r") else (name, False)
            cm = getattr(__import__("cmocean").cm, base)
            return cm.reversed() if rev else cm
        except Exception:
            pass
    return _LEGACY[role]


def roles():
    return sorted(_LEGACY)
