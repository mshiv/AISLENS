"""
slidestyle.py — shared palette, fonts and rcParams for talk figures.

Cover-plate palette (ink #0B2545, accent #1673A6) on a white ground, extended with three warm
tones for the coastal half. POS/NEG are the signed-response pair used in both halves:
warm = more ice lost / more water, cool = less. Figures carry no title -- the slide does.
"""
from __future__ import annotations

import os

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- palette
# Shiva's authoritative cover-plate palette, extended with three warm tones for the
# coastal half.  Figures and illustrations therefore share one colour system.
PAPER = "#FFFFFF"      # background / light ground
INK = "#0B2545"        # dominant hue -- all titles and body text
INK_DEEP = "#1B1B1B"   # darkest detail and shadow
INK_SOFT = "#4A5568"   # muted detail -- secondary labels
RULE = "#D7D7D3"       # surrounding neutral -- hairlines, spines
LINEWORK = "#B0C4DE"

ICE = "#1673A6"        # supporting accent -- Part I
ICE_TINT = "#D1E5F0"   # soft secondary tone
FIELD = "#EDF4FA"      # light supporting field

MARSH = "#C2703A"      # warm counter-hue -- Part II
MARSH_TINT = "#E3B183"  # soft warm tone
MARSH_DEEP = "#8A4B24"  # deep warm anchor

SEA = "#1673A6"        # shared / sea-level accent

# signed response: warm = more ice loss / more water, cool = less.
# Used identically in Antarctica and at Pin Point -- that is the point.
POS = MARSH
NEG = ICE
DIVERGING = LinearSegmentedColormap.from_list(
    "defense_div",
    [INK, ICE, ICE_TINT, PAPER, MARSH_TINT, MARSH, MARSH_DEEP],
    N=256,
)

_FONT_DIR = os.path.expanduser("~/Library/Fonts/_deck")


def _register_fonts() -> str:
    """Add the deck fonts if present; fall back gracefully if they are not."""
    from matplotlib import font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for fname in ("IBMPlexSans.ttf", "Newsreader.ttf"):
        path = os.path.join(_FONT_DIR, fname)
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
            except RuntimeError:
                pass
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("IBM Plex Sans", "Helvetica Neue", "Avenir Next", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "sans-serif"


def apply(scale: float = 1.0) -> str:
    """Install the deck rcParams. Returns the resolved sans family name."""
    family = _register_fonts()
    mpl.rcParams.update({
        "font.family": family,
        "font.size": 11 * scale,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.dpi": 220,
        "figure.dpi": 110,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11 * scale,
        "axes.labelsize": 11 * scale,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 10 * scale,
        "ytick.labelsize": 10 * scale,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 10 * scale,
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": False,
    })
    return family


def strip(ax, keep=("left", "bottom")):
    """Remove spines not in `keep`."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)

