#!/usr/bin/env python3
"""
flowline.py -- along-flow transects and grounding-line location.

Numpy only, so the HPC extractor can build the same sections without matplotlib.
"""
from __future__ import annotations

import numpy as np

RHO_I, RHO_O = 910.0, 1028.0


def build_flowline(x, y, sel, vx, vy, tree, bed, h0, step_m=2000.0,
                   n_up=500, n_down=120, smooth=9, tail_up_km=380.0, tail_down_km=40.0):
    """Transect traced along observed ice flow through the shelf centroid.

    The path is integrated up- and downstream through the observed velocity field,
    so it is along flow by construction. Returns (s, pts) with s increasing inland.
    """
    P = np.column_stack([x[sel], y[sel]])
    c = P.mean(axis=0)
    spd = np.hypot(vx, vy)

    # march until the condition is met, then run on a little: a fixed step count
    # never reaches grounded ice on a 500+ km shelf
    hflot_all = (RHO_O / RHO_I) * np.maximum(0.0, -bed)
    grounded_all = (h0 > 1.0) & (h0 > hflot_all)

    def march(p0, sign, n, want, tail_km):
        tail = int(tail_km * 1e3 / step_m)
        pts, p, hit = [], p0.copy(), -1
        for k in range(n):
            j = tree.query(p)[1]
            v = np.array([vx[j], vy[j]])
            m = np.hypot(*v)
            if not np.isfinite(m) or m <= 0:
                break
            p = p + sign * step_m * v / m
            pts.append(p.copy())
            if hit < 0 and want(tree.query(p)[1]):
                hit = k
            if hit >= 0 and k - hit >= tail:
                break
        return np.array(pts) if pts else np.empty((0, 2))

    # start from the fastest cell in the shelf: the centroid can sit on a slow margin
    idx_shelf = np.flatnonzero(sel)
    start = np.column_stack([x, y])[idx_shelf[np.argmax(spd[idx_shelf])]]
    # the inland tail must outrun the retreat itself
    up = march(start, -1.0, n_up, lambda j: grounded_all[j], tail_up_km)
    down = march(start, +1.0, n_down, lambda j: h0[j] <= 1.0, tail_down_km)
    pts = np.vstack([down[::-1], start[None, :], up])
    if smooth > 1 and pts.shape[0] > smooth:
        k = np.ones(smooth) / smooth
        pts = np.column_stack([np.convolve(pts[:, 0], k, "same"),
                               np.convolve(pts[:, 1], k, "same")])
        pts[:smooth], pts[-smooth:] = pts[smooth], pts[-smooth - 1]
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    return d - d[down.shape[0]], pts


def gl_position_main(s, h, hflot, min_run_km=15.0):
    """Seaward edge of the grounded body connected to the interior. (s_gl, slope).

    The transect ends in the grounded interior, so the run reaching the inland end
    is the ice sheet proper and its seaward edge is the grounding line. Runs shorter
    than min_run_km are pinning points and are ignored.
    """
    grounded = (h > 1.0) & (h > hflot)
    if not grounded.any():
        return np.nan, np.nan
    d = np.diff(grounded.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if grounded[0]:
        starts = np.r_[0, starts]
    if grounded[-1]:
        ends = np.r_[ends, grounded.size]
    runs = [(a, b) for a, b in zip(starts, ends)
            if s[b - 1] - s[a] >= min_run_km * 1e3]
    if not runs:
        return np.nan, np.nan
    a, _ = runs[-1]              # most inland surviving run = the ice sheet
    if a == 0:
        return s[0], np.nan      # grounded from the seaward end of the section
    g = h - hflot
    g = np.where(h > 1.0, g, -np.abs(g) - 1.0)
    g0, g1 = g[a - 1], g[a]
    if g1 == g0:
        return s[a], np.nan
    t = -g0 / (g1 - g0)
    return s[a - 1] + t * (s[a] - s[a - 1]), (g1 - g0) / (s[a] - s[a - 1])
