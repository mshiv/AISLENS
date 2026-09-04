#!/usr/bin/env python3
"""Audit every ice shelf against the production fb_A background-melt curve.

The audit is for transparent panel selection.  It does not refit the production
parameterization.  Scores favour well-sampled, visually legible examples whose
production slope makes basal mass balance more negative with increasing draft,
whose breakpoint lies within the observed draft range, and whose curve follows
the means of populated draft bins.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from fig_methods_piecewise_background_melt import (
    REPO, SHELF_GEOMETRY, descriptive_bins, evaluate_curve, read_observations,
    read_parameters, region_names,
)


def safe_r(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 3 or np.nanstd(x[good]) == 0 or np.nanstd(y[good]) == 0:
        return np.nan
    return float(pearsonr(x[good], y[good]).statistic)


def r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    good = np.isfinite(observed) & np.isfinite(predicted)
    if np.count_nonzero(good) < 3:
        return np.nan
    denominator = np.sum((observed[good] - np.mean(observed[good])) ** 2)
    if denominator <= 0:
        return np.nan
    return float(1.0 - np.sum((observed[good] - predicted[good]) ** 2) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO / "reports/dissertation/figures/tierA/draft_dependence_shelf_audit.csv"),
    )
    args = parser.parse_args()

    geometry = gpd.read_file(SHELF_GEOMETRY).to_crs(3031)
    available = set(geometry.name.astype(str))
    shelves = [name for name in region_names()[33:] if name in available]
    parameters, _, _, _ = read_parameters(shelves)
    observations = read_observations(geometry, shelves)

    rows = []
    for shelf in shelves:
        draft, melt = observations[shelf]
        pars = parameters[shelf]
        if draft.size < 20:
            continue
        prediction = evaluate_curve(draft, pars)
        bx, by = descriptive_bins(draft, melt, n_bins=100, min_count=20)
        bp = evaluate_curve(bx, pars) if bx.size else np.asarray([])
        d05, d95 = np.nanpercentile(draft, [5, 95])
        interior = bool(d05 < pars["dmin"] < d95) if int(round(pars["p"])) == 0 else False
        deep = draft >= pars["dmin"] if int(round(pars["p"])) == 0 else np.ones_like(draft, bool)
        raw_iqr = float(np.subtract(*np.nanpercentile(melt, [75, 25])))
        rmse = float(np.sqrt(np.nanmean((melt - prediction) ** 2)))
        # The score is a ranking aid, not a reported statistic.
        bin_r2 = r2(by, bp) if bx.size >= 3 else np.nan
        desirable = int(round(pars["p"])) == 0 and pars["a1"] < 0 and interior
        score = (
            (2.0 * max(-1.0, np.nan_to_num(bin_r2, nan=-1.0)))
            + max(0.0, -np.nan_to_num(safe_r(draft[deep], melt[deep]), nan=0.0))
            + 0.12 * np.log10(max(draft.size, 1))
            + (0.35 if desirable else 0.0)
        )
        rows.append({
            "shelf": shelf,
            "parameterization": "constant" if int(round(pars["p"])) == 1 else "capped_piecewise_linear",
            "n_cells": int(draft.size),
            "dmin_m": pars["dmin"],
            "slope_B_per_m": pars["a1"],
            "threshold_inside_5_95pct": interior,
            "raw_pearson_r": safe_r(draft, melt),
            "deep_pearson_r": safe_r(draft[deep], melt[deep]),
            "raw_curve_r2": r2(melt, prediction),
            "bin_curve_r2": bin_r2,
            "rmse_m_ice_yr-1": rmse,
            "rmse_over_observed_iqr": rmse / raw_iqr if raw_iqr > 0 else np.nan,
            "selection_score": score,
            "preferred_direction_and_breakpoint": desirable,
        })

    frame = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print("Top production piecewise examples (negative slope, interior breakpoint):")
    print(frame.loc[frame.preferred_direction_and_breakpoint,
                    ["shelf", "n_cells", "dmin_m", "deep_pearson_r", "bin_curve_r2",
                     "raw_curve_r2", "selection_score"]].head(25).to_string(index=False))
    print("\nConstant-form candidates with the weakest absolute draft correlation:")
    constants = frame.loc[frame.parameterization == "constant"].copy()
    constants["abs_r"] = constants.raw_pearson_r.abs()
    print(constants.sort_values(["abs_r", "n_cells"], ascending=[True, False])[
        ["shelf", "n_cells", "raw_pearson_r", "rmse_over_observed_iqr"]
    ].head(15).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
