#!/usr/bin/env python3
"""
make_legible_figures.py -- rebuild the figures under figstyle_legible, touching no script.

Each figure script is executed in-process with its own argv after the style patches are
installed, so the outputs differ only by style and palette. Originals are left where they are.
"""
from __future__ import annotations

import os, sys, runpy, argparse, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = f"{ROOT}/reports/dissertation/figures/legible"

# script, argv template.  {out} is the directory, {f} a file inside it.
JOBS = [
    ("fig_bedrock.py",                    ["--palette", "cmocean", "--out", "{out}/fig_bedrock.png"]),
    ("fig_mesh_fields.py",                ["--palette", "cmocean", "--contact", "--outdir", "{out}/mesh"]),
    ("fig_ice_ocean_system.py",           ["--outdir", "{out}"]),
    ("fig_mali_mesh.py",                  ["--outdir", "{out}"]),
    ("fig_ocean_variability.py",          ["--out", "{out}/fig_ocean_variability.png"]),
    ("fig_drift_timeseries.py",           ["--out", "{out}/fig_drift_timeseries.png"]),
    ("fig_drift_basin_map.py",            ["--out", "{out}/fig_drift_basin_map.png"]),
    ("fig_effect_hierarchy.py",           ["--out", "{out}/fig_effect_hierarchy.png"]),
    ("fig_generator_phase_randomization.py", ["--out", "{out}/fig_generator_phase_randomization.png"]),
    ("fig_gl_members.py",                 ["--outdir", "{out}"]),
    ("fig_gl_timing.py",                  []),
]


CHAPTER_JOBS = [
    ("fig_std_vs_mean.py",             ["--variable", "volumeAboveFloatation", "--out-dir", "{out}"]),
    ("fig_amplification_composite.py", ["--out-dir", "{out}"]),
    ("fig_dynamic_gating.py",          ["--out", "{out}/F8_dynamic_gating_SSP585.png"]),
]

CORAL = "/Users/smurugan9/research/coral"
CORAL_OUT = f"{CORAL}/reports/legible"
CORAL_JOBS = [
    ("fig_compound_drivers.py",  ["--outdir", "{out}"]),
    ("fig_fort_pulaski.py",      ["--outdir", "{out}"]),
    ("fig_matthew_forcing.py",   ["--outdir", "{out}"]),
    ("fig_river_discharge.py",   ["--out", "{out}/fig_river_discharge.png"]),
    ("fig_tide_context.py",      ["--outdir", "{out}"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="aislens", choices=["aislens", "coral", "chapter"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    jobs, base, scripts_dir = {
        "aislens": (JOBS, OUT, HERE),
        "coral":   (CORAL_JOBS, CORAL_OUT, f"{CORAL}/scripts"),
        "chapter": (CHAPTER_JOBS, f"{OUT}/chapter", HERE),
    }[a.set]
    if a.outdir is None:
        a.outdir = base
    os.makedirs(a.outdir, exist_ok=True)
    os.makedirs(f"{a.outdir}/mesh", exist_ok=True)

    import figstyle_legible as legible
    legible.apply()

    ok, bad = [], []
    for script, argv in jobs:
        if a.only and not any(o in script for o in a.only):
            continue
        path = os.path.join(scripts_dir, script)
        if not os.path.exists(path):
            bad.append((script, "missing")); continue
        sys.argv = [path] + [x.format(out=a.outdir) for x in argv]
        try:
            runpy.run_path(path, run_name="__main__")
            ok.append(script)
            print(f"  ok      {script}")
        except SystemExit:
            ok.append(script); print(f"  ok      {script}")
        except Exception as e:
            bad.append((script, f"{type(e).__name__}: {e}"))
            print(f"  FAILED  {script}  {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)

    print(f"\n{len(ok)} rebuilt, {len(bad)} failed -> {a.outdir}")
    for s, why in bad:
        print(f"  {s}: {why}")


if __name__ == "__main__":
    main()
