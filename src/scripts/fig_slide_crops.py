#!/usr/bin/env python3
"""
fig_slide_crops.py — crop chapter figures down to single-claim slide panels.

Chapter figures run to 9-13 panels and do not survive projection. This crops the panel that
carries the claim. A cropper, not a re-plotter: pixels come from figures already reviewed, so
nothing new enters the talk. --contact writes a sheet for checking the crop boxes, which are
fractions of each source image.
"""
from __future__ import annotations

import os, sys, argparse
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slidestyle as ds  # noqa: E402

AIS = os.path.abspath(os.path.join(HERE, "..", ".."))
CORAL = "/Users/smurugan9/research/coral"
OUT = f"{AIS}/reports/dissertation/figures/slides"

FIGS = f"{AIS}/reports/dissertation/figures"
CH1 = ("/Users/smurugan9/Documents/vaults/shadowfax-wiki/wiki/dissertation/"
       "overleaf_updates/chapter1_figures_2026-09-02/figures")

# slide, source, crop (l, t, r, b) as fractions, note
RECUTS = [
    ("S02_gmsl", f"{CH1}/ch1_srocc_to_fort_pulaski.png",
     (0.0, 0.0, 1.0, 0.48), "SROCC panel (a), global mean sea level"),
    ("S03_antarctic_contribution", f"{CH1}/ch1_srocc_to_fort_pulaski.png",
     (0.0, 0.48, 0.665, 1.0), "SROCC panel (b), Antarctic contribution"),
    ("S04_fort_pulaski", f"{CH1}/ch1_srocc_to_fort_pulaski.png",
     (0.63, 0.48, 1.0, 1.0), "Fort Pulaski scenario panel alone"),
    ("S08_observed_melt", f"{CH1}/ch1_adusumilli_fig1.png",
     (0.0, 0.0, 1.0, 1.0), "observed basal melt, full plate"),
    ("S10_experiment_design", f"{CH1}/ch1_emulation_framework.png",
     (0.0, 0.0, 1.0, 0.53), "Antarctic half of the emulation framework"),
    ("S11_mean_trajectory", f"{FIGS}/tierA/std_vs_mean_volumeAboveFloatation.png",
     (0.50, 0.0, 1.0, 1.0), "mean panel only"),
    ("S12_spread", f"{FIGS}/tierA/std_vs_mean_volumeAboveFloatation.png",
     (0.0, 0.0, 0.50, 1.0), "spread panel only"),
    ("S13_amplification_ts", f"{FIGS}/tierA/F24_amplification_composite.png",
     (0.0, 0.085, 1.0, 0.275), "the sigma-ratio time series, panel (a)"),
    ("S13_amplification_map", f"{FIGS}/tierA/F24_amplification_composite.png",
     (0.075, 0.315, 0.222, 0.497), "one Amundsen ratio map, year 50 — map only"),
    ("S16_dynamic_gating", f"{FIGS}/tierB/F8_dynamic_gating_SSP585.png",
     (0.0, 0.0, 1.0, 1.0), "where the spread lives"),
    ("S21_site", f"{CORAL}/reports/chapter4_final_local/coral_domain_context.png",
     (0.47, 0.0, 1.0, 1.0), "Pin Point in the Vernon River estuary, panel (b)"),
    ("S21_track", f"{CORAL}/reports/chapter4_final_local/coral_domain_context.png",
     (0.0, 0.0, 0.47, 1.0), "Matthew track and the nested domains, panel (a)"),
    # panel titles in the source overlap their right-hand neighbour, so these crop
    # below the title -- the slide names the intervention anyway
    ("S22_intervention_wall", f"{CORAL}/reports/figures/intervention_anatomy_v2.png",
     (0.030, 0.118, 0.250, 0.500), "floodwall — map and DEM section, panel A"),
    ("S22_intervention_marsh", f"{CORAL}/reports/figures/intervention_anatomy_v2.png",
     (0.012, 0.620, 0.245, 0.695), "marsh restoration — footprint only, panel E"),
    ("S23_model_chain", f"{CH1}/ch1_emulation_framework.png",
     (0.0, 0.53, 1.0, 1.0), "coastal half of the emulation framework"),
    ("S25_compound", f"{CORAL}/reports/chapter4_final_local/coral_compound_effect.png",
     (0.0, 0.0, 0.33, 1.0), "compound-effect map, panel (a)"),
    ("S25_compound_curve", f"{CORAL}/reports/chapter4_final_local/coral_compound_effect.png",
     (0.63, 0.0, 1.0, 0.72), "exceedance curve, panel (c)"),
    ("S27_unet", f"{CH1}/ch1_unet_schematic.png",
     (0.0, 0.0, 1.0, 1.0), "U-Net input stack and architecture"),
    ("S29_ood_ecdf", f"{CORAL}/reports/chapter4_remaining_v1/coral_emulator_compact.png",
     (0.0, 0.50, 0.93, 1.0), "the extrapolation ECDF, panel (e) — the best CORAL visual"),
    ("S29_ood_map", f"{CORAL}/reports/chapter4_remaining_v1/coral_emulator_compact.png",
     (0.690, 0.095, 0.900, 0.450), "worst unseen-SLR member, as an inset"),
]


def recut(src, box, pad=0.02):
    im = Image.open(src).convert("RGBA")
    w, h = im.size
    l, t, r, b = box
    crop = im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    cw, ch = crop.size
    m = int(round(pad * max(cw, ch)))
    ground = Image.new("RGB", (cw + 2 * m, ch + 2 * m), ds.PAPER)
    ground.paste(crop, (m, m), crop)
    return ground


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--contact", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    made, missing = [], []
    for slide, src, box, note in RECUTS:
        if a.only and not any(slide.startswith(o) for o in a.only):
            continue
        if not os.path.exists(src):
            missing.append((slide, src))
            continue
        img = recut(src, box)
        dst = f"{OUT}/recut_{slide}.png"
        img.save(dst, "PNG", optimize=True)
        made.append((slide, img.size, note))
        print(f"  {slide:26s} {img.size[0]:5d}×{img.size[1]:<5d}  {note}")

    if missing:
        print("\nMISSING SOURCES")
        for slide, src in missing:
            print(f"  {slide:26s} {src}")

    if a.contact and made:
        # one sheet so the crops can be checked at a glance
        thumbs = []
        for slide, _, _ in made:
            im = Image.open(f"{OUT}/recut_{slide}.png")
            im.thumbnail((520, 520))
            thumbs.append((slide, im))
        cols = 3
        rows = (len(thumbs) + cols - 1) // cols
        cw = max(t.size[0] for _, t in thumbs) + 24
        chh = max(t.size[1] for _, t in thumbs) + 44
        sheet = Image.new("RGB", (cols * cw, rows * chh), ds.PAPER)
        from PIL import ImageDraw
        d = ImageDraw.Draw(sheet)
        for i, (slide, t) in enumerate(thumbs):
            x, y = (i % cols) * cw + 12, (i // cols) * chh + 32
            sheet.paste(t, (x, y))
            d.text((x, y - 22), slide, fill=ds.INK)
        sheet.save(f"{OUT}/recut_contact_sheet.png", "PNG")
        print(f"\nwrote {OUT}/recut_contact_sheet.png")

    print(f"\n{len(made)} recut · {len(missing)} missing")


if __name__ == "__main__":
    main()
