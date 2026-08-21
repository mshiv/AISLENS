# Chapter 3 figures — full runbook (from zero)

Assumes **no `ensembleStats_*.nc` exist**. Everything below is reproducible from raw
member output.

Two independent tracks:

- **Track A — local, time series.** Runs on the laptop from `globalStats.nc` /
  `regionalStats.nc`. Already working; ~2 minutes.
- **Track B — HPC, spatial maps.** Needs `output_state` / `output_flux` on scratch.
  Three stages: per-member reduction → ensemble statistics → plots.

Figure numbering follows the wiki note *AISLENS — Chapter 3 argument spine and figure plan*.

---

## Path map

### Laptop (`/Users/smurugan9/research/aislens/AISLENS`)

| what | where |
|---|---|
| ensemble diagnostics (input) | `data/MALI/diagnostics/ENSEMBLES/<ENS>/<MEMBER>/globalStats.nc`, `regionalStats.nc` |
| mesh (for maps) | `data/MALI/AIS_4to20km_r01_20220907.regionMask_ismip6.nc` |
| **all chapter figures (output)** | `reports/dissertation/figures/` |
| driver | `src/scripts/make_dissertation_figures.sh` |

### HPC (`hcoda1/6/smurugan9`)

| what | where |
|---|---|
| code | `data/dev/AISLENS` → `$CODE_ROOT` |
| raw member output | `scratch/AISLENS/data/MALI/ENSEMBLES/<ENS>/<MEMBER>/output/` |
| job scripts | `data/jobs/aislens/` (copy the `.sbatch` files here) |
| figures + stats root | `scratch/AISLENS/data/figures/MALI/` → `$FIG_PARENT` |

**Generated on HPC, in order:**

| stage | file | location |
|---|---|---|
| 1. per-member time-average | `output_flux_all_timesteps_<YEAR>_tAvg.nc` | *inside each member's* `output/` |
| 1. per-member rate | `output_flux_all_timesteps_dhdt_<N>yr.nc` | *inside each member's* `output/` |
| 2. ensemble statistics | `ensembleStats_<YEAR>.nc` | `$FIG_PARENT/<ENS>/ensemble_stats/` |
| 3. maps | `*.png` | `$FIG_PARENT/<ENS>/ensemble_maps/` |
| 3. σ-ratio maps | `sigmaRatio_thickness_<YEAR>.nc` + `*.png` | `$FIG_PARENT/sigma_ratio/<NUM>_over_<DEN>/` |

> ⚠️ **Two different stats layouts exist in the archive.** The processing scripts write
> `$FIG_PARENT/<ENS>/ensemble_stats/`, but `aislens_plot_dhdt_ratio_*.sbatch` reads
> `$FIG_PARENT/ensemble_stats/<ENS>/` — reversed. The processing layout is authoritative
> because it is what actually gets written; `aislens_dissertation_sigma_ratio.sbatch`
> follows it. Fix the old ratio scripts' path if you reuse them.

---

## Track A — local figures (do this first, it is free)

```bash
cd /Users/smurugan9/research/aislens/AISLENS
bash src/scripts/make_dissertation_figures.sh
```

Writes 29 PNGs plus two markdown tables into `reports/dissertation/figures/`:

```
tierA/   F4 topline, percentile band, F5 amplification, sigma-vs-mean, F6 melt3x
tierB/   F7 covariance, spread budget, F8 gating, GL migration, rate, Jourdain
tierC/   distribution snapshots + convergence/dispersion/skewness x 4 ensembles
tables/  frozen_results.md          <- canonical numbers for the chapter
         readiness_diagnostics.md   <- metric/timing/covariance/twin-gate output
_logs/   one log per figure
```

Every figure runs independently; failures are reported at the end and never abort the run.
Override the input root with `AISLENS_ENSEMBLES_ROOT=... bash src/scripts/...`.

---

## Track B — HPC spatial maps

### Stage 0 — sync code and jobs

```bash
# from the laptop
rsync -av src/ smurugan9@login-phoenix.pace.gatech.edu:~/data/dev/AISLENS/src/
rsync -av src/pace-jobs/aislens/aislens_dissertation_*.sbatch \
          smurugan9@login-phoenix.pace.gatech.edu:~/data/jobs/aislens/
```

### Stage 1+2 — per-member reduction and ensemble statistics

One array job per ensemble. Each does **both** stages: it builds the per-member
`tAvg` and `dhdt` files, then reduces them across members into `ensembleStats_<YEAR>.nc`.

```bash
cd ~/data/jobs/aislens
sbatch aislens_dissertation_ensemble_processing_ssp585.sbatch
sbatch aislens_dissertation_ensemble_processing_var10x.sbatch
sbatch aislens_dissertation_ensemble_processing_ctrl.sbatch
sbatch aislens_dissertation_ensemble_processing_ssp126.sbatch
sbatch aislens_dissertation_ensemble_processing_3x.sbatch
```

All five can run concurrently — they touch different ensembles.

| script | ensemble | members | years | array |
|---|---|---:|---|---|
| `..._ssp585.sbatch` | `SSP585` | 10 | 2000–2300 / 10 | `0-30` |
| `..._var10x.sbatch` | `SSP585_varScaled10x` | **15** | 2000–2300 / 10 | `0-30` |
| `..._ctrl.sbatch` | `CTRL` | 10 | 2000–2300 / 10 | `0-30` |
| `..._ssp126.sbatch` | `SSP126` | 10 | 2000–2300 / 10 | `0-30` |
| `..._3x.sbatch` | `SSP585-3X` | 10 | 2000–**2190** / 10 | `0-19` |

**Why these differ from the archive scripts.** `aislens_mali_ensemble_processing_fast_ssp585.sbatch`
is misnamed — its `ENSEMBLE_DIR` is `SSP585_varScaled10x`, so there was **no config for
plain SSP585 at all**. The old CTRL config used `START_YEAR=2050`, which references `dhdt`
to the wrong epoch, and CTRL/SSP126 used 25- and 50-year increments, too coarse to share a
year grid with the others. The dissertation configs fix all of that: every ensemble uses
`START_YEAR=2000`, `YEAR_INCREMENT=10`, so all five land on the same year grid — required
for the between-ensemble ratio in stage 3. varScaled10x now lists all **15** members
(the old file listed 10). SSP585-3X stops at 2190 because its shortest member ends at 2194.6.

**Runtime:** hours. Stage 1 is the expensive part (NCO over every member-year).
Re-running is cheap — existing `tAvg`/`dhdt`/stats files are skipped unless `FORCE_STATS=1`.

**Verify before moving on:**
```bash
for E in SSP585 SSP585_varScaled10x CTRL SSP126 SSP585-3X; do
  echo "$E: $(ls $FIG_PARENT/$E/ensemble_stats/ensembleStats_*.nc 2>/dev/null | wc -l) stats files"
done
ncdump -h $FIG_PARENT/SSP585/ensemble_stats/ensembleStats_2200.nc | grep -E 'thickness_|dhdt_'
```
Expect `thickness_mean/min/max/range/std` and the same for `dhdt`, plus `xCell`, `yCell`, `dcEdge`.

### Stage 3a — per-ensemble maps (S1, S5)

`PROCESS_PLOTTING=1` is already set, so stage 1+2 calls `plot_ensemble_maps.py`
automatically and writes to `$FIG_PARENT/<ENS>/ensemble_maps/`. These are the σ / range /
mean maps **with grounding-line overlays** — figures S1 and S5.

To re-plot without recomputing, set `FORCE_STATS=0` and re-submit; or call
`plot_ensemble_maps.py` directly with `--ensemble_files`, `--years`, `--variables`,
`--run_dirs`, `--run_names`, `--save_base`.

### Stage 3b — between-ensemble σ ratio (S2)

**Requires stage 2 complete for both SSP585 and varScaled10x.**

```bash
sbatch aislens_dissertation_sigma_ratio.sbatch
```

Builds `sigmaRatio_thickness_<YEAR>.nc` containing `thickness_std_num`,
`thickness_std_den` and their ratio, then plots via the existing
`plot_ensemble_maps_ratio.py`. Output: `$FIG_PARENT/sigma_ratio/SSP585_varScaled10x_over_SSP585/`.

Edit at the top of the script: `NUM_ENS`, `DEN_ENS`, `N_NUM`, `N_DEN`, `VARIABLE`
(`thickness` or `dhdt`), `YEARS`. It skips any year whose stats are missing and says so.

### Stage 4 — bring the maps home

```bash
# from the laptop
mkdir -p reports/dissertation/figures/spatial
rsync -av --include='*/' --include='*.png' --exclude='*' \
  smurugan9@login-phoenix.pace.gatech.edu:~/scratch/AISLENS/data/figures/MALI/ \
  reports/dissertation/figures/spatial/
```

---

## What each quantity means

**`dhdt` is referenced to `START_YEAR`, not centred.**
```
dhdt(cell, year) = [thickness(year) - thickness(2000)] / (year - 2000)
```
So it is the *mean* thinning rate over 2000→year, not an instantaneous rate. That is the
right quantity for an altimetry comparison over a matching interval, but it is **not** the
same as MALI's instantaneous `dHdt` field, and the two should not be mixed in one figure.

**σ from the NCO pipeline is the population standard deviation.**
```
sigma_pop = sqrt(rms^2 - mean^2)        # divides by N
sigma_sample = sigma_pop * sqrt(N/(N-1))   # what freeze_results_table.py reports
```
5.4 % low at N=10, 3.6 % at N=15. Harmless within one map; it matters when a map σ is
quoted beside a table σ, and it does **not** cancel in a ratio of ensembles with different
N — which is why `aislens_dissertation_sigma_ratio.sbatch` applies
`sqrt(N_num/(N_num-1))/sqrt(N_den/(N_den-1))` = 0.9820 for 15-over-10.

**`range = max - min`** is reported beside σ because at N=10 it is often the more honest
summary; for a Gaussian, E[range] ≈ 3.08 σ at N=10.

---

## Order of work

1. **Track A now** — 2 minutes, gives the frozen table and every time-series figure.
2. **Stage 1+2 for SSP585 and varScaled10x** — these two unlock the σ-ratio, the highest-value
   new figure.
3. **Stage 3b** as soon as those two finish.
4. **CTRL, SSP126, 3X** processing — needed for the full five-ensemble map set, but nothing
   depends on them.

## Known gaps

- The old `aislens_plot_dhdt_ratio_*.sbatch` scripts read the reversed stats path (above).
- `fig_jourdain_model_universe.py` takes no `--out-dir`; it writes to its own default under
  `reports/figures/`. Copy its output into `tierB/` manually.
- Neither new sbatch has been executed — treat the first submission as a smoke test and
  check the `.err` file before launching the rest.
