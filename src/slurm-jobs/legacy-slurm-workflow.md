# AISLENS Legacy SLURM Workflow Reference

This file is a record of the older SLURM workflow used before the refactor.
It is ***not*** the active workflow. For current runs, start with [refactored/](refactored/) and the README in that directory.

## What changed

The original SLURM job set was split into a smaller refactored workflow under [refactored/](refactored/).
The main cleanup was:

- consolidating related scripts into phase-based jobs
- moving shared setup into `00-common-header.sh`
- moving environment settings into `refactored/config/`
- keeping the active scripts in one place and moving old variants to `archive/`

## Current status

- Active workflow: `src/slurm-jobs/refactored/`
- Legacy workflow history: this file
- Archived scripts: `src/slurm-jobs/archive/`
- Legacy helper scripts: `src/bash-utils/`

## Refactor summary

The active refactored workflow currently covers these phases:

1. Data preparation
2. Draft dependence calculation
3. Interpolation to MPAS-LI
4. Forcing generation
5. Trend extraction
6. Trend and variability combination
7. MALI dH/dt processing
8. Visualization

The refactored set uses 14 consolidated scripts plus a shared header and config directory.

## What was consolidated

The following groups were merged into the refactored workflow:

- separate time-mean scripts into one phase-based script with mode flags
- multiple draft-dependence scripts into one comprehensive script
- multiple forcing-generation scripts into a smaller set of phase-driven jobs
- multiple trend-extraction scripts into one clear workflow path
- multiple dH/dt and plotting scripts into unified phase-based jobs

## Historical notes

The old workflow also included many one-off scripts, debug variants, and scenario-specific duplicates.
Those are preserved in the archive for reference, but they are no longer part of the active workflow.

A few patterns were kept because they were useful:

- node-local or cached preprocessing behavior
- array-job support for large batch runs
- separate handling for CTRL, SSP585, and SSP126 where needed
- NCO-based utilities for simple file transformations

## Where to look next

- Use [src/slurm-jobs/README.md](README.md) for a short pointer into the current workflow
- Use [refactored/README.md](refactored/README.md) for the current active scripts
- Use [docs/slurm-workflow-reference.md](../docs/slurm-workflow-reference.md) for the broader workflow reference and script map

## Legacy script counts

Before refactoring, the workflow had 43 active scripts and 62 archived scripts.
The refactor reduced the active workflow to a smaller consolidated set centered on `refactored/`.
