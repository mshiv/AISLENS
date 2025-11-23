#!/usr/bin/env python3
"""
Main workflow orchestrator for AISLENS forcing generation.

Runs: 1) Initialize dirs  2) Prepare data  3) Draft dependence & forcings (parallel)  4) Regrid

Usage: python main.py [--skip-dirs] [--serial]
"""

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time

from aislens.config import config
from aislens.utils import collect_directories, initialize_directories, setup_logging

logger = logging.getLogger(__name__)


def run_script(script_name):
    """Run a Python script and log timing."""
    logger.info(f"Running {script_name}...")
    start = time()
    result = subprocess.run([sys.executable, Path(__file__).parent / script_name], capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")
    logger.info(f"{script_name} completed ({time() - start:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the complete AISLENS workflow')
    parser.add_argument('--skip-dirs', action='store_true', help='Skip directory initialization')
    parser.add_argument('--serial', action='store_true', help='Run serially instead of parallel')
    args = parser.parse_args()
    
    setup_logging(Path(config.DIR_PROCESSED), "main_workflow")
    Path(config.DIR_PROCESSED).mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("AISLENS WORKFLOW - FULL PIPELINE")
    logger.info("="*80)
    workflow_start = time()
    
    # Step 0: Initialize directories
    if not args.skip_dirs:
        logger.info("Step 0: Initializing directories...")
        initialize_directories(collect_directories(config))
    
    # Step 1: Prepare data
    logger.info("Step 1: Preparing data...")
    run_script("prepare_data.py")
    
    # Steps 2-3: Draft dependence & forcings
    if args.serial:
        logger.info("Step 2: Calculating draft dependence...")
        run_script("calculate_draft_dependence.py")
        logger.info("Step 3: Generating forcings...")
        run_script("generate_forcings.py")
    else:
        logger.info("Steps 2-3: Running in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1, f2 = executor.submit(run_script, "calculate_draft_dependence.py"), \
                     executor.submit(run_script, "generate_forcings.py")
            f1.result(), f2.result()
    
    # Step 4: Regrid
    logger.info("Step 4: Regridding to MALI grid...")
    run_script("regrid_to_mali.py")
    
    logger.info("="*80)
    logger.info(f"WORKFLOW COMPLETE ({time() - workflow_start:.1f}s)")
    logger.info("="*80)
