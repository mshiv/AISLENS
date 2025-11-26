#!/usr/bin/env python3
"""
Parallel comprehensive draft dependence calculation with multi-parameter testing.
"""

import argparse
import logging
import sys
from pathlib import Path
from time import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
import geopandas as gpd
from tqdm import tqdm

from aislens.config import config
from aislens.utils import write_crs, setup_logging
from aislens.dataprep import dedraft_catchment_comprehensive

logger = logging.getLogger(__name__)


def define_parameter_sets():
    """Define parameter combinations for testing."""
    return {
        'standard': {
            'min_r2_threshold': 0.005,
            'min_correlation': -0.7,
            'ruptures_penalty': 0.5,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'mean',
            'model_selection': 'threshold_intercept',
            'description': 'Standard settings (continuous, permissive)'
        },
        'permissive': {
            'min_r2_threshold': 0.001,
            'min_correlation': -0.9,
            'ruptures_penalty': 0.3,
            'n_bins': 50,
            'min_points_per_bin': 3,
            'noisy_fallback': 'mean',
            'model_selection': 'threshold_intercept',
            'description': 'Very permissive - catch more linear relationships'
        },
        'strict': {
            'min_r2_threshold': 0.05,
            'min_correlation': 0.3,
            'ruptures_penalty': 1.0,
            'n_bins': 50,
            'min_points_per_bin': 10,
            'noisy_fallback': 'zero',
            'model_selection': 'threshold_intercept',
            'description': 'Strict quality thresholds'
        },
        'sensitive_changepoint': {
            'min_r2_threshold': 0.005,
            'min_correlation': -0.7,
            'ruptures_penalty': 0.1,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'mean',
            'model_selection': 'threshold_intercept',
            'description': 'Very sensitive changepoint detection'
        },
        'robust_changepoint': {
            'min_r2_threshold': 0.005,
            'min_correlation': -0.7,
            'ruptures_penalty': 2.0,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'mean',
            'model_selection': 'threshold_intercept',
            'description': 'Robust changepoint detection (fewer breakpoints)'
        },
        'fine_binning': {
            'min_r2_threshold': 0.005,
            'min_correlation': -0.7,
            'ruptures_penalty': 0.5,
            'n_bins': 100,
            'min_points_per_bin': 3,
            'noisy_fallback': 'mean',
            'model_selection': 'threshold_intercept',
            'description': 'Fine spatial resolution'
        },
        # Variants derived from fine_binning to reduce noisy fits or adjust resolution
        'fb_A': {
            'min_r2_threshold': 0.005,
            'min_correlation': -0.7,
            'ruptures_penalty': 0.5,
            'n_bins': 100,
            'min_points_per_bin': 5,
            'noisy_fallback': 'mean',
            'model_selection': 'threshold_intercept',
            'description': 'fine_binning variant: higher min_points_per_bin (safer)'
        },
        'fb_B': {
            'min_r2_threshold': 0.005,
            'min_correlation': -0.7,
            'ruptures_penalty': 0.5,
            'n_bins': 75,
            'min_points_per_bin': 4,
            'noisy_fallback': 'mean',
            'model_selection': 'threshold_intercept',
            'description': 'fine_binning variant: moderate bin count, slightly safer'
        },
        'fb_C': {
            'min_r2_threshold': 0.005,
            'min_correlation': -0.7,
            'ruptures_penalty': 0.8,
            'n_bins': 100,
            'min_points_per_bin': 3,
            'noisy_fallback': 'mean',
            'model_selection': 'threshold_intercept',
            'description': 'fine_binning variant: stronger changepoint penalty'
        },
        # Variants derived from sensitive_changepoint to reduce over-sensitivity
        'sc_A': {
            'min_r2_threshold': 0.005,
            'min_correlation': -0.7,
            'ruptures_penalty': 0.2,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'mean',
            'model_selection': 'threshold_intercept',
            'description': 'sensitive_changepoint variant: slightly less sensitive'
        },
        'sc_B': {
            'min_r2_threshold': 0.005,
            'min_correlation': -0.7,
            'ruptures_penalty': 0.3,
            'n_bins': 50,
            'min_points_per_bin': 5,
            'noisy_fallback': 'mean',
            'model_selection': 'threshold_intercept',
            'description': 'sensitive_changepoint variant: moderate penalty and safer bin counts'
        },
    }


def load_existing_parameters(shelf_name: str, save_dir: Path) -> dict:
    """
    Load existing parameters for a shelf from its parameter files.
    
    Args:
        shelf_name: Name of ice shelf
        save_dir: Directory containing parameter files
        
    Returns:
        Dictionary with draft parameters (minDraft, constantValue, paramType, alpha0, alpha1)
    """
    param_files = {
        'minDraft': save_dir / f"draftDepenBasalMelt_minDraft_{shelf_name}.nc",
        'constantValue': save_dir / f"draftDepenBasalMelt_constantMeltValue_{shelf_name}.nc",
        'paramType': save_dir / f"draftDepenBasalMelt_paramType_{shelf_name}.nc",
        'alpha0': save_dir / f"draftDepenBasalMeltAlpha0_{shelf_name}.nc",
        'alpha1': save_dir / f"draftDepenBasalMeltAlpha1_{shelf_name}.nc"
    }
    
    draft_params = {}
    
    try:
        # Load each parameter file and extract the scalar value
        for param_key, file_path in param_files.items():
            if file_path.exists():
                ds = xr.open_dataset(file_path)
                var_name = list(ds.data_vars)[0]
                
                # Extract the most common value (mode) as the representative parameter
                values = ds[var_name].values.flatten()
                valid_values = values[~np.isnan(values)]
                
                if len(valid_values) > 0:
                    # Use mode for paramType, mean for others
                    if param_key == 'paramType':
                        unique, counts = np.unique(valid_values, return_counts=True)
                        param_value = float(unique[np.argmax(counts)])
                    else:
                        param_value = float(np.mean(valid_values))
                    draft_params[param_key] = param_value
                else:
                    draft_params[param_key] = 0.0
                    
                ds.close()
            else:
                draft_params[param_key] = 0.0
                
    except Exception as e:
        logger.warning(f"Error loading parameters for {shelf_name}: {e}")
        # Return default values on error
        return {
            'minDraft': 0.0,
            'constantValue': 0.0,
            'paramType': 1,
            'alpha0': 0.0,
            'alpha1': 0.0
        }
    
    return draft_params


def process_single_shelf(args_tuple):
    """Process a single ice shelf in parallel."""
    (shelf_idx, shelf_name, icems_path, satobs_path, config_dict, 
     save_dir, param_dict) = args_tuple
    
    try:
        icems = gpd.read_file(icems_path).to_crs(config_dict['CRS_TARGET'])
        satobs = xr.open_dataset(satobs_path)
        # Ensure satellite flux units are converted to SI (kg m-2 s-1) in worker processes
        try:
            flux_var = config_dict.get('SATOBS_FLUX_VAR')
            draft_var = config_dict.get('SATOBS_DRAFT_VAR')
            if flux_var in satobs and satobs.attrs.get('units', '') == 'm of ice per year':
                # Use centralized conversion factor passed in config_dict when available
                # Build conversion factor from provided RHO_ICE and SECONDS_PER_YEAR
                rho = config_dict.get('RHO_ICE', None)
                s_per_yr = config_dict.get('SECONDS_PER_YEAR', None)
                if rho is not None and s_per_yr is not None:
                    myr_to_si = float(rho) / float(s_per_yr)
                else:
                    # Fallback to project convention (shouldn't happen if config provided)
                    myr_to_si = 910.0 / (365.0*24*3600)
                satobs[flux_var] = satobs[flux_var] * myr_to_si
                satobs[flux_var].attrs['units'] = 'kg m^-2 s^-1'
                if draft_var in satobs:
                    satobs[draft_var].attrs['units'] = 'm'
        except Exception:
            # Be resilient in worker processes; proceed without converting if anything goes wrong
            logger.debug('Worker: SATOBS unit conversion failed or not applicable; continuing without conversion')

        satobs = write_crs(satobs, config_dict['CRS_TARGET'])
        
        class SimpleConfig:
            def __init__(self, config_dict):
                for k, v in config_dict.items():
                    setattr(self, k, v)
        
        cfg = SimpleConfig(config_dict)
        
        result = dedraft_catchment_comprehensive(
            shelf_idx, icems, satobs, cfg,
            save_dir=save_dir,
            weights=None,
            weight_power=0.25,
            save_pred=True,
            save_coefs=True,
            **param_dict
        )
        
        if isinstance(result, dict) and 'full_results' in result and 'draft_params' in result:
            # Write a small scalar netCDF with aggregated parameters for downstream tools
            try:
                params = result.get('draft_params', {})
                # Ensure keys exist and are scalars (use np.nan if missing)
                minDraft = float(params.get('minDraft', np.nan)) if params.get('minDraft', None) is not None else np.nan
                constantValue = float(params.get('constantValue', np.nan)) if params.get('constantValue', None) is not None else np.nan
                alpha0 = float(params.get('alpha0', np.nan)) if params.get('alpha0', None) is not None else np.nan
                alpha1 = float(params.get('alpha1', np.nan)) if params.get('alpha1', None) is not None else np.nan
                paramType = int(params.get('paramType', -1)) if params.get('paramType', None) is not None else -1

                scalar_ds = xr.Dataset(
                    {
                        'draftDepenBasalMelt_minDraft': ((), minDraft),
                        'draftDepenBasalMelt_constantMeltValue': ((), constantValue),
                        'draftDepenBasalMeltAlpha0': ((), alpha0),
                        'draftDepenBasalMeltAlpha1': ((), alpha1),
                        'draftDepenBasalMelt_paramType': ((), paramType)
                    }
                )
                # Add sensible units so downstream tools can interpret scalars
                # Worker processes now convert SATOBS to SI mass-flux, so scalars are in kg m-2 s-1
                scalar_ds['draftDepenBasalMelt_minDraft'].attrs['units'] = 'm'
                scalar_ds['draftDepenBasalMelt_constantMeltValue'].attrs['units'] = 'kg m^-2 s^-1'
                scalar_ds['draftDepenBasalMeltAlpha0'].attrs['units'] = 'kg m^-2 s^-1'
                # Slope units: represent change in mass-flux per unit draft -> kg m^-3 s^-1
                scalar_ds['draftDepenBasalMeltAlpha1'].attrs['units'] = 'kg m^-3 s^-1'
                scalar_ds['draftDepenBasalMelt_paramType'].attrs['description'] = 'parameterization type code (0=linear,1=constant, etc)'
                scalar_ds.attrs['shelf_name'] = shelf_name
                # Write to save_dir with a clear scalar filename; keep existing grid outputs untouched
                scalar_file = Path(save_dir) / f'draftDepenBasalMelt_params_{shelf_name}.nc'
                scalar_ds.to_netcdf(scalar_file)
                scalar_ds.close()
            except Exception as e:
                logger.debug(f"Failed to write scalar params for {shelf_name}: {e}")
            return {
                'success': True,
                'shelf_name': shelf_name,
                'shelf_idx': shelf_idx,
                'full_results': result['full_results'],
                'draft_params': result['draft_params']
            }
        else:
            return {
                'success': False,
                'shelf_name': shelf_name,
                'shelf_idx': shelf_idx,
                'error': 'Invalid result structure'
            }
            
    except Exception as e:
        return {
            'success': False,
            'shelf_name': shelf_name,
            'shelf_idx': shelf_idx,
            'error': str(e)
        }


def calculate_parallel(icems, satobs, config_obj, param_dict, 
                      start_index=33, end_index=None, n_workers=None,
                      output_dir=None, processed_dir=None, skip_existing=True,
                      rerun_shelf_names=None):
    """
    Parallel comprehensive draft dependence calculation.
    """
    logger.info("="*80)
    logger.info("PARALLEL COMPREHENSIVE DRAFT DEPENDENCE CALCULATION")
    logger.info("="*80)
    
    total_start = time()
    
    if output_dir is None:
        output_dir = config_obj.DIR_ICESHELF_DEDRAFT_SATOBS / "comprehensive"
    else:
        output_dir = Path(output_dir)
    
    if processed_dir is None:
        processed_dir = config_obj.DIR_PROCESSED / "draft_dependence_changepoint"
    else:
        processed_dir = Path(processed_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Interim directory: {output_dir}")
    logger.info(f"Processed directory: {processed_dir}")
    logger.info(f"Parameters: {param_dict}")
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)
    
    logger.info(f"Using {n_workers} parallel workers")
    
    if end_index is None:
        end_index = len(icems)
    
    # Default shelf index selection
    shelf_indices = list(range(start_index, min(end_index, len(icems))))
    shelf_names = [icems.name.values[i] for i in shelf_indices]

    # If a list of shelf names to rerun is provided, override selection to those shelves
    if rerun_shelf_names:
        rerun_indices = []
        rerun_names_clean = [str(s).strip() for s in rerun_shelf_names]
        for name in rerun_names_clean:
            matches = list(np.where(icems.name.values == name)[0])
            if matches:
                rerun_indices.append(int(matches[0]))
            else:
                logger.warning(f"Requested rerun shelf not found in masks: {name}")
        if len(rerun_indices) == 0:
            logger.error("No valid rerun shelves found; aborting rerun request")
        else:
            shelf_indices = rerun_indices
            shelf_names = [icems.name.values[i] for i in shelf_indices]
    
    logger.info(f"Processing {len(shelf_indices)} ice shelves (indices {start_index}-{end_index-1})")

    if skip_existing:
        logger.info("Checking for existing files...")
        param_names = ['draftDepenBasalMelt_minDraft', 'draftDepenBasalMelt_constantMeltValue',
                      'draftDepenBasalMelt_paramType', 'draftDepenBasalMeltAlpha0', 'draftDepenBasalMeltAlpha1']

        shelves_to_skip = []
        for i, (idx, name) in enumerate(zip(shelf_indices, shelf_names)):
            all_exist = all((output_dir / f"{pn}_{name}.nc").exists() for pn in param_names)
            # If specific rerun_shelf_names provided, do not skip those shelves even when files exist
            if all_exist:
                if rerun_shelf_names and name in rerun_shelf_names:
                    logger.info(f"Forcing recompute for requested shelf: {name}")
                else:
                    shelves_to_skip.append((idx, name))

        logger.info(f"  Found {len(shelves_to_skip)} shelves with complete files")

        skip_indices = {idx for idx, _ in shelves_to_skip}
        shelf_indices = [idx for idx in shelf_indices if idx not in skip_indices]
        shelf_names = [icems.name.values[i] for i in shelf_indices]

        logger.info(f"  Will process {len(shelf_indices)} shelves")
    else:
        shelves_to_skip = []
    
    config_dict = {
        'CRS_TARGET': config_obj.CRS_TARGET,
        'SATOBS_FLUX_VAR': config_obj.SATOBS_FLUX_VAR,
        'SATOBS_DRAFT_VAR': config_obj.SATOBS_DRAFT_VAR,
        'SORRM_FLUX_VAR': getattr(config_obj, 'SORRM_FLUX_VAR', None),
        'SORRM_DRAFT_VAR': getattr(config_obj, 'SORRM_DRAFT_VAR', None),
        'TIME_DIM': getattr(config_obj, 'TIME_DIM', 'time'),
        'DATA_ATTRS': getattr(config_obj, 'DATA_ATTRS', {}),
        'RHO_ICE': getattr(config_obj, 'RHO_ICE', None),
        'SECONDS_PER_YEAR': getattr(config_obj, 'SECONDS_PER_YEAR', None)
    }
    
    icems_path = str(config_obj.FILE_ICESHELFMASKS)
    satobs_path = str(config_obj.FILE_PAOLO23_SATOBS_PREPARED)
    
    args_list = [
        (idx, name, icems_path, satobs_path, config_dict, output_dir, param_dict)
        for idx, name in zip(shelf_indices, shelf_names)
    ]
    
    logger.info(f"Starting parallel processing with {n_workers} workers...")
    all_results = {}
    all_draft_params = {}
    
    processed_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_shelf = {executor.submit(process_single_shelf, args): args[1] for args in args_list}
        
        for future in tqdm(as_completed(future_to_shelf), total=len(args_list), desc="Processing shelves"):
            shelf_name = future_to_shelf[future]
            try:
                result = future.result()
                
                if result['success']:
                    all_results[result['shelf_name']] = result['full_results']
                    all_draft_params[result['shelf_name']] = result['draft_params']
                    processed_count += 1
                else:
                    logger.warning(f"Failed: {result['shelf_name']} - {result['error']}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Exception processing {shelf_name}: {e}")
                failed_count += 1
    
    # Load existing parameters from files
    if skip_existing and shelves_to_skip:
        logger.info(f"Loading parameters from {len(shelves_to_skip)} existing shelf files...")
        for idx, name in shelves_to_skip:
            # Actually load the parameters from the files
            draft_params = load_existing_parameters(name, output_dir)
            all_draft_params[name] = draft_params
            
            # Create a results entry (mark as skipped but include paramType info)
            all_results[name] = {
                'is_meaningful': draft_params.get('paramType', 1) == 0,  # linear = meaningful
                'correlation': np.nan,
                'r2': np.nan,
                'shelf_name': name,
                'skipped': True
            }
            
            logger.debug(f"  Loaded {name}: paramType={draft_params.get('paramType', 'unknown')}, "
                        f"minDraft={draft_params.get('minDraft', 0):.2f}")
    
    total_time = time() - total_start
    
    logger.info("="*80)
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"  Processed: {processed_count}")
    logger.info(f"  Loaded from existing: {len(shelves_to_skip) if skip_existing else 0}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    if processed_count > 0:
        logger.info(f"  Average time per shelf: {total_time/processed_count:.2f}s")
    logger.info("="*80)
    
    if all_results:
        from calculate_draft_dependence_comprehensive_fast import (
            create_comprehensive_summary, merge_comprehensive_parameters
        )
        
        logger.info("Creating summary...")
        create_comprehensive_summary(all_results, all_draft_params, output_dir)
        
        logger.info("Merging parameters...")
        merge_comprehensive_parameters(all_draft_params, icems, satobs, config_obj, output_dir, processed_dir)
    
    return all_results, all_draft_params


def main():
    parser = argparse.ArgumentParser(
        description='Parallel comprehensive draft dependence calculation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of parallel workers (default: auto-detect, max 8)')
    parser.add_argument('--start-index', type=int, default=33,
                        help='Start processing from ice shelf index (default: 33)')
    parser.add_argument('--end-index', type=int, default=None,
                        help='End processing at ice shelf index (default: None = all)')
    parser.add_argument('--skip-existing', dest='skip_existing', action='store_true', default=True,
                        help='Skip shelves with existing files (default: True)')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                        help='Reprocess all shelves')
    parser.add_argument('--interim-dir', type=str, default=None,
                        help='Custom interim directory')
    parser.add_argument('--processed-dir', type=str, default=None,
                        help='Custom processed directory')
    parser.add_argument('--parameter-sets', nargs='+', default=None,
                        help='Parameter set names to test (e.g., standard permissive strict)')
    parser.add_argument('--test-all-sets', action='store_true',
                        help='Test all available parameter sets')
    parser.add_argument('--rerun-shelves', nargs='+', default=None,
                        help='Specific shelf names to re-run (overrides start/end selection)')
    parser.add_argument('--rerun-file', type=str, default=None,
                        help='Path to a file with one shelf name per line to re-run')
    
    args = parser.parse_args()
    
    interim_dir = Path(args.interim_dir) if args.interim_dir else config.DIR_ICESHELF_DEDRAFT_SATOBS
    output_dir = interim_dir / "comprehensive"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(output_dir, "calculate_draft_dependence_comprehensive_parallel")
    
    logger.info("PARALLEL COMPREHENSIVE DRAFT DEPENDENCE CALCULATOR")
    logger.info(f"Command: {' '.join(sys.argv)}")
    
    logger.info("Loading data...")
    satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)
    if config.SATOBS_FLUX_VAR in satobs and satobs.attrs.get('units', '') == 'm of ice per year':
        satobs[config.SATOBS_FLUX_VAR] = satobs[config.SATOBS_FLUX_VAR] * (config.RHO_ICE / config.SECONDS_PER_YEAR)
        satobs[config.SATOBS_FLUX_VAR].attrs['units'] = 'kg m^-2 s^-1'
    satobs = write_crs(satobs, config.CRS_TARGET)
    
    icems = gpd.read_file(config.FILE_ICESHELFMASKS).to_crs(config.CRS_TARGET)
    
    all_param_sets = define_parameter_sets()
    
    if args.test_all_sets:
        param_set_names = list(all_param_sets.keys())
    elif args.parameter_sets:
        param_set_names = args.parameter_sets
    else:
        param_set_names = ['standard']
    
    logger.info(f"Will process {len(param_set_names)} parameter set(s): {', '.join(param_set_names)}")
    
    for set_name in param_set_names:
        if set_name not in all_param_sets:
            logger.error(f"Unknown parameter set: {set_name}")
            continue
        
        param_set = all_param_sets[set_name]
        logger.info(f"\n{'='*80}")
        logger.info(f"PARAMETER SET: {set_name}")
        logger.info(f"Description: {param_set['description']}")
        logger.info(f"{'='*80}\n")
        
        set_interim_dir = interim_dir / set_name if len(param_set_names) > 1 else interim_dir
        set_processed_dir = (Path(args.processed_dir) / set_name) if args.processed_dir and len(param_set_names) > 1 else args.processed_dir
        
        param_dict = {k: v for k, v in param_set.items() if k != 'description'}
        
        # Build rerun list if provided
        rerun_list = None
        if args.rerun_file:
            try:
                with open(args.rerun_file, 'r') as fh:
                    rerun_list = [ln.strip() for ln in fh if ln.strip()]
            except Exception as e:
                logger.error(f"Failed to read rerun file {args.rerun_file}: {e}")
        elif args.rerun_shelves:
            rerun_list = args.rerun_shelves

        results, draft_params = calculate_parallel(
            icems, satobs, config,
            param_dict=param_dict,
            start_index=args.start_index,
            end_index=args.end_index,
            n_workers=args.n_workers,
            output_dir=set_interim_dir / "comprehensive" if set_interim_dir else None,
            processed_dir=set_processed_dir,
            skip_existing=args.skip_existing,
            rerun_shelf_names=rerun_list
        )
    
    logger.info("\nAll processing complete!")


if __name__ == "__main__":
    main()