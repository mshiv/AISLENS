# Draft Dependence Parameter Testing Workflow

## Overview

This document describes the workflow for testing different parameter combinations in the draft dependence analysis and validating the results. The goal is to find parameters that classify more ice shelves as having linear draft-melt relationships rather than constant (noisy) relationships.

## The Problem

The initial draft dependence analysis was too conservative - only about 30-40% of ice shelves were getting linear parameterizations (paramType=0), with the rest classified as "noisy" and given constant parameterizations (paramType=1). Tuning the parameters can get more linear relationships while maintaining reasonable quality.

## Key Parameters That Control Classification

These are the main parameters that can be adjusted:

```python
min_r2_threshold=0.1        # Lower → more linear classifications
min_correlation=0.2         # Lower → more linear classifications
ruptures_penalty=1.0        # Lower → more sensitive changepoint detection
n_bins=50                   # Affects resolution of draft binning
min_points_per_bin=5        # Fewer → allows sparser data
```

## Testing Workflow

### Step 1: Run the Base Analysis

Start by running the comprehensive draft dependence calculation with current parameters:

```bash
cd /Users/smurugan9/research/aislens/AISLENS/src/scripts
python calculate_draft_dependence_comprehensive.py
```

This creates output files in:
```
data/processed/draft_dependence_changepoint/
├── comprehensive_summary.csv                          # Key file to check
└── ruptures_draftDepenBasalMelt_parameters.nc        # Final merged parameters
```

### Step 2: Test Alternative Parameters

A parameter testing script can try different combinations:

```bash
python simple_parameter_tester.py
```

This tests 4 different parameter sets:
- **original**: Baseline parameters (for comparison)
- **permissive**: Lower thresholds (R²≥0.05, |corr|≥0.1)
- **very_permissive**: Very low thresholds (R²≥0.02, |corr|≥0.05)
- **sensitive_changepoint**: More aggressive changepoint detection

Results go to:
```
data/interim/iceshelf_dedraft_satobs/parameter_tests/
├── parameter_sets.json                    # Parameter definitions
├── results_summary.json                   # Summary statistics
├── original/
│   └── comprehensive/
│       └── comprehensive_summary.csv      # Detailed results per shelf
├── permissive/
└── visualizations/
    └── draft_dependence_comparison_*.png  # Validation plots
```

### Step 3: Create Validation Plots

Visualize the results to check if the parameterizations make physical sense:

```bash
# Visualize original parameters
python visualize_draft_dependence.py --parameter_set original --create_summary

# Visualize permissive parameters
python visualize_draft_dependence.py --parameter_set permissive

# Test with fewer shelves first (faster)
python visualize_draft_dependence.py --parameter_set original --max_shelves 12
```

The plots show:
- Observed data (black scatter points)
- Predicted relationship (orange line for linear, red for constant)
- Threshold draft depth (red dashed line)
- Quality metrics (MSE, R²)

### Step 4: Analyze Specific Results

Use the inspection script to understand why shelves are classified as they are:

```bash
# Overview of classifications
python inspect_draft_dependence.py --parameter_set original

# Check specific ice shelf
python inspect_draft_dependence.py --parameter_set permissive --shelf_name "Pine Island"

# Create threshold sensitivity analysis
python inspect_draft_dependence.py --parameter_set permissive --create_sensitivity_plot
```

### Step 5: Choose Final Parameters

Based on the results, pick the parameter set that gives the best balance between coverage and quality. Look for:

- **High coverage**: 50-70% linear parameterizations
- **Physical reasonableness**: Positive slopes, sensible thresholds
- **Visual validation**: Clear trends in scatter plots
- **Important shelves**: Key ice shelves (Pine Island, Thwaites, Ross) get linear parameterizations

### Step 6: Apply to Production

Once parameters are chosen, update the main script:

```python
# In calculate_draft_dependence_comprehensive.py
all_results, all_draft_params = calculate_draft_dependence_comprehensive(
    icems, satobs, config,
    min_r2_threshold=0.03,      # Chosen from testing
    min_correlation=0.1,        # Chosen from testing
    ruptures_penalty=0.8,       # Chosen from testing
    n_bins=50,
    min_points_per_bin=4,
    noisy_fallback='zero',
    model_selection='best'
)
```

## Output Files and What They Tell You

### comprehensive_summary.csv
Shows each ice shelf's classification and metrics:
- `is_meaningful`: Whether linear relationship was detected
- `correlation`: Pearson correlation coefficient
- `r2`: R² value for linear fit
- `threshold_draft`: Detected changepoint depth
- `paramType`: Final classification (0=linear, 1=constant)

### Scatter Plot Grids
Visual validation of fits:
- Melt rate on X-axis (m/yr)
- Draft depth on Y-axis (m, inverted)
- Clear trends indicate good linear relationships
- Scattered data suggests constant parameterization is appropriate

### parameter_set_comparison.png
Bar charts comparing different parameter sets:
- Number of meaningful relationships detected
- Distribution of linear vs constant parameterizations
- Helps identify which parameter set gives best coverage

## Expected Results from Different Parameter Sets

| Parameter Set | Linear Coverage | Notes |
|--------------|----------------|-------|
| Conservative (original) | ~30-50% | High quality, limited coverage |
| Moderate | ~40-55% | Balanced approach |
| Permissive | ~55-70% | Good coverage, acceptable quality |
| Very Permissive | ~70-90% | Maximum coverage, may include noise |

## Customizing Parameters Further

If the predefined parameter sets don't work well, custom combinations can be created in `simple_parameter_tester.py`:

```python
'my_custom': {
    'min_r2_threshold': 0.04,
    'min_correlation': 0.12,
    'ruptures_penalty': 0.9,
    'n_bins': 60,
    'min_points_per_bin': 4,
    'noisy_fallback': 'zero',
    'model_selection': 'best',
    'description': 'Custom parameters for specific needs'
}
```

## Troubleshooting Common Issues

### Imports Failing
Make sure the script is run from the right directory and the aislens package is installed:
```bash
cd /Users/smurugan9/research/aislens/AISLENS
pip install -e .
```

### No Data in Plots
Check that ice shelf masks and satellite data are properly loaded. Look at the console output for data loading messages.

### Poor Quality Fits
If many linear fits look unreasonable:
- Increase `min_r2_threshold` (be more strict)
- Increase `ruptures_penalty` (reduce false changepoints)
- Check specific problematic shelves with inspection script

### Still Too Few Linear Parameterizations
If more coverage is needed:
- Lower `min_correlation` further
- Reduce `min_points_per_bin`
- Try `n_bins=25` for small ice shelves

## Current Best Practice

After extensive testing, the recommended workflow is:

1. **Start with permissive parameters** (R²≥0.03, |corr|≥0.1)
2. **Generate validation plots** for visual quality check
3. **Inspect key ice shelves** (Pine Island, Thwaites, Ross, Filchner-Ronne)
4. **Adjust parameters** if needed based on specific requirements
5. **Document choices** in analysis notes for reproducibility

This typically gives 55-70% linear parameterizations with acceptable quality for Antarctic ice sheet modeling applications.
