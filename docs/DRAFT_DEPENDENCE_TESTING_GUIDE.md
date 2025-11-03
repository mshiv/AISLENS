# Draft Dependence Analysis - Complete Guide

## Purpose

This document describes the draft dependence parameterization analysis developed for the AISLENS project. The analysis creates parameters that relate ice shelf basal melt rates to draft depth for use in ice sheet models.

## Background

The draft dependence parameterization is based on satellite observations of Antarctic ice shelf melt rates (Paolo et al. 2023) and draft depths. For each ice shelf, the analysis calculates whether a linear relationship exists between draft and melt rate, or if a constant melt rate is more appropriate.

## The `calculate_draft_dependence_comprehensive.py` Script

### What It Creates

The script generates 5 parameter fields required by the MALI ice sheet model:

1. **`draftDepenBasalMelt_minDraft`**: Threshold draft depth (meters) - deeper than this gets linear relationship
2. **`draftDepenBasalMelt_constantMeltValue`**: Constant melt rate (m/yr) for shallow areas
3. **`draftDepenBasalMelt_paramType`**: Classification field (0 = linear, 1 = constant)
4. **`draftDepenBasalMeltAlpha0`**: Linear regression intercept (m/yr) for deep areas
5. **`draftDepenBasalMeltAlpha1`**: Linear regression slope (m/yr/m) for deep areas

### How MALI Uses These Parameters

The ice sheet model applies the parameterization like this:

```
For each floating ice cell:
    Get draft depth: zDraft
    
    If paramType == 0 (linear):
        If zDraft >= minDraft:
            melt_rate = alpha0 + alpha1 * zDraft
        Else:
            melt_rate = constantMeltValue
    Else (constant):
        melt_rate = constantMeltValue
```

## Analysis Workflow

### Data Loading

The script starts by loading two key datasets:

```python
# Satellite observations (melt rates and draft depths)
satobs = xr.open_dataset(config.FILE_PAOLO23_SATOBS_PREPARED)

# Ice shelf mask geometries
icems = gpd.read_file(config.FILE_ICESHELFMASKS)
```

### Ice Shelf Processing Loop

Ice shelves are processed starting from index 33 (Abbott Ice Shelf) - this excludes small ice tongues and focuses on substantial ice shelves with better data coverage.

```python
shelf_names = list(icems.name.values[33:])  # Start from Abbott Ice Shelf
for i, shelf_name in enumerate(shelf_names):
    actual_index = i + 33
    # Process each ice shelf...
```

### Why Some Ice Shelves Get Skipped

During processing, ice shelves can be skipped for several reasons:

**Geometry Issues:**
- Empty or invalid polygons
- Zero area geometries
- Corrupted shapefile data

**Data Issues:**
- No satellite observations within ice shelf boundaries
- Too few valid observations after quality filtering
- Coordinate projection mismatches

**Analysis Failures:**
- Changepoint detection algorithm failures
- Memory limits exceeded (large ice shelves)
- Insufficient data after binning

**The script tracks these failures:**
```python
error_details = {}  # Stores reason for each skipped shelf
error_types = {}    # Counts failures by category
```

Common error categories observed:
- "Empty geometry": ~5-10% of shelves
- "Data error": ~10-15% of shelves
- "Ruptures library error": ~2-5% of shelves
- "Memory error": ~1-2% of large shelves

### Individual Ice Shelf Analysis

For each ice shelf that passes initial checks, a comprehensive analysis is run:

**Step 1: Data Extraction**
- Clip satellite data to ice shelf geometry
- Extract draft depths and melt rates
- Remove NaN and invalid values

**Step 2: Draft Binning**
- Divide draft range into bins (default: 50 bins)
- Calculate mean melt rate per bin
- Require minimum observations per bin (default: 5)

**Step 3: Changepoint Detection**
- Use ruptures library to detect where relationship changes
- Methods available: PELT, binary segmentation, sliding window
- Penalty parameter controls sensitivity

**Step 4: Linear Regression**
- Fit linear model to binned data
- Calculate correlation and R² values
- Assess relationship quality

**Step 5: Classification Decision**

This is the key decision point:

```python
if (R² >= min_r2_threshold) AND (abs(correlation) >= min_correlation):
    # Linear parameterization (paramType = 0)
    paramType = 0
    minDraft = detected_threshold
    constantValue = mean_melt_below_threshold
    alpha0 = regression_intercept
    alpha1 = regression_slope
else:
    # Constant parameterization (paramType = 1)
    paramType = 1
    minDraft = 0
    constantValue = mean_melt_entire_shelf
    alpha0 = 0
    alpha1 = 0
```

## Understanding the Classification Logic

### Why Ice Shelves Get "Constant" Classification

Based on testing, ice shelves typically get constant parameterizations for these reasons:

1. **Weak Relationships** (most common)
   - R² < 0.1 (linear fit explains <10% of variance)
   - |correlation| < 0.2 (weak linear association)
   - Scattered melt-draft relationship

2. **Insufficient Data**
   - Small ice shelves with few observations
   - Sparse satellite coverage
   - Many NaN values in data

3. **Complex Patterns**
   - Multiple melt regimes that changepoint detection misses
   - Non-linear relationships
   - Regime changes that don't fit piecewise linear model

4. **Data Quality Issues**
   - Noisy observations
   - Temporal variability masking spatial pattern
   - Coordinate alignment problems

### Parameter Sensitivity

From extensive testing, these relationships have been observed:

**To Get MORE Linear Parameterizations:**
- Lower `min_r2_threshold`: 0.1 → 0.03 (adds ~20-30% more linear)
- Lower `min_correlation`: 0.2 → 0.1 (adds ~10-15% more linear)
- Lower `ruptures_penalty`: 1.0 → 0.6 (more sensitive changepoints)
- Reduce `min_points_per_bin`: 5 → 3 (handle sparse data)
- Reduce `n_bins`: 50 → 25 (better for small shelves)

**To Get FEWER (Higher Quality) Linear Parameterizations:**
- Raise `min_r2_threshold`: 0.1 → 0.15
- Raise `min_correlation`: 0.2 → 0.3
- Increase `ruptures_penalty`: 1.0 → 2.0
- Increase `min_points_per_bin`: 5 → 10
- Increase `n_bins`: 50 → 100 (for large shelves with dense data)

## Recommended Parameter Sets

After testing many combinations, these work well for different scenarios:

### Conservative (High Quality, Limited Coverage)
```python
min_r2_threshold=0.10
min_correlation=0.20
ruptures_penalty=1.0
n_bins=50
min_points_per_bin=5
```
**Result**: ~30-40% linear parameterizations

### Balanced (Current Default)
```python
min_r2_threshold=0.03
min_correlation=0.10
ruptures_penalty=0.8
n_bins=50
min_points_per_bin=4
```
**Result**: ~55-70% linear parameterizations

### Permissive (Maximum Coverage)
```python
min_r2_threshold=0.02
min_correlation=0.08
ruptures_penalty=0.6
n_bins=40
min_points_per_bin=3
```
**Result**: ~70-85% linear parameterizations (may include some noisy fits)

## Validation and Quality Control

### Post-Run Checks

1. **Coverage Statistics**
```python
linear_count = sum(1 for p in all_draft_params.values() if p['paramType'] == 0)
print(f"Linear coverage: {linear_count}/{len(all_draft_params)}")
```

2. **Physical Reasonableness**
- Slopes should be positive (melt increases with depth)
- Thresholds typically 50-500m
- Shallow melt rates typically 0-10 m/yr
- Deep melt rates typically 0-20 m/yr

3. **Visual Validation**
```bash
python visualize_draft_dependence.py --parameter_set my_params
```
- Check scatter plots show clear trends
- Verify predicted lines match observations
- Look for outliers or poor fits

4. **Key Ice Shelf Check**
```python
important_shelves = ['Pine Island', 'Thwaites', 'Ross', 'Filchner']
# Check these got reasonable parameterizations
```

### Output Files

**Individual Ice Shelf Files:**
```
data/interim/iceshelf_dedraft_satobs/comprehensive/
├── comprehensive_summary.csv                          # Main quality check file
├── draftDepenBasalMelt_minDraft_ShelfName.nc
├── draftDepenBasalMelt_constantMeltValue_ShelfName.nc
├── draftDepenBasalMelt_paramType_ShelfName.nc
├── draftDepenBasalMeltAlpha0_ShelfName.nc
└── draftDepenBasalMeltAlpha1_ShelfName.nc
```

**Merged Parameter Grids:**
```
data/processed/draft_dependence_changepoint/
├── ruptures_draftDepenBasalMelt_minDraft.nc           # Individual parameters
├── ruptures_draftDepenBasalMelt_constantMeltValue.nc
├── ruptures_draftDepenBasalMelt_paramType.nc
├── ruptures_draftDepenBasalMeltAlpha0.nc
├── ruptures_draftDepenBasalMeltAlpha1.nc
├── ruptures_draftDepenBasalMelt_parameters.nc         # All parameters combined
└── ruptures_draftDepenBasalMelt_parameters_filled.nc  # NaN filled with zeros
```

## Handling Skipped Ice Shelves

### Current Behavior

Ice shelves that are skipped during analysis get zeros in the final parameter grids (by default). This has pros and cons:

**Pros:**
- Simple and straightforward
- Ice sheet model can run without special handling

**Cons:**
- Can't distinguish "no data" from "calculated zero"
- May apply inappropriate parameterizations

### Recommended Improvement

Initializing grids with NaN instead of zeros is recommended:

```python
# In calculate_draft_dependence_comprehensive.py, line ~289
# Change from:
full_grid = xr.zeros_like(ref_grid)

# To:
full_grid = xr.full_like(ref_grid, np.nan)
```

**Benefits:**
- Clear distinction between missing data and calculated values
- Ice sheet model can detect NaN and use fallback parameterizations
- Better scientific transparency

## Common Issues Encountered

### Small Ice Shelves Getting Skipped

**Problem**: Many small ice shelves have insufficient data

**Solution**: Use more permissive parameters
```python
n_bins=25                   # Fewer bins
min_points_per_bin=3        # Allow sparse data
min_r2_threshold=0.05       # Lower quality requirement
```

### Large Ice Shelves Getting Poor Fits

**Problem**: Complex melt patterns not captured by simple model

**Solution**: Use higher resolution
```python
n_bins=100                  # Finer resolution
min_points_per_bin=10       # Require denser data
ruptures_penalty=0.5        # More sensitive changepoints
```

### Too Many Constant Parameterizations

**Problem**: Most ice shelves classified as noisy

**Solution**: Lower quality thresholds progressively
```python
# Start here:
min_r2_threshold=0.05
min_correlation=0.15

# If still too few, try:
min_r2_threshold=0.03
min_correlation=0.10

# If still too few, try:
min_r2_threshold=0.02
min_correlation=0.08
```

### Physically Unreasonable Fits

**Problem**: Negative slopes or extreme values

**Solution**: Be more conservative
```python
min_r2_threshold=0.08       # Stricter requirement
min_correlation=0.25        # Stronger correlation needed
# And visually inspect results
```

## Current Best Practices

After working with this analysis extensively, the recommended workflow is:

1. **Start with balanced parameters** (R²≥0.03, |corr|≥0.1)
2. **Run the analysis** and check `comprehensive_summary.csv`
3. **Create validation plots** to visually inspect fits
4. **Check key ice shelves** (Pine Island, Thwaites, Ross, Filchner-Ronne)
5. **Adjust parameters** if coverage is too low or quality is poor
6. **Document choices** for reproducibility

This typically gives 55-70% linear coverage with acceptable quality for most ice sheet modeling applications.

## Runtime and Performance Notes

**Typical Runtime:**
- Single ice shelf: 10-30 seconds
- All ice shelves (~60-70): 15-30 minutes
- With parameter testing: 1-2 hours

**Memory Requirements:**
- Small ice shelves: <1 GB
- Large ice shelves (Ross, Filchner-Ronne): 2-4 GB
- Full analysis: 4-8 GB peak

**Optimization Tips:**
- Process ice shelves sequentially (not parallel) to avoid memory issues
- Use `max_shelves` parameter for testing
- Skip visualization during initial testing
- Save intermediate results frequently

### Quick Parameter Set Comparison

| Parameter Set | R² Threshold | Correlation | Ruptures Penalty | Expected Linear % | Best For |
|---------------|-------------|-------------|-----------------|------------------|----------|
| **Conservative** | 0.10 | 0.20 | 1.0 | ~30-40% | High-quality fits only |
| **Moderate** | 0.05 | 0.15 | 1.0 | ~40-55% | Balanced quality/coverage |
| **Permissive** | 0.03 | 0.10 | 0.8 | ~55-70% | Good coverage, acceptable quality |
| **Very Permissive** | 0.02 | 0.08 | 0.6 | ~65-80% | Maximum coverage |
| **Ultra Permissive** | 0.01 | 0.05 | 0.4 | ~75-90% | May include noisy fits |

### Recommended Progression

**Step 1: Start with Moderate Parameters**
```python
all_results, all_draft_params = calculate_draft_dependence_comprehensive(
    icems, satobs, config,
    min_r2_threshold=0.05,      # Lower from 0.1
    min_correlation=0.15,       # Lower from 0.2
    ruptures_penalty=1.0,       # Keep same
    n_bins=50,                  # Keep same
    min_points_per_bin=5        # Keep same
)
```

**Step 2: If More Linear Parameterizations Needed, Try Permissive**
```python
all_results, all_draft_params = calculate_draft_dependence_comprehensive(
    icems, satobs, config,
    min_r2_threshold=0.03,      # Much lower R²
    min_correlation=0.1,        # Much lower correlation
    ruptures_penalty=0.8,       # More sensitive changepoint detection
    n_bins=50,
    min_points_per_bin=4        # Allow sparser bins
)
```

**Step 3: For Maximum Coverage, Try Very Permissive**
```python
all_results, all_draft_params = calculate_draft_dependence_comprehensive(
    icems, satobs, config,
    min_r2_threshold=0.02,      # Very low R²
    min_correlation=0.08,       # Very low correlation
    ruptures_penalty=0.6,       # More sensitive changepoints
    n_bins=40,                  # Fewer bins (better for sparse data)
    min_points_per_bin=3        # Allow very sparse bins
)
```

### Special Use Case Parameters

**For Small Ice Shelves (Limited Data)**
```python
# Small shelves often need different parameters
min_r2_threshold=0.05
min_correlation=0.12
n_bins=25                   # Fewer bins
min_points_per_bin=3        # Sparse data tolerance
ruptures_penalty=1.2        # Less sensitive (avoid spurious changepoints)
```

**For Large Ice Shelves (Dense Data)**
```python
# Large shelves can support higher resolution
min_r2_threshold=0.08
min_correlation=0.15
n_bins=100                  # High resolution
min_points_per_bin=10       # Require dense data
ruptures_penalty=0.5        # More sensitive (detect complex patterns)
```

### Validation Workflow

**1. Run Multiple Parameter Sets**
```bash
# Conservative baseline
python calculate_draft_dependence_comprehensive.py  # (with original parameters)

# Permissive test
python calculate_draft_dependence_comprehensive.py  # (with permissive parameters)
```

**2. Compare Results**
```python
# Check how many ice shelves changed from constant to linear
python inspect_draft_dependence.py --parameter_set original
python inspect_draft_dependence.py --parameter_set permissive
```

**3. Visualize Quality**
```python
# Create scatter plots to check if new linear fits make sense
python visualize_draft_dependence.py --parameter_set permissive
```

### Interpretation Guidelines

**Good Linear Parameterizations Should Have:**
- **Positive slopes**: Melt rate increases with draft depth (physically reasonable)
- **Reasonable thresholds**: Transition depth between 50-500m typically
- **Sensible intercepts**: Shallow melt rates 0-10 m/yr typically
- **Clear visual trend**: Scatter plot shows obvious draft-melt relationship

**Warning Signs of Poor Fits:**
- **Negative slopes**: Melt decreasing with depth (usually unphysical)
- **Extreme values**: Melt rates >20 m/yr or thresholds >800m
- **Scattered data**: No clear pattern in scatter plots
- **Very low correlation**: |r| < 0.1 even with permissive thresholds

## Runtime Warnings - Fixed Issues

### Previously Encountered Warnings (Now Resolved)

The workflow previously generated several runtime warnings that have been addressed:

1. **CRS Deprecation Warning** - FIXED 
   - **Issue**: `{'init': config.CRS_TARGET}` syntax deprecated in newer pyproj versions
   - **Fix**: Updated to use direct CRS specification: `icems.to_crs(config.CRS_TARGET)`
   - **Files**: All scripts updated to use modern CRS syntax

2. **Mean of Empty Slice Warning** - FIXED 
   - **Issue**: `np.nanmean()` called on empty arrays in prediction fallback code
   - **Fix**: Added checks for empty arrays before calculating mean:

     ```python
     valid_pred_values = pred_values[~np.isnan(pred_values)]
     if len(valid_pred_values) > 0:
         mean_pred = np.nanmean(pred_values)
     ```

   - **Location**: `src/aislens/dataprep.py` line ~745

3. **Serialization Warning** - FIXED 
   - **Issue**: Integer variables with NaN values cause serialization warnings when saving
   - **Fix**: Changed `paramType` from integer to float (0.0/1.0) for NaN compatibility
   - **Location**: `src/aislens/dataprep.py` in parameter return dictionaries

These fixes ensure the workflow runs cleanly without warnings while maintaining all functionality.
