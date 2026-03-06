# Dataset Merger: Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [The Problem: Why Simple Merge Doesn't Work](#the-problem-why-simple-merge-doesnt-work)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [How It Works Internally](#how-it-works-internally)
6. [Usage Examples](#usage-examples)
7. [Point-in-Time Correctness](#point-in-time-correctness)
8. [Common Patterns](#common-patterns)

---

## Overview

The `DatasetMerger` is a utility for combining xarray datasets with different time frequencies while maintaining data integrity and point-in-time correctness. It's specifically designed for financial data workflows where you need to merge:

- **Monthly price data** (CRSP) with **annual fundamentals** (Compustat)
- **Daily returns** with **quarterly earnings**
- Any combination of different-frequency panel data

The merger works with Hindsight's multi-dimensional time structure (`year`, `month`, `day`, `hour`) rather than a single flattened time axis.

**Location:** `src/pipeline/data_handler/merge.py`

---

## The Problem: Why Simple Merge Doesn't Work

### Hindsight's Data Structure

Hindsight stores panel data as xarray Datasets with dimensions:

```
Dimensions:  (year: 5, month: 12, day: 1, hour: 1, asset: 1000)
Coordinates:
  * year     (year) int64 2016 2017 2018 2019 2020
  * month    (month) int64 1 2 3 4 5 6 7 8 9 10 11 12
  * day      (day) int64 1
  * hour     (hour) int64 0
  * asset    (asset) int64 10001 10002 10003 ...
  * time     (year, month, day, hour) datetime64[ns] ...
Data variables:
    ret      (year, month, day, hour, asset) float64 ...
    me       (year, month, day, hour, asset) float64 ...
```

### The Frequency Mismatch Problem

**CRSP (Monthly):**
```
Dimensions: (year: 5, month: 12, day: 1, hour: 1, asset: 5000)
Variables: ret, me, prc, vol, ...
```

**Compustat (Annual):**
```
Dimensions: (year: 5, month: 1, day: 1, hour: 1, asset: 4000)
Variables: seq, txditc, at, ...
```

A naive `xr.merge(crsp, compustat)` fails because:

1. **Different `month` dimensions**: CRSP has 12 months, Compustat has 1
2. **Different `asset` sets**: Not all CRSP stocks have Compustat coverage
3. **No time alignment**: Annual data needs to be broadcast across months
4. **Point-in-time violation**: Using Dec 2019 data in Jan 2020 is look-ahead bias

### What DatasetMerger Solves

1. **Broadcasts** annual data across all 12 months
2. **Aligns** on the `asset` dimension (keeping left's assets)
3. **Forward-fills** missing values appropriately
4. **Applies time offsets** for point-in-time correctness
5. **Namespaces** variables to avoid collisions (`seq` → `comp_seq`)

---

## Core Concepts

### MergeSpec: The Configuration Object

`MergeSpec` is a dataclass that describes how to merge one dataset into another:

```python
@dataclass
class MergeSpec:
    right_name: str                           # Name of dataset to merge
    on: Union[str, List[str]] = "asset"       # Join dimension(s)
    time_alignment: TimeAlignment = FFILL     # How to align time
    time_offset_months: int = 0               # Lag/lead offset
    ffill_limit: Optional[int] = None         # Max forward-fill periods
    prefix: str = ""                          # Variable name prefix
    suffix: str = ""                          # Variable name suffix
    variables: Optional[List[str]] = None     # Variables to include
    drop_vars: Optional[List[str]] = None     # Variables to exclude
```

### TimeAlignment: How to Handle Time Differences

```python
class TimeAlignment(Enum):
    EXACT = "exact"      # Only match exact timestamps (rare for cross-freq)
    FFILL = "ffill"      # Forward-fill: carry last value forward
    BFILL = "bfill"      # Backward-fill: carry next value backward
    NEAREST = "nearest"  # Use nearest available value
    AS_OF = "as_of"      # Point-in-time: like ffill but with offset logic
```

**When to use each:**

| Alignment | Use Case |
|-----------|----------|
| `FFILL` | Default for most merges; annual data fills forward through months |
| `AS_OF` | Same as FFILL but semantically indicates point-in-time intent |
| `BFILL` | Rare; when you want future data to fill backward |
| `EXACT` | When frequencies match exactly |
| `NEAREST` | When you want closest available value |

### MergeMethod: Join Types

```python
class MergeMethod(Enum):
    LEFT = "left"     # Keep all assets from left dataset
    RIGHT = "right"   # Keep all assets from right dataset
    INNER = "inner"   # Keep only assets in both datasets
    OUTER = "outer"   # Keep all assets from both datasets
```

**Default is `LEFT`** - this keeps all CRSP stocks even if they don't have Compustat coverage (those get NaN for Compustat fields).

---

## API Reference

### DatasetMerger Class

```python
class DatasetMerger:
    def merge(
        self,
        left: xr.Dataset,           # Primary dataset (higher frequency)
        right: xr.Dataset,          # Secondary dataset (to merge in)
        spec: MergeSpec,            # Merge configuration
        method: MergeMethod = LEFT  # Join type
    ) -> xr.Dataset:
        """Merge two datasets according to specification."""
        
    def merge_multiple(
        self,
        base: xr.Dataset,                    # Primary dataset
        datasets: Dict[str, xr.Dataset],     # Named datasets to merge
        specs: List[MergeSpec],              # One spec per dataset
        method: MergeMethod = LEFT
    ) -> xr.Dataset:
        """Merge multiple datasets into base."""
```

### Convenience Function

```python
def merge_datasets(
    base: xr.Dataset,
    datasets: Dict[str, xr.Dataset],
    merge_config: List[Dict[str, Any]]  # Config dicts instead of MergeSpec
) -> xr.Dataset:
    """
    Merge using plain dictionaries (for YAML-driven configs).
    
    Example config:
    [
        {
            'right_name': 'compustat',
            'on': 'asset',
            'time_alignment': 'as_of',
            'time_offset_months': 6,
            'prefix': 'comp_'
        }
    ]
    """
```

---

## How It Works Internally

### Step-by-Step Merge Process

When you call `merger.merge(left, right, spec)`, here's what happens:

#### Step 1: Prepare Right Dataset (`_prepare_right_dataset`)

```python
# If variables specified, select only those
if spec.variables is not None:
    right = right[spec.variables]

# If drop_vars specified, remove those
if spec.drop_vars is not None:
    right = right.drop_vars(spec.drop_vars)

# Apply prefix/suffix to avoid name collisions
if spec.prefix or spec.suffix:
    right = right.rename({var: f"{prefix}{var}{suffix}" for var in right.data_vars})
```

**Example:**
```python
# Before: right has variables ['seq', 'txditc', 'at']
# With spec.prefix='comp_', spec.variables=['seq', 'at']
# After: right has variables ['comp_seq', 'comp_at']
```

#### Step 2: Expand to Time Grid (`_expand_to_time_grid`)

This is the core logic. For annual-to-monthly:

```python
# 1. Reindex year dimension to match left's years
result = right.reindex(year=left.coords['year'], method=None)

# 2. If right has no 'month' dimension, broadcast across all months
if 'month' not in result.dims:
    result = result.expand_dims(month=left.coords['month'])
# If right has fewer months, reindex
elif result.sizes['month'] < left.sizes['month']:
    result = result.reindex(month=left.coords['month'], method=None)

# 3. Same for 'day' and 'hour' dimensions

# 4. Apply forward-fill along time dimensions
for dim in ['year', 'month', 'day', 'hour']:
    if dim in result.dims:
        result = result.ffill(dim=dim, limit=ffill_limit)
```

**Visual example:**

```
Annual data (before):
         Year 2019  Year 2020
Asset A:   100        200

After expand_dims(month=[1..12]):
         Year 2019                    Year 2020
         M1  M2  M3 ... M12           M1  M2  M3 ... M12
Asset A: 100 NaN NaN ... NaN          200 NaN NaN ... NaN

After ffill(dim='month'):
         Year 2019                    Year 2020
         M1  M2  M3 ... M12           M1  M2  M3 ... M12
Asset A: 100 100 100 ... 100          200 200 200 ... 200
```

#### Step 3: Apply Time Offset (`_apply_offset_mask`)

If `time_offset_months > 0`, we need to shift which year's data is available when:

```python
# For offset=6 (data available 6 months after fiscal year end):
# - Months 1-6 of year Y use data from year Y-1
# - Months 7-12 of year Y use data from year Y

cutoff_month = offset_months % 12  # = 6

for var in right.data_vars:
    # Shift data by 1 year
    shifted = da.shift(year=1)
    
    # Use current year data for months > cutoff, shifted for months <= cutoff
    use_current = month_coord > cutoff_month
    combined = xr.where(use_current, da, shifted)
```

**Visual example with offset=6:**

```
Before offset (wrong - look-ahead bias):
         Year 2020
         Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
Asset A: 200  200  200  200  200  200  200  200  200  200  200  200
         ^^^ This uses 2020 data in Jan 2020, but 2020 annual report
             isn't published until ~March 2021!

After offset (correct - point-in-time):
         Year 2020
         Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
Asset A: 100  100  100  100  100  100  200  200  200  200  200  200
         ^^^ Uses 2019 data (available) ^^^ Uses 2020 data (now available)
```

#### Step 4: Merge on Asset (`_merge_on_asset`)

```python
# Reindex right to match left's assets
right = right.reindex(asset=left.coords['asset'], method=None)

# Merge using xarray's merge
result = xr.merge([left, right], join='left', compat='override')
```

Assets in left but not in right get NaN for right's variables.

---

## Usage Examples

### Basic: Merge Compustat into CRSP

```python
from src.pipeline.data_handler import DatasetMerger, MergeSpec, TimeAlignment

# Load datasets
crsp = load_crsp_monthly()      # (year, month, day, hour, asset)
comp = load_compustat_annual()  # (year, month=1, day, hour, asset)

# Create merger
merger = DatasetMerger()

# Define merge specification
spec = MergeSpec(
    right_name='compustat',
    on='asset',
    time_alignment=TimeAlignment.FFILL,
    prefix='comp_'
)

# Merge
merged = merger.merge(crsp, comp, spec)

# Result has: ret, me, prc, ..., comp_seq, comp_txditc, comp_at
```

### With Point-in-Time Offset

```python
spec = MergeSpec(
    right_name='compustat',
    on='asset',
    time_alignment=TimeAlignment.AS_OF,
    time_offset_months=6,  # Data available 6 months after fiscal year end
    prefix='comp_',
    variables=['seq', 'txditc', 'at']  # Only these variables
)

merged = merger.merge(crsp, comp, spec)
```

### Merge Multiple Datasets

```python
merger = DatasetMerger()

datasets = {
    'compustat': comp_annual,
    'ibes': ibes_quarterly,
}

specs = [
    MergeSpec(
        right_name='compustat',
        on='asset',
        time_alignment=TimeAlignment.AS_OF,
        time_offset_months=6,
        prefix='comp_'
    ),
    MergeSpec(
        right_name='ibes',
        on='asset',
        time_alignment=TimeAlignment.FFILL,
        prefix='ibes_'
    ),
]

merged = merger.merge_multiple(crsp, datasets, specs)
```

### Using Config Dictionaries (for YAML)

```python
from src.pipeline.data_handler import merge_datasets

config = [
    {
        'right_name': 'compustat',
        'on': 'asset',
        'time_alignment': 'as_of',
        'time_offset_months': 6,
        'prefix': 'comp_',
        'variables': ['seq', 'txditc', 'at']
    }
]

merged = merge_datasets(crsp, {'compustat': comp}, config)
```

---

## Point-in-Time Correctness

### Why It Matters

In backtesting and factor construction, using data before it was actually available creates **look-ahead bias**. This inflates performance metrics unrealistically.

**Example: Book-to-Market Ratio**

The Fama-French methodology:
1. Use book equity from fiscal year ending in calendar year t-1
2. Use market equity from December of year t-1
3. Form portfolios at the end of June in year t
4. Hold portfolios from July t to June t+1

This means December 2019 book equity is used starting July 2020, not January 2020.

### How DatasetMerger Handles It

The `time_offset_months` parameter shifts when data becomes "available":

```python
spec = MergeSpec(
    right_name='compustat',
    time_offset_months=6,  # 6-month lag
    ...
)
```

**What happens internally:**

For each variable in the right dataset:
1. Create a shifted version (`da.shift(year=1)`)
2. For months 1-6: use the shifted (previous year's) data
3. For months 7-12: use the current year's data

**Timeline visualization:**

```
Fiscal Year 2019 data (ends Dec 2019):
├── Published: ~March 2020 (10-K filing deadline)
├── Conservative availability: June 2020 (offset=6)
└── Used in merged dataset: July 2020 - June 2021

In the merged dataset:
Year 2020, Months 1-6:  Uses 2018 fiscal year data
Year 2020, Months 7-12: Uses 2019 fiscal year data
Year 2021, Months 1-6:  Uses 2019 fiscal year data
Year 2021, Months 7-12: Uses 2020 fiscal year data
```

---

## Common Patterns

### Pattern 1: FF3 CRSP + Compustat Merge

```python
spec = MergeSpec(
    right_name='compustat',
    on='asset',
    time_alignment=TimeAlignment.AS_OF,
    time_offset_months=6,
    prefix='comp_',
    variables=['seq', 'txditc', 'pstkrv', 'pstkl', 'pstk', 'at']
)

merged = merger.merge(crsp_monthly, compustat_annual, spec)

# Now compute book equity
# BE = SEQ + TXDITC - PS (where PS = coalesce(pstkrv, pstkl, pstk, 0))
```

### Pattern 2: Select Specific Variables

```python
# Only merge specific Compustat variables
spec = MergeSpec(
    right_name='compustat',
    variables=['seq', 'at'],  # Only these
    prefix='comp_'
)
```

### Pattern 3: Exclude Variables

```python
# Merge all except certain variables
spec = MergeSpec(
    right_name='compustat',
    drop_vars=['indfmt', 'datafmt', 'popsrc', 'consol'],  # Exclude these
    prefix='comp_'
)
```

### Pattern 4: Inner Join (Only Common Assets)

```python
merged = merger.merge(
    crsp, comp, spec,
    method=MergeMethod.INNER  # Only keep assets in both
)
```

### Pattern 5: Limit Forward-Fill

```python
spec = MergeSpec(
    right_name='compustat',
    time_alignment=TimeAlignment.FFILL,
    ffill_limit=12,  # Only fill up to 12 months
    ...
)
```

---

## Summary

| Feature | Description |
|---------|-------------|
| **Multi-frequency merge** | Annual → Monthly, Quarterly → Daily, etc. |
| **Point-in-time** | `time_offset_months` prevents look-ahead bias |
| **Variable namespacing** | `prefix`/`suffix` avoid collisions |
| **Flexible selection** | `variables` and `drop_vars` control what's merged |
| **Join types** | LEFT, RIGHT, INNER, OUTER |
| **Fill strategies** | FFILL, BFILL, NEAREST, EXACT, AS_OF |
| **Pure xarray** | No pandas in the merge logic |

The `DatasetMerger` is the foundation for building complex multi-source pipelines like Fama-French factor construction, where CRSP returns, Compustat fundamentals, and other data sources must be combined with proper time alignment.

