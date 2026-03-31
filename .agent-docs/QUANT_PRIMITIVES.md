# Quantitative Primitives

This document covers the six core quantitative primitives used for cross-sectional analysis and portfolio construction. Three are **formula functions** (registered in the AST system) and three are **pipeline processors** (registered in the DataHandler system).

---

## 1. `cs_rank` — Cross-Sectional Percentile Rank

### Purpose

Ranks all assets from 0.0 (lowest) to 1.0 (highest) at each time step. Used for relative comparisons across the asset universe (e.g., "this stock is in the 80th percentile by market cap").

### Python Signature

```python
# src/data/ast/functions.py
@register_function(category="cross_sectional")
def cs_rank(data: xr.DataArray) -> xr.DataArray
```

### YAML Usage

In a formula expression:

```yaml
features:
  operations:
    - name: "ranks"
      formulas:
        me_rank:
          expression: "cs_rank($me)"
```

Or directly via `FormulaManager`:

```python
result, ds = formula_manager.evaluate("cs_rank", dataset, overrides={"data": "me"})
```

### NaN Behavior

- **NaN inputs produce NaN outputs.** NaN assets are excluded from rank computation; they receive NaN rank, not zero.
- If fewer than 2 valid assets exist at a time step, all ranks are NaN.
- No silent zeroing — NaN is preserved throughout.

### Known Edge Cases

- Ties are broken by `np.argsort(np.argsort(...))`, which gives a natural ordering but is not stable across identical values.
- When passed a `Dataset` instead of `DataArray`, dispatches via `_dispatch_cross_sectional` and applies to each numeric variable with an `asset` dimension.
- Not JAX-compatible — uses `numpy` argsort internally via `xr.apply_ufunc`.

---

## 2. `cs_quantile` — Cross-Sectional Quantile

### Purpose

Computes the q-th quantile across the asset dimension at each time step. Returns a **scalar per time step** (no `asset` dimension). Typically used to compute breakpoints for portfolio sorting.

### Python Signature

```python
# src/data/ast/functions.py
@register_function(category="cross_sectional")
def cs_quantile(data: xr.DataArray, q: float) -> xr.DataArray
```

### YAML Usage

```yaml
features:
  operations:
    - name: "breakpoints"
      formulas:
        me_median:
          expression: "cs_quantile($me, 0.5)"
        bm_p30:
          expression: "cs_quantile($bm, 0.3)"
        bm_p70:
          expression: "cs_quantile($bm, 0.7)"
```

### NaN Behavior

- Uses `skipna=True` — NaN values are ignored when computing the quantile.
- All-NaN time steps return NaN.
- The result has no `asset` dimension — xarray broadcasting handles expansion when used with `assign_bucket` or `where`.

### Known Edge Cases

- `q` must be a scalar float in [0, 1]. Passing a DataArray or array raises `TypeError`.
- Only works on `DataArray`, not `Dataset`. Passing a Dataset raises `TypeError`.
- The `quantile` coordinate is dropped from the result to avoid downstream alignment issues.

---

## 3. `assign_bucket` — Breakpoint-Based Bin Assignment

### Purpose

Assigns each asset to an integer bin based on pre-computed breakpoint thresholds. Supports 1–3 breakpoints (2–4 bins). The primary tool for creating portfolio group labels from quantile breakpoints.

### Python Signature

```python
# src/data/ast/functions.py
@register_function(category="cross_sectional")
def assign_bucket(
    data: xr.DataArray,
    bp1,            # First breakpoint (scalar or DataArray without 'asset' dim)
    bp2=None,       # Optional second breakpoint
    bp3=None,       # Optional third breakpoint
) -> xr.DataArray
```

### YAML Usage

Two-bin sort (e.g., size median split):

```yaml
assign_bucket($me, cs_quantile($me_nyse, 0.5))
# Result: 0 for below-median, 1 for above-median
```

Three-bin sort (e.g., value terciles):

```yaml
assign_bucket($bm, cs_quantile($bm_nyse, 0.3), cs_quantile($bm_nyse, 0.7))
# Result: 0 for bottom 30%, 1 for middle 40%, 2 for top 30%
```

### NaN Behavior

- **NaN data → NaN bucket.** No silent zeroing.
- Non-NaN data is compared to each breakpoint via `data > bp`. The result is the count of breakpoints exceeded.
- Breakpoints from `cs_quantile` have no `asset` dim; xarray broadcasts automatically.

### Known Edge Cases

- Maximum of 3 breakpoints (4 bins). For more bins, use `CrossSectionalSort` processor instead.
- Breakpoints must be ordered ascending for correct bin assignment.
- When all values equal the breakpoint, `data > bp` is False, so they fall in the lower bin (no `>=` comparison).

---

## 4. `CrossSectionalSort` — Pipeline Processor for Portfolio Sorting

### Purpose

Sorts assets into portfolio bins at each time step based on a signal variable. Supports scoped breakpoints (e.g., calculate breakpoints on NYSE stocks only, apply to all stocks). More flexible than `assign_bucket` for pipeline use.

### Python Signature

```python
# src/pipeline/data_handler/processors.py
@dataclass
class CrossSectionalSort(Processor):
    signal: str = ""                        # Signal variable to sort on
    n_bins: int = 0                         # Number of equal quantile bins
    name: str = "sort"                      # Processor name
    scope: Optional[str] = None             # Boolean mask for breakpoint calculation
    labels: Optional[List[Any]] = None      # Custom bin labels
    quantiles: Optional[List[float]] = None # Custom interior breakpoints (overrides n_bins)
```

### YAML Usage

```yaml
preprocessing:
  infer:
    - type: "sort"
      signal: "me_june"
      n_bins: 2
      scope: "is_nyse"         # breakpoints from NYSE stocks only
      name: "size_sort"

    - type: "sort"
      signal: "beme_june"
      n_bins: 3
      quantiles: [0.3, 0.7]   # custom breakpoints: 30th and 70th percentile
      scope: "is_nyse"
      name: "bm_sort"
```

**Output:** Creates `{signal}_port` variable (e.g., `me_june_port`) with integer bin labels 0 to n_bins-1.

### NaN Behavior

- Assets with NaN signal receive NaN port assignment.
- If `scope` is provided, NaN values in the scope mask are treated as False.
- Breakpoints are computed with `np.nanquantile` (NaN-safe).
- If all scoped values are NaN, all ports are NaN for that time step.

### Known Edge Cases

- **`quantiles` overrides `n_bins`**: When `quantiles=[0.3, 0.7]` is provided, `n_bins` is ignored and 3 bins are created.
- **Breakpoint edge extension**: Lower bound extended by `-1e-9`, upper bound by `+1e-9` to ensure all values fall within bins.
- Uses `np.digitize` internally — bins include the left edge, exclude the right (except last bin).
- The processor modifies the input dataset in place (mutates `ds`). This is by design for pipeline efficiency.

---

## 5. `PortfolioReturns` — Portfolio Return Aggregation

### Purpose

Computes value-weighted or equal-weighted returns for portfolios defined by one or more grouping variables. Aggregates individual asset returns within each portfolio group at each time step.

### Python Signature

```python
# src/pipeline/data_handler/processors.py
@dataclass
class PortfolioReturns(Processor):
    groupby: List[str] = field(default_factory=list)  # Grouping variables (port labels)
    returns_var: str = "ret"                           # Returns variable name
    weights_var: Optional[str] = None                  # Weights variable (None = equal weight)
    name: str = "port_ret"                             # Processor name
```

### YAML Usage

```yaml
preprocessing:
  infer:
    - type: "port_ret"
      groupby: ["me_june_port", "beme_june_port"]
      returns_var: "ret"
      weights_var: "me_lag1"      # value-weighted by lagged market equity
      name: "ff_port_ret"
```

**Output:** Creates `port_ret_{joined_groupby}` variable (e.g., `port_ret_me_june_port_beme_june_port`) with dimensions renamed to `{groupby}_dim` (e.g., `me_june_port_dim`, `beme_june_port_dim`). The `asset` dimension is collapsed.

### NaN Behavior

- Returns and weights must both be non-NaN for an asset to contribute to a portfolio.
- `valid_mask = ~np.isnan(ret) & ~np.isnan(weights)` filters both.
- If denominator (sum of weights) is zero for a group at a time step, the portfolio return is NaN.
- Group variables with NaN are excluded via the validity mask.

### Known Edge Cases

- **Single groupby**: output has one portfolio dimension `{groupby[0]}_dim`.
- **Multiple groupby**: creates composite integer IDs via linear indexing. Assumes integer bins with small cardinality (< 1000 per variable). Unstacks to multi-dimensional output using `pd.MultiIndex.from_product`.
- **Group variable cardinality**: inferred from `max().item()`. If group variables have gaps (e.g., bins 0 and 2 but no 1), the missing group gets NaN returns.
- The `FutureWarning` about `pd.MultiIndex` in xarray coordinate assignment occurs in multi-groupby mode.

---

## 6. `FactorSpread` — Long-Short Factor Construction

### Purpose

Computes long-short factor return spreads from a portfolio return array. Selects long and short legs using dimension coordinate selectors, optionally averages over a cross-dimension, then returns long minus short.

### Python Signature

```python
# src/pipeline/data_handler/processors.py
@dataclass
class FactorSpread(Processor):
    source: str = ""                               # Source portfolio returns variable
    factors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    name: str = "factor_spread"                    # Processor name
```

Each factor config has:

```python
{
    "long": {dim_name: coordinate_value},    # .sel() kwargs for long leg
    "short": {dim_name: coordinate_value},   # .sel() kwargs for short leg
    "average_over": "dim_name"               # optional: average over this dim before subtraction
}
```

### YAML Usage

```yaml
preprocessing:
  infer:
    - type: "factor_spread"
      source: "port_ret_me_june_port_beme_june_port"
      factors:
        SMB:
          long: {me_june_port_dim: 0}       # small stocks
          short: {me_june_port_dim: 1}      # big stocks
          average_over: "beme_june_port_dim" # average across B/M groups
        HML:
          long: {beme_june_port_dim: 2}     # high B/M
          short: {beme_june_port_dim: 0}    # low B/M
          average_over: "me_june_port_dim"  # average across size groups
      name: "ff_factors"
```

**Output:** One variable per factor (e.g., `ds['SMB']`, `ds['HML']`) with only time dimensions remaining — no asset or portfolio dimensions.

### NaN Behavior

- Uses `.sel()` on the portfolio return array. If the selected coordinate doesn't exist, raises `KeyError`.
- `mean(dim=..., skipna=True)` when averaging over the cross-dimension — NaN values are skipped.
- If both long and short legs are NaN at a time step, the spread is NaN.

### Known Edge Cases

- **Source variable must already exist**: produced by a prior `PortfolioReturns` processor. If missing, raises `KeyError` with a descriptive message listing available variables.
- **Dimension names must use `_dim` suffix**: consistent with `PortfolioReturns` output naming convention (e.g., `me_june_port_dim`, not `me_june_port`).
- `average_over` can be a string or list of strings. If omitted, no averaging is performed before subtraction.
- The result has the same time dimensions as the source but no portfolio or asset dimensions.

---

## Typical Workflow: Factor Construction Pipeline

The six primitives chain together in a standard factor construction workflow:

```
1. Formula functions compute signals:
   cs_rank($me) → me_rank
   cs_quantile($me_nyse, 0.5) → me_median
   assign_bucket($me, me_median) → me_bucket

2. CrossSectionalSort creates portfolio labels:
   sort(signal="me_june", n_bins=2, scope="is_nyse") → me_june_port

3. PortfolioReturns aggregates:
   port_ret(groupby=["me_port","bm_port"], returns_var="ret") → port_ret_me_port_bm_port

4. FactorSpread computes long-short:
   factor_spread(source="port_ret_...", factors={SMB: ...}) → SMB, HML
```

The `examples/ff3_model.yaml` file demonstrates this complete workflow for constructing three-factor model returns.

---

## See also

- [FORMULA_SYSTEM.md](./FORMULA_SYSTEM.md) — Full list of registered functions, including `cs_rank`, `cs_quantile`, `assign_bucket`
- [PIPELINE_SYSTEM.md](./PIPELINE_SYSTEM.md) — Processor stage model and how `CrossSectionalSort`, `PortfolioReturns`, `FactorSpread` are configured in YAML
