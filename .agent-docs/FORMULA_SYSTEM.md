# Formula System

The formula system provides YAML-defined financial computations that are parsed into an AST, resolved for dependencies, and evaluated against xarray Datasets. Functions are registered globally via decorator and invoked within formula expressions.

---

## YAML Formula Definition Schema

Formula definitions live in `src/data/ast/definitions/*.yaml`. Each file follows `schema.yaml`:

```yaml
formula_name:
  description: "Human-readable description"
  expression: "sma($close, 20)"        # the formula expression
  return_type: "float"                  # expected result dtype
  variables:                            # maps $-prefixed names to dataset variables
    close: "close"
  functions:                            # lists function names used
    - sma
  modules:                              # optional extra modules (e.g. generators)
    - src.data.generators.weights
```

**Expression syntax:**

- `$var_name` references a dataset variable (resolved via `variables` mapping)
- `func_name(arg1, arg2, ...)` calls a registered function
- Standard arithmetic: `+`, `-`, `*`, `/`, `**`
- Parentheses for grouping
- Numeric literals: `1`, `3.14`, `-2.5`

**Example from `technical.yaml`:**

```yaml
sma:
  description: "Simple Moving Average"
  expression: "sma($price, $window)"
  return_type: "float"
  variables:
    price: "close"
  functions:
    - sma
```

---

## Registered Functions by Category

All functions are registered via `@register_function(category=...)` in `src/data/ast/functions.py`. They are available in formula expressions as `func_name(args)`.

### Arithmetic

| Function | Signature | Description |
|----------|-----------|-------------|
| `sum` | `sum(data, dim=None)` | Sum along dimension |
| `sqrt` | `sqrt(x)` | Square root |
| `abs` | `abs(x)` | Absolute value |
| `log` | `log(x)` | Natural logarithm |
| `exp` | `exp(x)` | Exponential (e^x) |

### Statistical

| Function | Signature | Description |
|----------|-----------|-------------|
| `std` | `std(data, dim=None)` | Standard deviation |
| `var` | `var(data, dim=None)` | Variance |
| `min` | `min(data, dim=None)` | Minimum |
| `max` | `max(data, dim=None)` | Maximum |
| `mean` | `mean(data, dim=None)` | Arithmetic mean |

### Temporal (Rolling)

| Function | Signature | Description |
|----------|-----------|-------------|
| `sma` | `sma(data, window)` | Simple moving average |
| `ema` | `ema(data, window)` | Exponential moving average |
| `rma` | `rma(data, window)` | Relative moving average (Wilder's) |
| `wma` | `wma(data, window, weights=None)` | Weighted moving average |
| `gain` | `gain(data, window)` | Rolling gain (positive changes) |
| `loss` | `loss(data, window)` | Rolling loss (negative changes) |
| `moving_median` | `moving_median(data, window)` | Rolling median |
| `moving_mode` | `moving_mode(data, window)` | Rolling mode |
| `shift` | `shift(data, periods=1)` | Business-day-aware temporal shift |
| `returns` | `returns(price_data, periods=1)` | Simple returns (price_t / price_{t-n} - 1) |
| `rolling_sum` | `rolling_sum(data, window)` | Rolling sum |
| `triple_exponential_smoothing` | `triple_exponential_smoothing(data, alpha=0.2, beta=0.1, gamma=0.1)` | Holt-Winters triple smoothing |
| `adaptive_ema` | `adaptive_ema(data, smoothing_factors)` | Adaptive EMA with time-varying smoothing |

### Cross-Sectional

| Function | Signature | Description |
|----------|-----------|-------------|
| `cs_rank` | `cs_rank(data)` | Percentile rank across assets [0, 1] |
| `cs_quantile` | `cs_quantile(data, q)` | q-th quantile across assets (returns scalar per time step) |
| `cs_demean` | `cs_demean(data)` | Subtract cross-sectional mean |
| `assign_bucket` | `assign_bucket(data, bp1, bp2=None, bp3=None)` | Assign assets to bins by breakpoints |
| `month_coord` | `month_coord(data)` | Broadcast month coordinate to data shape |

### Conditional

| Function | Signature | Description |
|----------|-----------|-------------|
| `coalesce` | `coalesce(a, b)` | First non-NaN value |
| `gt` | `gt(a, b)` | a > b |
| `lt` | `lt(a, b)` | a < b |
| `ge` | `ge(a, b)` | a >= b |
| `le` | `le(a, b)` | a <= b |
| `eq` | `eq(a, b)` | a == b |
| `nan_const` | `nan_const()` | Returns NaN scalar |
| `where` | `where(condition, true_val, false_val)` | Conditional selection |

---

## FormulaManager — Dependency Resolution and Evaluation

```python
# src/data/ast/manager.py
class FormulaManager:
    def __init__(self, definitions_dir=None)
    def load_directory(self, directory) -> None
    def load_file(self, file_path) -> None
    def add_formula(self, name, definition) -> None
    def get_formula(self, name) -> Dict[str, Any]
    def evaluate(self, name, dataset, overrides=None, ...) -> Tuple[result, xr.Dataset]
    def evaluate_bulk(self, names, dataset, ...) -> xr.Dataset
    def evaluate_all_loaded(self, dataset, ...) -> xr.Dataset
    def get_formula_dependencies(self, name) -> Set[str]
    def get_dependency_chain(self, name) -> List[str]
    def compile_all_formulas_as_functions(self) -> None
```

**Dependency resolution process:**

1. `FormulaManager.load_directory()` loads all YAML files from `src/data/ast/definitions/`
2. Each formula's `expression` is parsed to extract `$variable` references
3. If a variable maps to another formula name, that formula is added as a dependency
4. `get_dependency_chain(name)` performs topological sort to determine evaluation order
5. `evaluate()` walks the chain, evaluating each formula in dependency order
6. Results are attached to the output dataset progressively

**Evaluation flow:**

```
FormulaManager.evaluate("rsi", dataset)
  → get_dependency_chain("rsi")  → ["gain", "loss", "rsi"]
  → evaluate "gain":  gain($close, 14)  → result attached to dataset
  → evaluate "loss":  loss($close, 14)  → result attached to dataset
  → evaluate "rsi":   100 - (100 / (1 + $gain / $loss))
  → return (result_dataarray, updated_dataset)
```

---

## Grammar Limitations

| Limitation | Detail |
|------------|--------|
| **No comparison operators in grammar** | `>`, `<`, `==` are not in the CFG. Use registered functions `gt()`, `lt()`, `eq()` instead. |
| **No boolean logic** | No `and`, `or`, `not` operators. Chain conditionals via `where()`. |
| **No multi-output formulas** | Each formula produces a single DataArray. Bollinger Bands (3 outputs) requires 3 separate formulas. |
| **No inline expressions in YAML** | `evaluate_bulk` cannot handle inline `expression:` keys from pipeline YAML. Formulas must be pre-defined or loaded via `add_formula()`. |
| **Single-variable functions** | Temporal functions (sma, ema, etc.) operate on one variable at a time. No multivariate rolling operations. |
| **No string operations** | Formula system operates on numeric data only. |

---

## Formula YAML Definition Files

| File | Category | Formulas defined |
|------|----------|-----------------|
| `technical.yaml` | Technical indicators | sma, ema, rma, wma, rsi, macd, bollinger, atr, etc. |
| `marketchars.yaml` | Market characteristics | log_return, volatility, turnover, etc. |
| `composite.yaml` | Composite indicators | Built from other formulas |
| `cross_sectional.yaml` | Cross-sectional | cs_rank, cs_zscore, relative_strength |
| `ts_dependence.yaml` | Time-series dependence | Autocorrelation, partial autocorrelation |
| `schema.yaml` | Schema definition | Documents the YAML structure |

---

## How to Add a New Formula Function

1. **Add to `src/data/ast/functions.py`:**

```python
@register_function(category="temporal", description="My custom indicator")
def my_indicator(data, window, param=1.0):
    """Docstring with Args/Returns."""
    return _dispatch_rolling(some_kernel, data, window=window, func_name='my_indicator', param=param)
```

2. **The function is immediately available** in formula expressions as `my_indicator($close, 20, 1.5)`.

3. **Optionally add a YAML formula definition** in `src/data/ast/definitions/`:

```yaml
my_indicator:
  description: "My custom indicator"
  expression: "my_indicator($price, $window)"
  return_type: "float"
  variables:
    price: "close"
  functions:
    - my_indicator
```

---

## See also

- [DATA_LAYER.md](./DATA_LAYER.md) — `.dt` accessor, rolling operations that formulas dispatch to
- [PIPELINE_SYSTEM.md](./PIPELINE_SYSTEM.md) — `FormulaEval` processor that runs formulas in the pipeline
- [QUANT_PRIMITIVES.md](./QUANT_PRIMITIVES.md) — Cross-sectional functions (cs_rank, cs_quantile, assign_bucket)
