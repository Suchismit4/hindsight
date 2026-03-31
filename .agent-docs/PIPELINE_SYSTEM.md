# Pipeline System

The pipeline system provides a declarative YAML-driven workflow for data processing and model training. It wraps the existing `DataHandler`, `ModelAdapter`, and `FormulaManager` infrastructure with caching, orchestration, and a three-stage processor model (shared/learn/infer).

---

## Processor Stage Model

Every pipeline has three processor stages, managed by `DataHandler`:

```
                  ┌──────────────┐
                  │   Raw Data   │
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │    Shared    │   Transform-only, runs once on full data
                  │  processors  │   (e.g., forward fill, formula eval)
                  └──────┬───────┘
                         │
           ┌─────────────┴─────────────┐
           │ INDEPENDENT mode          │
    ┌──────▼──────┐             ┌──────▼──────┐
    │    Learn    │             │    Infer    │
    │ fit + trans │             │ transform   │
    └─────────────┘             └─────────────┘
      LEARN View                  INFER View
```

| Stage | Execution | State | Typical use |
|-------|-----------|-------|-------------|
| **shared** | Transform-only on full data | Stateless | Forward fill, feature engineering, formula evaluation |
| **learn** | `fit()` on train slice → `transform()` on both train and infer | Stateful (mean, std, etc.) | Cross-sectional z-score normalization |
| **infer** | Transform-only on infer slice | Stateless | Portfolio sorting, returns aggregation, factor spreads |

**Pipeline modes:**

| Mode | Flow | Use case |
|------|------|----------|
| `INDEPENDENT` | shared → learn (branch), shared → infer (branch) | Standard ML preprocessing |
| `APPEND` | shared → infer → learn (sequential) | When learn needs to see infer outputs |

---

## PipelineSpec YAML Schema

The complete YAML specification structure (from `src/pipeline/spec/schema.py`):

```yaml
spec_version: "1.0"                    # Schema version
name: "pipeline_name"                  # Unique pipeline identifier
version: "1.0"                         # Pipeline version string

time_range:
  start: "YYYY-MM-DD"                 # Start date (inclusive)
  end: "YYYY-MM-DD"                   # End date (inclusive)

data:                                  # DataSourceSpec (one or more)
  source_name:
    provider: "wrds|crypto|open_bb"    # Data provider name
    dataset: "equity/crsp"             # Dataset within provider
    frequency: "H|D|M|Y"              # Data frequency
    columns: ["col1", "col2"]          # Optional column subset to load
    filters:                           # Django-style filters at DataFrame level
      shrcd__in: [10, 11]
    external_tables:                   # Side-table merge configs
      - path: "/path/to/table.sas7bdat"
        type: "replace|lookup"
        on: "permno"
        time_column: "dlstdt"
        from_column: "dlret"
        to_column: "ret"
        columns: ["col1"]
    processors:                        # Source-level post-processing
      transforms:
        - type: "set_coordinates"
          coord_type: "permco"
        - type: "fix_market_equity"
        - type: "preferred_stock"

merge_base: "crsp"                     # Optional: base source for ordered merges
merges:                                # Optional: ordered merge configuration
  - right_name: "compustat"
    on: "asset"
    time_alignment: "exact|ffill|bfill|nearest|as_of"
    time_offset_months: 6
    ffill_limit: null
    prefix: "comp_"
    suffix: ""
    variables: ["seq", "txditc"]
    drop_vars: []

features:                              # FeaturesSpec
  operations:
    - name: "operation_name"           # Unique operation name
      depends_on: ["prior_op"]         # Dependency for topological ordering
      formulas:
        formula_name:
          - {window: 20}              # Config overrides (list of dicts)
          - {window: 50}
        inline_formula:
          expression: "$seq + $txditc" # Inline expression (requires add_formula)

preprocessing:                         # PreprocessingSpec
  mode: "independent|append"
  shared:
    - type: "processor_type"
      name: "processor_name"
      # ... processor-specific params
  learn:
    - type: "processor_type"
      name: "processor_name"
      # ... processor-specific params
  infer:
    - type: "processor_type"
      name: "processor_name"
      # ... processor-specific params

model:                                 # ModelSpec
  adapter: "sklearn"                   # Adapter type
  type: "RandomForestRegressor"        # Model class name
  params:                              # Constructor kwargs
    n_estimators: 50
    max_depth: 14
  features: ["feature1", "feature2"]   # Input variable names
  target: "close"                      # Label variable name
  adapter_params:
    output_var: "close_pred"           # Prediction variable name
    use_proba: false                   # Use predict_proba vs predict
  walk_forward:                        # SegmentConfig parameters
    train_span_hours: 720
    infer_span_hours: 24
    step_hours: 24
    gap_hours: 0
    start: "YYYY-MM-DDTHH:MM:SS"      # Optional, defaults to data bounds
    end: "YYYY-MM-DDTHH:MM:SS"        # Optional
  runner_params:
    overlap_policy: "last|first"       # How overlapping segments are resolved

metadata:                              # Optional documentation
  description: "Pipeline description"
  author: "Author"
  tags: ["tag1", "tag2"]
```

---

## Registered Processors

All processors inherit from `ProcessorContract` in `src/pipeline/data_handler/core.py` and are implemented in `src/pipeline/data_handler/processors.py`.

| YAML `type` | Class | Stage | Stateful | Purpose |
|-------------|-------|-------|----------|---------|
| `cs_zscore` | `CSZScore` | learn | Yes | Cross-sectional z-score normalization |
| `per_asset_ffill` | `PerAssetFFill` | shared | No | Forward-fill per asset along time |
| `formula_eval` | `FormulaEval` | shared | No | Evaluate AST formulas on dataset |
| `sort` | `CrossSectionalSort` | infer | No | Sort assets into portfolio bins by signal |
| `port_ret` | `PortfolioReturns` | infer | No | Compute value-weighted or equal-weighted portfolio returns |
| `factor_spread` | `FactorSpread` | infer | No | Compute long-short factor spreads |

### CSZScore

```python
CSZScore(name="normalizer", vars=["close", "volume"], out_suffix="_norm", eps=1e-8)
```

- **fit**: Computes per-asset mean and std across all time dimensions
- **transform**: Creates `{var}{out_suffix}` = (var - mean) / (std + eps)
- **State**: `CSZScoreState(variables, means, stds)` — lightweight dataclass

### PerAssetFFill

```python
PerAssetFFill(name="ffill", vars=["close", "volume"])
```

- Stateless. Stacks time dims, forward-fills, unstacks.
- If `vars=None`, applies to all numeric variables.

### FormulaEval

```python
FormulaEval(
    name="features",
    formula_configs={"sma": [{"window": 20}], "rsi": [{"window": 14}]},
    use_jit=True,
    defs_dir=None,
    assign_in_place=True,
)
```

- Wraps `FormulaManager.evaluate_bulk()`. Compiles formulas once, caches the callable.
- Optionally applies `jax.jit` for performance.

### CrossSectionalSort

```python
CrossSectionalSort(signal="me_june", n_bins=2, scope="is_nyse", quantiles=[0.3, 0.7])
```

- Sorts assets into bins at each time step using `np.digitize`.
- `scope`: boolean mask variable to restrict breakpoint calculation (e.g., NYSE-only).
- `quantiles`: custom interior breakpoints; if omitted, uses `n_bins` equal quantiles.
- Output: `{signal}_port` variable with integer bin labels.

### PortfolioReturns

```python
PortfolioReturns(groupby=["me_port", "bm_port"], returns_var="ret", weights_var="me_lag1")
```

- Computes weighted returns per portfolio group.
- Single groupby: output dim is `{groupby[0]}_dim`.
- Multiple groupby: uses composite integer IDs, unstacks to multi-dimensional output.
- Output: `port_ret_{joined_groupby}` variable.

### FactorSpread

```python
FactorSpread(
    source="port_ret_me_port_bm_port",
    factors={
        "SMB": {"long": {"me_port_dim": 0}, "short": {"me_port_dim": 1}, "average_over": "bm_port_dim"},
        "HML": {"long": {"bm_port_dim": 2}, "short": {"bm_port_dim": 0}, "average_over": "me_port_dim"},
    },
)
```

- Selects long/short legs from portfolio return array using `.sel()`.
- Optionally averages over a cross-dimension.
- Output: one variable per factor (e.g., `SMB`, `HML`) with only time dimensions.

---

## How to Add a New Processor

1. **Create a processor class** in `src/pipeline/data_handler/processors.py`:

```python
@dataclass
class MyProcessor(Processor):
    name: str = "my_proc"
    my_param: float = 1.0

    def fit(self, ds: xr.Dataset) -> ProcessorState:
        # Learn parameters from data (or return None if stateless)
        state = {"learned_value": ds["var"].mean().item()}
        return state

    def transform(self, ds: xr.Dataset, state=None) -> xr.Dataset:
        # Apply transformation
        out = ds.copy()
        if state:
            out["var_processed"] = ds["var"] - state["learned_value"]
        return out
```

2. **Register in `ProcessorRegistry`** (`src/pipeline/spec/processor_registry.py`):

```python
ProcessorRegistry._registry['my_proc'] = MyProcessor
```

3. **Use in YAML:**

```yaml
preprocessing:
  learn:
    - type: "my_proc"
      name: "my_instance"
      my_param: 2.0
```

---

## Pipeline Caching (L2–L5)

The `GlobalCacheManager` provides content-addressable caching at each pipeline stage:

| Stage | Key | What's cached | Storage |
|-------|-----|---------------|---------|
| L2 | hash(data_config) | Raw xr.Dataset | `.nc` + `.attrs.pkl` + `.meta.json` |
| L3 | hash(formula_config, L2_key) | Feature Dataset | `.nc` + `.attrs.pkl` + `.meta.json` |
| L4 | hash(preprocessing_config, L3_key) | (Dataset, DataHandler) | `.pkl` + `.meta.json` |
| L5 | hash(model_config + walk_forward, L4_key) | (Predictions, ModelRunnerResult) | `.pkl` + `.meta.json` |

Cache keys are 16-character hex SHA-256 hashes. Parent keys propagate — changing L2 data invalidates all downstream caches.

The `.attrs.pkl` companion file preserves attributes that NetCDF cannot serialize (e.g., `TimeSeriesIndex`).

---

## See also

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Overall system data-flow, dimension contract, layer map
- [WALK_FORWARD.md](./WALK_FORWARD.md) — SegmentConfig, WalkForwardRunner, ModelRunner, leakage contract
- [FORMULA_SYSTEM.md](./FORMULA_SYSTEM.md) — Formula definitions and registered functions used by FormulaEval
- [QUANT_PRIMITIVES.md](./QUANT_PRIMITIVES.md) — CrossSectionalSort, PortfolioReturns, FactorSpread in detail
