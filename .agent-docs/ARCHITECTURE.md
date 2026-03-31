# Architecture

Hindsight is a financial backtesting and research library built on **xarray** for panel data and **JAX** for accelerated rolling computations. All datasets share a fixed multi-dimensional time layout. The pipeline system provides a declarative YAML-driven workflow from raw data to model predictions.

---

## Canonical Data-Flow Diagram

```
Raw Data (CSV / SAS / API)
       │
       ▼
  from_table()          ← src/data/loaders/table.py
       │                   Converts pd.DataFrame → xr.Dataset
       ▼                   with dims (year, month, day, hour, asset)
  xr.Dataset
  + .dt accessor        ← src/data/core/struct.py
       │                   sel(), shift(), rolling(), compute_mask()
       ▼
  DataHandler           ← src/pipeline/data_handler/handler.py
       │                   Orchestrates shared → learn → infer processors
       │                   (CSZScore, PerAssetFFill, FormulaEval,
       │                    CrossSectionalSort, PortfolioReturns, FactorSpread)
       ▼
  WalkForwardRunner     ← src/pipeline/walk_forward/execution.py
       │                   Splits data into temporal segments
       │                   Fits learn processors per segment
       ▼
  ModelRunner           ← src/pipeline/model/runner.py
       │                   Wraps ModelAdapter (SklearnAdapter)
       │                   Trains + predicts per walk-forward segment
       ▼
  Results
  (xr.Dataset with predictions, factor spreads, portfolio returns)
```

---

## Dimension Contract

Every dataset produced by `from_table()` and consumed by the pipeline adheres to this dimension layout:

| Dimension | dtype   | Meaning                            | Present when              |
|-----------|---------|------------------------------------|---------------------------|
| `year`    | `int64` | Calendar year                      | Always                    |
| `month`   | `int64` | Calendar month (1–12)              | Always                    |
| `day`     | `int64` | Calendar day (1–31)                | Always                    |
| `hour`    | `int64` | Hour of day (0–23)                 | Always (0 for daily data) |
| `asset`   | `int64` | Asset identifier (permno, ticker…) | Always                    |

**Key invariants:**

- The `hour` dimension always exists. For daily or lower frequencies, it is fixed to `0`.
- `from_table()` sets `day=1` for monthly data and `month=1, day=1` for yearly data.
- A `time` coordinate of dtype `datetime64[ns]` is attached as a non-dimension coordinate computed from `(year, month, day, hour)`.
- The `.dt` accessor (registered on both `xr.Dataset` and `xr.DataArray`) provides `sel()`, `shift()`, `rolling()`, `compute_mask()`, and `to_time_indexed()` to bridge between multi-dim and flat time representations.

---

## Layer Boundary Table

| Layer            | Package path                          | Responsibility                                                                 | Key entry points                                       |
|------------------|---------------------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------|
| **Data loading** | `src/data/loaders/`                   | Load raw data from CSV/SAS/API, convert to xr.Dataset via `from_table()`       | `from_table()`, `DataManager`, `CacheManager`          |
| **AST / Formulas** | `src/data/ast/`                     | Parse and evaluate YAML-defined formula expressions using a registered function registry | `FormulaManager`, `FormulaParser`, `register_function` |
| **Core ops**     | `src/data/core/`                      | Multi-dim time accessor, JAX-backed rolling ops, cache, provider registry       | `.dt` accessor, `TimeSeriesOps.u_roll`, `Rolling`      |
| **Processors**   | `src/pipeline/data_handler/`          | Stateful/stateless data transforms (shared/learn/infer stages)                  | `DataHandler`, `CSZScore`, `PerAssetFFill`, `FormulaEval`, `CrossSectionalSort`, `PortfolioReturns`, `FactorSpread` |
| **Walk-forward** | `src/pipeline/walk_forward/`          | Temporal segmentation and segment-by-segment execution                          | `SegmentConfig`, `SegmentPlan`, `WalkForwardRunner`    |
| **Model**        | `src/pipeline/model/`                 | Model adapter interface, walk-forward model training                            | `ModelAdapter`, `SklearnAdapter`, `ModelRunner`        |
| **Pipeline spec**| `src/pipeline/spec/`                  | Declarative YAML pipeline specification and caching                             | `PipelineSpec`, `PipelineExecutor`, `GlobalCacheManager`|
| **Backtester**   | `src/backtester/`                     | Event-based strategy backtesting engine (partially integrated)                  | `EventBasedStrategy`, `BacktestState`                  |

---

## Global Configuration

- **JAX float64**: `os.environ["JAX_ENABLE_X64"] = "1"` is set in `src/__init__.py` at import time. All JAX operations use `float64`.
- **Cache directories**: `initialize_cache_directories()` runs on import of `src.data`, creating filesystem dirs under `~/data/cache/` for all registered providers.
- **Provider self-registration**: Importing a loader package (e.g. `src.data.loaders.wrds`) triggers `register_provider()` which populates the global `_PROVIDER_REGISTRY`. `DataManager` reads from this registry at init.

---

## Module Hierarchy

```
src/
├── __init__.py              # Public API; sets JAX float64
├── data/
│   ├── core/
│   │   ├── types.py         # FrequencyType enum, TimeSeriesIndex
│   │   ├── struct.py        # .dt xarray accessor
│   │   ├── rolling.py       # Rolling window adapter (JAX u_roll)
│   │   ├── operations/      # TimeSeriesOps: shift(), u_roll(), kernels
│   │   ├── cache.py         # Two-level NetCDF cache (L1 raw, L2 post)
│   │   └── provider.py      # Provider registry
│   ├── loaders/
│   │   ├── table.py         # from_table() — the core converter
│   │   ├── abstracts/       # BaseDataSource ABC
│   │   ├── wrds/            # CRSP, Compustat SAS loaders
│   │   ├── crypto/          # Local CSV crypto loader
│   │   └── open_bb/         # OpenBB generic loader
│   ├── managers/
│   │   ├── data_manager.py  # DataManager — user-facing data loading
│   │   └── config_schema.py # DEPRECATED config schema
│   ├── ast/
│   │   ├── grammar.py       # Context-free grammar for formulas
│   │   ├── parser.py        # Formula parser + evaluator
│   │   ├── functions.py     # Registered function registry
│   │   ├── manager.py       # FormulaManager (YAML load, dep resolve)
│   │   ├── nodes.py         # AST node types
│   │   └── definitions/     # YAML formula definitions
│   ├── processors/          # Data-level post-processors (set_permno, etc.)
│   ├── filters/             # Django-style filter DSL
│   ├── generators/          # Weight generators (linear, exp, ALMA, etc.)
│   └── configs/             # Built-in named data configurations
├── pipeline/
│   ├── data_handler/
│   │   ├── core.py          # View, PipelineMode, ProcessorContract
│   │   ├── handler.py       # DataHandler orchestrator
│   │   ├── processors.py    # CSZScore, PerAssetFFill, FormulaEval, CrossSectionalSort, PortfolioReturns, FactorSpread
│   │   ├── config.py        # HandlerConfig
│   │   └── merge.py         # DatasetMerger (multi-frequency joins)
│   ├── walk_forward/
│   │   ├── segments.py      # Segment, SegmentPlan, SegmentConfig
│   │   ├── planning.py      # make_plan(), expand_plan_coverage()
│   │   └── execution.py     # WalkForwardRunner
│   ├── model/
│   │   ├── adapter.py       # ModelAdapter ABC, SklearnAdapter
│   │   └── runner.py        # ModelRunner
│   ├── spec/
│   │   ├── schema.py        # PipelineSpec, DataSourceSpec, ModelSpec, etc.
│   │   ├── parser.py        # YAML → PipelineSpec
│   │   ├── executor.py      # PipelineExecutor
│   │   └── processor_registry.py
│   └── cache/
│       ├── manager.py       # GlobalCacheManager (L2–L5)
│       ├── metadata.py      # Cache metadata
│       └── stages.py        # CacheStage enum
└── backtester/
    ├── core.py              # EventBasedStrategy ABC
    ├── struct.py            # Order types, BacktestState
    └── metrics/             # Performance metrics
```

---

## See also

- [DATA_LAYER.md](./DATA_LAYER.md) — `from_table()`, `DataManager`, `CacheManager`, `.dt` accessor details
- [PIPELINE_SYSTEM.md](./PIPELINE_SYSTEM.md) — Processor stage model, PipelineSpec schema, registered processors
- [WALK_FORWARD.md](./WALK_FORWARD.md) — SegmentConfig, WalkForwardRunner vs ModelRunner, leakage contract
