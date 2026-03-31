# Architecture

How the Hindsight system works — components, relationships, data flows, invariants.

---

## System Overview

Hindsight is a JAX-powered, xarray-backed quantitative finance research library. Its primary data structure is an xarray Dataset with multi-dimensional time coordinates `(year, month, day[, hour], asset)` instead of a flat datetime index.

## Layer Map

```
src/
├── __init__.py              # Public API: DataManager, FrequencyType, TimeSeriesIndex
├── data/                    # Data loading and transformation layer
│   ├── core/                # Types, struct (.dt accessor), rolling, operations, cache, provider
│   ├── loaders/             # table.py (from_table), abstracts/base.py (BaseDataSource ABC)
│   │   ├── wrds/            # WRDS loaders: crsp, compustat, generic
│   │   ├── crypto/          # Local crypto loader
│   │   └── open_bb/         # OpenBB generic loader
│   ├── managers/            # DataManager (user entry point), config_schema
│   ├── configs/             # Built-in named configs (crypto_standard, equity_standard)
│   ├── processors/          # Post-processing registry
│   ├── filters/             # Django-style filter DSL
│   ├── generators/          # Weight/window generators
│   └── ast/                 # Formula evaluation: grammar, parser, functions, manager, nodes
├── pipeline/                # Processing pipeline and walk-forward analysis
│   ├── data_handler/        # ProcessorContract, DataHandler, processors, merge
│   ├── walk_forward/        # SegmentPlan, WalkForwardRunner, planning, execution
│   ├── model/               # ModelAdapter ABC, SklearnAdapter, ModelRunner
│   ├── cache/               # Content-addressable pipeline cache
│   └── spec/                # YAML-based pipeline spec system (PipelineSpec, PipelineExecutor)
└── backtester/              # Event-based strategy, order types, metrics
```

## Data Flow

```
Raw Data (CSV/WRDS/OpenBB)
    → from_table() / DataManager
    → xarray Dataset (year, month, day[, hour], asset)
    → DataHandler (shared → learn → infer processors)
    → WalkForwardRunner / ModelRunner
    → Results (predictions, portfolio returns)
```

## Dimension Contract

| Dimension | Required | Notes |
|-----------|----------|-------|
| year      | Yes      | Integer coordinate |
| month     | Yes      | Integer coordinate |
| day       | Yes      | Integer coordinate |
| hour      | No       | Only for intraday data |
| asset     | Yes      | String coordinate |

## Key Invariants

- JAX configured for float64 globally at import time
- Provider registration via `_PROVIDER_REGISTRY` — loaders self-register on module import
- Pipeline processor stages: shared (stateless), learn (fit+transform), infer (inference-only)
- Walk-forward leakage contract: learn stage sees only training window data
- Formula system: YAML-defined expressions evaluated against xarray variables
- Cache: NetCDF-backed, hash-keyed from request parameters

## Current Repository Structure (Pre-Reorganization)

```
hindsight/
├── src/                  # Library source (PRODUCTION)
├── dev/                  # Tests + AI artifacts + archive (MIXED)
├── examples/             # Pipeline YAML examples (PRODUCTION)
├── docs/                 # Architecture docs (PRODUCTION)
├── source/               # Stale Sphinx tree (DOCS_STALE)
├── build/                # Generated Sphinx output (GENERATED)
├── .github/              # CI workflows + copilot config (MIXED)
├── [AI artifacts at root] # DUMP.md, AGENTS.md, CLAUDE.md, etc.
└── [Dev scratch at root]  # notebooks, PNGs, logs, etc.
```

## Target Repository Structure (Post-Reorganization)

```
hindsight/
├── .agent/               # gitignored — AI context, prompts, dumps
├── .agent-docs/          # TRACKED — hierarchical LLM-readable docs
├── .factory/             # gitignored — Droid skills and config
├── src/                  # TRACKED — library source (untouched)
├── tests/                # TRACKED — canonical test suite (from dev/)
├── examples/             # TRACKED — pipeline YAML examples
├── docs/                 # TRACKED — human-readable architecture docs
├── README.md             # TRACKED
├── requirements.txt      # TRACKED
├── conftest.py           # TRACKED
└── pytest.ini            # TRACKED
```
