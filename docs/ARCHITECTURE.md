# Hindsight Architecture

## What This Page Covers

This page explains the system model that the tracked codebase actually
implements today: how data is loaded, shaped, merged, transformed, cached, and
fed into walk-forward model execution. It focuses on the boundaries between the
main packages rather than documenting every class.

## When To Read It

Read this page when you are new to the repository, trying to understand where a
bug belongs, or need the mental model before editing YAML specs, formulas, or
pipeline code.

## Core Ideas

### One shared data model

Hindsight is organized around `xarray` datasets whose calendar structure is
kept explicit rather than flattened away. Most pipeline-facing datasets use:

```text
(year, month, day, hour, asset)
```

That contract matters because the same dataset can move through feature
engineering, cross-sectional processors, time-aware merges, and walk-forward
evaluation without repeatedly reinterpreting the time axis.

The broad flow is:

```text
provider -> data loader -> xr.Dataset -> formulas/processors -> walk-forward/model
```

### The main layers

| Layer | Primary path | Responsibility |
| --- | --- | --- |
| Data loading | `src/data/loaders/`, `src/data/managers/` | Read raw data, normalize columns, and return `xr.Dataset` outputs |
| Formula system | `src/data/ast/` | Parse and evaluate formula expressions and formula definitions |
| Data handling | `src/pipeline/data_handler/` | Apply shared, learn, and infer processors over datasets |
| Merge utilities | `src/pipeline/data_handler/merge.py` | Combine datasets with different frequencies and publication lags |
| Pipeline spec | `src/pipeline/spec/` | Parse YAML, orchestrate stage execution, and coordinate caches |
| Modeling | `src/pipeline/model/` | Wrap estimators and run walk-forward training and prediction |
| Cache | `src/pipeline/cache/` | Persist outputs at each pipeline stage with content-addressable keys |

### The executor is orchestration, not a second implementation

`PipelineExecutor` does not reimplement data loading, formulas, or model
training. It coordinates existing subsystems in stages:

1. Load each named data source.
2. Merge sources if the spec defines multi-source merges.
3. Compute feature operations.
4. Build a `DataHandler` and materialize preprocessing views.
5. If a `model` block is present, build a walk-forward plan and run `ModelRunner`.

That distinction is important when debugging. Problems with feature values
usually live in the formula system or upstream data; problems with per-segment
training usually live in the model runner or handler configuration; problems
with cache reuse usually live in stage configuration or parent keys.

### Layer boundaries that matter in practice

#### Data loading and merge behavior

Each named source in a pipeline spec becomes its own load request. The executor
builds a cache configuration from provider, dataset, date range, filters,
external tables, columns, frequency, and source-level processors. If there is
more than one source, the executor can merge them using the declarative merge
config from the spec.

The dataset merger is where point-in-time alignment lives. It expands lower
frequency data onto the left dataset's calendar and can shift data availability
with `time_offset_months`.

#### Feature engineering

Features are grouped into named operations. Each operation can depend on earlier
operations, and the executor topologically orders them before evaluation. That
keeps feature stages composable and cacheable without forcing users into a
single monolithic formula block.

#### Preprocessing

Preprocessing is handled by `DataHandler`, which exposes three conceptual
pipelines:

- `shared`: transform-only steps applied before branching
- `learn`: steps that fit on the learn path and keep learned state
- `infer`: transform-only steps for inference and post-processing

The resulting views are exposed as `RAW`, `LEARN`, and `INFER`. In practice,
this is the boundary between "feature creation" and "pipeline-time data
transformation".

#### Modeling

Model execution is walk-forward by design. The executor converts the YAML
`model.walk_forward` block into a `SegmentConfig`, builds a `SegmentPlan`, then
runs a fresh model instance per segment through `ModelRunner`.

The current executor only implements the `sklearn` adapter path. The schema is
worded broadly enough to accommodate more adapters later, but the checked-in
executor raises an error for non-`sklearn` adapters today.

### Cache stages

The pipeline cache is structured as five stages:

| Stage | Meaning |
| --- | --- |
| `L1_RAW` | Raw loader outputs before higher-level post-processing |
| `L2_POSTPROCESSED` | Loaded pipeline data sources after source-level config has been applied |
| `L3_FEATURES` | Feature outputs after formula operations |
| `L4_PREPROCESSED` | Preprocessed dataset plus handler state |
| `L5_MODEL` | Prediction dataset plus model runner result |

For datasets that carry attributes NetCDF cannot store directly, the cache
manager also writes companion `.attrs.pkl` files and restores those attributes
on load.

### What is and is not central

The repository supports factor-style and portfolio-style workflows, but the
architecture is not centered on any single example. `examples/ff3_model.yaml`
is useful because it exercises merges, formulas, infer-stage processors, and
portfolio outputs. It is still an example, not a special-case architecture path.

## Practical Examples

### Architecture map from a simple modeled pipeline

The bundled crypto example touches most of the main layers:

```text
examples/pipeline_specs/crypto_momentum_baseline.yaml
    -> PipelineSpec parsing
    -> single-source load through DataManager
    -> formula operations
    -> DataHandler preprocessing
    -> sklearn adapter + ModelRunner walk-forward execution
```

### Architecture map from a factor workflow

The FF3 example uses a different slice of the system:

```text
examples/ff3_model.yaml
    -> multi-source load
    -> point-in-time merge via DatasetMerger
    -> formulas for book equity and June signals
    -> infer-stage processors for sorting, portfolio returns, and factor spreads
    -> no model block required
```

That contrast is useful: the same data model supports both learned prediction
pipelines and portfolio construction workflows.

## Common Pitfalls

- Treating the `asset` dimension as the whole problem. Many bugs are really time-alignment problems in `(year, month, day, hour)`, not asset-join problems.
- Assuming the executor supports every adapter named in comments or schema docstrings. The current execution path is `sklearn` only.
- Treating preprocessing as a generic transform list. The distinction between `shared`, `learn`, and `infer` changes what state is fit and when it is reused.
- Flattening the time axis too early in mental models. Internally, model adapters flatten for estimator APIs, but the pipeline itself keeps the panel layout explicit as long as possible.
- Treating `.agent/` artifacts as product documentation. They are not part of the human-facing architecture surface.

## Read Next

- [PIPELINE_SYSTEM.md](./PIPELINE_SYSTEM.md) for the YAML contract, supported processors, and model execution details
- [DATASET_MERGER.md](./DATASET_MERGER.md) for time alignment and point-in-time merge semantics
- [INDEX.md](./INDEX.md) for the documentation map
