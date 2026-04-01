# Hindsight Pipeline System

## What This Page Covers

This page explains the tracked YAML pipeline surface: how a `PipelineSpec` is
structured, what each stage does, which processor and model paths are supported
by the checked-in executor, and what output artifacts you get back.

It does not try to dump every schema field. For exhaustive key-by-key reference,
start from [`../src/pipeline/spec/schema.py`](../src/pipeline/spec/schema.py)
(`PipelineSpec`) and the YAML under `src/data/ast/definitions/` after you
understand the mental model.

## When To Read It

Read this page before writing a new pipeline spec, when debugging a stage
boundary, or when deciding whether a workflow belongs in features,
preprocessing, infer-stage processors, or the model block.

## Core Ideas

### The pipeline stages

The executor runs a pipeline in up to four major stages:

1. `data`: load one or more named sources
2. `features`: evaluate formula operations in dependency order
3. `preprocessing`: build `DataHandler` views from shared, learn, and infer processors
4. `model`: run walk-forward training and prediction through `ModelRunner`

Each stage has its own cache key and can be reused when upstream inputs and
configuration stay the same.

### The top-level spec shape

The current schema centers on these top-level sections:

```yaml
spec_version: "1.0"
name: "pipeline_name"
version: "1.0"

time_range:
  start: "YYYY-MM-DD"
  end: "YYYY-MM-DD"

data: {}
merge_base: "optional_source_name"
merges: []
features: {}
preprocessing: {}
model: {}
metadata: {}
```

Not every section is required for every workflow. A factor-construction spec can
stop after infer-stage processors. A modeled pipeline usually uses every section
from `data` through `model`.

### Data sources and merges

Each entry under `data` is named and loaded independently. A source specifies:

- `provider`
- `dataset`
- optional `frequency`
- optional `filters`
- optional `external_tables`
- optional `columns`
- optional source-level `processors`

If you have multiple sources, `merge_base` and `merges` define how they are
combined before feature engineering. Merge logic is handled by
`DatasetMerger`; see [DATASET_MERGER.md](./DATASET_MERGER.md) for the semantics.

### Feature operations are named and ordered

Feature engineering is grouped into named operations. Each operation contains a
`formulas` mapping and may declare `depends_on`.

That structure matters for two reasons:

- it gives the executor a stable topological order
- it makes feature caching granular enough to reuse earlier operations

### Preprocessing is not just a flat transform list

`DataHandler` distinguishes three processor lists:

| Stage | Purpose |
| --- | --- |
| `shared` | Transform-only steps applied before branching |
| `learn` | Steps that fit state on the learn path and store that state |
| `infer` | Transform-only steps used on the inference path |

The `mode` field controls whether `learn` and `infer` branch from the shared
view independently or whether infer output feeds into the learn path.

### Supported processors in the registry

The processor registry currently exposes these YAML types:

| YAML type | Meaning |
| --- | --- |
| `cs_zscore` | Cross-sectional z-score normalization |
| `per_asset_ffill` | Forward-fill selected variables per asset |
| `formula_eval` | Evaluate formulas from preprocessing rather than from the features block |
| `cross_sectional_sort` / `sort` | Assign portfolio buckets from a signal |
| `portfolio_returns` / `port_ret` | Aggregate grouped returns |
| `factor_spread` | Construct long-short spreads from portfolio return outputs |

The short aliases `sort` and `port_ret` exist because the bundled factor example
uses them.

### The model path that exists today

The schema is broad enough to describe multiple adapter types, but the executor
currently implements one adapter path: `sklearn`.

In practice, the model stage does the following:

1. validates `features` and `target`
2. requires a `walk_forward` block
3. builds a `SegmentConfig` and `SegmentPlan`
4. resolves the requested sklearn estimator from a small set of sklearn modules
5. creates a fresh `SklearnAdapter` per segment
6. runs `ModelRunner`

If `adapter` is anything other than `sklearn`, the executor raises a
not-yet-implemented error.

### What comes back from execution

`PipelineExecutor.execute()` returns an `ExecutionResult` with:

- `data`: loaded source datasets by source name
- `features_data`: merged dataset after feature operations
- `preprocessed_data`: dataset after preprocessing
- `model_predictions`: dataset after the model stage, when present
- `cache_keys`: per-stage cache keys
- `fitted_model`: legacy alias for the runner artifact
- `model_runner_result`: canonical model artifact

One subtlety matters for users inspecting results: `learned_states` is typed
broadly in `ExecutionResult`, but the executor currently assigns the
preprocessing `DataHandler` instance there when a preprocessing stage runs. That
means the useful follow-up object is the handler, not a plain dict keyed by
processor name.

## Practical Examples

### A minimal modeled pipeline

The bundled crypto spec is the cleanest end-to-end example:

```yaml
data:
  crypto_prices:
    provider: "crypto"
    dataset: "spot/binance"
    frequency: "H"

features:
  operations:
    - name: "momentum_indicators"
      formulas:
        sma:
          - {window: 20}
          - {window: 50}

preprocessing:
  mode: "independent"
  shared:
    - type: "per_asset_ffill"
      name: "forward_fill"
      vars: ["close", "volume"]
  learn:
    - type: "cs_zscore"
      name: "normalizer"
      vars: ["close", "volume", "sma_ww20", "sma_ww50", "ema_ww12", "ema_ww26"]
      out_suffix: "_norm"

model:
  adapter: "sklearn"
  type: "LinearRegression"
  features: ["close_norm", "volume_norm", "sma_ww20_norm", "sma_ww50_norm", "ema_ww12_norm", "ema_ww26_norm"]
  target: "close"
  walk_forward:
    train_span_hours: 120
    infer_span_hours: 24
    step_hours: 24
```

That pipeline is exercised by [`examples/run_minimal_example.py`](../examples/run_minimal_example.py)
and the full multi-spec walkthrough in `dev/examples/run_pipeline_example.py` (local `dev/` tree).

### A factor-style pipeline without a model

`dev/examples/ff3_model.yaml` shows the other important path: infer-stage processors
can be the end product.

In that spec:

- the data stage loads CRSP and Compustat
- the merge stage applies point-in-time alignment
- the features stage computes book equity and June signals
- infer-stage processors sort, forward-fill buckets, compute portfolio returns, and construct factor spreads

That is still a pipeline spec, even though it never enters the model stage.

### Interpreting the example runner output

The bundled example script prints summaries like:

```python
print(list(result.data.keys()))
print(len(result.features_data.data_vars))
print(result.model_runner_result.attrs)
```

If preprocessing ran, `result.learned_states` is best treated as the
preprocessing handler object, which exposes learned state lists and cached views.

## Common Pitfalls

- Assuming feature formulas and preprocessing formulas are interchangeable. They use related machinery, but they sit at different pipeline boundaries and therefore different cache stages.
- Putting portfolio construction logic in the model block. Sorting, grouped returns, and factor spreads belong in infer-stage processors.
- Expecting `adapter: lightgbm`, `adapter: pytorch`, or custom model registries to work because the schema or comments mention them. The executor currently supports `sklearn` only.
- Omitting `walk_forward` in a modeled pipeline. The executor requires it.
- Reading `learned_states` as a processor-state dict. In the current execution path it is the preprocessing `DataHandler`.
- Using absolute machine-specific paths in commands or docs. Examples in this repository should stay relative to the repo.

## Read Next

- [ARCHITECTURE.md](./ARCHITECTURE.md) for the system-level layer map
- [DATASET_MERGER.md](./DATASET_MERGER.md) for multi-source and point-in-time merge behavior
- [INDEX.md](./INDEX.md) for the documentation map

