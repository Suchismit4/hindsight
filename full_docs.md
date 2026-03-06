# Hindsight Pipeline Framework - Complete Documentation

**Version:** 1.0
**Last Updated:** December 2025

---

## Table of Contents

### Part I: Getting Started
1. [Overview](#1-overview)
2. [Installation and Setup](#2-installation-and-setup)
3. [Data Loading with DataManager](#3-data-loading-with-datamanager)
4. [Data Handler and Processing Pipeline](#4-data-handler-and-processing-pipeline)
5. [Feature Engineering with Processors](#5-feature-engineering-with-processors)
6. [Walk-Forward Analysis and Temporal Segmentation](#6-walk-forward-analysis-and-temporal-segmentation)
7. [Model Integration and Execution](#7-model-integration-and-execution)
8. [Pipeline Execution and Results Analysis](#8-pipeline-execution-and-results-analysis)

### Part II: Complete Workflow Example
9. [Complete Workflow Example](#9-complete-workflow-example)

### Part III: API Reference
10. [Pipeline API Reference](#10-pipeline-api-reference)
11. [Data Handler API Reference](#11-data-handler-api-reference)
12. [Walk-Forward Analysis API Reference](#12-walk-forward-analysis-api-reference)
13. [Model Integration API Reference](#13-model-integration-api-reference)

**📘 For detailed API documentation with complete method signatures and docstrings, see: [full_docs_api.md](full_docs_api.md)**

### Part IV: Advanced Topics
14. [Pipeline System with YAML Specifications](#14-pipeline-system-with-yaml-specifications)
15. [Architecture Deep Dive](#15-architecture-deep-dive)
16. [Dataset Merger for Multi-Frequency Data](#16-dataset-merger-for-multi-frequency-data)

### Part V: Appendix
17. [Building the Documentation](#17-building-the-documentation)

---

# Part I: Getting Started

## 1. Overview

### Hindsight Pipeline Framework Overview

The Hindsight Pipeline Framework is a Python library for multi-asset, time-series research. It provides well-defined layers for transforming large xarray-based datasets, planning walk-forward segments, and integrating machine learning models without leaking future information.

#### What This Guide Covers

- How the pipeline is structured around "how" (data transformations) and "when" (temporal segmentation).
- Core modules that form the public API.
- Data flow from raw datasets to aggregated predictions.
- A short example showing the typical control flow.

#### Architectural Separation

**Data transformations (the "how")**
Managed by `src.pipeline.data_handler`. A `DataHandler` applies processors in three stages:

- `shared` processors run once on the entire dataset. They are stateless or cache their output for both training and inference paths.
- `learn` processors fit on training data and produce state objects (typically xarray Datasets) that are later reused when transforming inference slices.
- `infer` processors run transform-only operations after the learn stage to finish inference-specific adjustments.

Typical processors live in `src.pipeline.data_handler.processors`. They operate on xarray objects so they can broadcast across dimensions such as `asset` and the flattened calendar index.

**Temporal segmentation (the "when")**
Exposed through `src.pipeline.walk_forward`. Key abstractions:

- `Segment` defines train/infer windows for a single iteration.
- `SegmentPlan` is an ordered list of segments.
- `SegmentConfig` lets you describe a rolling or expanding schedule; `make_plan` converts that configuration into a `SegmentPlan` while clipping to available data and optionally inserting gaps.

`WalkForwardRunner` takes a `DataHandler` and a `SegmentPlan`. It applies the handler to each segment, respecting the train/infer boundaries, capturing learned states per segment, and aggregating processed inference panels back into a global Dataset.

**Model integration**
Resides in `src.pipeline.model`. `ModelRunner` mirrors `WalkForwardRunner` but adds per-segment model fitting and prediction. `ModelAdapter` objects (for example, `SklearnAdapter`) act as bridges between xarray slices and model-specific APIs. A factory function instantiates a fresh adapter per segment to avoid cross-segment leakage.

#### Core Data Structures

The framework consistently uses `xarray.Dataset` and `xarray.DataArray` objects that share a hierarchical calendar:

- `year`, `month`, `day` (and optionally `hour`) dims form a rectangular grid.
- `asset` dimension indexes securities.
- `time` coordinate holds datetime64 values; `time_flat` (if present) is a flattened index aligned with stacked views.

Processors, planners, and adapters all rely on this layout to perform vectorized operations without copying large arrays.

#### End-to-End Flow

1. Load or construct an xarray dataset (see `DataManager` utilities under `src.data`).
2. Configure a `HandlerConfig` with the processors you need. Instantiate `DataHandler` with the raw dataset.
3. Describe your walk-forward schedule using `SegmentConfig` and call `make_plan`.
4. Choose an execution path:
   - `WalkForwardRunner` if you only need processed datasets per segment.
   - `ModelRunner` if you also want model predictions aggregated back into a Dataset.
5. Inspect results: both runners return objects with the aggregated output, per-segment metadata, and optional learned states.

#### Concise Example

```python
import numpy as np
from src.pipeline import (
    DataHandler, HandlerConfig, PipelineMode,
    PerAssetFFill, CSZScore, make_plan, SegmentConfig, ModelRunner
)
from src.pipeline.model import SklearnAdapter
from sklearn.ensemble import RandomForestRegressor

# Assume ``raw_ds`` is an xarray.Dataset loaded elsewhere.

config = HandlerConfig(
    shared=[PerAssetFFill(name="ffill")],
    learn=[CSZScore(name="norm", vars=["close"])],
    mode=PipelineMode.INDEPENDENT,
    feature_cols=["close_csz"],
    label_cols=["target"]
)
handler = DataHandler(base=raw_ds, config=config)

seg_cfg = SegmentConfig(
    start=np.datetime64("2020-01-01"),
    end=np.datetime64("2023-01-01"),
    train_span=np.timedelta64(365, "D"),
    infer_span=np.timedelta64(30, "D"),
    step=np.timedelta64(30, "D"),
    gap=np.timedelta64(1, "D")
)
plan = make_plan(seg_cfg, ds_for_bounds=raw_ds)

def factory():
    return SklearnAdapter(
        model=RandomForestRegressor(),
        handler=handler,
        output_var="pred"
    )

runner = ModelRunner(
    handler=handler,
    plan=plan,
    model_factory=factory,
    feature_cols=["close_csz"],
    label_col="target"
)
result = runner.run()
predictions = result.pred_ds["pred"]
```

#### Where to Go Next

- `getting_started/data_loading` describes dataset requirements.
- `getting_started/data_handler` and `getting_started/feature_engineering` dive into processor design.
- `getting_started/walk_forward` documents segment planning and execution internals.
- `getting_started/model_integration` covers adapters and model runners in more depth.

---

## 2. Installation and Setup

### Building the Documentation

To build the HTML documentation:

1. Activate the conda environment with Sphinx installed:
   ```bash
   conda activate jax
   ```

2. Build the documentation:
   ```bash
   make html
   ```

3. Serve the documentation locally (recommended for VM/remote development):
   ```bash
   python3 serve_docs.py
   ```
   This will start a local HTTP server on port 8000. VS Code will automatically port-forward the URL for viewing in your browser.

   Alternatively, you can open the documentation directly:
   ```bash
   python3 open_docs.py
   ```

The generated documentation will be available in `build/html/index.html`.

---

## 3. Data Loading with DataManager

Every pipeline run begins with an xarray `Dataset`. Hindsight ships with loaders in `src.data` that convert raw sources (e.g., WRDS extracts, Parquet archives, cached NetCDF files) into the canonical layout expected by the processing stack. If you are wiring in your own data, focus on matching that layout.

### Dataset Shape

Hindsight expects a rectangular calendar with explicit calendar dimensions:

- `year`, `month`, `day` (and optionally `hour`) define the calendar grid.
- `asset` indexes instruments (tickers, pairs, etc.).
- `time` coordinate provides datetime64 values, aligned with the stacked calendar.
- `time_flat` (if present) is a 1-D index that matches the stacked calendar order.

All variable arrays should align with these dimensions. Numeric series such as `close` or `volume` are stored as `DataArray` objects with shape `(year, month, day, asset, ...)`.

### Loading via DataManager

`DataManager` (`src.data.DataManager`) exposes helper methods for loading built-in datasets or constructing pipelines from configuration files. A common workflow looks like this:

```python
from src.data import DataManager

dm = DataManager()
raw_ds = dm.load(
    source="wrds_eod",    # Loader key configured in data YAML
    table="daily_equities",
    start="2020-01-01",
    end="2023-01-01",
    assets=["AAPL", "MSFT"],
    cache=True             # Reuse prepared dataset if available
)
```

`raw_ds` is now an xarray `Dataset` adhering to the structure above. If the loader cannot populate certain calendar slots (weekends, market holidays), those coordinates appear as NaT entries in `time`; downstream processors handle them, typically via forward-fill or masking.

### Validation Checklist

Before handing the dataset to `DataHandler`:

- **Inspect dimensions**: `raw_ds.sizes` should show expected `year`/`month`/`day`/`asset` counts.
- **Confirm coordinates**: `raw_ds.time` should contain monotonic datetime64 values; `asset` coordinate should be unique identifiers.
- **Check dtype**: Ensure numeric features are floating point; convert integers if necessary to avoid unintended casting.
- **Handle missing data**: Leave NaNs in place; processors such as `PerAssetFFill` or `CSZScore` are designed to address them.

### Merging Multiple Sources

It is common to stitch together multiple datasets (e.g., prices + fundamentals). Because everything is xarray-based, merging retains aligned coordinates:

```python
price_ds = dm.load(source="prices", table="crypto_hourly")
factor_ds = dm.load(source="factors", table="crypto_factors")
combined = price_ds.merge(factor_ds, join="outer")
```

When merging, ensure both datasets share the same calendar. Use `reindex` or `align` to resolve mismatches before merging.

### Custom Loaders

If you have a proprietary data source, implement a loader that follows the patterns under `src.data.loaders`:

1. Parse raw files into pandas `DataFrame` objects.
2. Pivot to bring symbol identifiers to columns.
3. Convert to xarray via `to_xarray()` and reshape into the canonical calendar dims.
4. Attach metadata (e.g., exchange, frequency) via Dataset attributes if needed.

By respecting these conventions, your datasets seamlessly plug into `DataHandler` and the rest of the pipeline.

---

## 4. Data Handler and Processing Pipeline

`DataHandler` orchestrates the "how" side of the framework. It accepts an xarray `Dataset` plus a `HandlerConfig` describing how to transform the dataset across three stages: shared, learn, and infer. These stages mirror qlib's design but are tailored for multi-dimensional xarray workflows.

### HandlerConfig Parameters

`HandlerConfig` (`src.pipeline.data_handler.config`) exposes the following fields:

- `shared`: List of processors applied once to the full dataset. Use this for stateless operations (e.g., `PerAssetFFill`) or expensive transforms you want cached.
- `learn`: Processors that fit on training data and output state objects. During inference, the stored state is reused to transform new slices. `CSZScore` is a typical learn-stage processor.
- `infer`: Transform-only processors applied after the learn stage on inference slices. Use this for inference-specific post-processing.
- `mode`: `PipelineMode.INDEPENDENT` (default) keeps learn and infer branches separate; `PipelineMode.APPEND` feeds the infer branch output into the learn branch.
- `feature_cols` / `label_cols`: Optional lists that describe semantic groups. `handler.fetch` relies on these names to assemble feature/label datasets.

### Example Configuration

```python
from src.pipeline import HandlerConfig, PipelineMode
from src.pipeline.data_handler.processors import PerAssetFFill, FormulaEval, CSZScore

handler_config = HandlerConfig(
    shared=[
        PerAssetFFill(name="ffill", vars=["close", "volume"]),
        FormulaEval(
            name="formulas",
            formula_configs={"rsi": [{"window": 14}], "sma": [{"window": 20}]},
            static_context={"price": "close"}
        )
    ],
    learn=[CSZScore(name="norm", vars=["rsi", "sma"])],
    infer=[],
    mode=PipelineMode.INDEPENDENT,
    feature_cols=["rsi_norm", "sma_norm"],
    label_cols=["target_return"]
)
```

### Pipeline Execution Model

When you instantiate `DataHandler(base=raw_ds, config=handler_config)`, it builds lazily. The first call to `handler.view` triggers the following steps (cached afterwards):

1. **Shared stage**: Each processor's `transform` method runs on the full dataset. Outputs are stored in the handler cache under `shared_view`.

2. **Branch** depending on `mode`:
   - `INDEPENDENT`: Shared output feeds two separate pipelines (learn, infer).
   - `APPEND`: Shared output is passed through infer processors first; the result is then fed into learn processors.

3. **Learn stage** (if applicable): For each processor, call `fit_transform` on the appropriate training slice. The returned states (xarray Datasets) are kept in `handler.learn_states`.

4. **Infer stage**: Run transform-only operations using the cached states when needed.

### Accessing Views

`handler.view` accepts `View.RAW`, `View.LEARN`, or `View.INFER`:

- `RAW`: Original dataset (no transformations).
- `LEARN`: Shared + learn stages applied. If learn states have not been computed yet, they're created on demand.
- `INFER`: Shared + infer stages applied; may reuse learn states depending on mode.

To pull semantic column groups, use `handler.fetch`:

```python
from src.pipeline.data_handler import View

feature_ds = handler.fetch(View.LEARN, ["features"])
label_ds = handler.fetch(View.LEARN, ["labels"])
```

### State Management

Learn-stage processors must return compact state datasets (usually a few arrays with summary statistics). `DataHandler` preserves state order so you can zip processors with their states. Walk-forward runners reuse these states when transforming inference slices to guarantee temporal isolation.

### Pipeline Modes

- `PipelineMode.INDEPENDENT` (default): Shared output branches into learn and infer pipelines separately. Inference never sees learn-stage outputs.
- `PipelineMode.APPEND`: Shared output flows through infer processors first; the result is then passed into the learn pipeline. Choose this if you need inference transformations to feed into learning.

### Custom Processors

To create new transformations, subclass `src.pipeline.data_handler.processors.Processor` (which itself implements `ProcessorContract`):

- Implement `fit` to compute state (return an xarray `Dataset`).
- Implement `transform` to apply the transformation using the state.
- Optionally override `fit_transform` for efficiency.

Remember to operate on xarray objects and preserve coordinates. For example, to produce a demeaned series per asset, align dimensions carefully so broadcasting works as expected.

### Integrations

`WalkForwardRunner` and `ModelRunner` call `handler._apply_pipeline` internally for each segment. They never mutate the base dataset. This makes `DataHandler` reusable across multiple segment plans or models as long as the underlying dataset remains constant.

---

## 5. Feature Engineering with Processors

Hindsight's feature engineering layer is built on `Processor` objects. A processor is a reusable transformation that consumes an xarray `Dataset` and returns a modified dataset while optionally producing a compact state (for learn-stage processors). Processors live in `src.pipeline.data_handler.processors` and implement `ProcessorContract` defined in `core.py`.

### Processor Stages

The DataHandler pipeline invokes processors in three ordered stages:

- **Shared**: Runs once on the entire dataset before any temporal slicing. Use this for operations that should be cached (formula evaluation, per-asset fills).
- **Learn**: Runs on segment-specific training windows. Each processor fits on the training slice, returns a state `Dataset`, and then transforms the slice (and eventually the inference slice) using that state.
- **Infer**: Runs transform-only operations on the inference slice after learn-stage transformations.

### Built-in Processors

The framework ships with several processors tuned for common financial tasks:

**PerAssetFFill**
Stateless processor that forward-fills NaNs independently per asset along the flattened time index.

**CSZScore**
Cross-sectional z-score normalization. During `fit` it calculates per-timestamp mean and standard deviation across assets, storing the stats in a state `Dataset`. `transform` then normalizes each asset using those stats.

**FormulaEval**
Wraps the AST formula engine in `src.data.ast`. It evaluates declarative formulas (e.g., RSI, moving averages) and merges the outputs into the dataset. Can optionally JIT compile evaluations with JAX.

### Configuring Processors

You attach processors to stages via `HandlerConfig`. Example:

```python
from src.pipeline import HandlerConfig, PipelineMode
from src.pipeline.data_handler.processors import PerAssetFFill, FormulaEval, CSZScore

formulas = {
    "rsi": [{"window": 14}, {"window": 21}],
    "sma": [{"window": 20}]
}

config = HandlerConfig(
    shared=[
        PerAssetFFill(vars=["close", "volume"]),
        FormulaEval(
            formula_configs={
                "rsi": [{"window": 14}, {"window": 21}],
                "sma": [{"window": 20}]
            },
            static_context={"price": "close"}
        )
    ],
    learn=[CSZScore(vars=["rsi_w14", "rsi_w21", "sma_w20"])]
)
```

### FormulaEval Details

- **Formula definitions** live under `src/data/ast/definitions` (YAML). Each entry describes parameters, dependencies, and output naming conventions.
- **Static context** provides aliases and constants accessible to formulas (e.g., mapping `price` to `close`).
- **Outputs** merge into the dataset (e.g., `rsi_w14`). Use `assign_in_place=False` plus `prefix` if you want namespaced outputs instead.
- **Performance**: `use_jit=True` compiles evaluation with JAX; best used when the dataset is large and formulas are reused.

### Custom Processor Development

When you need bespoke logic, extend `Processor`:

```python
from dataclasses import dataclass
import xarray as xr
from src.pipeline.data_handler.processors import Processor

@dataclass
class DemeanByAsset(Processor):
    name: str
    var: str

    def fit(self, ds: xr.Dataset) -> xr.Dataset:
        means = ds[self.var].mean(dim="asset", skipna=True)
        return xr.Dataset({f"{self.name}_mean": means})

    def transform(self, ds: xr.Dataset, state: xr.Dataset = None) -> xr.Dataset:
        if state is None:
            state = self.fit(ds)
        demeaned = ds[self.var] - state[f"{self.name}_mean"]
        return ds.assign({f"{self.var}_{self.name}": demeaned})
```

Place the new processor in a stage depending on whether it needs to be fitted. If `transform` does not rely on a fitted state, the processor is effectively stateless and can run in the shared or infer stage.

### Best Practices

- **Coordinate safety**: Always rely on xarray broadcasting. When stacking/unstacking, preserve coordinates so downstream processors continue to align correctly.
- **State compactness**: Store only necessary statistics in state Datasets. Large states increase memory usage during walk-forward execution.
- **Naming**: Adopt predictable output naming (e.g., suffix-based) so configurations and downstream consumers can reference generated variables easily.
- **Statistical validity**: If a transformation requires fitting (e.g., normalization, PCA), it belongs in the learn stage to prevent leakage.

---

## 6. Walk-Forward Analysis and Temporal Segmentation

Hindsight separates temporal logic from data processing. The `walk_forward` package wires this up by describing how training and inference windows advance through time. This section explains the core abstractions and how they interact with `DataHandler`.

### Core Abstractions

**Segment**
Immutable descriptor of one walk-forward step. Each segment carries `train_start`, `train_end`, `infer_start`, and `infer_end` timestamps. Train boundaries are inclusive; inference end is exclusive to avoid overlap between consecutive segments.

**SegmentPlan**
Ordered collection of segments. It offers validation helpers, coverage summaries, and iteration semantics.

**SegmentConfig**
Declarative specification used to generate a `SegmentPlan`. It defines the global start/end range, training window length, inference window length, step size, optional gap, and whether to clip to dataset boundaries.

### Generating Plans

`make_plan` builds a plan from a `SegmentConfig`. It can optionally inspect a dataset for boundary clipping, ensuring segments do not extend beyond data availability.

```python
import numpy as np
from src.pipeline import make_plan, SegmentConfig

cfg = SegmentConfig(
    start=np.datetime64("2020-01-01"),
    end=np.datetime64("2023-01-01"),
    train_span=np.timedelta64(365, "D"),
    infer_span=np.timedelta64(30, "D"),
    step=np.timedelta64(30, "D"),
    gap=np.timedelta64(1, "D"),
    clip_to_data=True
)

# ``dataset`` refers to the xarray Dataset you plan to process.
plan = make_plan(cfg, ds_for_bounds=dataset)
issues = plan.validate()
if issues:
    raise ValueError(issues)
```

### Execution Flow

`WalkForwardRunner` orchestrates data processing over a `SegmentPlan`. Given a `DataHandler` and a plan, it:

1. Computes the shared view once (using the handler's shared processors).
2. Stacks the shared dataset for efficient integer slicing.
3. For each segment:
   - Identifies train and infer slices by integer bounds (avoiding expensive datetime indexing).
   - Applies learn processors in fit/transform mode on the train slice, capturing states.
   - Applies learn processors in transform-only mode plus infer processors on the inference slice.
   - Adds inference outputs to a global buffer according to the overlap policy ("last" or "first").
4. Unstacks the buffers back into the original calendar structure and returns a `WalkForwardResult` with the aggregated dataset and per-segment metadata.

### State Handling

Each learn-stage processor produces an xarray `Dataset` containing the fitted parameters (means, variances, etc.). `WalkForwardRunner` stores these states alongside the segment metadata so you can inspect how parameters evolve over time. This also ensures inference slices reuse the exact statistics learned on the corresponding training window.

### Overlap Policies

Segments may overlap on inference windows (e.g., rolling windows). `WalkForwardRunner` supports two strategies:

- `last` (default): Later segments overwrite earlier predictions where they overlap.
- `first`: First segment to produce a value keeps it; later segments only fill holes.

### Advanced Utilities

**expand_plan_coverage**
Extend an existing plan to cover earlier or later periods using the same configuration.

**optimize_plan_for_dataset**
Remove segments that do not meet minimum sample requirements for training or inference based on actual data coverage.

**run_segments**
Execute the pipeline segment-by-segment and return the individual `SegmentResult` objects instead of the aggregated dataset.

### Integration with ModelRunner

`ModelRunner` reuses the same segmentation logic but introduces model fitting/prediction on top of the processed slices. Understanding the base `WalkForwardRunner` is essential because `ModelRunner` piggybacks on its shared view and state management routines.

---

## 7. Model Integration and Execution

The `model` package extends walk-forward execution by fitting predictive models on each segment and aggregating forecasts. Rather than binding the pipeline to a specific ML library, Hindsight uses adapters that translate between xarray slices and model APIs.

### ModelAdapter Interface

`ModelAdapter` (`src.pipeline.model.adapter`) defines the minimal contract:

- `fit(ds, features, label=None, sample_weight=None)` trains the underlying model using variables extracted from `ds`.
- `predict(ds, features)` returns an xarray `DataArray` aligned with the segment's inference slice.
- `partial_fit` (optional) enables incremental training.
- `get_state` / `load_state` (optional) provide persistence hooks.

Adapters are responsible for converting the xarray slice into the format expected by the model (typically flattening `(time, asset)` into `(n_samples, n_features)` and dropping NaNs) and mapping predictions back to the original coordinates.

### SklearnAdapter

`SklearnAdapter` wraps any scikit-learn estimator exposing `fit`/`predict`. It uses the handler's `to_arrays_for_model` helper to extract feature and label arrays. Example:

```python
from sklearn.ensemble import RandomForestRegressor
from src.pipeline.model import SklearnAdapter

adapter = SklearnAdapter(
    model=RandomForestRegressor(n_estimators=100, random_state=42),
    handler=handler,
    output_var="score",
    drop_nan_rows=True
)
```

During `fit`, the adapter:

1. Converts the training dataset to `X` (`T x N x J`) and `y` arrays.
2. Flattens to `(T*N, J)` while removing rows with NaNs (if `drop_nan_rows` is True).
3. Calls the estimator's `fit` method (with optional sample weights).

`predict` performs the reverse process, reshaping predictions back to the full time/asset grid while preserving NaNs where the model could not produce values.

### ModelRunner Workflow

`ModelRunner` coordinates data processing, segmentation, and model execution:

1. Compute shared view once.
2. Pre-stack the dataset and pre-compute train/infer integer slices for all segments.
3. For each segment:
   - Slice train and infer windows from the shared view.
   - Fit learn processors on the train slice and produce states.
   - Transform the inference slice using the stored states and infer processors.
   - Instantiate a fresh adapter via `model_factory` and fit/predict on the processed slices.
   - Scatter predictions into a global buffer using the configured overlap policy.
4. Unstack the global buffer to produce an aggregated `Dataset` with the specified `output_var`.

The result is a `ModelRunnerResult` containing `pred_ds` (predictions), `segment_states` (learn-stage states), and run metadata.

### Overlap Policies

Identical to `WalkForwardRunner`: `last` overwrites earlier predictions with later ones where they overlap; `first` keeps the earliest prediction and only fills gaps with subsequent segments.

### Debugging Controls

`ModelRunner` accepts optional parameters to narrow execution during development:

- `debug_start` / `debug_end`: Limit the inference window in the global buffer.
- `debug_asset`: Restrict processing to a single asset for quicker iteration.

These flags ensure you can inspect a slice of the workflow without running the full backtest.

### Custom Adapters

To support other libraries (PyTorch, TensorFlow, statsmodels), subclass `ModelAdapter`:

```python
from src.pipeline.model import ModelAdapter
import torch

class TorchAdapter(ModelAdapter):
    def __init__(self, network, handler):
        self.network = network
        self.handler = handler

    def fit(self, ds, features, label=None, sample_weight=None):
        X, y = self.handler.to_arrays_for_model(ds, features, [label] if label else None)
        # Convert to tensors, train network...
        return self

    def predict(self, ds, features):
        X, _, idx = self.handler.to_arrays_for_model(ds, features, None, return_indexer=True)
        # Run inference, map back using idx
        # Return xarray.DataArray with proper coordinates
```

### Factory Functions

`model_factory` can be either an adapter instance or a callable returning a fresh adapter. Passing a callable is recommended so each segment gets a clean model instance, preventing leakage of learned parameters.

### Putting It Together

The following snippet assumes you have already prepared `handler`, `plan`, and `handler_config` as described in earlier sections (for example, by loading a dataset with `DataManager` and building a `SegmentPlan`).

```python
from src.pipeline.model import ModelRunner

def make_adapter():
    return SklearnAdapter(
        model=RandomForestRegressor(n_estimators=50, random_state=0),
        handler=handler,
        output_var="score"
    )

runner = ModelRunner(
    handler=handler,
    plan=plan,
    model_factory=make_adapter,
    feature_cols=handler_config.feature_cols,
    label_col=handler_config.label_cols[0],
    overlap_policy="last"
)

result = runner.run()
pred_ds = result.pred_ds
```

From here, you can merge `pred_ds` with the original dataset, compute metrics, or feed predictions into downstream systems.

---

## 8. Pipeline Execution and Results Analysis

After configuring processors and segments, the final step is to execute the pipeline and analyze outputs. The runners handle execution; your job is to interpret the resulting datasets.

### WalkForwardRunner Usage

`WalkForwardRunner` applies the data pipeline across a `SegmentPlan` without involving any models. Use it when you want processed inference panels for analysis or as a precursor to custom evaluation code.

```python
from src.pipeline import WalkForwardRunner

wf_runner = WalkForwardRunner(handler=handler, plan=plan)
wf_result = wf_runner.run()
```

`wf_result` is a `WalkForwardResult` with:

- `processed_ds`: Aggregated inference Dataset containing transformed variables.
- `segment_states`: List of dicts capturing per-segment metadata (state counts, durations, overlap decisions).
- `attrs`: Run-level metadata (overlap policy, segment count, timestamp).

### ModelRunner Usage

`ModelRunner` builds on `WalkForwardRunner` to include model training and prediction.

```python
from src.pipeline.model import ModelRunner

model_runner = ModelRunner(
    handler=handler,
    plan=plan,
    model_factory=make_adapter,
    feature_cols=handler_config.feature_cols,
    label_col=handler_config.label_cols[0],
    overlap_policy="last"
)

model_result = model_runner.run()
```

`model_result.pred_ds` mirrors the calendar of the input dataset and contains the prediction variable configured in the adapter. `segment_states` includes both learn-stage processor states and optional adapter state summaries if the adapter populates them.

### Analyzing Outputs

**Inspect coverage**
Confirm the prediction span: `model_result.pred_ds.time.min()` / `max()` and asset coverage via `pred_ds.asset`.

**Merge with raw or processed data**
Combine predictions with other variables:

```python
merged = raw_ds.merge(model_result.pred_ds)
```

**Compute metrics**
Use xarray-friendly metrics (correlations, mean returns) or convert to pandas for custom evaluation.

**Per-segment diagnostics**
Iterate `model_result.segment_states` to examine how many samples or states were learned per segment. Useful for spotting segments that had insufficient data.

### Presentation Tips

- The datasets are n-dimensional; slice to a smaller subset (e.g., `sel(asset="BTCUSDT")`) when plotting.
- Preserve metadata by using xarray operations (`merge`, `assign_coords`) rather than converting to numpy prematurely.
- Persist outputs with `to_netcdf` for reproducibility.

---

# Part II: Complete Workflow Example

## 9. Complete Workflow Example

This example stitches together all major components: loading data, configuring processors, planning walk-forward segments, fitting a model, and analyzing predictions.

### Full Script

```python
import numpy as np
import xarray as xr

from src.data import DataManager
from src.pipeline import (
    DataHandler, HandlerConfig, PipelineMode,
    PerAssetFFill, FormulaEval, CSZScore,
    SegmentConfig, make_plan
)
from src.pipeline.model import ModelRunner, SklearnAdapter
from sklearn.ensemble import RandomForestRegressor

# 1. Load data (hourly cryptocurrency prices in this example)
dm = DataManager()
raw_ds = dm.load(source="crypto_standard", table="crypto_prices")

# 2. Configure processors
formulas = {
    "sma": [{"window": 100, "wilder_weights": True}, {"window": 200, "wilder_weights": True}],
    "rsi": [{"window": 14}],
    "price_ret_var": [{"p": 1}, {"p": 3}, {"p": 6}]
}

handler_config = HandlerConfig(
    shared=[
        PerAssetFFill(name="ffill", vars=["close", "volume"]),
        FormulaEval(
            name="formulas_core",
            formula_configs=formulas,
            static_context={"price": "close", "prc": "close"},
            use_jit=False
        )
    ],
    learn=[CSZScore(name="norm", vars=["sma_ww100", "sma_ww200", "rsi", "price_ret_var_p1"] )],
    infer=[],
    mode=PipelineMode.INDEPENDENT,
    feature_cols=[
        "sma_ww100_norm",
        "sma_ww200_norm",
        "rsi_norm",
        "price_ret_var_p1_norm"
    ],
    label_cols=["fwd_return_p5"]
)

handler = DataHandler(base=raw_ds, config=handler_config)

# 3. Plan walk-forward segments
segment_cfg = SegmentConfig(
    start=np.datetime64("2020-01-22T18:00:00"),
    end=np.datetime64("2023-12-01T00:00:00"),
    train_span=np.timedelta64(24 * 5, "h"),
    infer_span=np.timedelta64(24 * 1, "h"),
    step=np.timedelta64(24 * 1, "h"),
    gap=np.timedelta64(0, "h"),
    clip_to_data=True
)
plan = make_plan(segment_cfg, ds_for_bounds=raw_ds)

# 4. Define adapter factory
def make_adapter():
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=14,
        min_samples_leaf=10,
        bootstrap=True,
        max_samples=0.5,
        n_jobs=-1,
        random_state=0
    )
    return SklearnAdapter(
        model=model,
        handler=handler,
        output_var="score",
        use_proba=False
    )

# 5. Execute model runner
runner = ModelRunner(
    handler=handler,
    plan=plan,
    model_factory=make_adapter,
    feature_cols=handler_config.feature_cols,
    label_col=handler_config.label_cols[0],
    overlap_policy="last"
)
results = runner.run()

# 6. Analyze predictions
pred_ds = results.pred_ds
merged = raw_ds.merge(pred_ds)
btc_panel = merged.sel(asset="BTCUSDT")

corr = xr.corr(btc_panel["score"], btc_panel["fwd_return_p5"], dim="time_flat").item()
print(f"Prediction correlation: {corr:.4f}")
```

### Explanation

1. **Load**: `DataManager` returns an xarray `Dataset` with the canonical calendar layout.
2. **Processors**: Shared stage fills gaps and computes formulas; learn stage normalizes a subset of indicators.
3. **Segmentation**: `SegmentConfig` describes a rolling window; `make_plan` clips it to data availability.
4. **Adapter**: `SklearnAdapter` wraps a random forest and handles xarray-to-numpy conversions.
5. **Execution**: `ModelRunner` processes each segment, fits the model, aggregates predictions, and records per-segment states.
6. **Analysis**: Merge predictions with raw data and calculate diagnostic metrics (e.g., correlation between predicted score and forward returns).

This workflow can be adapted by plugging in different processors, segment schedules, or adapters depending on your research goals.

---

# Part III: API Reference

## 10. Pipeline API Reference

### Core Classes and Functions

The `src.pipeline` module provides the main entry point for the Hindsight Pipeline Framework, offering convenient access to all core components through a unified interface.

**Classes:**
- `DataHandler` - Main data processing orchestrator
- `HandlerConfig` - Configuration for data processing pipeline
- `View` - Enum for accessing different data views (RAW, LEARN, INFER)
- `PipelineMode` - Enum for pipeline modes (INDEPENDENT, APPEND)
- `Segment` - Single walk-forward segment definition
- `SegmentPlan` - Collection of segments
- `SegmentConfig` - Configuration for generating segment plans

**Functions:**
- `make_plan` - Generate a SegmentPlan from a SegmentConfig
- `expand_plan_coverage` - Extend an existing plan to cover more periods
- `optimize_plan_for_dataset` - Remove segments with insufficient data

**Data Processing Components:**
- `ProcessorContract` - Abstract interface for all processors
- `Processor` - Base class for processor implementations
- `CSZScore` - Cross-sectional z-score normalization processor
- `PerAssetFFill` - Per-asset forward-fill processor
- `FormulaEval` - Formula evaluation processor

**Walk-Forward Execution:**
- `WalkForwardRunner` - Execute data processing over temporal segments
- `SegmentResult` - Results for a single segment
- `WalkForwardResult` - Aggregated results from all segments

---

## 11. Data Handler API Reference

The `src.pipeline.data_handler` module provides the data processing pipeline functionality, including processors, configuration, and orchestration components.

**Main Classes:**
- `DataHandler` - Pipeline orchestrator
- `HandlerConfig` - Pipeline configuration

**Core Types and Enums:**
- `View` - Data view access modes (RAW, LEARN, INFER)
- `PipelineMode` - Processing modes (INDEPENDENT, APPEND)
- `ProcessorContract` - Abstract processor interface

**Processors:**
- `Processor` - Base processor class
- `CSZScore` - Cross-sectional z-score normalization
- `PerAssetFFill` - Forward-fill missing values per asset
- `FormulaEval` - Evaluate formulas on datasets

---

## 12. Walk-Forward Analysis API Reference

The `src.pipeline.walk_forward` module provides comprehensive walk-forward analysis capabilities for temporal segmentation and execution.

**Main Classes:**
- `Segment` - Single walk-forward segment
- `SegmentPlan` - Ordered collection of segments
- `SegmentConfig` - Configuration for segment generation
- `WalkForwardRunner` - Execute pipeline over segments
- `SegmentResult` - Results for single segment
- `WalkForwardResult` - Aggregated results

**Planning Functions:**
- `make_plan` - Generate SegmentPlan from SegmentConfig
- `expand_plan_coverage` - Extend plan to additional periods
- `optimize_plan_for_dataset` - Remove insufficient segments

---

## 13. Model Integration API Reference

The `src.pipeline.model` module provides model integration capabilities for seamless integration of machine learning models with the pipeline framework.

**Main Classes:**
- `ModelAdapter` - Abstract adapter interface
- `SklearnAdapter` - Scikit-learn model adapter
- `ModelRunner` - Execute model training over segments
- `ModelRunnerResult` - Model training and prediction results

---

# Part IV: Advanced Topics

## 14. Hindsight Pipeline System


## Overview

The Hindsight Pipeline System is an end-to-end framework for building, training, and deploying quantitative trading strategies. It provides:

- **Declarative YAML specifications** for defining complete pipelines
- **Hierarchical caching (L1-L5)** with content-addressable keys
- **Automatic cache reuse** across different pipeline configurations
- **JIT-compatible operations** using JAX for performance
- **Learned state persistence** for preprocessing and models

### Key Features

- 🚀 **6-18x speedup** from intelligent caching
- 📊 **End-to-end workflow**: Data → Features → Preprocessing → Model → Predictions
- 🔄 **Automatic cache invalidation** when dependencies change
- 🎯 **Content-addressable caching** ensures correctness
- 🧩 **Extensible design** for custom processors and models

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline Specification (YAML)               │
│  - Data sources                                                  │
│  - Feature formulas                                              │
│  - Preprocessing steps                                           │
│  - Model configuration                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Executor                         │
│  Orchestrates execution with caching at each stage              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                     ↓                      ↓
   ┌─────────┐          ┌──────────┐          ┌──────────┐
   │ L2: Data│          │L3:Feature│          │L4: Preproc│
   │ Loading │    →     │Engineer. │    →     │& Normalize│
   └─────────┘          └──────────┘          └──────────┘
        ↓
   ┌──────────┐
   │ L5: Model│
   │ Training │
   └──────────┘
        ↓
   ┌──────────────┐
   │ Predictions  │
   └──────────────┘
```

### Component Hierarchy

```
src/pipeline/
├── cache/                    # Caching infrastructure
│   ├── manager.py           # GlobalCacheManager
│   ├── metadata.py          # MetadataManager
│   └── stages.py            # CacheStage enum (L1-L5)
├── spec/                     # Pipeline specifications
│   ├── schema.py            # PipelineSpec, ModelSpec, etc.
│   ├── parser.py            # YAML → Python objects
│   ├── executor.py          # PipelineExecutor
│   ├── result.py            # ExecutionResult
│   ├── processor_registry.py # Processor instantiation
│   └── model_registry.py    # Model instantiation
└── data_handler/            # Data processing
    ├── core.py              # DataHandler
    └── processors.py        # CSZScore, PerAssetFFill, etc.
```

---

## Pipeline Stages

### L1: Raw Data (Internal)
- Raw xarray Datasets directly from data loaders
- Not directly cached (used internally by loaders)

### L2: Post-Processed Data
**What it does:**
- Loads data from providers (crypto, WRDS, OpenBB, etc.)
- Applies data-level processors (filters, merges, etc.)
- Creates xarray Datasets with proper dimensions

**Cache key based on:**
- Provider and dataset name
- Time range
- Filters and processors
- Frequency

**Example output:**
```python
<xarray.Dataset>
Dimensions:  (year: 1, month: 12, day: 31, hour: 24, asset: 253)
Variables:
  - open, high, low, close, volume
```

### L3: Features
**What it does:**
- Evaluates formulas using the AST system
- Computes technical indicators (SMA, EMA, RSI, etc.)
- Per-operation caching for fine-grained reuse

**Cache key based on:**
- Formula definitions and parameters
- L2 cache key (parent data)

**Example output:**
```python
<xarray.Dataset>
Dimensions:  (year: 1, month: 12, day: 31, hour: 24, asset: 253)
Variables:
  - open, high, low, close, volume  # Original
  - sma_ww20, sma_ww50              # Computed features
  - ema_ww12, ema_ww26
  - rsi
```

### L4: Preprocessed
**What it does:**
- Applies shared processors (forward fill, etc.)
- Fits learn processors (normalization, scaling)
- Stores learned states for inference
- Applies infer processors (transform-only)

**Cache key based on:**
- Preprocessing configuration
- L3 cache key (parent features)

**Learned states:**
- Normalization parameters (mean, std)
- Scaling factors
- Any stateful transformations

**Example output:**
```python
<xarray.Dataset>
Dimensions:  (year: 1, month: 12, day: 31, hour: 24, asset: 253)
Variables:
  - open, high, low, close, volume     # Original
  - sma_ww20, sma_ww50, ema_ww12, ...  # Features
  - close_norm, volume_norm, ...       # Normalized
  
Learned States:
  normalizer: {mu__close, sd__close, mu__volume, sd__volume, ...}
```

### L5: Model Predictions
**What it does:**
- Creates sklearn model instance from spec
- Wraps model in `SklearnAdapter` for consistent interface
- Trains model on preprocessed data using adapter
- Generates predictions through adapter
- Caches both predictions and fitted adapter

**Cache key based on:**
- Model type and hyperparameters
- Feature list and target variable
- Adapter parameters
- L4 cache key (parent preprocessed data)

**Integration with existing infrastructure:**
- Uses `SklearnAdapter` from `src.pipeline.model.adapter`
- Uses `DataHandler` for consistent data stacking
- Compatible with `ModelRunner` for walk-forward validation

**Example output:**
```python
<xarray.Dataset>
Dimensions:  (year: 1, month: 12, day: 31, hour: 24, asset: 253)
Variables:
  - [all previous variables]
  - close_pred                        # Model predictions

Fitted Adapter:
  SklearnAdapter(model=LinearRegression(...), ...)
```

---

## Caching System

### Content-Addressable Keys

Each cache key is a hash of:
1. **Stage configuration**: All parameters for that stage
2. **Parent cache keys**: Dependencies from previous stages

This ensures:
- ✅ Same config + same inputs = same cache key (deterministic)
- ✅ Changed config = different cache key (automatic invalidation)
- ✅ Changed inputs = different cache key (propagates changes)

### Cache Key Computation

```python
def _compute_key(stage, config, parent_keys):
    """
    Compute content-addressable cache key.
    
    Args:
        stage: CacheStage enum (L2, L3, L4, L5)
        config: Configuration dict for this stage
        parent_keys: List of cache keys this depends on
    
    Returns:
        16-character hex hash
    """
    # Normalize config (sort keys, handle special types)
    normalized = _normalize_config(config)
    
    # Create deterministic representation
    content = {
        'stage': stage.value,
        'config': normalized,
        'parents': sorted(parent_keys)
    }
    
    # Hash with SHA256
    content_str = json.dumps(content, sort_keys=True)
    hash_obj = hashlib.sha256(content_str.encode())
    
    return hash_obj.hexdigest()[:16]
```

### Cache Hierarchy Example

```
Pipeline A: crypto_momentum_baseline
  L2: a40d6ea463d5443f  (crypto/spot/binance, 2023-01-01 to 2023-12-31)
    ↓
  L3: 71e0190e64738beb  (sma[20,50], ema[12,26], rsi[14])
    ↓
  L4: 6154db95785f9a2c  (ffill + cs_zscore[close,volume,sma,ema])
    ↓
  L5: 3f8a2b1c4d5e6f7a  (linear model, 6 features)

Pipeline B: crypto_momentum_enhanced
  L2: a40d6ea463d5443f  ← REUSED (same data source)
    ↓
  L3: 9c3d7f2a8b4e1f5d  (additional features: sma[100], ema[50])
    ↓
  L4: 2e9f8c1b3a5d7f4e  (same preprocessing, different features)
    ↓
  L5: 8d4c2f1a9b3e7f5c  (lightgbm model, 8 features)
```

### Cache Storage

```
~/data/hindsight_cache/
├── l2_post/
│   ├── a40d6ea463d5443f.nc          # Dataset
│   ├── a40d6ea463d5443f.attrs.pkl   # Non-NetCDF attrs (TimeSeriesIndex, etc.)
│   └── a40d6ea463d5443f.meta.json   # Metadata
├── l3_features/
│   ├── 71e0190e64738beb.nc
│   ├── 71e0190e64738beb.attrs.pkl
│   └── 71e0190e64738beb.meta.json
├── l4_prep/
│   ├── 6154db95785f9a2c.pkl         # (dataset, DataHandler)
│   └── 6154db95785f9a2c.meta.json
└── l5_model/
    ├── 3f8a2b1c4d5e6f7a.pkl         # (predictions, ModelRunnerResult)
    └── 3f8a2b1c4d5e6f7a.meta.json
```

- Each `.nc` file may be accompanied by an `.attrs.pkl` that preserves attributes NetCDF cannot encode (notably the `TimeSeriesIndex` used by the `.dt` accessor). When loading from cache, these extras are rehydrated automatically.

---

## YAML Specification

### Complete Spec Structure

```yaml
spec_version: "1.0"
name: "pipeline_name"
version: "1.0"

# Global time range
time_range:
  start: "YYYY-MM-DD"
  end: "YYYY-MM-DD"

# Data sources (L2)
data:
  source_name:
    provider: "crypto" | "wrds" | "open_bb"
    dataset: "spot/binance" | "crsp_daily" | ...
    frequency: "H" | "D" | "M"
    filters: {}
    processors: []

# Feature engineering (L3)
features:
  operations:
    - name: "operation_name"
      formulas:
        formula_name:
          - {param1: value1, param2: value2}
          - {param1: value3}

# Preprocessing (L4)
preprocessing:
  mode: "independent" | "append"
  
  shared:
    - type: "processor_type"
      name: "processor_name"
      # processor-specific params
  
  learn:
    - type: "processor_type"
      name: "processor_name"
      # processor-specific params
  
  infer:
    - type: "processor_type"
      # transform-only processors

# Model training (L5)
model:
  adapter: "sklearn" | "lightgbm" | "pytorch" | ...
  type: "LinearRegression" | "RandomForestRegressor" | any supported model
  params:
    # model-specific hyperparameters passed to the constructor
  features:
    - "feature1"
    - "feature2"
  target: "target_variable"
  adapter_params:
    output_var: "predictions"  # name of prediction variable
    use_proba: false           # use predict_proba instead of predict
  walk_forward:
    train_span_hours: 120
    infer_span_hours: 24
    step_hours: 24
    gap_hours: 0
    start: "YYYY-MM-DDTHH:MM:SS"  # optional, defaults to dataset bounds
    end:   "YYYY-MM-DDTHH:MM:SS"  # optional
  runner_params:
    overlap_policy: "last"  # or "first"

# Metadata (optional)
metadata:
  description: "Pipeline description"
  author: "Author name"
  tags: ["tag1", "tag2"]
```

### Available Processors

**per_asset_ffill**: Forward fill missing values per asset
```yaml
- type: "per_asset_ffill"
  name: "ffill"
  vars: ["close", "volume"]
```

**cs_zscore**: Cross-sectional z-score normalization
```yaml
- type: "cs_zscore"
  name: "normalizer"
  vars: ["close", "volume"]
  out_suffix: "_norm"
  eps: 1e-8
```

**formula_eval**: Evaluate formulas (for complex preprocessing)
```yaml
- type: "formula_eval"
  name: "custom_features"
  formulas:
    log_return:
      - {}
```

### Available Models

The model system integrates with sklearn through the existing `SklearnAdapter` infrastructure. Any sklearn model can be used by specifying its class name.

**LinearRegression**: Simple linear regression
```yaml
model:
  type: "LinearRegression"
  params:
    fit_intercept: true
  adapter_params:
    output_var: "predictions"
```

**RandomForestRegressor**: Random forest ensemble
```yaml
model:
  type: "RandomForestRegressor"
  params:
    n_estimators: 100
    max_depth: 10
    n_jobs: -1
  adapter_params:
    output_var: "predictions"
```

**Other sklearn models**: Any sklearn estimator with `.fit()` and `.predict()` methods
```yaml
model:
  type: "Ridge"  # or "Lasso", "ElasticNet", "GradientBoostingRegressor", etc.
  params:
    alpha: 1.0
  adapter_params:
    output_var: "predictions"
```

The system automatically searches common sklearn modules:
- `sklearn.linear_model`
- `sklearn.ensemble`
- `sklearn.tree`
- `sklearn.svm`
- `sklearn.neighbors`

---

## Usage Guide

### Basic Usage

```python
from src.pipeline.cache import GlobalCacheManager
from src.pipeline.spec import SpecParser, PipelineExecutor
from src.data.managers.data_manager import DataManager

# Setup
cache_manager = GlobalCacheManager(cache_root="~/data/hindsight_cache")
data_manager = DataManager()
executor = PipelineExecutor(cache_manager, data_manager)

# Load and execute pipeline
spec = SpecParser.load_from_yaml("my_pipeline.yaml")
result = executor.execute(spec)

# Access results
print(f"Data: {list(result.data.keys())}")
print(f"Features: {list(result.features_data.data_vars)}")
print(f"Predictions: {result.model_predictions['close_pred']}")

# Access learned states (for inference)
normalizer_state = result.learned_states['normalizer']

# Access fitted model (for inference)
runner = result.model_runner_result
print(runner.attrs)
```

### Running the Example

```bash
# Navigate to examples directory
cd /home/suchismit/projects/hindsight/examples

# Set PYTHONPATH
export PYTHONPATH=/home/suchismit/projects/hindsight:$PYTHONPATH

# Run example (requires JAX environment)
conda activate jax
python run_pipeline_example.py
```

### Expected Output

```
================================================================================
Part 1: Execute Baseline Pipeline
================================================================================

Loading spec: crypto_momentum_baseline.yaml

Pipeline: crypto_momentum_baseline v1.0
  Data: ['crypto_prices']
  Features: 1 operation(s)
  Preprocessing: 1 learn processor(s)
  Model: linear

Executing baseline pipeline (all cache misses expected)...

================================================================================
Executing Pipeline: crypto_momentum_baseline
================================================================================

Loading data sources...
  Source: crypto_prices
Cache miss: l2_post/a40d6ea463d5... - computing...
  Shape: {'year': 1, 'month': 12, 'day': 31, 'hour': 24, 'asset': 253}

================================================================================
Feature Engineering Stage
================================================================================

Operation: momentum_indicators
Cache miss: l3_features/71e0190e6473... - computing...
  Formulas computed: ['sma_ww20', 'sma_ww50', 'ema_ww12', 'ema_ww26', 'rsi']

================================================================================
Preprocessing Stage
================================================================================

Cache miss: l4_prep/6154db95785f... - computing...
  Applying shared processor: forward_fill
  Applying learn processor: normalizer

================================================================================
Model Training/Prediction Stage
================================================================================

Cache miss: l5_model/3f8a2b1c4d5e... - computing...
  Training sklearn/LinearRegression model via ModelRunner...
  Walk-forward segments: 100%|████████████████████████████████| 365/365 [00:05<00:00, 72.1segment/s]
    Segments processed: 365
    Model trained successfully

Execution Summary:
  Total time: 8.45s
  Pipeline: crypto_momentum_baseline v1.0
  Data sources: ['crypto_prices']
  Features computed: 10 variables
  Preprocessed variables: 16
  Model type: linear
  Predictions: close_pred

================================================================================
Part 2: Re-execute Baseline Pipeline
================================================================================

Executing same pipeline again (all cache hits expected)...

Cache hit: l2_post/a40d6ea463d5... - loading...
Cache hit: l3_features/71e0190e6473... - loading...
Cache hit: l4_prep/6154db95785f... - loading...
Cache hit: l5_model/3f8a2b1c4d5e... - loading...

Execution Summary:
  Total time: 0.52s

Performance comparison:
  First execution:  8.45s
  Second execution: 0.52s
  Speedup: 16.3x

Cache verification:
  data: ✓ MATCH
  features: ✓ MATCH
  preprocessing: ✓ MATCH
  model: ✓ MATCH

================================================================================
Part 3: Execute Enhanced Pipeline
================================================================================

Loading spec: crypto_momentum_enhanced.yaml

Pipeline: crypto_momentum_enhanced v2.0
  Data: ['crypto_prices'] (SAME as baseline)
  Features: 2 operation(s) (EXTENDED)
  Model: lightgbm (DIFFERENT)

Expected cache behavior:
  - Data (L2): HIT (same data source)
  - Features (L3): PARTIAL (reuse baseline features, compute new ones)
  - Preprocessing (L4): MISS (different feature set)
  - Model (L5): MISS (different model type)

Cache hit: l2_post/a40d6ea463d5... - loading...
Cache hit: l3_features/71e0190e6473... - loading... (momentum_indicators)
Cache miss: l3_features/9c3d7f2a8b4e... - computing... (volatility_indicators)
Cache miss: l4_prep/2e9f8c1b3a5d... - computing...
Cache miss: l5_model/8d4c2f1a9b3e... - computing...
  Walk-forward segments: 100%|████████████████████████████████| 365/365 [00:08<00:00, 45.1segment/s]

Execution Summary:
  Total time: 3.21s

================================================================================
Summary
================================================================================

Key Takeaways:
  1. First execution computes all stages (8.45s)
  2. Second execution hits all caches (0.52s, 16.3x speedup)
  3. Different specs reuse shared stages automatically
  4. Cache keys are content-addressable
  5. Learned states and ModelRunner results are cached
```

---

## Under the Hood

This section provides detailed explanations of the internal mechanics, data flows, and implementation details of the pipeline system.

### Pipeline Execution Flow

```python
def execute(spec: PipelineSpec) -> ExecutionResult:
    """
    Execute complete pipeline with caching.
    
    Flow:
    1. Data Loading (L2)
    2. Feature Engineering (L3)
    3. Preprocessing (L4)
    4. Model Training (L5)
    """
    
    # Stage 1: Data Loading (L2)
    data, data_key = _execute_data_stage(spec)
    # Computes: hash(data_config)
    
    # Stage 2: Feature Engineering (L3)
    features, features_key = _execute_features_stage(spec, data, data_key)
    # Computes: hash(features_config, data_key)
    
    # Stage 3: Preprocessing (L4)
    preprocessed, prep_key, handler = _execute_preprocessing_stage(
        spec, features, features_key
    )
    # Computes: hash(preprocessing_config, features_key)
    # Returns: (dataset, DataHandler with learned states)
    
    # Stage 4: Model Training (L5)
    predictions, model_key, runner_result = _execute_model_stage(
        spec, preprocessed, prep_key, handler
    )
    # Computes: hash(model_config + walk_forward, prep_key)
    # Returns: (predictions, ModelRunnerResult)
    
    return ExecutionResult(
        data=data,
        features_data=features,
        preprocessed_data=preprocessed,
        model_predictions=predictions,
        learned_states=handler,      # DataHandler
        fitted_model=runner_result,   # Legacy alias
        model_runner_result=runner_result,
        cache_keys={
            'data': data_key,
            'features': features_key,
            'preprocessing': prep_key,
            'model': model_key
        }
    )
```

### DataHandler: Deep Dive

#### What is DataHandler?

`DataHandler` is the core preprocessing orchestrator in Hindsight. It manages:
- **Processor pipelines**: Shared, learn, and infer stages
- **State management**: Learned parameters (normalization stats, etc.)
- **View management**: Different data perspectives (RAW, LEARN, INFER)
- **Caching**: Intermediate transformation results

#### DataHandler Internal Structure

```python
@dataclass
class DataHandler:
    """
    Central orchestrator for data processing pipelines.
    
    Attributes:
        base: xr.Dataset - Raw input data
        config: HandlerConfig - Pipeline configuration
        cache: Dict - Cached views and states
        learn_states: List - Fitted processor states
        infer_states: List - Inference processor states
    """
    base: xr.Dataset
    config: HandlerConfig
    cache: Dict = field(default_factory=dict)
    learn_states: List = field(default_factory=list)
    infer_states: List = field(default_factory=list)
    
    def build(self) -> None:
        """
        Execute the complete pipeline.
        
        Internal flow:
        1. Apply shared processors (transform-only)
        2. Branch into learn and infer paths
        3. Fit learn processors on data
        4. Transform data through both paths
        5. Cache all views and states
        """
        # Step 1: Apply shared processors
        shared_view = self.base
        for processor in self.config.shared:
            shared_view = processor.transform(shared_view)
        self.cache['shared_view'] = shared_view
        
        # Step 2: Apply learn processors (fit + transform)
        learn_view = shared_view
        self.learn_states = []
        for processor in self.config.learn:
            # Fit the processor
            processor.fit(learn_view)
            # Transform the data
            learn_view = processor.transform(learn_view)
            # Store learned state
            state = processor.get_state()
            self.learn_states.append(state)
        self.cache['learn_view'] = learn_view
        
        # Step 3: Apply infer processors (transform-only)
        infer_view = shared_view
        for processor in self.config.infer:
            infer_view = processor.transform(infer_view)
        self.cache['infer_view'] = infer_view
    
    def view(self, which: View) -> xr.Dataset:
        """
        Get a specific data view.
        
        Views:
        - RAW: Original base dataset
        - LEARN: Shared + Learn transformations
        - INFER: Shared + Infer transformations
        """
        if which == View.RAW:
            return self.base
        if which == View.LEARN:
            return self.cache['learn_view']
        if which == View.INFER:
            return self.cache['infer_view']
```

#### How Pipeline Uses DataHandler

**Step 1: Parse YAML Preprocessing Spec**

```yaml
preprocessing:
  mode: "independent"
  shared:
    - type: "per_asset_ffill"
      name: "forward_fill"
      vars: ["close", "volume"]
  learn:
    - type: "cs_zscore"
      name: "normalizer"
      vars: ["close", "volume", "sma_ww20"]
      out_suffix: "_norm"
```

**Step 2: Instantiate Processors via ProcessorRegistry**

```python
from src.pipeline.spec.processor_registry import ProcessorRegistry

# ProcessorRegistry maps YAML types to processor classes
shared_processors = [
    ProcessorRegistry.create_processor({
        'type': 'per_asset_ffill',
        'name': 'forward_fill',
        'vars': ['close', 'volume']
    })
]
# Result: [PerAssetFFill(name='forward_fill', vars=['close', 'volume'])]

learn_processors = [
    ProcessorRegistry.create_processor({
        'type': 'cs_zscore',
        'name': 'normalizer',
        'vars': ['close', 'volume', 'sma_ww20'],
        'out_suffix': '_norm'
    })
]
# Result: [CSZScore(name='normalizer', vars=[...], out_suffix='_norm')]
```

**Step 3: Create HandlerConfig**

```python
from src.pipeline.data_handler import HandlerConfig, PipelineMode

handler_config = HandlerConfig(
    shared=shared_processors,     # [PerAssetFFill(...)]
    learn=learn_processors,       # [CSZScore(...)]
    infer=[],                     # []
    mode=PipelineMode.INDEPENDENT,
    feature_cols=[],              # Not used in preprocessing-only
    label_cols=[]                 # Not used in preprocessing-only
)
```

**Step 4: Create and Build DataHandler**

```python
from src.pipeline.data_handler import DataHandler

# Create handler
handler = DataHandler(base=features_data, config=handler_config)

# Build pipeline - this executes all transformations
handler.build()
```

**What happens inside `handler.build()`:**

```python
# 1. Apply shared processors (transform-only)
shared_view = features_data
for processor in [PerAssetFFill(...)]:
    shared_view = processor.transform(shared_view)
    # PerAssetFFill.transform():
    #   - For each asset in the dataset
    #   - Forward-fill 'close' and 'volume' variables
    #   - Fills NaN values with last valid observation
    #   - Result: close and volume have fewer NaNs

# 2. Apply learn processors (fit + transform)
learn_view = shared_view
learn_states = []

processor = CSZScore(name='normalizer', vars=['close', 'volume', 'sma_ww20'], out_suffix='_norm')

# Fit: Learn normalization parameters
processor.fit(learn_view)
# CSZScore.fit():
#   - For each variable (close, volume, sma_ww20)
#   - Compute cross-sectional mean and std at each time step
#   - mean[t] = mean across all assets at time t
#   - std[t] = std across all assets at time t
#   - Store in processor.state = {'mean': {...}, 'std': {...}}

# Transform: Apply normalization
learn_view = processor.transform(learn_view)
# CSZScore.transform():
#   - For each variable
#   - Create new variable: var_norm = (var - mean) / std
#   - close_norm = (close - mean_close) / std_close
#   - volume_norm = (volume - mean_volume) / std_volume
#   - sma_ww20_norm = (sma_ww20 - mean_sma) / std_sma
#   - Add new variables to dataset

# Store learned state
learn_states.append(processor.get_state())
# Returns: {'mean': xr.Dataset, 'std': xr.Dataset}

# 3. Cache results
handler.cache['shared_view'] = shared_view
handler.cache['learn_view'] = learn_view
handler.cache['learn_states'] = learn_states
```

**Step 5: Get Preprocessed Data**

```python
from src.pipeline.data_handler import View

# Get the LEARN view (all transformations applied)
preprocessed_data = handler.view(View.LEARN)

# preprocessed_data now contains:
# - Original variables: open, high, low, close, volume, ema_ww12, etc.
# - Forward-filled: close, volume (modified in-place)
# - Normalized: close_norm, volume_norm, sma_ww20_norm (new variables)
```

**Step 6: Cache Handler and Data**

```python
# The executor caches BOTH dataset and handler
cache_manager.save_with_state(
    stage=CacheStage.L4_PREPROCESSED,
    key=cache_key,
    data=preprocessed_data,      # xr.Dataset
    state=handler                # DataHandler with learned states
)

# On cache hit, both are restored:
preprocessed_data, handler = cache_manager.load_with_state(cache_key)

# The handler can be used for inference:
# - Apply same transformations to new data
# - Use cached learn_states (no refitting)
new_data_preprocessed = handler.view(View.INFER)
```

### ModelAdapter: Deep Dive

#### What is ModelAdapter?

`ModelAdapter` provides a uniform interface for machine learning models, handling:
- **Data conversion**: xarray ↔ numpy
- **Model training**: Fit on training data
- **Prediction**: Generate predictions on new data
- **State management**: Fitted model parameters

#### ModelAdapter Internal Structure

```python
class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Attributes:
        model: The underlying model (sklearn, lightgbm, pytorch, etc.)
        handler: DataHandler for data stacking/unstacking
        output_var: Name for prediction variable
    """
    
    def __init__(self, model, handler, output_var='predictions', **kwargs):
        self.model = model
        self.handler = handler
        self.output_var = output_var
    
    @abstractmethod
    def fit(self, ds: xr.Dataset, features: List[str], label: str):
        """
        Train the model.
        
        Internal flow:
        1. Stack features from xarray to numpy
        2. Stack labels from xarray to numpy
        3. Remove NaN rows
        4. Fit model on clean data
        """
        pass
    
    @abstractmethod
    def predict(self, ds: xr.Dataset, features: List[str]) -> xr.DataArray:
        """
        Generate predictions.
        
        Internal flow:
        1. Stack features from xarray to numpy
        2. Predict using fitted model
        3. Unstack predictions back to xarray
        """
        pass
    
    def stack_features(self, ds: xr.Dataset, features: List[str]) -> np.ndarray:
        """
        Convert xarray features to numpy array.
        
        Process:
        1. Extract specified variables from dataset
        2. Stack all dimensions into (samples, features)
        3. Handle multi-dimensional time (year, month, day, hour)
        """
        # Extract feature arrays
        feature_arrays = [ds[var] for var in features]
        
        # Stack dimensions
        # From: (year, month, day, hour, asset) per feature
        # To: (samples, n_features) where samples = year*month*day*hour*asset
        stacked = []
        for arr in feature_arrays:
            # Flatten all dimensions
            flat = arr.values.ravel()
            stacked.append(flat)
        
        # Combine features
        X = np.column_stack(stacked)
        # Shape: (n_samples, n_features)
        
        return X
    
    def unstack_predictions(self, predictions: np.ndarray, 
                           template: xr.DataArray) -> xr.DataArray:
        """
        Convert numpy predictions back to xarray.
        
        Process:
        1. Reshape flat predictions to match template shape
        2. Create DataArray with same coordinates
        3. Name it according to output_var
        """
        # Reshape to match template
        pred_shaped = predictions.reshape(template.shape)
        
        # Create DataArray
        pred_da = xr.DataArray(
            pred_shaped,
            coords=template.coords,
            dims=template.dims,
            name=self.output_var
        )
        
        return pred_da


class SklearnAdapter(ModelAdapter):
    """
    Adapter for scikit-learn models.
    """
    
    def fit(self, ds: xr.Dataset, features: List[str], label: str):
        """
        Fit sklearn model.
        """
        # Stack features
        X = self.stack_features(ds, features)
        # Shape: (n_samples, n_features)
        
        # Stack labels
        y = self.stack_features(ds, [label])
        # Shape: (n_samples, 1)
        y = y.ravel()  # Flatten to 1D
        
        # Remove NaNs
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        # Fit model
        self.model.fit(X_clean, y_clean)
    
    def predict(self, ds: xr.Dataset, features: List[str]) -> xr.DataArray:
        """
        Generate predictions.
        """
        # Stack features
        X = self.stack_features(ds, features)
        
        # Predict
        predictions = self.model.predict(X)
        # Shape: (n_samples,)
        
        # Unstack to xarray
        template = ds[features[0]]  # Use first feature as shape template
        pred_da = self.unstack_predictions(predictions, template)
        
        return pred_da
```

#### How Pipeline Uses ModelAdapter

**Step 1: Parse YAML Model Spec**

```yaml
model:
  adapter: "sklearn"
  type: "RandomForestRegressor"
  params:
    n_estimators: 50
    max_depth: 14
  features: ["close_norm", "volume_norm", "sma_ww20_norm"]
  target: "close"
  adapter_params:
    output_var: "close_pred"
```

**Step 2: Dynamically Import and Instantiate Model**

```python
import importlib

# Import model class
module = importlib.import_module('sklearn.ensemble')
model_class = getattr(module, 'RandomForestRegressor')

# Instantiate with params
model = model_class(n_estimators=50, max_depth=14)
# Result: RandomForestRegressor(n_estimators=50, max_depth=14)
```

**Step 3: Create DataHandler for Adapter**

```python
# The adapter needs a handler for data stacking
adapter_handler = DataHandler(
    base=preprocessed_data,
    config=HandlerConfig(
        shared=[],
        learn=[],
        infer=[],
        mode=PipelineMode.INDEPENDENT,
        feature_cols=['close_norm', 'volume_norm', 'sma_ww20_norm'],
        label_cols=['close']
    )
)
adapter_handler.build()
```

**Step 4: Create SklearnAdapter**

```python
from src.pipeline.model.adapter import SklearnAdapter

adapter = SklearnAdapter(
    model=model,                      # RandomForestRegressor
    handler=adapter_handler,          # DataHandler for stacking
    output_var='close_pred'           # Prediction variable name
)
```

**Step 5: Fit Model**

```python
features = ['close_norm', 'volume_norm', 'sma_ww20_norm']
target = 'close'

adapter.fit(ds=preprocessed_data, features=features, label=target)
```

**What happens inside `adapter.fit()`:**

```python
# 1. Stack features
X = adapter.stack_features(preprocessed_data, features)
# Process:
#   - Extract close_norm, volume_norm, sma_ww20_norm from dataset
#   - Each has shape: (year=1, month=12, day=31, hour=24, asset=253)
#   - Flatten: 1 * 12 * 31 * 24 * 253 = 2,260,928 samples
#   - Stack: (2260928, 3) for 3 features
# Result: X.shape = (2260928, 3)

# 2. Stack target
y = adapter.stack_features(preprocessed_data, [target])
# Process:
#   - Extract 'close' variable
#   - Flatten same way as features
# Result: y.shape = (2260928, 1)
y = y.ravel()  # Flatten to 1D: (2260928,)

# 3. Remove NaN rows
valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
# Find rows where neither features nor target have NaN
X_clean = X[valid_mask]
y_clean = y[valid_mask]
# Result: X_clean.shape ≈ (2000000, 3) after removing ~260k NaN rows

# 4. Fit model
model.fit(X_clean, y_clean)
# RandomForestRegressor trains on clean data
# Learns: 50 decision trees with max_depth=14
# Stores: Tree structures, split points, leaf values
```

**Step 6: Generate Predictions**

```python
pred_da = adapter.predict(ds=preprocessed_data, features=features)
```

**What happens inside `adapter.predict()`:**

```python
# 1. Stack features (same as fit)
X = adapter.stack_features(preprocessed_data, features)
# Result: X.shape = (2260928, 3)

# 2. Predict
predictions = model.predict(X)
# RandomForestRegressor generates predictions
# For each sample, average predictions from 50 trees
# Result: predictions.shape = (2260928,)

# 3. Unstack to xarray
template = preprocessed_data['close']
# template.shape = (1, 12, 31, 24, 253)

pred_da = adapter.unstack_predictions(predictions, template)
# Process:
#   - Reshape: (2260928,) → (1, 12, 31, 24, 253)
#   - Create xr.DataArray with same coords as template
#   - Name it 'close_pred'
# Result: xr.DataArray with dims (year, month, day, hour, asset)
```

**Step 7: Add Predictions to Dataset**

```python
output_ds = preprocessed_data.copy()
output_ds['close_pred'] = pred_da

# output_ds now contains:
# - All original variables
# - All preprocessed variables (*_norm)
# - Predictions (close_pred)
```

**Step 8: Cache Adapter and Predictions**

```python
cache_manager.save_with_state(
    stage=CacheStage.L5_MODEL,
    key=cache_key,
    data=output_ds,      # xr.Dataset with predictions
    state=adapter        # Fitted SklearnAdapter
)

# On cache hit:
predictions, adapter = cache_manager.load_with_state(cache_key)

# The adapter can generate predictions on new data:
new_predictions = adapter.predict(new_data, features)
```

### ModelRunner Execution (Actual Pipeline)

In the executor, these adapter steps run inside `ModelRunner.run()`:

1. **Segment planning** – construct a `SegmentConfig` from `model.walk_forward` (train/infer/span/step/gap/start/end) and build a `SegmentPlan` via `make_plan`.
2. **Per-segment adapters** – define a factory that instantiates a fresh adapter/model for each segment while reusing the preprocessing `DataHandler` (with cached learn states).
3. **Progress reporting** – iterate the plan with a tqdm progress bar (`"Walk-forward segments"`), fit learn processors on the train slice, apply learned states to the inference slice, fit the segment adapter, and scatter predictions back using `runner_params.overlap_policy`.
4. **Cached artifact** – receive a `ModelRunnerResult` (`pred_ds`, `segment_states`, `attrs`) which is cached alongside the predictions dataset so cache hits restore both immediately.

### Cache Manager: get_or_compute

```python
def get_or_compute(stage, config, parent_keys, compute_fn):
    """
    Get cached result or compute and cache.
    
    This is the core caching primitive used by all stages.
    """
    # 1. Compute content-addressable key
    cache_key = _compute_key(stage, config, parent_keys)
    
    # 2. Try to load from cache
    cache_path = _get_cache_path(stage, cache_key)
    if cache_path.exists():
        result = _load(cache_path, stage)
        metadata_manager.update_access(cache_key, stage)
        print(f"Cache hit: {stage.value}/{cache_key[:12]}...")
        return result, cache_key
    
    # 3. Cache miss - compute
    print(f"Cache miss: {stage.value}/{cache_key[:12]}...")
    result = compute_fn()
    
    # 4. Save to cache
    cache_path = _save(result, stage, cache_key)
    
    # 5. Record metadata
    metadata = CacheMetadata(
        key=cache_key,
        stage=stage.value,
        config=config,
        parent_keys=parent_keys,
        size_bytes=cache_path.stat().st_size,
        created_at=datetime.now()
    )
    metadata_manager.record_cache(metadata)
    
    return result, cache_key
```

### Feature Engineering: Per-Operation Caching

```python
def _execute_features_stage(spec, data, data_key):
    """
    Execute feature engineering with per-operation caching.
    
    Each operation is cached independently, enabling fine-grained reuse.
    """
    merged_data = data
    
    for operation in spec.features.operations:
        # Build config for this operation
        config = {
            'operation_name': operation.name,
            'formulas': operation.formulas
        }
        
        # Compute with caching
        def compute_fn():
            return _compute_formulas(operation.formulas, merged_data)
        
        result, op_key = cache_manager.get_or_compute(
            stage=CacheStage.L3_FEATURES,
            config=config,
            parent_keys=[data_key],
            compute_fn=compute_fn
        )
        
        # Merge results
        merged_data = xr.merge([merged_data, result])
    
    # Compute final cache key for all features
    features_key = _compute_key(
        CacheStage.L3_FEATURES,
        {'operations': [op.to_dict() for op in spec.features.operations]},
        [data_key]
    )
    
    return merged_data, features_key
```

### Preprocessing: State Persistence

```python
def _execute_preprocessing_stage(spec, features, features_key):
    """
    Execute preprocessing with learned state caching.
    
    Returns both transformed dataset AND learned states.
    """
    config = {
        'mode': spec.preprocessing.mode,
        'shared': spec.preprocessing.shared,
        'learn': spec.preprocessing.learn,
        'infer': spec.preprocessing.infer
    }
    
    def compute_fn():
        # Apply shared processors (stateless)
        data = features
        for processor in shared_processors:
            data = processor.transform(data)
        
        # Apply learn processors (fit_transform)
        learned_states = {}
        for processor in learn_processors:
            data, state = processor.fit_transform(data)
            learned_states[processor.name] = state
        
        # Apply infer processors (transform-only)
        for processor in infer_processors:
            data = processor.transform(data)
        
        return data, learned_states
    
    # Use specialized caching for (dataset, states) tuple
    preprocessed, states, prep_key = cache_manager.get_or_compute_with_state(
        stage=CacheStage.L4_PREPROCESSED,
        config=config,
        parent_keys=[features_key],
        compute_fn=compute_fn
    )
    
    return preprocessed, prep_key, states
```

### Model Training: Model Persistence

```python
def _execute_model_stage(spec, preprocessed, prep_key):
    """
    Execute model training with model caching.
    
    Returns both predictions AND fitted model.
    """
    config = {
        'model_type': spec.model.type,
        'params': spec.model.params,
        'features': spec.model.features,
        'target': spec.model.target
    }
    
    def compute_fn():
        # Extract features and target
        X = extract_features(preprocessed, spec.model.features)
        y = preprocessed[spec.model.target].values
        
        # Remove NaN samples
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        # Train model
        model = ModelRegistry.create_model(spec.model)
        model.fit(X_clean, y_clean)
        
        # Generate predictions
        predictions = np.full(len(X), np.nan)
        predictions[valid_mask] = model.predict(X_clean)
        
        # Add predictions to dataset
        output_ds = preprocessed.copy()
        output_ds[f'{spec.model.target}_pred'] = create_dataarray(predictions)
        
        return output_ds, model
    
    # Use specialized caching for (predictions, model) tuple
    predictions, model, model_key = cache_manager.get_or_compute_with_state(
        stage=CacheStage.L5_MODEL,
        config=config,
        parent_keys=[prep_key],
        compute_fn=compute_fn
    )
    
    return predictions, model_key, model
```

---

## Examples

### Example 1: Baseline Pipeline

**File:** `examples/pipeline_specs/crypto_momentum_baseline.yaml`

```yaml
spec_version: "1.0"
name: "crypto_momentum_baseline"
version: "1.0"

time_range:
  start: "2023-01-01"
  end: "2023-12-31"

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
        ema:
          - {window: 12}
          - {window: 26}
        rsi:
          - {window: 14}

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
  params:
    fit_intercept: true
  features:
    - "close_norm"
    - "volume_norm"
    - "sma_ww20_norm"
    - "sma_ww50_norm"
    - "ema_ww12_norm"
    - "ema_ww26_norm"
  target: "close"
  adapter_params:
    output_var: "close_pred"
  walk_forward:
    train_span_hours: 120
    infer_span_hours: 24
    step_hours: 24
    gap_hours: 0
  runner_params:
    overlap_policy: "last"
```

**What it does:**
1. Loads hourly crypto data for 2023
2. Computes momentum indicators (SMA, EMA, RSI)
3. Forward fills missing values
4. Normalizes features using cross-sectional z-score
5. Trains linear regression model to predict close price

### Example 2: Enhanced Pipeline

**File:** `examples/pipeline_specs/crypto_momentum_enhanced.yaml`

```yaml
spec_version: "1.0"
name: "crypto_momentum_enhanced"
version: "2.0"

time_range:
  start: "2023-01-01"
  end: "2023-12-31"

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
        ema:
          - {window: 12}
          - {window: 26}
        rsi:
          - {window: 14}
    
    - name: "volatility_indicators"
      formulas:
        sma:
          - {window: 100}
        ema:
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
      vars: ["close", "volume", "sma_ww20", "sma_ww50", "sma_ww100", "ema_ww12", "ema_ww26", "ema_ww50"]
      out_suffix: "_norm"

model:
  adapter: "sklearn"
  type: "RandomForestRegressor"
  params:
    n_estimators: 50
    max_depth: 14
    min_samples_leaf: 10
    n_jobs: -1
    random_state: 0
  features:
    - "close_norm"
    - "volume_norm"
    - "sma_ww20_norm"
    - "sma_ww50_norm"
    - "sma_ww100_norm"
    - "ema_ww12_norm"
    - "ema_ww26_norm"
    - "ema_ww50_norm"
  target: "close"
  adapter_params:
    output_var: "close_pred"
  walk_forward:
    train_span_hours: 120
    infer_span_hours: 24
    step_hours: 24
    gap_hours: 0
  runner_params:
    overlap_policy: "last"
```

**What it does:**
1. Loads same data as baseline (reuses L2 cache)
2. Computes baseline features + additional volatility indicators
3. Same preprocessing (reuses baseline normalization for shared features)
4. Trains RandomForest model with more features

**Cache behavior:**
- L2 (Data): ✅ HIT (same data source)
- L3 (Features): ⚡ PARTIAL (reuses baseline, computes new)
- L4 (Preprocessing): ❌ MISS (different feature set)
- L5 (Model): ❌ MISS (different model type)

---

## Performance Characteristics

### Execution Times (Typical)

| Stage | First Run | Cache Hit | Speedup |
|-------|-----------|-----------|---------|
| Data Loading (L2) | 1-2s | 0.1s | 10-20x |
| Features (L3) | 3-4s | 0.1s | 30-40x |
| Preprocessing (L4) | 0.5s | 0.05s | 10x |
| Model Training (L5) | 1-2s | 0.1s | 10-20x |
| **Total** | **6-9s** | **0.5-1s** | **6-18x** |

### Cache Storage (Typical)

| Stage | Size per Entry | Notes |
|-------|----------------|-------|
| L2 (Data) | 50-200 MB | Depends on time range and assets |
| L3 (Features) | 100-400 MB | Includes all computed features |
| L4 (Preprocessing) | 100-400 MB | Dataset + learned states |
| L5 (Model) | 100-400 MB | Predictions + fitted model |

### Memory Usage

- **Peak memory**: ~2-4 GB for typical crypto dataset (1 year, 250 assets, hourly)
- **Streaming**: Not yet implemented (loads full dataset into memory)
- **JAX**: Uses GPU if available, falls back to CPU

---

## Best Practices

### 1. Organize Specs by Strategy

```
pipeline_specs/
├── momentum/
│   ├── baseline_v1.yaml
│   ├── baseline_v2.yaml
│   └── enhanced_v1.yaml
├── mean_reversion/
│   └── ...
└── ml_ensemble/
    └── ...
```

### 2. Use Meaningful Names

```yaml
name: "crypto_momentum_baseline"  # Good
name: "test1"                      # Bad

preprocessing:
  learn:
    - type: "cs_zscore"
      name: "normalizer"           # Good
      name: "proc1"                # Bad
```

### 3. Version Your Specs

```yaml
name: "crypto_momentum"
version: "1.0"  # Baseline
version: "2.0"  # Enhanced features
version: "3.0"  # Different model
```

### 4. Document with Metadata

```yaml
metadata:
  description: "Baseline momentum strategy using linear regression"
  author: "Your Name"
  tags: ["crypto", "momentum", "baseline"]
  notes: "Uses 6 normalized features for prediction"
```

### 5. Clear Cache Periodically

```bash
# Clear all cache
rm -rf ~/data/hindsight_cache

# Clear specific stage
rm -rf ~/data/hindsight_cache/l5_model
```

### 6. Monitor Cache Size

```python
stats = cache_manager.get_stats()
print(f"Total cache size: {stats['total_size_mb']:.2f} MB")
print(f"Total entries: {stats['total_entries']}")
```

---

## Troubleshooting

### Issue: Cache not hitting when it should

**Symptom:** Same spec runs twice, but second run doesn't hit cache

**Causes:**
1. Non-deterministic config (e.g., random seeds, timestamps)
2. Data changed between runs
3. Cache was cleared

**Solution:**
- Ensure all config is deterministic
- Check cache directory exists
- Verify cache keys match between runs

### Issue: Out of memory

**Symptom:** Process killed or OOM error

**Causes:**
1. Dataset too large for available RAM
2. Too many features computed at once
3. Model too large

**Solutions:**
- Reduce time range
- Reduce number of assets
- Use smaller model
- Increase system RAM

### Issue: Slow feature computation

**Symptom:** Feature stage takes very long

**Causes:**
1. Complex formulas (e.g., nested operations)
2. Large window sizes
3. Many assets

**Solutions:**
- Simplify formulas
- Reduce window sizes
- Use JAX JIT compilation (already enabled)
- Check for inefficient formula definitions

### Issue: Model training fails

**Symptom:** Error during model stage

**Causes:**
1. Missing features in dataset
2. All NaN values after preprocessing
3. Model library not installed (LightGBM, XGBoost)

**Solutions:**
- Verify feature names match between preprocessing and model
- Check for sufficient valid data after NaN removal
- Install required model libraries: `pip install lightgbm xgboost`

---

## Extension Points

### Adding Custom Processors

```python
# 1. Create processor class
from src.pipeline.data_handler.processors import Processor

class MyCustomProcessor(Processor):
    def __init__(self, name: str, my_param: float):
        super().__init__(name)
        self.my_param = my_param
    
    def fit(self, ds: xr.Dataset) -> xr.Dataset:
        # Learn parameters from data
        state = compute_state(ds, self.my_param)
        return state
    
    def transform(self, ds: xr.Dataset, state=None) -> xr.Dataset:
        # Apply transformation
        return apply_transform(ds, state, self.my_param)

# 2. Register processor
from src.pipeline.spec import ProcessorRegistry

ProcessorRegistry.register('my_custom', MyCustomProcessor)

# 3. Use in YAML
preprocessing:
  learn:
    - type: "my_custom"
      name: "custom_proc"
      my_param: 0.5
```

### Adding Custom Models

```python
# 1. Create model class
from src.pipeline.spec.model_registry import BaseModel

class MyCustomModel(BaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Train model
        self.params_ = train_model(X, y, **self.params)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Make predictions
        return make_predictions(X, self.params_)

# 2. Register model
from src.pipeline.spec import ModelRegistry

ModelRegistry.register('my_custom_model', MyCustomModel)

# 3. Use in YAML
model:
  type: "my_custom_model"
  params:
    param1: value1
```

---

## Conclusion

The Hindsight Pipeline System provides a complete, production-ready framework for quantitative trading strategy development. Key benefits:

- ✅ **Declarative**: Define pipelines in YAML, not code
- ✅ **Fast**: 6-18x speedup from intelligent caching
- ✅ **Correct**: Content-addressable keys ensure cache correctness
- ✅ **Flexible**: Extensible processors and models
- ✅ **Complete**: End-to-end from data to predictions

For questions or issues, refer to the source code in `src/pipeline/` or run the example in `examples/run_pipeline_example.py`.



---

## 15. Pipeline System Architecture: Deep Dive  


---

## Overview

The pipeline specification system is a **declarative abstraction layer** built on top of Hindsight's existing infrastructure. It provides YAML-based configuration for end-to-end machine learning workflows while leveraging battle-tested components like `DataHandler`, `ModelAdapter`, and `FormulaManager`.

**Key Principle**: The system **wraps, not replaces**. Every component in the spec system delegates to existing, proven infrastructure.

```
┌──────────────────────────────────────────────────────────────┐
│                    YAML Specification                         │
│  Declarative "what" - user intent, configuration             │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                   Pipeline Executor                           │
│  Orchestration "when" - caching, ordering, coordination      │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              Existing Infrastructure                          │
│  Implementation "how" - DataHandler, ModelAdapter, etc.      │
└──────────────────────────────────────────────────────────────┘
```

---

## Design Philosophy

### 1. Abstraction, Not Replacement

The YAML spec system does **not** duplicate or replace existing infrastructure:

- **DataHandler**: Used as-is for all preprocessing
- **ModelAdapter**: Used as-is for all model training
- **FormulaManager**: Used as-is for all feature engineering
- **DataManager**: Used as-is for all data loading

What the spec system **adds**:
- Hierarchical caching (L2-L5)
- Content-addressable cache keys
- Declarative YAML interface
- Automatic dependency tracking
- Cross-pipeline cache reuse

### 2. Separation of Concerns

The system maintains clear boundaries:

| Layer | Responsibility | Example |
|-------|---------------|---------|
| **YAML Spec** | What to compute | "Normalize close and volume" |
| **Executor** | When and where to compute | "Check cache, if miss compute, then cache" |
| **Infrastructure** | How to compute | "Use CSZScore processor on specified vars" |

### 3. Composability

Every stage is independently cacheable and reusable:

```
Pipeline A: Data → Features → Preprocessing → Model A
Pipeline B: Data → Features → Preprocessing → Model B
                    ↑
              Cache reused here
```

---

## Core Components

### Component Hierarchy

```
PipelineExecutor
├── GlobalCacheManager (L2-L5 caching)
├── DataManager (data loading)
├── FormulaManager (feature engineering)
├── ProcessorRegistry (preprocessing)
└── ModelAdapter (model training)
```

### File Structure

```
src/pipeline/
├── spec/
│   ├── schema.py          # YAML schema definitions
│   ├── parser.py          # YAML → Python objects
│   ├── executor.py        # Pipeline orchestration
│   ├── result.py          # Execution results
│   └── processor_registry.py  # Dynamic processor creation
├── cache/
│   ├── manager.py         # Cache operations
│   ├── metadata.py        # JSON-based metadata
│   └── stages.py          # Cache level definitions
├── data_handler/
│   ├── handler.py         # DataHandler (preprocessing)
│   ├── processors.py      # Processor implementations
│   └── config.py          # Handler configuration
└── model/
    ├── adapter.py         # ModelAdapter interface
    └── runner.py          # Walk-forward validation
```

---

## Data Flow: End-to-End

### High-Level Flow

```
YAML Spec → Parser → Executor → Infrastructure → Results
```

### Detailed Flow with Caching

```
1. YAML Parsing
   ├─ Load YAML file
   ├─ Validate schema
   ├─ Create PipelineSpec object
   └─ Return spec

2. Data Loading (L2)
   ├─ Build data config from spec
   ├─ Compute cache key: hash(data_config)
   ├─ Check cache
   │  ├─ HIT: Load from cache
   │  └─ MISS: Load via DataManager
   ├─ Apply data-level processors
   ├─ Save to cache (xr.Dataset)
   └─ Return (dataset, cache_key)

3. Feature Engineering (L3)
   ├─ Topological sort of operations
   ├─ For each operation:
   │  ├─ Compute cache key: hash(formula_config, parent_keys)
   │  ├─ Check cache
   │  │  ├─ HIT: Load from cache
   │  │  └─ MISS: Compute via FormulaManager
   │  ├─ Save to cache (xr.Dataset)
   │  └─ Return (dataset, cache_key)
   └─ Merge all operations

4. Preprocessing (L4)
   ├─ Build preprocessing config
   ├─ Compute cache key: hash(preprocessing_config, L3_key)
   ├─ Check cache
   │  ├─ HIT: Load (dataset, handler) from cache
   │  └─ MISS: Create DataHandler
   │     ├─ Instantiate processors via ProcessorRegistry
   │     ├─ Create HandlerConfig
   │     ├─ Create DataHandler(base=data, config=config)
   │     ├─ handler.build()  # Fit learn processors
   │     ├─ Get handler.view(View.LEARN)  # Apply all transforms
   │     └─ Return (dataset, handler)
   ├─ Save to cache (dataset, handler)
   └─ Return (preprocessed_data, cache_key, handler)

5. Model Training (L5)
   ├─ Build walk-forward configuration (train/infer/step/gap)
   ├─ Compute cache key: hash(model_config + walk_forward, L4_key)
   ├─ Check cache
   │  ├─ HIT: Load (predictions, ModelRunnerResult) from cache
   │  └─ MISS: Train via ModelRunner
   │     ├─ Dynamically import model class and create adapter factory
   │     ├─ Generate SegmentPlan (`make_plan`) from walk-forward config
   │     ├─ Instantiate `ModelRunner` with preprocessing handler + factory
   │     ├─ Runner iterates segments with tqdm progress, fitting fresh adapter per segment
   │     ├─ Predictions aggregated and unstacked back to Dataset
   │     └─ Return (predictions_dataset, runner_result)
   ├─ Save to cache (predictions_dataset, runner_result)
   └─ Return (predictions, cache_key, runner_result)
```

---

## DataHandler Integration

### What is DataHandler?

`DataHandler` is Hindsight's core preprocessing orchestrator. It manages:
- **Processor pipelines**: Shared, learn, and infer stages
- **State management**: Learned parameters (e.g., normalization stats)
- **View management**: Different data views (RAW, LEARN, INFER)
- **Caching**: Intermediate results

### DataHandler Architecture

```
DataHandler
├── base: xr.Dataset (raw input data)
├── config: HandlerConfig
│   ├── shared: List[Processor]  # Transform-only, no fitting
│   ├── learn: List[Processor]   # Fit on train, transform on all
│   ├── infer: List[Processor]   # Transform-only on inference
│   └── mode: PipelineMode (INDEPENDENT or APPEND)
├── cache: Dict
│   ├── shared_view: xr.Dataset
│   ├── learn_view: xr.Dataset
│   ├── infer_view: xr.Dataset
│   └── learn_states: List[Any]  # Fitted processor states
└── methods:
    ├── build()         # Execute pipeline, fit processors
    ├── view(View)      # Get specific view
    └── fetch(View, cols)  # Get specific columns from view
```

### Pipeline Modes

#### INDEPENDENT Mode
```
                    ┌─────────────┐
                    │  Raw Data   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Shared    │  (transform-only)
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼──────┐          ┌──────▼──────┐
       │    Learn    │          │    Infer    │
       │ (fit+trans) │          │  (trans)    │
       └─────────────┘          └─────────────┘
         LEARN View               INFER View
```

#### APPEND Mode
```
       ┌─────────────┐
       │  Raw Data   │
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │   Shared    │
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │    Infer    │
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │    Learn    │  (sees infer outputs)
       └─────────────┘
         LEARN View
```

### How Spec System Uses DataHandler

#### Step 1: Parse YAML Preprocessing Spec

```yaml
preprocessing:
  mode: "independent"
  shared:
    - type: "per_asset_ffill"
      name: "forward_fill"
      vars: ["close", "volume"]
  learn:
    - type: "cs_zscore"
      name: "normalizer"
      vars: ["close", "volume", "sma_ww20"]
      out_suffix: "_norm"
  infer: []
```

#### Step 2: Instantiate Processors

```python
# In executor._compute_preprocessing()

from src.pipeline.spec.processor_registry import ProcessorRegistry

# ProcessorRegistry dynamically creates processor instances
shared_processors = [
    ProcessorRegistry.create_processor({
        'type': 'per_asset_ffill',
        'name': 'forward_fill',
        'vars': ['close', 'volume']
    })
]
# Returns: [PerAssetFFill(name='forward_fill', vars=['close', 'volume'])]

learn_processors = [
    ProcessorRegistry.create_processor({
        'type': 'cs_zscore',
        'name': 'normalizer',
        'vars': ['close', 'volume', 'sma_ww20'],
        'out_suffix': '_norm'
    })
]
# Returns: [CSZScore(name='normalizer', vars=[...], out_suffix='_norm')]
```

#### Step 3: Create HandlerConfig

```python
from src.pipeline.data_handler import HandlerConfig, PipelineMode

# Map YAML mode string to PipelineMode enum
mode_map = {
    'independent': PipelineMode.INDEPENDENT,
    'append': PipelineMode.APPEND,
}
mode = mode_map['independent']  # PipelineMode.INDEPENDENT

# Create config
handler_config = HandlerConfig(
    shared=shared_processors,     # [PerAssetFFill(...)]
    learn=learn_processors,       # [CSZScore(...)]
    infer=[],                     # []
    mode=mode,                    # PipelineMode.INDEPENDENT
    feature_cols=[],              # Not used in preprocessing-only mode
    label_cols=[]                 # Not used in preprocessing-only mode
)
```

#### Step 4: Create and Build DataHandler

```python
from src.pipeline.data_handler import DataHandler

# Create handler with input data and config
handler = DataHandler(base=features_data, config=handler_config)

# Build the pipeline
handler.build()
```

**What happens in `handler.build()`:**

1. **Apply shared processors** (transform-only):
   ```python
   shared_view = base_data
   for processor in shared_processors:
       shared_view = processor.transform(shared_view)
   # Result: close and volume are forward-filled
   ```

2. **Apply learn processors** (fit + transform):
   ```python
   learn_view = shared_view
   learn_states = []
   for processor in learn_processors:
       # Fit on the data
       processor.fit(learn_view)
       # Transform the data
       learn_view = processor.transform(learn_view)
       # Store learned state
       learn_states.append(processor.get_state())
   # Result: close_norm, volume_norm, sma_ww20_norm created
   # States: {mean: [...], std: [...]} for each variable
   ```

3. **Cache results**:
   ```python
   handler.cache['shared_view'] = shared_view
   handler.cache['learn_view'] = learn_view
   handler.cache['learn_states'] = learn_states
   ```

#### Step 5: Get Preprocessed Data

```python
from src.pipeline.data_handler import View

# Get the LEARN view (shared + learn transformations applied)
preprocessed_data = handler.view(View.LEARN)

# This dataset now contains:
# - Original variables: open, high, low, close, volume, etc.
# - Forward-filled: close, volume (in-place)
# - Normalized: close_norm, volume_norm, sma_ww20_norm (new variables)
```

#### Step 6: Cache Handler and Data

```python
# The executor caches BOTH the dataset and the handler
cache_manager.save(
    stage=CacheStage.L4_PREPROCESSED,
    key=cache_key,
    data=preprocessed_data,      # xr.Dataset
    state=handler                # DataHandler with learned states
)

# On cache hit, both are restored:
preprocessed_data, handler = cache_manager.load(...)
```

### Why Cache the Handler?

The `DataHandler` contains critical information:

1. **Learned States**: Normalization parameters, fill strategies, etc.
   ```python
   handler.cache['learn_states'][0]  # CSZScore state
   # {'mean': array([...]), 'std': array([...])}
   ```

2. **Pipeline Configuration**: Exact processors and order
   ```python
   handler.config.shared  # [PerAssetFFill(...)]
   handler.config.learn   # [CSZScore(...)]
   ```

3. **Inference Capability**: Can transform new data consistently
   ```python
   # Later, on new data:
   new_preprocessed = handler.view(View.INFER)
   # Uses cached learn_states, no refitting
   ```

---

## ModelAdapter Integration

### What is ModelAdapter?

`ModelAdapter` is Hindsight's interface for machine learning models. It provides:
- **Uniform API**: Same interface for sklearn, LightGBM, PyTorch, etc.
- **Data handling**: Automatic stacking/unstacking of xarray data
- **Prediction wrapping**: Converts model outputs back to xarray
- **State management**: Fitted model parameters

### ModelAdapter Architecture

```
ModelAdapter (Abstract Base Class)
├── model: Any (sklearn, lightgbm, pytorch, etc.)
├── handler: DataHandler (for data stacking)
├── output_var: str (prediction variable name)
└── methods:
    ├── fit(ds, features, label)      # Train model
    ├── predict(ds, features)         # Generate predictions
    ├── stack_features(ds, features)  # xarray → numpy
    └── unstack_predictions(preds)    # numpy → xarray

SklearnAdapter (Concrete Implementation)
├── Inherits from ModelAdapter
├── Wraps any sklearn-compatible model
├── Handles 2D numpy arrays
└── Supports classification and regression
```

### How Spec System Uses ModelAdapter

#### Step 1: Parse YAML Model Spec

```yaml
model:
  adapter: "sklearn"
  type: "RandomForestRegressor"
  params:
    n_estimators: 50
    max_depth: 14
    random_state: 42
  features:
    - "close_norm"
    - "volume_norm"
    - "sma_ww20_norm"
    - "sma_ww50_norm"
    - "ema_ww12_norm"
    - "ema_ww26_norm"
  target: "close"
  adapter_params:
    output_var: "close_pred"
    use_proba: false
```

#### Step 2: Dynamically Import Model Class

```python
# In executor._compute_model()

import importlib

adapter_type = "sklearn"  # From spec
model_type = "RandomForestRegressor"  # From spec

# Try common sklearn modules
sklearn_modules = [
    'sklearn.linear_model',
    'sklearn.ensemble',
    'sklearn.tree',
    'sklearn.svm',
    'sklearn.neighbors',
]

model_class = None
for module_name in sklearn_modules:
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, model_type):
            model_class = getattr(module, model_type)
            break
    except ImportError:
        continue

# Result: model_class = sklearn.ensemble.RandomForestRegressor
```

#### Step 3: Instantiate Model

```python
# Get params from spec
params = {
    'n_estimators': 50,
    'max_depth': 14,
    'random_state': 42
}

# Create model instance
model = model_class(**params)
# Result: RandomForestRegressor(n_estimators=50, max_depth=14, ...)
```

#### Step 4: Create DataHandler for Adapter

```python
from src.pipeline.data_handler import DataHandler, HandlerConfig, PipelineMode

# The adapter needs a DataHandler for consistent data stacking
# We create a minimal one (no processors, just for stacking)
handler_config = HandlerConfig(
    shared=[],
    learn=[],
    infer=[],
    mode=PipelineMode.INDEPENDENT,
    feature_cols=features,  # ['close_norm', 'volume_norm', ...]
    label_cols=[target]     # ['close']
)

handler = DataHandler(base=preprocessed_data, config=handler_config)
handler.build()
```

**Why does the adapter need a DataHandler?**

The adapter uses the handler's stacking logic to convert xarray → numpy:

```python
# xarray.Dataset with dimensions (time, asset)
# → numpy array with shape (n_samples, n_features)

# The handler knows how to:
# 1. Extract specified variables
# 2. Stack time and asset dimensions
# 3. Handle NaNs consistently
# 4. Unstack predictions back to xarray
```

#### Step 5: Create ModelAdapter

```python
from src.pipeline.model.adapter import SklearnAdapter

# Get adapter params
adapter_params = {
    'output_var': 'close_pred',
    'use_proba': False
}

# Create adapter
adapter = SklearnAdapter(
    model=model,                      # RandomForestRegressor instance
    handler=handler,                  # DataHandler for stacking
    output_var='close_pred',          # Name for predictions
    use_proba=False                   # Regression, not classification
)
```

#### Step 6: Fit Model

```python
features = ['close_norm', 'volume_norm', 'sma_ww20_norm', ...]
target = 'close'

# Fit the model
adapter.fit(
    ds=preprocessed_data,  # xr.Dataset
    features=features,     # List of feature variable names
    label=target           # Target variable name
)
```

**What happens in `adapter.fit()`:**

1. **Stack features**:
   ```python
   # Extract feature variables from dataset
   feature_arrays = [preprocessed_data[var] for var in features]
   
   # Stack using handler
   X = handler.stack_features(preprocessed_data, features)
   # Shape: (n_samples, n_features)
   # Example: (2260928, 6) for 253 assets × 8928 timesteps
   ```

2. **Stack target**:
   ```python
   y = handler.stack_features(preprocessed_data, [target])
   # Shape: (n_samples, 1)
   ```

3. **Remove NaNs**:
   ```python
   # Find rows with no NaNs
   valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
   X_clean = X[valid_mask]
   y_clean = y[valid_mask].ravel()
   ```

4. **Fit model**:
   ```python
   model.fit(X_clean, y_clean)
   # RandomForestRegressor is now fitted
   ```

#### Step 7: Generate Predictions

```python
# Generate predictions
pred_da = adapter.predict(
    ds=preprocessed_data,
    features=features
)
```

**What happens in `adapter.predict()`:**

1. **Stack features** (same as fit):
   ```python
   X = handler.stack_features(preprocessed_data, features)
   ```

2. **Predict**:
   ```python
   predictions = model.predict(X)
   # Shape: (n_samples,)
   ```

3. **Unstack to xarray**:
   ```python
   # Convert back to xarray.DataArray
   pred_da = handler.unstack_predictions(
       predictions,
       template=preprocessed_data[target],  # Use target shape as template
       output_var='close_pred'
   )
   # Result: xr.DataArray with dims (time, asset)
   ```

4. **Return**:
   ```python
   return pred_da  # xr.DataArray named 'close_pred'
   ```

#### Step 8: Add Predictions to Dataset

```python
# Create output dataset with predictions
output_ds = preprocessed_data.copy()
output_ds['close_pred'] = pred_da

# Now output_ds contains:
# - All original variables
# - All preprocessed variables (*_norm)
# - Predictions (close_pred)
```

#### Step 9: Cache Adapter and Predictions

```python
# Cache both predictions and fitted adapter
cache_manager.save(
    stage=CacheStage.L5_MODEL,
    key=cache_key,
    data=output_ds,      # xr.Dataset with predictions
    state=adapter        # Fitted SklearnAdapter
)

# On cache hit, both are restored:
predictions, adapter = cache_manager.load(...)

# The adapter can be used for inference on new data:
new_predictions = adapter.predict(new_data, features)
```

### ModelRunner Execution (Current)

While the above steps describe the adapter mechanics, the **pipeline executor now wraps them inside `ModelRunner`**:

1. **Walk-forward plan**: build `SegmentConfig` from `model.walk_forward` (train/infer/span/step/gap/start/end) and generate a `SegmentPlan` with `make_plan`.
2. **Factory per segment**: create a factory that instantiates a fresh adapter (and underlying sklearn model) for each segment. The preprocessing `DataHandler`—with learned states cached at L4—is reused so stacking/unstacking remain consistent.
3. **Progress reporting**: `ModelRunner.run()` iterates `plan` with `tqdm(desc="Walk-forward segments", unit="segment")`, printing live segment progress.
4. **Segment loop**:
   - fit learn processors on train slice (via handler),
   - transform inference slice using learned states,
   - fit adapter using factory-provided model,
   - predict on inference slice,
   - scatter predictions back into global buffer according to `runner_params.overlap_policy` (`"last"` by default).
5. **Results**: returns `ModelRunnerResult(pred_ds, segment_states, attrs)`, which the cache stores alongside the predictions dataset. On cache hit both predictions and runner metadata are restored.

This keeps the YAML spec thin (declarative) while respecting the library’s existing walk-forward architecture (`example.py`).

---

## Caching System

### Cache Architecture

```
GlobalCacheManager
├── cache_dir: Path (e.g., /home/user/data/hindsight_cache/)
├── metadata_manager: MetadataManager
└── methods:
    ├── get_or_compute(stage, config, parent_keys, compute_fn)
    ├── get_or_compute_with_state(...)  # For handlers/adapters
    ├── compute_key(stage, config, parent_keys)
    └── get_stats()

Cache Directory Structure:
cache_dir/
├── l2_post/
│   ├── a40d6ea463d5443f.nc          # Dataset
│   ├── a40d6ea463d5443f.attrs.pkl   # Non-NetCDF attrs
│   └── a40d6ea463d5443f.meta.json   # Metadata
├── l3_features/
│   ├── 26d8ca3930ff1665.nc
│   ├── 26d8ca3930ff1665.attrs.pkl
│   └── 26d8ca3930ff1665.meta.json
├── l4_prep/
│   ├── d167e13d1b36a8f2.pkl         # (dataset, DataHandler)
│   └── d167e13d1b36a8f2.meta.json
└── l5_model/
    ├── 8b08547aa0704c1e.pkl         # (predictions, ModelRunnerResult)
    └── 8b08547aa0704c1e.meta.json
```

- **Attribute preservation:** when datasets carry objects NetCDF cannot serialize (e.g., `coords['time'].attrs['indexes']` holding a `TimeSeriesIndex`), the cache manager strips them before writing the `.nc`, stores them in the companion `.attrs.pkl`, and automatically restores them on load. Callers see the exact original dataset structure, including dt-accessor metadata.

### Content-Addressable Keys

Cache keys are computed as:

```python
def compute_key(stage, config, parent_keys):
    """
    Compute content-addressable cache key.
    
    Args:
        stage: CacheStage enum (L2, L3, L4, L5)
        config: Dict of configuration for this stage
        parent_keys: List of cache keys from dependencies
        
    Returns:
        str: 16-character hex hash
    """
    # Serialize config to stable JSON
    config_json = json.dumps(config, sort_keys=True)
    
    # Combine with parent keys
    combined = f"{stage.value}:{config_json}:{':'.join(parent_keys)}"
    
    # Hash
    hash_obj = hashlib.sha256(combined.encode())
    return hash_obj.hexdigest()[:16]
```

**Example:**

```python
# L2 (Data Loading)
L2_key = hash("l2_post:{'provider':'crypto','dataset':'spot/binance',...}:")
# Result: "a40d6ea463d5443f"

# L3 (Features)
L3_key = hash("l3_features:{'formulas':{'sma':[{...}],...}}:a40d6ea463d5443f")
# Result: "26d8ca3930ff1665"

# L4 (Preprocessing)
L4_key = hash("l4_prep:{'shared':[...],'learn':[...]}:26d8ca3930ff1665")
# Result: "d167e13d1b36a8f2"

# L5 (Model)
L5_key = hash("l5_model:{'type':'RandomForest',...}:d167e13d1b36a8f2")
# Result: "8b08547aa0704c1e"
```

### Cache Reuse Logic

```python
def get_or_compute(stage, config, parent_keys, compute_fn):
    """
    Get from cache or compute if missing.
    
    Flow:
    1. Compute cache key from config + parent_keys
    2. Check if key exists in cache
    3. If HIT: Load and return
    4. If MISS: Call compute_fn(), save, return
    """
    # Compute key
    key = compute_key(stage, config, parent_keys)
    
    # Check cache
    if exists(cache_dir / stage.value / f"{key}.nc"):
        print(f"Cache hit: {stage.value}/{key[:16]}...")
        data = load_from_cache(key)
        metadata_manager.update_access(key)  # Update access time
        return data, key
    
    # Cache miss - compute
    print(f"Cache miss: {stage.value}/{key[:16]}... - computing...")
    data = compute_fn()
    
    # Save to cache
    save_to_cache(key, data)
    metadata_manager.save(key, {
        'stage': stage.value,
        'config': config,
        'parent_keys': parent_keys,
        'created_at': datetime.now(),
        'size_bytes': get_size(data)
    })
    
    return data, key
```

### State Caching (L4 and L5)

For stages that produce both data and state (handlers, adapters):

```python
def get_or_compute_with_state(stage, config, parent_keys, compute_fn):
    """
    Get from cache or compute, handling state objects.
    
    Returns:
        Tuple[xr.Dataset, Any, str]: (data, state, cache_key)
    """
    key = compute_key(stage, config, parent_keys)
    
    # Check cache
    data_path = cache_dir / stage.value / f"{key}.nc"
    state_path = cache_dir / stage.value / f"{key}_state.pkl"
    
    if data_path.exists() and state_path.exists():
        print(f"Cache hit: {stage.value}/{key[:16]}...")
        data = xr.open_dataset(data_path)
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        return data, state, key
    
    # Cache miss - compute
    print(f"Cache miss: {stage.value}/{key[:16]}... - computing...")
    data, state = compute_fn()  # Returns (dataset, handler/adapter)
    
    # Save both
    data.to_netcdf(data_path)
    with open(state_path, 'wb') as f:
        pickle.dump(state, f)
    
    return data, state, key
```

### Cache Invalidation

Caches are automatically invalidated when:

1. **Config changes**: Different parameters → different key
2. **Parent changes**: Different input data → different key
3. **Code changes**: Manual cache clear required

```python
# Example: Changing a parameter invalidates downstream caches

# Original
config_A = {'type': 'cs_zscore', 'vars': ['close']}
L4_key_A = hash(config_A + L3_key)  # "d167e13d1b36a8f2"

# Modified
config_B = {'type': 'cs_zscore', 'vars': ['close', 'volume']}
L4_key_B = hash(config_B + L3_key)  # "f8a2c9d4e1b7063a" (different!)

# L5 caches that depend on L4_key_A are not reused
```

---

## Processor Registry

### What is ProcessorRegistry?

A factory for dynamically creating processor instances from YAML configuration.

### Implementation

```python
# src/pipeline/spec/processor_registry.py

class ProcessorRegistry:
    """
    Factory for creating processor instances from configuration.
    """
    
    # Map YAML type names to processor classes
    _registry = {
        'cs_zscore': CSZScore,
        'per_asset_ffill': PerAssetFFill,
        'formula_eval': FormulaEval,
        # Add more as needed
    }
    
    @classmethod
    def create_processor(cls, config: Dict[str, Any]) -> Processor:
        """
        Create a processor instance from configuration.
        
        Args:
            config: Dict with 'type' and processor-specific params
            
        Returns:
            Processor instance
            
        Example:
            config = {
                'type': 'cs_zscore',
                'name': 'normalizer',
                'vars': ['close', 'volume'],
                'out_suffix': '_norm'
            }
            processor = ProcessorRegistry.create_processor(config)
            # Returns: CSZScore(name='normalizer', vars=[...], ...)
        """
        processor_type = config.get('type')
        if processor_type not in cls._registry:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        # Get processor class
        processor_class = cls._registry[processor_type]
        
        # Extract params (everything except 'type')
        params = {k: v for k, v in config.items() if k != 'type'}
        
        # Instantiate
        return processor_class(**params)
```

### Usage in Executor

```python
# In executor._compute_preprocessing()

from src.pipeline.spec.processor_registry import ProcessorRegistry

# YAML config
preprocessing_spec = {
    'shared': [
        {'type': 'per_asset_ffill', 'name': 'ffill', 'vars': ['close']}
    ],
    'learn': [
        {'type': 'cs_zscore', 'name': 'norm', 'vars': ['close'], 'out_suffix': '_norm'}
    ]
}

# Create processors
shared_processors = [
    ProcessorRegistry.create_processor(cfg)
    for cfg in preprocessing_spec['shared']
]
# Result: [PerAssetFFill(name='ffill', vars=['close'])]

learn_processors = [
    ProcessorRegistry.create_processor(cfg)
    for cfg in preprocessing_spec['learn']
]
# Result: [CSZScore(name='norm', vars=['close'], out_suffix='_norm')]
```

---

## Under the Hood: Detailed Mechanics

### Complete Execution Trace

Let's trace a complete pipeline execution with detailed internal operations.

#### YAML Spec

```yaml
spec_version: "1.0"
name: "example_pipeline"

time_range:
  start: "2023-01-01"
  end: "2023-12-31"

data:
  crypto_prices:
    provider: "crypto"
    dataset: "spot/binance"
    frequency: "H"

features:
  operations:
    - name: "indicators"
      formulas:
        sma: [{window: 20}]
        rsi: [{window: 14}]

preprocessing:
  mode: "independent"
  shared:
    - type: "per_asset_ffill"
      vars: ["close"]
  learn:
    - type: "cs_zscore"
      vars: ["close", "sma_ww20"]
      out_suffix: "_norm"

model:
  adapter: "sklearn"
  type: "LinearRegression"
  params: {fit_intercept: true}
  features: ["close_norm", "sma_ww20_norm"]
  target: "close"
```

#### Execution Trace

```python
# 1. PARSE YAML
# --------------
spec = SpecParser.load_from_yaml("example_pipeline.yaml")
# Result: PipelineSpec object with all fields populated

# 2. CREATE EXECUTOR
# ------------------
executor = PipelineExecutor(
    cache_manager=GlobalCacheManager(cache_dir="/path/to/cache"),
    data_manager=DataManager(),
    formula_manager=FormulaManager()
)

# 3. EXECUTE PIPELINE
# -------------------
result = executor.execute(spec)

# 3.1. DATA LOADING (L2)
# ----------------------
# Build config
data_config = {
    'provider': 'crypto',
    'dataset': 'spot/binance',
    'frequency': 'H',
    'time_range': {'start': '2023-01-01', 'end': '2023-12-31'}
}

# Compute cache key
L2_key = cache_manager.compute_key(
    stage=CacheStage.L2_POSTPROCESSED,
    config=data_config,
    parent_keys=[]
)
# Result: "a40d6ea463d5443f"

# Check cache
if not cache_manager.exists(L2_key):
    # MISS - Load data
    raw_data = data_manager.load_builtin(
        provider='crypto',
        dataset='spot/binance',
        start='2023-01-01',
        end='2023-12-31'
    )
    # Result: xr.Dataset with dims (year, month, day, hour, asset)
    #         Variables: open, high, low, close, volume
    #         Shape: (1, 12, 31, 24, 253)
    
    # Save to cache
    cache_manager.save(L2_key, raw_data)

# Load from cache
dataset = cache_manager.load(L2_key)

# 3.2. FEATURE ENGINEERING (L3)
# -----------------------------
# For operation "indicators"
formula_config = {
    'name': 'indicators',
    'formulas': {
        'sma': [{'window': 20}],
        'rsi': [{'window': 14}]
    }
}

# Compute cache key
L3_key = cache_manager.compute_key(
    stage=CacheStage.L3_FEATURES,
    config=formula_config,
    parent_keys=[L2_key]
)
# Result: "26d8ca3930ff1665"

# Check cache
if not cache_manager.exists(L3_key):
    # MISS - Compute features
    features_data = formula_manager.evaluate_bulk(
        formula_names={'sma': [{'window': 20}], 'rsi': [{'window': 14}]},
        context={'_dataset': dataset}
    )
    # Internal operations:
    # 1. Parse formula "sma" with window=20
    # 2. Call sma(dataset, window=20)
    #    - dataset.dt.rolling(dim='time', window=20).reduce(mean)
    #    - Compute mask on-demand from dataset
    #    - Apply rolling mean with mask
    #    - Return xr.DataArray named "sma_ww20"
    # 3. Parse formula "rsi" with window=14
    # 4. Call rsi(dataset, window=14)
    #    - Compute gain and loss
    #    - Apply EMA
    #    - Return xr.DataArray named "rsi"
    # 5. Merge into dataset
    
    # Result: dataset with new variables sma_ww20, rsi
    
    # Save to cache
    cache_manager.save(L3_key, features_data)

# Load from cache
features_data = cache_manager.load(L3_key)

# 3.3. PREPROCESSING (L4)
# -----------------------
preprocessing_config = {
    'mode': 'independent',
    'shared': [{'type': 'per_asset_ffill', 'vars': ['close']}],
    'learn': [{'type': 'cs_zscore', 'vars': ['close', 'sma_ww20'], 'out_suffix': '_norm'}]
}

# Compute cache key
L4_key = cache_manager.compute_key(
    stage=CacheStage.L4_PREPROCESSED,
    config=preprocessing_config,
    parent_keys=[L3_key]
)
# Result: "d167e13d1b36a8f2"

# Check cache
if not cache_manager.exists(L4_key):
    # MISS - Create DataHandler
    
    # Step 1: Instantiate processors
    shared_processors = [
        PerAssetFFill(name='ffill', vars=['close'])
    ]
    learn_processors = [
        CSZScore(name='norm', vars=['close', 'sma_ww20'], out_suffix='_norm')
    ]
    
    # Step 2: Create config
    handler_config = HandlerConfig(
        shared=shared_processors,
        learn=learn_processors,
        infer=[],
        mode=PipelineMode.INDEPENDENT
    )
    
    # Step 3: Create handler
    handler = DataHandler(base=features_data, config=handler_config)
    
    # Step 4: Build pipeline
    handler.build()
    # Internal operations:
    # 1. Apply shared processors:
    #    - PerAssetFFill.transform(features_data)
    #    - For each asset, forward-fill 'close' variable
    #    - Result: shared_view with filled close values
    # 2. Apply learn processors:
    #    - CSZScore.fit(shared_view)
    #      - Compute mean and std for 'close' and 'sma_ww20'
    #      - Across assets (cross-sectional)
    #      - Store in processor state
    #    - CSZScore.transform(shared_view)
    #      - Create 'close_norm' = (close - mean) / std
    #      - Create 'sma_ww20_norm' = (sma_ww20 - mean) / std
    #    - Result: learn_view with normalized variables
    # 3. Cache views:
    #    - handler.cache['shared_view'] = shared_view
    #    - handler.cache['learn_view'] = learn_view
    #    - handler.cache['learn_states'] = [cszscore_state]
    
    # Step 5: Get preprocessed data
    preprocessed_data = handler.view(View.LEARN)
    # Result: dataset with close_norm, sma_ww20_norm
    
    # Save to cache (data + handler)
    cache_manager.save_with_state(L4_key, preprocessed_data, handler)

# Load from cache
preprocessed_data, handler = cache_manager.load_with_state(L4_key)

# 3.4. MODEL TRAINING (L5)
# ------------------------
model_config = {
    'adapter': 'sklearn',
    'type': 'LinearRegression',
    'params': {'fit_intercept': True},
    'features': ['close_norm', 'sma_ww20_norm'],
    'target': 'close'
}

# Compute cache key
L5_key = cache_manager.compute_key(
    stage=CacheStage.L5_MODEL,
    config=model_config,
    parent_keys=[L4_key]
)
# Result: "8b08547aa0704c1e"

# Check cache
if not cache_manager.exists(L5_key):
    # MISS - Train model
    
    # Step 1: Import model class
    from sklearn.linear_model import LinearRegression
    
    # Step 2: Instantiate model
    model = LinearRegression(fit_intercept=True)
    
    # Step 3: Create handler for adapter
    adapter_handler = DataHandler(
        base=preprocessed_data,
        config=HandlerConfig(
            shared=[],
            learn=[],
            infer=[],
            mode=PipelineMode.INDEPENDENT,
            feature_cols=['close_norm', 'sma_ww20_norm'],
            label_cols=['close']
        )
    )
    adapter_handler.build()
    
    # Step 4: Create adapter
    adapter = SklearnAdapter(
        model=model,
        handler=adapter_handler,
        output_var='close_pred'
    )
    
    # Step 5: Fit model
    adapter.fit(
        ds=preprocessed_data,
        features=['close_norm', 'sma_ww20_norm'],
        label='close'
    )
    # Internal operations:
    # 1. Stack features:
    #    X = adapter_handler.stack_features(preprocessed_data, features)
    #    - Extract close_norm and sma_ww20_norm
    #    - Stack (year, month, day, hour, asset) → (samples, features)
    #    - Shape: (2260928, 2) for 253 assets × 8928 timesteps
    # 2. Stack target:
    #    y = adapter_handler.stack_features(preprocessed_data, ['close'])
    #    - Shape: (2260928, 1)
    # 3. Remove NaNs:
    #    valid = ~(np.isnan(X).any(1) | np.isnan(y).any(1))
    #    X_clean = X[valid]  # Shape: (~2000000, 2)
    #    y_clean = y[valid].ravel()
    # 4. Fit:
    #    model.fit(X_clean, y_clean)
    #    - LinearRegression learns coefficients
    #    - coef_ = [w1, w2], intercept_ = b
    
    # Step 6: Predict
    predictions = adapter.predict(
        ds=preprocessed_data,
        features=['close_norm', 'sma_ww20_norm']
    )
    # Internal operations:
    # 1. Stack features (same as fit)
    # 2. Predict:
    #    y_pred = model.predict(X)  # Shape: (2260928,)
    # 3. Unstack:
    #    pred_da = adapter_handler.unstack_predictions(
    #        y_pred,
    #        template=preprocessed_data['close']
    #    )
    #    - Reshape (2260928,) → (1, 12, 31, 24, 253)
    #    - Create xr.DataArray with same coords as 'close'
    #    - Name it 'close_pred'
    
    # Step 7: Add to dataset
    output_ds = preprocessed_data.copy()
    output_ds['close_pred'] = predictions
    
    # Save to cache (predictions + adapter)
    cache_manager.save_with_state(L5_key, output_ds, adapter)

# Load from cache
predictions, runner_result = cache_manager.load_with_state(L5_key)

# 4. RETURN RESULTS
# -----------------
result = ExecutionResult(
    spec=spec,
    raw_data=dataset,
    features_data=features_data,
    preprocessed_data=preprocessed_data,
    model_predictions=predictions,
    fitted_model=runner_result,      # Legacy alias
    model_runner_result=runner_result,
    learned_states=handler,
    cache_keys={
        'data': L2_key,
        'features': L3_key,
        'preprocessing': L4_key,
        'model': L5_key
    },
    execution_time=elapsed_time
)

return result
```

---

## Future Extensions

### 1. Walk-Forward Validation

Integration with `ModelRunner` for time-series cross-validation:

```yaml
model:
  adapter: "sklearn"
  type: "RandomForestRegressor"
  walk_forward:
    train_span_hours: 720   # 30 days
    infer_span_hours: 24    # 1 day
    step_hours: 24          # 1 day step
    min_train_hours: 168   # 7 days minimum
```

```python
# Future implementation in executor
from src.pipeline.model.runner import ModelRunner, make_plan

# Create segment plan
plan = make_plan(
    train_span_hours=720,
    infer_span_hours=24,
    step_hours=24,
    ds_for_bounds=preprocessed_data
)

# Model factory
def create_adapter():
    model = create_model_from_spec(spec.model)
    return SklearnAdapter(model=model, handler=handler, ...)

# Run walk-forward
runner = ModelRunner(
    handler=preprocessing_handler,
    plan=plan,
    model_factory=create_adapter,
    feature_cols=spec.model.features,
    label_col=spec.model.target
)

results = runner.run()
# Returns: List of (train_segment, infer_segment, predictions, metrics)
```

### 2. Additional Model Adapters

#### LightGBM Adapter

```python
# src/pipeline/model/adapter.py

class LightGBMAdapter(ModelAdapter):
    """Adapter for LightGBM models."""
    
    def fit(self, ds, features, label):
        import lightgbm as lgb
        
        X, y = self.stack_data(ds, features, label)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Train
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.get('num_boost_round', 100)
        )
    
    def predict(self, ds, features):
        X = self.stack_features(ds, features)
        predictions = self.model.predict(X)
        return self.unstack_predictions(predictions, ds, features[0])
```

```yaml
# YAML usage
model:
  adapter: "lightgbm"
  type: "LGBMRegressor"
  params:
    num_leaves: 31
    learning_rate: 0.05
    num_boost_round: 100
```

#### PyTorch Adapter

```python
# src/pipeline/model/adapter.py

class PyTorchAdapter(ModelAdapter):
    """Adapter for PyTorch models."""
    
    def __init__(self, model_class, model_params, training_params, **kwargs):
        super().__init__(**kwargs)
        self.model_class = model_class
        self.model_params = model_params
        self.training_params = training_params
    
    def fit(self, ds, features, label):
        import torch
        import torch.nn as nn
        
        X, y = self.stack_data(ds, features, label)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create model
        self.model = self.model_class(**self.model_params)
        
        # Training loop
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_params['learning_rate']
        )
        criterion = nn.MSELoss()
        
        for epoch in range(self.training_params['epochs']):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
    
    def predict(self, ds, features):
        import torch
        
        X = self.stack_features(ds, features)
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        
        return self.unstack_predictions(predictions, ds, features[0])
```

```yaml
# YAML usage
model:
  adapter: "pytorch"
  type: "LSTM"
  params:
    input_size: 6
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
  training_params:
    learning_rate: 0.001
    epochs: 100
    batch_size: 32
```

### 3. Ensemble Models

```yaml
model:
  adapter: "ensemble"
  type: "VotingRegressor"
  models:
    - adapter: "sklearn"
      type: "RandomForestRegressor"
      params: {n_estimators: 50}
      weight: 0.4
    - adapter: "lightgbm"
      type: "LGBMRegressor"
      params: {num_leaves: 31}
      weight: 0.3
    - adapter: "sklearn"
      type: "LinearRegression"
      params: {}
      weight: 0.3
  features: ["close_norm", "volume_norm"]
  target: "close"
```

### 4. Hyperparameter Optimization

```yaml
model:
  adapter: "sklearn"
  type: "RandomForestRegressor"
  hyperopt:
    method: "optuna"
    n_trials: 100
    params:
      n_estimators: [50, 100, 200, 500]
      max_depth: [5, 10, 15, 20]
      min_samples_split: [2, 5, 10]
    metric: "mse"
    cv_folds: 5
```

### 5. Feature Selection

```yaml
preprocessing:
  mode: "independent"
  learn:
    - type: "feature_selector"
      method: "mutual_info"
      n_features: 10
      vars: ["close_norm", "volume_norm", "sma_*", "ema_*"]
```

---

## Summary

The pipeline system is a **thin orchestration layer** that:

✅ **Wraps existing infrastructure**
- DataHandler for preprocessing
- ModelAdapter for models
- FormulaManager for features

✅ **Adds hierarchical caching**
- L2: Data loading
- L3: Feature engineering
- L4: Preprocessing
- L5: Model training

✅ **Provides declarative interface**
- YAML-based configuration
- Content-addressable caching
- Automatic dependency tracking

✅ **Maintains compatibility**
- Uses existing processor interface
- Uses existing adapter interface
- Integrates with walk-forward validation

✅ **Enables extensibility**
- Easy to add new adapters
- Easy to add new processors
- Easy to add new features

The system does **not** replace or duplicate existing infrastructure - it orchestrates and enhances it with caching and declarative configuration.


---

## 16. Dataset Merger: Comprehensive Documentation


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



---

# Part V: Appendix

## 17. Building the Documentation

### Documentation Structure

#### Getting Started Guide
- **Overview**: Framework architecture and core principles
- **Data Loading**: Using DataManager to load financial datasets
- **Data Handler**: Configuring and using the data processing pipeline
- **Feature Engineering**: Working with processors and formula evaluation
- **Walk-Forward Analysis**: Temporal segmentation and robust backtesting
- **Model Integration**: Integrating ML models with the pipeline
- **Execution and Analysis**: Running complete workflows and analyzing results

#### API Reference
- **Pipeline API**: Main entry point and core classes
- **Data Handler API**: Data processing pipeline components
- **Walk-Forward API**: Temporal segmentation and execution
- **Model API**: Model integration and adapters

#### Examples
- **Complete Workflow**: End-to-end example following `example.py`

### Key Features Highlighted

The documentation focuses on the high-level abstractions and public API that users need to understand:

- **Separation of "How" and "When"**: Clear distinction between data processing and temporal logic
- **Three-Stage Processing**: Shared, learn, and infer processor stages
- **Temporal Validity**: Prevention of lookahead bias through proper state management
- **Walk-Forward Analysis**: Robust backtesting with configurable temporal segments
- **Model Integration**: Seamless integration of ML models with the pipeline

### Areas Not Covered (Marked as TBA)

The documentation intentionally avoids implementation details that are "under the hood":
- AST system internals and formula evaluation implementation
- Data module internals (loaders, processors, core operations)
- Backtester implementation details
- Rolling operations and masking specifics

These topics may be covered in future detailed developer documentation.

---

**End of Documentation**

For questions or issues, refer to the source code in `src/pipeline/` or run the example in `examples/run_pipeline_example.py`.

