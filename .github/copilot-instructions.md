# Hindsight Copilot Instructions

## 1. Overview

Hindsight is a Python-based backtesting library for financial analysis. Its architecture is designed for modularity, statistical robustness (preventing lookahead bias), and performance. The core of the library is built around `xarray` for efficient multi-dimensional data manipulation.

The project's main goal is to provide a structured environment for financial research, from data processing and feature engineering to walk-forward backtesting and model integration.

## 2. Core Architecture

The architecture separates data processing ("how") from temporal analysis ("when"). The main components are in `src/`.

### The Pipeline (`src/pipeline`)

This is the orchestration engine. It connects data handling, temporal segmentation, and model execution.

1.  **`DataHandler` (`src/pipeline/data_handler`)**: Manages all data processing. It uses a crucial three-stage pipeline to prevent lookahead bias:
    *   **Shared Processors**: Applied to the entire dataset.
    *   **Learn Processors**: Fitted only on training data (`fit_transform`).
    *   **Infer Processors**: Applied to test data using parameters from the learn stage (`transform`).
    *   This pattern is enforced by the `ProcessorContract` ABC, similar to `scikit-learn`. Key implementations are in `src/pipeline/data_handler/processors.py`.

2.  **`Walk-Forward Engine` (`src/pipeline/walk_forward`)**: Manages the temporal aspects of the backtest.
    *   It slices data into training, testing, and validation segments based on a `SegmentPlan`.
    *   The `make_plan` function in `src/pipeline/walk_forward/planning.py` is the entry point for creating a temporal plan.

3.  **`Model Integration` (`src/pipeline/model`)**: A bridge for integrating ML models. Adapters like `SklearnAdapter` wrap models to conform to the pipeline's `xarray` data structures.

### Data System (`src/data`)

This is a comprehensive system for data loading, caching, and feature engineering.

*   **Semantic Configuration**: Data sources and processing steps are defined in YAML files (e.g., `src/data/configs/crypto_standard.yaml`). These are parsed by `DataConfig` dataclasses (`src/data/managers/config_schema.py`).
*   **Two-Level Caching (`src/data/core/cache.py`)**:
    *   **L1 Cache**: Stores raw, unprocessed data.
    *   **L2 Cache**: Stores fully processed data, ready for analysis. Cache keys are generated from a hash of the configuration, ensuring data integrity.
*   **AST-based Feature Engineering (`src/data/ast/`)**: A powerful feature is the ability to define complex financial indicators in YAML files (e.g., `src/data/ast/definitions/technical.yaml`). The `FormulaManager` parses and evaluates these, with optional JAX-based JIT compilation for performance.

## 3. Developer Workflows

### Environment Setup

The project uses a `requirements.txt` file for dependencies.

```bash
# It is recommended to use a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Documentation

The documentation is built using Sphinx.

```bash
# Build the HTML documentation
make html
```

The output will be in the `build/html/` directory. You can view it by opening `build/html/index.html`. The `serve_docs.py` and `open_docs.py` scripts can also be used.

### Experimentation and Debugging

The `dev/` directory is the primary space for experimentation. It contains Jupyter notebooks, sample data, and test scripts. A common workflow is to use notebooks in `dev/` to interact with the `src/` library, test new ideas, and debug components.

## 4. Key Conventions

*   **Stateless Processors**: When adding a new data processor, follow the `ProcessorContract` in `src/pipeline/data_handler/core.py`. If the processor has state (e.g., learns parameters), ensure it is fitted only on training data.
*   **`xarray` as the Standard**: The library uses `xarray.Dataset` and `xarray.DataArray` as the primary data structures. Ensure any new components are compatible with this standard.
*   **Use the AST for Features**: When implementing new financial indicators or features, prefer defining them in a YAML file under `src/data/ast/definitions/`. This leverages the existing, optimized formula evaluation engine.
*   **Semantic Data Configuration**: For new data sources, use the YAML-based semantic configuration system. Create a new config file in `src/data/configs/`.

## 5. Source File Summaries

### `src/`

- **`__init__.py`**: Initializes the `src` module.
- **`__pycache__/`**: Contains compiled Python files.
- **`backtester/`**: Handles the backtesting engine.
- **`data/`**: Manages data loading, processing, and feature engineering.
- **`pipeline/`**: Orchestrates workflows, including data handling, temporal segmentation, and model execution.

### `src/backtester/`

- **`__init__.py`**: Initializes the `backtester` module.
- **`core.py`**: Contains the core logic for the backtesting engine.
- **`struct.py`**: Defines data structures or utilities for the backtesting process.
- **`metrics/`**: Contains modules for evaluating backtesting results.

#### `src/backtester/metrics/`

- **`__init__.py`**: Initializes the `metrics` module.
- **`standard.py`**: Implements standard metrics for evaluating backtesting results.

### `src/data/`

- **`README.md`**: Provides an overview of the `data` module.
- **`__init__.py`**: Initializes the `data` module.
- **`ast/`**: Handles Abstract Syntax Tree (AST)-based feature engineering.
- **`configs/`**: Contains configuration files for data sources and processing.
- **`core/`**: Core utilities for data handling, such as caching and management.
- **`filters/`**: Implements filtering mechanisms for data.
- **`generators/`**: Likely generates data or features.
- **`loaders/`**: Handles data loading from various sources.
- **`managers/`**: Manages data configurations and orchestration.
- **`processors/`**: Contains data processing modules.

#### `src/data/ast/`

- **`README.md`**: Explains the purpose of the `ast` module.
- **`__init__.py`**: Initializes the `ast` module.
- **`definitions/`**: Contains YAML files defining financial indicators or formulas.
- **`functions.py`**: Implements functions for AST-based operations.
- **`grammar.py`**: Defines the grammar for parsing AST expressions.
- **`manager.py`**: Manages AST-based feature engineering, including parsing and evaluation.
- **`nodes.py`**: Defines the nodes used in the AST structure.
- **`parser.py`**: Parses AST expressions into executable structures.
- **`visualization.py`**: Provides tools for visualizing AST structures.

##### `src/data/ast/definitions/`

- **`composite.yaml`**: Defines composite financial indicators or features.
- **`marketchars.yaml`**: Defines market characteristics or related indicators.
- **`schema.yaml`**: Provides a schema for validating AST definitions.
- **`technical.yaml`**: Defines technical indicators like RSI, WMA, or ALMA.
- **`ts_dependence.yaml`**: Defines time-series dependencies or related features.

### `src/pipeline/`

- **`REFACTORING_SUMMARY.md`**: Documents recent refactoring efforts in the pipeline module.
- **`__init__.py`**: Initializes the `pipeline` module.
- **`data_handler/`**: Manages data processing within the pipeline.
- **`model/`**: Handles model integration and execution.
- **`walk_forward/`**: Manages temporal segmentation and walk-forward analysis.

#### `src/pipeline/data_handler/`

- **`__init__.py`**: Initializes the `data_handler` module.
- **`config.py`**: Handles configuration for data processing.
- **`core.py`**: Defines core concepts like `ProcessorContract` and processing modes.
- **`handler.py`**: Implements the main `DataHandler` class for orchestrating data processing.
- **`processors.py`**: Contains implementations of data processors, such as normalization or feature evaluation.

#### `src/pipeline/model/`

- **`__init__.py`**: Initializes the `model` module.
- **`adapter.py`**: Adapts machine learning models to the pipeline's data structures (e.g., `xarray.Dataset`).
- **`runner.py`**: Manages the training and prediction lifecycle of models within the pipeline.

#### `src/pipeline/model/adapter.py

This file provides a thin wrapper around machine learning models to integrate them into the pipeline. Key components include:

1. **`ModelAdapter` Base Class**:
   - **Purpose**:
     - Defines a stable interface for integrating models into the pipeline.
     - Handles conversion between `xarray.Dataset` slices and `numpy` arrays.
   - **Methods**:
     - `fit`: Trains the model using the provided dataset and features.
     - `partial_fit`: Supports incremental training, falling back to `fit` by default.
     - `predict`: Generates predictions for the given dataset and features.
     - `get_state` and `load_state`: Optional hooks for model persistence, allowing saving and loading of model parameters.

2. **`SklearnAdapter` Class**:
   - **Purpose**:
     - Wraps scikit-learn-like estimators to conform to the pipeline's interface.
   - **Parameters**:
     - `model`: The scikit-learn-like estimator.
     - `handler`: A `DataHandler` instance for consistent stacking rules.
     - `output_var`: Name of the output variable in the `xarray` dataset.
     - `use_proba`: If `True`, uses `predict_proba` and selects a specific column.
     - `proba_index`: Column index to take from `predict_proba`.
     - `drop_nan_rows`: If `True`, drops rows with non-finite features before calling the model.
   - **Implementation Details**:
     - Converts 3D arrays (time, asset, feature) into 2D arrays for scikit-learn compatibility.
     - Handles missing values minimally, relying on upstream processors for imputation.
     - Maps predictions back to the original dataset structure after flattening.

**Architectural Choices**:
- **Separation of Concerns**:
  - The adapter focuses solely on interfacing with models, leaving data preprocessing and postprocessing to other components.
- **Extensibility**:
  - The `ModelAdapter` base class allows easy integration of other model types by subclassing.
- **Compatibility with `xarray`**:
  - Ensures that predictions align with the multi-dimensional structure of the input data.

**Design Philosophy**:
- The module emphasizes modularity and reusability, enabling seamless integration of diverse machine learning models into the pipeline.

#### `src/pipeline/walk_forward/`

- **`__init__.py`**: Initializes the `walk_forward` module.
- **`execution.py`**: Implements the `WalkForwardRunner` for executing pipelines over temporal segments.
- **`planning.py`**: Provides functions for creating temporal plans, such as `make_plan`.
- **`segments.py`**: Defines data structures for temporal segmentation, like `Segment` and `SegmentPlan`.

#### `src/pipeline/walk_forward/planning.py`

This file provides utilities for generating walk-forward segment plans. Key components include:

1. **Purpose**:
   - Handles the complex logic of creating temporally consistent segments while respecting data boundaries and avoiding lookahead bias.
   - Follows qlib's pattern of separating temporal logic from data processing, enabling flexible backtesting workflows.

2. **Key Functions**:
   - `_time_min_max`:
     - Extracts the minimum and maximum valid datetime bounds from a dataset's time coordinate.
     - Handles invalid calendar entries (e.g., NaT) resulting from rectangular grid unstacking operations.
     - Ensures robust handling of time coordinates in both 1D and ND grids.
   - `_clip_timestamp`:
     - Clips a timestamp to be within specified bounds.
     - Ensures timestamps remain within valid temporal ranges for segment planning.

3. **Integration with Segments**:
   - The module integrates with `Segment`, `SegmentPlan`, and `SegmentConfig` classes to define and manage temporal segments for walk-forward analysis.

**Architectural Choices**:
- **Separation of Concerns**:
  - Temporal logic is decoupled from data processing, ensuring modularity and reusability.
- **Robust Time Handling**:
  - Functions like `_time_min_max` and `_clip_timestamp` ensure that temporal boundaries are handled consistently and accurately.

**Design Philosophy**:
- The module emphasizes flexibility and robustness, making it suitable for complex backtesting workflows that require precise temporal segmentation.

### `src/pipeline/walk_forward/execution.py`

This file implements the execution engine for walk-forward analysis, orchestrating the application of data processing pipelines across temporal segments. It follows the principle of separating temporal logic ("when") from data processing logic ("how").

#### Key Components

1. **`SegmentResult` Class**:
   - Encapsulates the results from processing a single walk-forward segment.
   - Attributes:
     - `segment`: The processed segment.
     - `ds_infer`: Fully processed inference dataset.
     - `learn_states`: States learned from training data.
   - Provides methods to retrieve training and inference periods and summarize learned states.

2. **`WalkForwardResult` Class**:
   - Aggregates the output from walk-forward analysis.
   - Attributes:
     - `processed_ds`: Aggregated dataset containing all inference periods.
     - `segment_states`: Summary of each processed segment.
     - `attrs`: Metadata about the walk-forward run.

3. **`WalkForwardRunner` Class**:
   - Manages the complete walk-forward execution workflow.
   - Attributes:
     - `handler`: Configured `DataHandler` with processing pipelines.
     - `plan`: Defines the temporal structure of the walk-forward analysis.
     - `_shared`: Cached shared processing view.
   - Methods:
     - `_ensure_shared`: Applies shared processors to the dataset and caches the result.
     - `_compute_bounds`: Computes integer slices for all segments.
     - `_masked_scatter_rows_inplace`: Handles in-place masked writes for efficient data updates.
     - `run`: Executes the complete walk-forward plan and returns aggregated results.
     - `run_segments`: Processes and returns individual segment results.
     - `run_single_segment`: Processes a single segment for debugging or testing.
     - `get_execution_summary`: Summarizes the execution plan.

#### Design Philosophy

- **Separation of Concerns**:
  - Temporal logic is decoupled from data processing, ensuring modularity and reusability.
- **Efficiency**:
  - Implements caching and efficient slicing to handle large datasets.
- **Flexibility**:
  - Supports both aggregated and segment-level results, enabling diverse analysis workflows.
- **Robustness**:
  - Prevents lookahead bias by ensuring inference data is processed using only training-time information.

#### Examples of Usage

- **Basic Walk-Forward Execution**:
  ```python
  runner = WalkForwardRunner(handler=handler, plan=plan)
  results = runner.run()
  print(f"Processed {len(results)} segments")
  ```

- **Advanced Analysis**:
  ```python
  results = runner.run()
  for result in results:
      summary = result.get_state_summary()
      print(f"Segment learned {summary['num_states']} processor states")
  ```

### `src/pipeline/data_handler/handler.py`

This file implements the `DataHandler` class, which orchestrates the complete data processing pipeline. It manages views, caching, and processor execution, separating data processing logic from temporal segmentation concerns.

#### Key Components

1. **`DataHandler` Class**:
   - **Purpose**: Central orchestrator for data processing pipelines.
   - **Attributes**:
     - `base`: The raw input dataset.
     - `config`: Configuration object defining the processing pipeline.
     - `cache`: Stores intermediate processing results.
     - `learn_states` and `infer_states`: Learned states from the respective pipelines.
   - **Methods**:
     - `build`: Executes the complete processing pipeline and caches results.
     - `view`: Retrieves specific views (RAW, LEARN, INFER) of the processed data.
     - `fetch`: Extracts subsets of variables from processed views.
     - `slice_time`: Efficiently slices datasets by time range.
     - `to_arrays_for_model`: Converts datasets to dense numpy arrays for ML models.
     - `fit_learn` and `transform_with_learn`: Fit and apply learn processors.
     - `apply_infer`: Applies infer processors in transform-only mode.
     - `_apply_pipeline`: Internal method for executing processor pipelines.

2. **Processing Pipeline Structure**:
   - **Shared Processors**: Run once on the full dataset (transform-only).
   - **Learn Processors**: Fit on training segments, transform on both train/infer.
   - **Infer Processors**: Transform-only on inference segments.

#### Design Philosophy

- **Separation of Concerns**:
  - Decouples data processing logic from temporal segmentation.
- **Efficiency**:
  - Implements lazy evaluation and caching to avoid recomputation.
- **Flexibility**:
  - Supports multiple views (RAW, LEARN, INFER) and semantic column groups.
- **Robustness**:
  - Handles invalid time entries and missing data gracefully.

#### Examples of Usage

- **Basic Configuration and Usage**:
  ```python
  from src.pipeline.data_handler import DataHandler, HandlerConfig
  from src.pipeline.data_handler.processors import CSZScore, PerAssetFFill

  config = HandlerConfig(
      shared=[PerAssetFFill(name="ffill")],
      learn=[CSZScore(name="norm", vars=["close", "volume"])],
      feature_cols=["close_csz", "volume_csz"]
  )
  handler = DataHandler(base=dataset, config=config)
  learn_view = handler.view(View.LEARN)
  features = handler.fetch(View.LEARN, ["features"])
  ```

- **Converting Data for Models**:
  ```python
  X, y = handler.to_arrays_for_model(
      ds=learn_view,
      feature_vars=["close_csz", "volume_csz"],
      label_vars=["target"]
  )
  ```
