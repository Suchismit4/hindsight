 # Hindsight Pipeline Framework Architecture
 
 ## 1. Core Philosophy
 
 The Hindsight Pipeline Framework is designed for robust and reproducible financial research, particularly for backtesting trading strategies. Its architecture is founded on a few key principles that address common pitfalls in financial modeling, such as lookahead bias, state management, and path dependency.
 
 ### Separation of "How" and "When"
 
 The most critical design principle, primarily embodied in the `pipeline` module, is the strict separation of **data processing logic ("how")** from **temporal segmentation ("when")**.
 
 -   **"How" (The Data Pipeline):** This concerns *what* transformations are applied to the data. It includes feature engineering, normalization, and formula calculations. This logic is encapsulated within the `DataHandler` and its `Processors`.
 -   **"When" (The Temporal Plan):** This concerns *which* time periods are used for training and inference. It involves creating walk-forward schedules, defining training/inference window sizes, and managing gaps. This logic is handled by the `walk_forward` module.
 
 This separation allows researchers to modify the backtesting schedule (e.g., change from a 1-year to a 2-year training window) without altering the feature engineering code, and vice-versa.
 
 ### Stateful Processing with Temporal Isolation
 
 The framework is built to handle stateful transformations (e.g., calculating normalization statistics) without introducing lookahead bias.
 
 -   **Stateful Processors:** Processors can learn parameters (a "state") from a training dataset.
 -   **Temporal Isolation:** When executing a walk-forward analysis, the state is *always* learned exclusively on the training data for a given segment. This learned state is then applied to transform both the training and inference data for that segment. A new state is learned for each subsequent segment. This is crucial for validity, as it ensures information from the future (the inference period) never leaks into the creation of features.
 
 ### Declarative, Composable Pipeline
 
 The entire workflow is defined declaratively through configuration objects (`HandlerConfig`, `SegmentConfig`). This makes pipelines easy to read, modify, and persist for reproducibility. Processors are composed into a sequence to form a complete data processing pipeline.
 
 ### Duality of Backtesting Approaches
 
 A key architectural feature is the presence of two distinct backtesting engines, each suited for different research goals:
 
 1.  **ML-Centric (`ModelRunner`):** A vectorized, bulk-processing engine designed to generate a complete panel of model predictions across all assets and time points. It excels at feature research, hyperparameter tuning, and evaluating models that produce cross-sectional scores.
 2.  **Event-Driven (`BacktestEngine`):** A more traditional, iterative engine that simulates the evolution of a single portfolio over time. It is path-dependent, managing cash, positions, and order execution. It is ideal for testing rule-based strategies and simulating portfolio-level performance.
 
 ## 2. Project Structure
 
 The project is organized into a `src` directory containing the core framework, a `dev` directory for examples and testing, and `build`/`source` for documentation.
 
 ```
 hindsight/
 ├── src/
 │   ├── pipeline/
 │   │   ├── data_handler/      # The "How" - Data Processing
 │   │   │   ├── __init__.py
 │   │   │   ├── core.py          # View, PipelineMode, ProcessorContract
 │   │   │   ├── config.py        # HandlerConfig class
 │   │   │   ├── handler.py       # DataHandler class
 │   │   │   └── processors.py    # FormulaEval, CSZScore, PerAssetFFill
 │   │   ├── walk_forward/      # The "When" - Temporal Segmentation
 │   │   │   ├── __init__.py
 │   │   │   ├── segments.py      # Segment, SegmentPlan, SegmentConfig
 │   │   │   ├── planning.py      # make_plan, optimize_plan_for_dataset
 │   │   │   └── execution.py     # WalkForwardRunner (data-only)
 │   │   └── model/             # ML Model Integration
 │   │       ├── __init__.py
 │   │       ├── adapter.py       # ModelAdapter, SklearnAdapter
 │   │       └── runner.py        # ModelRunner class
 │   ├── data/
 │   │   ├── ast/                 # Formula Engine (Abstract Syntax Tree)
 │   │   │   ├── __init__.py
 │   │   │   ├── manager.py       # FormulaManager class
 │   │   │   ├── parser.py        # Formula string parsing
 │   │   │   ├── nodes.py         # AST node definitions
 │   │   │   ├── functions.py     # Built-in and custom function registry
 │   │   │   ├── visualization.py # AST visualization tools
 │   │   │   └── definitions/     # YAML formula definitions
 │   │   ├── core/              # Core data utilities & JAX operations
 │   │   │   └── ...
 │   │   ├── loaders/             # Data loading implementations
 │   │   │   └── wrds/
 │   │   │       └── generic.py   # GenericWRDSDataLoader
 │   │   └── DataManager.py       # Data loading orchestration
 │   └── backtester/            # Event-Driven Backtesting Engine
 │       ├── __init__.py
 │       ├── core.py            # BacktestEngine, Broker, EventBasedStrategy
 │       ├── metrics.py         # Performance metrics (e.g., Sharpe Ratio)
 │       └── struct.py          # Core data structures (Order, BacktestState)
 ├── dev/
 │   └── ...
 └── ...
 ```
 
 ## 3. Component Deep Dive
 
 ### 3.1. Data Loading (`src/data/`)
 
 -   **`DataManager.py`:** The entry point for acquiring data. It abstracts away the specifics of data storage and provides a unified, cached interface for loading datasets.
 -   **`loaders/`:** Contains concrete loader implementations. For example, `GenericWRDSDataLoader` handles loading `.sas7bdat` files from WRDS, using multiprocessing for performance and applying pre-processing steps like date conversion and column filtering. This modular design allows for easy extension to other data sources (e.g., CSV files, SQL databases).
 
 ### 3.2. The Formula Engine (`src/data/ast/`)
 
 This is a powerful, self-contained module for defining, managing, and evaluating financial formulas. It serves as the engine for the `FormulaEval` processor.
 
 -   **`FormulaManager`:** The central class that orchestrates the entire formula system. It loads formula definitions from YAML files, validates them against a schema, and manages their evaluation. Its `evaluate_bulk` method is highly optimized for evaluating multiple formulas and configurations at once.
 -   **Dependency Resolution:** A key feature is its ability to manage dependencies between formulas. The `FormulaManager` builds a dependency graph and uses a topological sort (`_get_evaluation_order`) to determine the correct evaluation order. It supports two types of dependencies:
     -   **Functional Dependence:** A formula can call another formula like a function (e.g., `my_indicator(rsi(close, 14))`). This is enabled by compiling formulas into callable Python functions (`_compile_formula_as_function`).
     -   **Time-Series Dependence:** A formula can use the output of another formula as a time-series input variable (e.g., `sma($my_indicator, 10)` where `my_indicator` is the result of another formula).
 -   **`parser.py`:** Implements a `FormulaParser` that converts a formula string into a custom Abstract Syntax Tree (AST). It first replaces `$variable` syntax with a temporary, Python-valid name, then uses Python's native `ast` module for initial parsing, and finally converts the Python AST into the framework's custom `Node` objects.
 -   **`nodes.py`:** Defines the custom AST nodes (e.g., `BinaryOp`, `FunctionCall`, `DataVariable`). Each node has an `evaluate` method. The `DataVariable` node is the crucial link that resolves a `$variable` name by looking up the corresponding `DataArray` in the `_dataset` provided in the evaluation context.
 -   **`functions.py`:** Implements a function registry using the `@register_function` decorator. This allows users to easily extend the formula language with custom Python functions. The built-in functions (`sma`, `ema`, etc.) are designed to work with `xarray` and are often wrappers around high-performance, JAX-JIT-compiled numerical kernels.
 
 ### 3.3. The Data Pipeline (`src/pipeline/data_handler/`)
 
 This component represents the "how" of the ML-centric workflow.
 
 -   **`HandlerConfig`:** A dataclass that declaratively defines the entire processing pipeline, specifying lists of processors for the `shared`, `learn`, and `infer` stages.
 -   **`DataHandler`:** The pipeline orchestrator. It takes a base `xarray.Dataset` and a `HandlerConfig`. While it has a `build()` method for one-off processing, its main role in a backtest is to provide segment-specific processing via methods like `fit_learn` and `transform_with_learn`, which are called by the `ModelRunner`. Its `to_arrays_for_model` method is the critical bridge that converts a processed `xarray` slice into dense `numpy` arrays for an ML model.
 -   **`processors.py`:** Contains the atomic units of data transformation.
     -   `FormulaEval`: A stateless processor that wraps the `FormulaManager`. In its `fit` method, it pre-compiles a JIT-able evaluation function. In `transform`, it executes this function on the dataset and merges the results.
     -   `CSZScore`: A stateful processor for cross-sectional z-score normalization. Its `fit` method computes the mean and standard deviation across the `asset` dimension and stores them in a state `xr.Dataset`. `transform` then uses these stored statistics to normalize the data, ensuring no lookahead bias.
     -   `PerAssetFFill`: A simple, stateless processor for forward-filling missing values on a per-asset basis.
 
 ### 3.4. Temporal Segmentation (`src/pipeline/walk_forward/`)
 
 This component represents the "when" of the workflow.
 
 -   **`segments.py`:** Defines the core data structures: `Segment` (a single train/infer window), `SegmentPlan` (a list of `Segment`s), and `SegmentConfig` (a declarative way to define a walk-forward schedule).
 -   **`planning.py`:** Contains factory functions like `make_plan` which generates a `SegmentPlan` from a `SegmentConfig`, handling logic like `clip_to_data`.
 -   **`execution.py`:** Contains the `WalkForwardRunner`, a data-only runner that executes a `DataHandler` pipeline over a `SegmentPlan`. It is distinct from the `ModelRunner` and is used for generating processed datasets without a model.
 
 ### 3.5. ML Model Execution (`src/pipeline/model/`)
 
 This component integrates a machine learning model into the "how" and "when" pipeline.
 
 -   **`ModelRunner`:** The top-level orchestrator for a complete, ML-centric backtest. It uses an efficient **Gather-Scatter** pattern for execution:
     1.  **Setup:** It pre-stacks the shared dataset and pre-allocates a global, NaN-filled `DataArray` for the final predictions. It also pre-computes the integer-based slice boundaries for every segment for high performance.
     2.  **Loop (Gather):** For each `Segment` in the `SegmentPlan`:
         - It slices the training and inference data from the stacked shared view using the pre-computed integer slices.
         - It uses the `DataHandler` to apply `learn` and `infer` processors, producing fully processed data for the segment.
         - It calls the `model_factory` to create a *fresh, untrained* model instance, ensuring model isolation.
         - It fits the model on the training data and predicts on the inference data.
     3.  **Loop (Scatter):** The resulting segment prediction (a stacked `DataArray`) is written into the global prediction buffer using `_masked_scatter_rows_inplace`. This method uses precise integer indexing to handle overlaps and `NaN` values correctly, governed by the `overlap_policy`.
     4.  **Finalize:** After the loop, the global prediction buffer is unstacked back into the original multi-dimensional time format.
 -   **`ModelAdapter` (Adapter Pattern):** This abstraction decouples the `ModelRunner` from the specific API of any given ML library. `SklearnAdapter` is a concrete implementation that wraps any scikit-learn compatible model. It handles the final data conversion from the `(T, N, J)` numpy arrays provided by `DataHandler` to the 2D `(M, J)` arrays that scikit-learn models expect, including dropping of NaN rows.
 -   **`model_factory` (Factory Pattern):** The `ModelRunner` takes a factory function (e.g., `make_adapter`) to ensure a new, clean model is created for every walk-forward segment, preventing any state from leaking between segments.
 
 ### 3.6. Event-Driven Backtester (`src/backtester/`)
 
 This module provides a more traditional, path-dependent backtesting engine, separate from the ML-focused `ModelRunner`.
 
 -   **Purpose:** To simulate the evolution of a single portfolio over time, handling cash, positions, and order execution. It is ideal for rule-based strategies.
 -   **`BacktestEngine`:** The main orchestrator. It iterates through time step-by-step, feeding data windows to the strategy and instructing the `Broker` to execute trades.
 -   **`EventBasedStrategy`:** An abstract class that users implement. Its `next` method is called at each time step, receiving the latest market data and current portfolio state (`BacktestState`), and must return a list of `Order` objects to be executed.
 -   **`Broker`:** Manages the portfolio's state, including `cash` and `positions`. It queues `Order` objects from the strategy and executes them at the next time step's prices (e.g., next day's open), calculating commissions and updating the portfolio state. The `execute_orders_np` method is highly optimized to work with NumPy arrays for performance.
 -   **`struct.py` & `metrics.py`:** These files define the core data structures (`Order`, `BacktestState`) and performance metrics (e.g., Sharpe Ratio, Drawdown) used by the engine.
 
 ## 4. End-to-End Data Flow
 
 ### 4.1. `ModelRunner` (ML-Centric) Flow
 
 ```
 [Raw Data] -> [DataManager] -> [Base xarray.Dataset]
                                     |
                                     v
 [DataHandler] -> (runs shared processors) -> [Shared xarray.Dataset]
      |                                          |
      |                                          v
 [SegmentPlan] <-------------------------- [ModelRunner]
      |                                          |
      |                                          | (pre-stacks shared_ds, creates empty pred_buffer)
      |                                          |
      |--> For each Segment in Plan (Gather & Scatter):
      |    1. Slice train/infer data from stacked_shared_ds
      |    2. handler.fit_learn(train_ds) -> learn_states
      |    3. handler.transform_with_learn(infer_ds, learn_states) -> processed_infer_ds
      |    4. model_factory() -> new_model
      |    5. new_model.fit(processed_train_ds)
      |    6. new_model.predict(processed_infer_ds) -> segment_predictions (stacked DataArray)
      |    7. Scatter segment_predictions into global pred_buffer
      |
      v
 [Unstack pred_buffer] -> [Final Prediction Dataset]
      |
      v
 [ModelRunnerResult]
 ```
 
 ### 4.2. `BacktestEngine` (Event-Driven) Flow
 
 ```
 [Market Data] & [Characteristics Data]
      |
      v
 [BacktestEngine] -> (pre-computes data windows)
      |
      |--> For each time step `t`:
      |    1. Get data window for `t`
      |    2. strategy.next(window, current_state) -> [List of Orders]
      |    3. broker.queue_orders(orders)
      |    4. Get data for `t+1` (for execution)
      |    5. broker.execute_orders_np(data_t+1) -> updates cash & positions
      |    6. Update current_state
      |    7. Record portfolio value
      |
      v
 [Final BacktestState] & [Performance Metrics]
 ```
 
 ## 5. Abstractions and Design Patterns
 
 -   **Separation of Concerns:** The "How" (`DataHandler`) vs. "When" (`walk_forward`) is the primary architectural pattern for the ML pipeline.
 -   **Adapter Pattern:** `ModelAdapter` allows the framework to support various ML libraries without changing the core `ModelRunner` logic.
 -   **Factory Pattern:** The `model_factory` function ensures stateless, isolated model training in each walk-forward segment.
 -   **Strategy Pattern:** The `EventBasedStrategy` allows users to inject custom trading logic into the `BacktestEngine`.
 -   **Gather-Scatter Pattern:** Used by `ModelRunner` and `WalkForwardRunner` for efficient, overlap-aware aggregation of segment-based results into a global dataset.
 -   **Composition:** `Processor` objects are composed into a list within `HandlerConfig` to create a processing pipeline.
 -   **Declarative Configuration:** The entire ML pipeline is defined by configuration objects (`HandlerConfig`, `SegmentConfig`), making it explicit and reproducible.
 -   **Abstract Syntax Tree (AST):** The formula system uses an AST to provide a powerful, flexible, and extensible domain-specific language for feature engineering.
 
 ## 6. Future Work & Known Issues
 
 Based on the current codebase, several areas for future development are apparent:
 
 -   **Tighter Integration:** The `BacktestEngine` and `ModelRunner` pipelines are currently separate. Future work could involve creating a bridge, allowing the predictions from `ModelRunner` to be used as signals in an `EventBasedStrategy`.
 -   **Expanded Model Support:** Add more `ModelAdapter` implementations for popular deep learning libraries like PyTorch and TensorFlow.
 -   **Advanced Order Types:** The `backtester` could be expanded to support more complex order types, such as stop-loss, take-profit, and time-in-force orders.
 -   **AST Enhancements:** The `TODO`s in `parser.py` and `nodes.py` suggest plans for supporting multi-series expressions (e.g., for Bollinger Bands), a visitor pattern for AST transformations, and a more robust type validation system.
 -   **Performance:** While already a focus, further parallelization of the `ModelRunner` loop (processing multiple segments concurrently) could be explored, though it would require careful management of JAX/NumPy threading contexts.