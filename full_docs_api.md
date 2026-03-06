# Hindsight Pipeline Framework - Complete API Reference

**Version:** 1.0
**Last Updated:** December 2025

This document contains the complete API reference with detailed method signatures, parameters, and docstrings extracted from the Python source code.

---

## Table of Contents

1. [Data Handler API](#data-handler-api)
2. [Processors API](#processors-api)
3. [Walk-Forward Analysis API](#walk-forward-analysis-api)
4. [Model Integration API](#model-integration-api)
5. [Configuration Classes](#configuration-classes)
6. [Enums and Base Classes](#enums-and-base-classes)
7. [Utility Functions](#utility-functions)
8. [Additional Processors](#additional-processors)

---

# Data Handler API

## DataHandler

**Module:** `src.pipeline.data_handler.handler`

```python
@dataclass
class DataHandler:
    """
    Central orchestrator for data processing pipelines.

    The DataHandler manages the complete data processing workflow, including
    caching of intermediate results, pipeline execution, and view management.
    It separates "how data is processed" (processors) from "when" (segments),
    following qlib's architectural principles.

    Parameters
    ----------
    base : xr.Dataset
        The raw input dataset that serves as the foundation for all processing
    config : HandlerConfig
        Configuration object defining the processing pipeline

    Attributes
    ----------
    cache : dict
        Cache for storing intermediate processing results
    learn_states : list of ProcessorState
        Learned states from the learn processor pipeline
    infer_states : list of ProcessorState
        Learned states from the infer processor pipeline

    Notes
    -----
    The DataHandler supports three main views:
    - RAW: Original data after feature graph construction
    - LEARN: Data processed through shared + learn pipelines (for training)
    - INFER: Data processed through shared + infer pipelines (for inference)

    The handler uses lazy evaluation and caching to avoid recomputing expensive
    transformations. Views are built on-demand and cached for subsequent access.

    Processing Pipeline Structure:
    1. Shared processors: Run once on full dataset (transform-only)
    2. Learn processors: Fit on training segments, transform on both train/infer
    3. Infer processors: Transform-only on inference segments

    Examples
    --------
    >>> from src.pipeline.data_handler import DataHandler, HandlerConfig
    >>> from src.pipeline.data_handler.processors import CSZScore, PerAssetFFill
    >>>
    >>> config = HandlerConfig(
    ...     shared=[PerAssetFFill(name="ffill")],
    ...     learn=[CSZScore(name="norm", vars=["close", "volume"])],
    ...     feature_cols=["close_csz", "volume_csz"]
    ... )
    >>> handler = DataHandler(base=dataset, config=config)
    >>> learn_view = handler.view(View.LEARN)
    >>> features = handler.fetch(View.LEARN, ["features"])
    """
```

### Methods

#### `build()`

```python
def build(self) -> None:
    """
    Build all pipeline views and cache the results.

    This method executes the complete processing pipeline according to the
    configured mode and caches all intermediate results. It should be called
    before accessing views if not using lazy evaluation.

    The build process follows these steps:
    1. Initialize features cache with base dataset
    2. Apply shared processors (transform-only)
    3. Branch execution based on pipeline mode:
       - INDEPENDENT: Parallel learn and infer pipelines
       - APPEND: Sequential infer -> learn pipeline
    4. Cache final views and learned states
    """
```

#### `view(which)`

```python
def view(self, which: View) -> xr.Dataset:
    """
    Get a specific view of the processed data.

    Returns the requested data view, building the pipeline if necessary.
    Views are cached after first access for performance.

    Parameters
    ----------
    which : View
        The data view to retrieve (RAW, LEARN, or INFER)

    Returns
    -------
    xr.Dataset
        The requested data view

    Notes
    -----
    - RAW view returns the original base dataset
    - LEARN view returns data processed through shared + learn pipelines
    - INFER view returns data processed through shared + infer pipelines
    """
```

#### `fetch(which, col_set)`

```python
def fetch(self, which: View, col_set: Optional[Sequence[str]] = None) -> xr.Dataset:
    """
    Fetch specific columns from a data view.

    This method provides a convenient way to extract subsets of variables
    from processed views, with support for semantic column groups.

    Parameters
    ----------
    which : View
        The data view to fetch from
    col_set : Sequence[str], optional
        Column specifications. Can include:
        - Specific variable names
        - "feature"/"features" to get feature_cols
        - "label"/"labels" to get label_cols

    Returns
    -------
    xr.Dataset
        Dataset containing only the requested columns

    Notes
    -----
    If col_set contains "feature"/"features", the method will include all
    variables listed in config.feature_cols. Similarly for "label"/"labels"
    and config.label_cols.
    """
```

#### `shared_view()`

```python
def shared_view(self) -> xr.Dataset:
    """
    Get the shared processing view.

    Returns the dataset after shared processor application but before
    learn/infer branching. Useful for debugging and analysis.

    Returns
    -------
    xr.Dataset
        Dataset after shared processing
    """
```

#### `fit_learn(ds)`

```python
def fit_learn(self, ds: xr.Dataset) -> Tuple[xr.Dataset, List[ProcessorState]]:
    """
    Fit the learn processor pipeline on a dataset.

    This method fits all learn processors sequentially and returns both
    the transformed dataset and the learned states.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to fit processors on

    Returns
    -------
    tuple[xr.Dataset, list[ProcessorState]]
        Tuple of (transformed_dataset, list_of_states)
    """
```

#### `transform_with_learn(ds, states)`

```python
def transform_with_learn(self, ds: xr.Dataset, states: List[ProcessorState]) -> xr.Dataset:
    """
    Transform a dataset using pre-fitted learn processor states.

    This method applies learned transformations to new data using previously
    fitted processor states, enabling consistent processing across time segments.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to transform
    states : list[ProcessorState]
        List of learned states from fit_learn method

    Returns
    -------
    xr.Dataset
        Transformed dataset
    """
```

#### `apply_infer(ds)`

```python
def apply_infer(self, ds: xr.Dataset) -> xr.Dataset:
    """
    Apply the infer processor pipeline to a dataset.

    This method applies infer processors in transform-only mode, typically
    used for post-processing after learn transformations.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to process

    Returns
    -------
    xr.Dataset
        Processed dataset
    """
```

#### `to_arrays_for_model(ds, feature_vars, label_vars, ...)`

```python
def to_arrays_for_model(
    self,
    ds: xr.Dataset,
    feature_vars: Sequence[str],
    label_vars: Optional[Sequence[str]] = None,
    drop_invalid_time: bool = True,
    drop_all_nan_rows: bool = False,
    return_indexer: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]] | Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Convert dataset to dense numpy arrays for model training/inference.

    This method converts an unstacked dataset slice to the (T, N, J) format
    commonly used by machine learning models, with proper handling of invalid
    time entries and missing data.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to convert
    feature_vars : Sequence[str]
        Variable names to use as features
    label_vars : Sequence[str], optional
        Variable names to use as labels/targets
    drop_invalid_time : bool, default True
        Whether to drop rows where 'time' coordinate is NaT
    drop_all_nan_rows : bool, default False
        Whether to drop rows where all features across assets are NaN
    return_indexer : bool, default False
        Whether to return the indexer array

    Returns
    -------
    tuple[np.ndarray, np.ndarray or None] or tuple[np.ndarray, np.ndarray or None, np.ndarray]
        Tuple of (X, y) where:
        - X: Features array with shape (T, N, J)
        - y: Labels array with shape (T, N, K) or None if no label_vars
        - If return_indexer=True: (X, y, valid_idx) where valid_idx selects
          the kept time_index rows from the full stacked calendar

    Notes
    -----
    The conversion process:
    1. Stack time dimensions to create time_index
    2. Transpose to (time_index, asset, ...) order
    3. [Optionally] drop invalid times (NaT entries)
    4. Stack feature variables along last dimension
    5. [Optionally] drop rows with all-NaN features

    Invalid time entries typically result from unstacking operations that
    create rectangular grids from full day calendars.
    """
```

---

# Processors API

## Processor (Base Class)

**Module:** `src.pipeline.data_handler.processors`

```python
@dataclass
class Processor(ProcessorContract):
    """
    Base processor implementation with common functionality.

    This class provides the basic structure for processors following the
    ProcessorContract interface. All concrete processors should inherit from
    this class.

    Parameters
    ----------
    name : str
        Unique name identifier for this processor instance
    """
```

### Methods

#### `fit(ds)`

```python
def fit(self, ds: xr.Dataset) -> ProcessorState:
    """
    Learn parameters from the input dataset.

    Default implementation raises NotImplementedError. Subclasses must override.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to learn parameters from

    Returns
    -------
    ProcessorState
        Opaque state object containing learned parameters

    Raises
    ------
    NotImplementedError
        Must be implemented by subclasses
    """
```

#### `transform(ds, state)`

```python
def transform(
    self,
    ds: xr.Dataset,
    state: Optional[ProcessorState] = None,
) -> xr.Dataset:
    """
    Apply the transformation to the input dataset.

    Default implementation raises NotImplementedError. Subclasses must override.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to transform
    state : ProcessorState, optional
        State object returned by ``fit``. Processors should gracefully handle
        ``None`` when they can operate in stateless mode.

    Returns
    -------
    xr.Dataset
        Transformed dataset

    Raises
    ------
    NotImplementedError
        Must be implemented by subclasses
    """
```

## CSZScore

```python
@dataclass
class CSZScore(Processor):
    """
    Cross-sectional z-score normalization processor.

    The processor estimates per-asset means and standard deviations from the
    training data and reuses those statistics when transforming inference data.
    The learned state is a lightweight dataclass instead of an xr.Dataset so it
    can be stored without carrying time coordinates that would later conflict.

    Parameters
    ----------
    name : str
        Processor name identifier
    vars : list of str, optional
        List of variables to normalize. If None, applies to all numeric variables
        with asset dimension.
    out_suffix : str, default "_csz"
        Suffix to append to normalized variable names
    eps : float, default 1e-8
        Small constant added to standard deviation to avoid division by zero
    """
    vars: Optional[List[str]] = None
    out_suffix: str = "_csz"
    eps: float = 1e-8
```

### Methods

#### `fit(ds)`

```python
def fit(self, ds: xr.Dataset) -> CSZScoreState:
    """
    Learn per-asset statistics for each configured variable.

    Computes mean and standard deviation across all time dimensions
    for each asset, storing these statistics for later transformation.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to compute statistics from

    Returns
    -------
    CSZScoreState
        State containing means and standard deviations for each variable
    """
```

#### `transform(ds, state)`

```python
def transform(
    self,
    ds: xr.Dataset,
    state: Optional[ProcessorState] = None,
) -> xr.Dataset:
    """
    Apply learned statistics to normalize variables.

    Normalizes each configured variable using the learned mean and
    standard deviation, adding new variables with the configured suffix.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to normalize
    state : ProcessorState, optional
        Learned CSZScoreState. If None, fits on ds first.

    Returns
    -------
    xr.Dataset
        Dataset with added normalized variables
    """
```

## PerAssetFFill

```python
@dataclass
class PerAssetFFill(Processor):
    """
    Per-asset forward fill processor.

    This processor performs forward-fill (last observation carried forward) for each
    asset independently along chronological time. It is stateless and writes results
    in-place by default.

    Parameters
    ----------
    name : str, default "ffill"
        Processor name identifier
    vars : list of str, optional
        List of variables to forward fill. If None, applies to all numeric variables.

    Notes
    -----
    The processor is stateless; ``fit`` simply returns ``None``.
    """
    name: str = "ffill"
    vars: Optional[List[str]] = None
```

### Methods

#### `fit(ds)`

```python
def fit(self, ds: xr.Dataset) -> None:
    """
    Stateless processor – returns ``None``.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset (ignored)

    Returns
    -------
    None
        This processor has no learned state
    """
```

#### `transform(ds, state)`

```python
def transform(
    self,
    ds: xr.Dataset,
    state: Optional[ProcessorState] = None,
) -> xr.Dataset:
    """
    Apply forward fill transformation per asset.

    Forward fills missing values for each variable and asset independently
    along the time dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to forward fill
    state : ProcessorState, optional
        Unused for this stateless processor

    Returns
    -------
    xr.Dataset
        Dataset with forward-filled values
    """
```

---

# Walk-Forward Analysis API

## Segment

**Module:** `src.pipeline.walk_forward.segments`

```python
@dataclass(frozen=True)
class Segment:
    """
    A single walk-forward segment defining training and inference periods.

    This class represents a single step in walk-forward analysis, containing
    both the training period for model fitting and the inference period for
    out-of-sample evaluation. Training boundaries are inclusive, inference
    end is exclusive to prevent overlap between consecutive segments.

    Parameters
    ----------
    train_start : np.datetime64
        Start timestamp for training period (inclusive)
    train_end : np.datetime64
        End timestamp for training period (inclusive)
    infer_start : np.datetime64
        Start timestamp for inference period (inclusive)
    infer_end : np.datetime64
        End timestamp for inference period (exclusive)

    Notes
    -----
    The segment design ensures proper temporal separation:
    - Training period: [train_start, train_end] inclusive
    - Inference period: [infer_start, infer_end) exclusive at end
    - Typical pattern: train_end < infer_start (with optional gap)

    The inference end is exclusive to prevent overlap with the next segment's
    inference start when step equals infer_span and gap is zero.

    All timestamps should use consistent np.datetime64 units ('ns', 'ms', 's', etc.)
    for proper comparison and sorting operations.

    Examples
    --------
    >>> segment = Segment(
    ...     train_start=np.datetime64('2020-01-01'),
    ...     train_end=np.datetime64('2020-12-31'),
    ...     infer_start=np.datetime64('2021-01-01'),
    ...     infer_end=np.datetime64('2021-01-31')
    ... )
    >>> print(f"Training: {segment.train_start} to {segment.train_end}")
    >>> print(f"Inference: {segment.infer_start} to {segment.infer_end}")
    """
```

### Properties

#### `train_duration`

```python
@property
def train_duration(self) -> np.timedelta64:
    """
    Get the duration of the training period.

    Returns
    -------
    np.timedelta64
        Duration of training period
    """
```

#### `infer_duration`

```python
@property
def infer_duration(self) -> np.timedelta64:
    """
    Get the duration of the inference period.

    Returns
    -------
    np.timedelta64
        Duration of inference period
    """
```

#### `gap_duration`

```python
@property
def gap_duration(self) -> np.timedelta64:
    """
    Get the gap between training and inference periods.

    Returns
    -------
    np.timedelta64
        Gap duration (can be negative if periods overlap)
    """
```

### Methods

#### `overlaps_with(other)`

```python
def overlaps_with(self, other: "Segment") -> bool:
    """
    Check if this segment overlaps with another segment.

    Parameters
    ----------
    other : Segment
        Other segment to check overlap with

    Returns
    -------
    bool
        True if segments have any temporal overlap
    """
```

## SegmentPlan

```python
@dataclass
class SegmentPlan:
    """
    A collection of segments describing the complete walk-forward schedule.

    This class manages the complete sequence of walk-forward segments,
    providing iteration support and validation of the overall schedule.
    It ensures segments follow proper temporal ordering and provides
    utilities for schedule analysis.

    Parameters
    ----------
    segments : List[Segment], default empty
        List of segments in chronological order

    Notes
    -----
    The plan maintains segments in chronological order based on inference
    start times. This ensures proper temporal progression in walk-forward
    analysis and enables efficient validation of the overall schedule.
    """
    segments: List[Segment] = field(default_factory=list)
```

### Methods

#### `add_segment(segment)`

```python
def add_segment(self, segment: Segment) -> None:
    """
    Add a segment to the plan.

    Segments are automatically inserted in chronological order based
    on their inference start times.

    Parameters
    ----------
    segment : Segment
        Segment to add to the plan
    """
```

#### `validate(allow_overlaps)`

```python
def validate(self, allow_overlaps: bool = False) -> List[str]:
    """
    Validate the segment plan for temporal consistency.

    Parameters
    ----------
    allow_overlaps : bool, default False
        Whether to allow overlapping segments

    Returns
    -------
    List[str]
        List of validation warnings/errors (empty if valid)
    """
```

## SegmentResult

**Module:** `src.pipeline.walk_forward.execution`

```python
@dataclass
class SegmentResult:
    """
    Results from processing a single walk-forward segment.

    This class encapsulates the complete output from processing one segment,
    including the processed inference data and any learned processor states.
    It provides a clean interface for collecting and analyzing segment-level
    results in walk-forward workflows.

    Parameters
    ----------
    segment : Segment
        The segment that was processed to generate these results
    ds_infer : xr.Dataset
        Fully processed inference dataset after all pipeline stages:
        - Shared processors applied
        - Learn processors fitted on training data and applied to inference
        - Infer processors applied transform-only
    learn_states : List[ProcessorState]
        States learned from fitting processors on the training data.
        These are opaque objects whose structure depends on each processor.

    Notes
    -----
    The processing pipeline for each segment follows this sequence:
    1. Slice training and inference data from shared-processed dataset
    2. Fit learn processors on training slice -> learn_states
    3. Apply learn_states to inference slice (consistent transformation)
    4. Apply infer processors transform-only to inference slice -> ds_infer

    This design ensures that inference data is processed using only
    information available at training time, preventing lookahead bias
    while maintaining consistent feature engineering across time periods.
    """
```

### Properties

#### `inference_period`

```python
@property
def inference_period(self) -> tuple[np.datetime64, np.datetime64]:
    """
    Get the inference period for this result.

    Returns
    -------
    tuple[np.datetime64, np.datetime64]
        Tuple of (infer_start, infer_end) from the segment
    """
```

#### `training_period`

```python
@property
def training_period(self) -> tuple[np.datetime64, np.datetime64]:
    """
    Get the training period used to generate this result.

    Returns
    -------
    tuple[np.datetime64, np.datetime64]
        Tuple of (train_start, train_end) from the segment
    """
```

### Methods

#### `get_state_summary()`

```python
def get_state_summary(self) -> dict:
    """
    Get a summary of learned states for analysis.

    Returns
    -------
    dict
        Summary information about learned processor states
    """
```

## WalkForwardRunner

```python
@dataclass
class WalkForwardRunner:
    """
    Orchestrates walk-forward evaluation using an existing DataHandler.

    This class manages the complete walk-forward execution workflow,
    applying data processing pipelines across temporal segments with
    proper state management and efficient caching. It separates "when"
    (temporal segmentation) from "how" (data processing) following qlib's
    architectural principles.

    Parameters
    ----------
    handler : DataHandler
        Configured data handler with processing pipelines
    plan : SegmentPlan
        Walk-forward segment plan defining temporal structure
    overlap_policy : str, default "last"
        How to handle overlapping predictions ("last" or "first")
    return_segments : bool, default False
        Whether to return individual segment results

    Attributes
    ----------
    _shared : xr.Dataset, optional
        Cached shared processing view to avoid recomputation

    Notes
    -----
    The runner executes this strategy for each segment:
    1. Run shared processors once on full dataset (transform-only)
    2. For each segment:
       a. Fit learn processors on training slice to get states
       b. Apply learned states to inference slice (transform-only)
       c. Apply infer processors to inference slice (transform-only)
    3. Collect per-segment outputs as SegmentResult objects

    This design ensures:
    - No model integration (focuses purely on "when" logic)
    - Reuses DataHandler processors and semantics for consistency
    - Efficient caching of shared transformations
    - Proper temporal isolation preventing lookahead bias
    """
```

### Methods

#### `run(show_progress)`

```python
def run(self, show_progress: bool = True) -> WalkForwardResult:
    """
    Execute the complete walk-forward plan and return aggregated results.

    This method mirrors ModelRunner's gather>scatter pattern to properly
    handle rectangular expansion and NaN overlaps that occur when unstacking
    after slicing between irregular times.

    Parameters
    ----------
    show_progress : bool, default True
        Whether to show progress bar during execution

    Returns
    -------
    WalkForwardResult
        Aggregated results with processed dataset and segment metadata

    Notes
    -----
    Strategy:
    1. Compute shared view once
    2. Pre-stack shared for fast integer slicing and precompute segment bounds
    3. Create separate stacked buffers for each variable
    4. For each segment:
       a. Slice train and infer windows by integer search over stacked time vector
       b. Fit learn processors on train slice; transform infer slice using states
       c. Apply infer processors transform-only on infer slice
       d. Stack segment result and scatter into global buffers using precise indexing
    5. Unstack all buffers at the end and return aggregated dataset
    """
```

#### `run_segments(show_progress)`

```python
def run_segments(self, show_progress: bool = True) -> List[SegmentResult]:
    """
    Execute walk-forward analysis and return individual segment results.

    This method preserves the original behavior for backward compatibility
    while the main run() method now implements the gather>scatter pattern.

    Parameters
    ----------
    show_progress : bool, default True
        Whether to show progress bar during execution

    Returns
    -------
    List[SegmentResult]
        Individual results from each processed segment
    """
```

#### `run_single_segment(segment)`

```python
def run_single_segment(self, segment: Segment) -> SegmentResult:
    """
    Execute processing for a single segment.

    This method provides fine-grained control for processing individual
    segments, useful for debugging, testing, or custom workflows.

    Parameters
    ----------
    segment : Segment
        Single segment to process

    Returns
    -------
    SegmentResult
        Result from processing the segment

    Notes
    -----
    This method follows the same processing logic as run() but for
    a single segment. It's useful for:
    - Debugging specific time periods
    - Custom segment processing workflows
    - Testing processing logic on smaller datasets
    - Interactive analysis and development
    """
```

#### `get_execution_summary()`

```python
def get_execution_summary(self) -> dict:
    """
    Get summary information about the execution plan.

    Returns
    -------
    dict
        Summary statistics about the walk-forward execution including:
        - num_segments: Number of segments in the plan
        - total_training_period: Overall training period span
        - total_inference_period: Overall inference period span
        - avg_segment_gap: Average gap between segments
        - avg_train_duration: Average training window duration
        - avg_infer_duration: Average inference window duration
    """
```

---

# Model Integration API

## ModelAdapter

**Module:** `src.pipeline.model.adapter`

```python
class ModelAdapter:
    """
    Thin wrapper around any model to present a stable interface to the pipeline.
    The adapter is responsible only for converting xr.Dataset slices to numpy
    arrays, calling the underlying model, and mapping predictions back to xr.

    Conventions:
      - fit returns self so callers can write adapter.fit(...).predict(...)
      - predict returns an xr.DataArray named by output_var with dims aligned
        to the input dataset time/asset coords.
      - Missing values policy is intentionally minimal. Use processors upstream
        for real imputations. Here we support dropping invalid rows safely.
    """
```

### Methods

#### `fit(ds, features, label, sample_weight)`

```python
def fit(
    self,
    ds: xr.Dataset,
    features: Sequence[str],
    label: Optional[str] = None,
    sample_weight: Optional[xr.DataArray] = None,
) -> "ModelAdapter":
    """
    Train the model on the provided dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Training dataset
    features : Sequence[str]
        Feature variable names to use
    label : str, optional
        Label variable name for supervised learning
    sample_weight : xr.DataArray, optional
        Sample weights for training

    Returns
    -------
    ModelAdapter
        Self for method chaining

    Raises
    ------
    NotImplementedError
        Must be implemented by subclasses
    """
```

#### `predict(ds, features)`

```python
def predict(self, ds: xr.Dataset, features: Sequence[str]) -> xr.DataArray:
    """
    Generate predictions on the provided dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to generate predictions for
    features : Sequence[str]
        Feature variable names to use

    Returns
    -------
    xr.DataArray
        Predictions aligned with input dataset coordinates

    Raises
    ------
    NotImplementedError
        Must be implemented by subclasses
    """
```

#### `partial_fit(ds, features, label, sample_weight)`

```python
def partial_fit(
    self,
    ds: xr.Dataset,
    features: Sequence[str],
    label: Optional[str] = None,
    sample_weight: Optional[xr.DataArray] = None,
) -> "ModelAdapter":
    """
    Optional incremental training. Default falls back to fit.

    Parameters
    ----------
    ds : xr.Dataset
        Training dataset
    features : Sequence[str]
        Feature variable names
    label : str, optional
        Label variable name
    sample_weight : xr.DataArray, optional
        Sample weights

    Returns
    -------
    ModelAdapter
        Self for method chaining
    """
```

#### `get_state()`

```python
def get_state(self) -> Optional[Dict[str, Any]]:
    """
    Optional persistence hook. Return a JSON-serializable dict of model params.

    Returns
    -------
    dict or None
        Model state as dictionary, or None if not implemented
    """
```

#### `load_state(state)`

```python
def load_state(self, state: Dict[str, Any]) -> None:
    """
    Optional persistence hook to load params created by get_state.

    Parameters
    ----------
    state : dict
        Model state dictionary from get_state()
    """
```

## SklearnAdapter

```python
@dataclass
class SklearnAdapter(ModelAdapter):
    """
    Wrap a sklearn-like estimator exposing .fit(X, y) and .predict(X) or .predict_proba(X).

    Notes on shapes:
      - DataHandler.to_arrays_for_model emits X with shape (T, N, J) and optional y (T, N, K).
      - We flatten to 2D for sklearn: X2 = X.reshape(T*N, J) and y2 accordingly.
      - We drop rows where any feature is non-finite or y is non-finite if supervised.
      - Predictions are mapped back to a full stacked calendar and then unstacked.

    Parameters
    ----------
    model : Any
        The sklearn-like estimator.
    handler : DataHandler
        Used for consistent stacking rules via to_arrays_for_model.
    output_var : str, default "score"
        Name of the output variable in xr.
    use_proba : bool, default False
        If True, use predict_proba and select proba_index column.
    proba_index : int, default 1
        Column index to take from predict_proba.
    drop_nan_rows : bool, default True
        If True, drop rows with non-finite features before calling the model.
    """
    model: Any
    handler: DataHandler
    output_var: str = "score"
    use_proba: bool = False
    proba_index: int = 1
    drop_nan_rows: bool = True
```

### Methods

#### `fit(ds, features, label, sample_weight)`

```python
def fit(
    self,
    ds: xr.Dataset,
    features: Sequence[str],
    label: Optional[str] = None,
    sample_weight: Optional[xr.DataArray] = None,
) -> "SklearnAdapter":
    """
    Fit the sklearn model on the provided dataset.

    Converts dataset to numpy arrays, flattens to 2D, drops invalid rows,
    and calls the underlying model's fit method.

    Parameters
    ----------
    ds : xr.Dataset
        Training dataset
    features : Sequence[str]
        Feature variable names
    label : str, optional
        Label variable name for supervised learning
    sample_weight : xr.DataArray, optional
        Sample weights for training

    Returns
    -------
    SklearnAdapter
        Self for method chaining

    Raises
    ------
    ValueError
        If supervised fit is requested but no valid labels remain after preprocessing
    """
```

#### `predict(ds, features)`

```python
def predict(self, ds: xr.Dataset, features: Sequence[str]) -> xr.DataArray:
    """
    Generate predictions using the fitted sklearn model.

    Converts dataset to numpy arrays, generates predictions, and maps
    them back to xarray format with proper coordinate alignment.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to generate predictions for
    features : Sequence[str]
        Feature variable names

    Returns
    -------
    xr.DataArray
        Predictions with name=output_var, aligned with input coordinates

    Raises
    ------
    ValueError
        If stacked dataset has invalid dimensions
    AttributeError
        If use_proba=True but model lacks predict_proba method
    """
```

## ModelRunner

**Module:** `src.pipeline.model.runner`

```python
@dataclass
class ModelRunner:
    """
    Controller that mirrors WalkForwardRunner logic and plugs in a ModelAdapter.

    Strategy:
      1) Compute shared view once.
      2) Pre-stack shared for fast integer slicing and to precompute segment bounds.
      3) For each segment:
           a) Slice train and infer windows by integer search over the stacked time vector.
           b) Fit learn processors on train slice; transform infer slice using learn states.
           c) Apply infer processors transform-only on infer slice.
           d) Fit the model on processed train slice; predict on processed infer slice.
           e) Aggregate predictions into a single stacked array using the chosen policy.
      4) Unstack final prediction panel back to (year, month, day, hour, asset).

    Parameters
    ----------
    handler : DataHandler
        Configured data handler with processing pipelines
    plan : SegmentPlan
        Walk-forward segment plan
    model_factory : ModelAdapter or Callable[[], ModelAdapter]
        Factory function or instance for creating model adapters
    feature_cols : Sequence[str]
        Feature variable names to use
    label_col : str, optional
        Label variable name for supervised learning
    overlap_policy : str, default "last"
        How to handle overlapping predictions ("last" or "first")
    output_var : str, default "score"
        Name for prediction variable in output dataset
    return_model_states : bool, default True
        Whether to return model states in results
    debug_asset : str, optional
        Restrict processing to single asset for debugging
    debug_start : np.datetime64, optional
        Limit inference window start for debugging
    debug_end : np.datetime64, optional
        Limit inference window end for debugging
    """
```

### Methods

#### `run()`

```python
def run(self) -> ModelRunnerResult:
    """
    Execute the complete model training and prediction workflow.

    Runs walk-forward analysis with model training and prediction for each
    segment, aggregating results using the configured overlap policy.

    Returns
    -------
    ModelRunnerResult
        Results containing:
        - pred_ds: Dataset with predictions
        - segment_states: Per-segment metadata
        - attrs: Execution metadata

    Notes
    -----
    The execution follows this pattern:
    1. Compute and cache shared processing view
    2. Pre-stack dataset and compute segment boundaries
    3. For each segment:
       - Slice training and inference data
       - Fit learn processors and apply to inference
       - Create fresh model instance via factory
       - Train model on processed training data
       - Generate predictions on processed inference data
       - Scatter predictions into global buffer
    4. Unstack final predictions to original dimensions
    5. Return aggregated results with metadata

    Progress is shown via tqdm progress bar with segment-level updates.
    """
```

## ModelRunnerResult

```python
@dataclass
class ModelRunnerResult:
    """
    Final result of a modeling run.

    Attributes
    ----------
    pred_ds : xr.Dataset
        Dataset containing model predictions with configured output variable
    segment_states : List[Dict[str, Any]]
        Metadata for each processed segment including:
        - segment: Segment index
        - infer_start/infer_end: Inference period timestamps
        - infer_rows: Number of inference rows processed
        - num_learn_states: Number of learned processor states
        - status: Processing status
    attrs : Dict[str, Any]
        Run-level metadata including:
        - overlap_policy: Policy used for overlapping predictions
        - segments: Total number of segments
        - created_at_unix: Timestamp of creation
    """
    pred_ds: xr.Dataset
    segment_states: List[Dict[str, Any]] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
```

---



---

# Configuration Classes

## HandlerConfig

**Module:** `src.pipeline.data_handler.config`

```python
@dataclass
class HandlerConfig:
    """
    Configuration for DataHandler pipeline construction.
    
    This class defines the complete configuration for a data processing pipeline,
    including the processors for each stage and the execution mode. It follows
    qlib's pattern of separating "how data is processed" from "when".
    
    Parameters
    ----------
    shared : Sequence[Processor], default empty
        Processors that run once on the full dataset before segmentation.
        These are typically stateless transforms like data cleaning.
    learn : Sequence[Processor], default empty
        Processors that fit on training data and transform both train and inference.
        These learn parameters during training (e.g., normalization statistics).
    infer : Sequence[Processor], default empty
        Processors that run transform-only on inference data after learn processors.
        These are typically post-processing steps that don't require fitting.
    mode : PipelineMode, default INDEPENDENT
        Execution mode controlling how shared, learn, and infer pipelines combine.
    feature_cols : list of str, optional
        Names of columns to be treated as features for modeling.
    label_cols : list of str, optional
        Names of columns to be treated as labels/targets for modeling.
        
    Notes
    -----
    The pipeline execution follows this pattern:
    
    INDEPENDENT mode:
        shared -> learn (fit+transform on train, transform on infer)
        shared -> infer (transform-only)
        
    APPEND mode:
        shared -> infer -> learn (sequential, learn sees infer outputs)
        
    This design allows for flexible data processing workflows while maintaining
    clear separation between training and inference data flows.
    """
    shared: Sequence[Processor] = field(default_factory=list)
    learn: Sequence[Processor] = field(default_factory=list)
    infer: Sequence[Processor] = field(default_factory=list)
    mode: PipelineMode = PipelineMode.INDEPENDENT
    feature_cols: Optional[List[str]] = None
    label_cols: Optional[List[str]] = None
```

## SegmentConfig

**Module:** `src.pipeline.walk_forward.segments`

```python
@dataclass
class SegmentConfig:
    """
    Configuration for generating rolling walk-forward schedules.
    
    This class defines the parameters needed to automatically generate
    a sequence of walk-forward segments with consistent spacing and sizing.
    It supports flexible gap handling and optional data clipping for
    robust backtesting workflows.
    
    Parameters
    ----------
    start : np.datetime64
        First timestamp to consider for scheduling
    end : np.datetime64
        Last timestamp to consider for scheduling  
    train_span : np.timedelta64
        Length of each training window
    infer_span : np.timedelta64
        Length of each inference window
    step : np.timedelta64
        Shift between consecutive segments (typically equal to infer_span)
    gap : np.timedelta64, default 0
        Optional gap between train_end and infer_start to avoid leakage
    clip_to_data : bool, default True
        Whether to clip each segment to the dataset's valid time domain
        
    Notes
    -----
    The configuration generates segments following this pattern:
    1. Start at the configured start time
    2. Create training window of train_span duration  
    3. Add gap between training and inference
    4. Create inference window of infer_span duration
    5. Advance by step amount for next segment
    6. Repeat until end time is reached
    
    The clip_to_data option ensures segments don't extend beyond actual
    data availability when working with real datasets that may have
    irregular time coverage.
    
    Examples
    --------
    Monthly walk-forward with 12-month training, 1-month inference:
    
    >>> config = SegmentConfig(
    ...     start=np.datetime64('2020-01-01'),
    ...     end=np.datetime64('2023-12-31'),
    ...     train_span=np.timedelta64(365, 'D'),  # 12 months
    ...     infer_span=np.timedelta64(30, 'D'),   # 1 month
    ...     step=np.timedelta64(30, 'D'),         # 1 month step
    ...     gap=np.timedelta64(1, 'D'),           # 1 day gap
    ...     clip_to_data=True
    ... )
    """
    start: np.datetime64
    end: np.datetime64
    train_span: np.timedelta64
    infer_span: np.timedelta64
    step: np.timedelta64
    gap: np.timedelta64 = np.timedelta64(0, 'D')
    clip_to_data: bool = True
```

---

# Enums and Base Classes

## View

**Module:** `src.pipeline.data_handler.core`

```python
class View(Enum):
    """
    Enumeration of data processing views in the pipeline.
    
    This enum defines the different data processing contexts,
    separating training and inference data flows.
    
    Attributes
    ----------
    RAW : str
        Raw data loaded as-is after feature graph, no train/infer specialization
    LEARN : str
        Fit+transform path used for training with full processor fitting
    INFER : str
        Transform-only path used for prediction with pre-fitted processors
    """
    RAW = "raw"
    LEARN = "learn"
    INFER = "infer"
```

## PipelineMode

**Module:** `src.pipeline.data_handler.core`

```python
class PipelineMode(Enum):
    """
    Enumeration of pipeline execution modes.
    
    This enum defines how the shared, learn, and infer processor pipelines
    are combined and executed.
    
    Attributes
    ----------
    INDEPENDENT : str
        Two independent branches (shared -> learn, shared -> infer)
    APPEND : str
        Sequential execution (shared -> infer -> learn, where learn sees extra steps)
    """
    INDEPENDENT = "independent"
    APPEND = "append"
```

## ProcessorContract

**Module:** `src.pipeline.data_handler.core`

```python
class ProcessorContract:
    """
    Abstract base class defining the processor contract for xarray I/O.
    
    This class establishes the interface that all processors must implement,
    following the scikit-learn fit/transform pattern adapted for xarray datasets.
    
    The processor contract follows these conventions:
    - fit(ds) -> ProcessorState: learn parameters and return an opaque state object
    - transform(ds, state=None) -> xr.Dataset: apply the transformation using state
    - fit_transform(ds) -> (output, state): convenience method combining both
    
    State management conventions:
    - state_ds stores only parameters needed at inference time
    - Keep state compact and aligned by dimensions when possible
    - Use namespaced parameter names: f"{self.name}_param__{var}"
    
    Variable handling conventions:
    - Default to writing new variables with suffix/prefix for safety
    - Provide parameter to allow in-place replacement when desired
    """
```

### Methods

#### `fit(ds: xr.Dataset) -> ProcessorState`

Learn parameters from the input dataset.

**Parameters:**
- `ds` (xr.Dataset): Input dataset to learn parameters from

**Returns:**
- `ProcessorState`: Opaque processor state containing only the parameters needed at inference time

#### `transform(ds: xr.Dataset, state: Optional[ProcessorState] = None) -> xr.Dataset`

Apply the transformation to the input dataset.

**Parameters:**
- `ds` (xr.Dataset): Input dataset to transform
- `state` (ProcessorState, optional): State returned by fit. If None, processor must behave as stateless.

**Returns:**
- `xr.Dataset`: Transformed dataset

#### `fit_transform(ds: xr.Dataset) -> Tuple[xr.Dataset, ProcessorState]`

Convenience method that combines fit and transform operations.

**Parameters:**
- `ds` (xr.Dataset): Input dataset to fit and transform

**Returns:**
- `tuple`: (transformed_dataset, state_object)

---

# Utility Functions

## make_plan

**Module:** `src.pipeline.walk_forward.planning`

```python
def make_plan(config: SegmentConfig, ds_for_bounds: Optional[xr.Dataset] = None) -> SegmentPlan:
    """
    Generate a complete walk-forward segment plan from configuration.
    
    This function creates a contiguous walk-forward schedule across the
    specified time range, with optional clipping to dataset boundaries
    to ensure segments don't extend beyond available data.
    
    Parameters
    ----------
    config : SegmentConfig
        Configuration parameters for segment generation
    ds_for_bounds : xr.Dataset, optional
        Dataset to use for boundary clipping if config.clip_to_data is True.
        If provided, segments will be clipped to the dataset's valid time domain.
        
    Returns
    -------
    SegmentPlan
        Generated plan containing all walk-forward segments
        
    Raises
    ------
    ValueError
        If step size is not positive
    FileNotFoundError
        If ds_for_bounds is required but not provided
        
    Notes
    -----
    The generation process follows these steps:
    1. Normalize all timestamps to nanosecond precision for consistency
    2. Extract dataset bounds if clipping is enabled
    3. Generate segments by advancing cursor in step increments:
       - Create training window: [cursor, cursor + train_span]
       - Add gap: training_end + gap
       - Create inference window: [train_end + gap, train_end + gap + infer_span]
       - Clip to dataset bounds if enabled
       - Skip segments with collapsed windows after clipping
    4. Continue until cursor exceeds end time
    
    The clipping process ensures no segment extends beyond actual data
    availability, which is crucial for robust backtesting with real datasets
    that may have irregular coverage or business day calendars.
    
    Examples
    --------
    Basic monthly walk-forward plan:
    
    >>> config = SegmentConfig(
    ...     start=np.datetime64('2020-01-01'),
    ...     end=np.datetime64('2020-06-30'),
    ...     train_span=np.timedelta64(90, 'D'),
    ...     infer_span=np.timedelta64(30, 'D'),
    ...     step=np.timedelta64(30, 'D'),
    ...     gap=np.timedelta64(1, 'D')
    ... )
    >>> plan = make_plan(config)
    >>> print(f"Generated {len(plan)} segments")
    
    Plan with dataset boundary clipping:
    
    >>> plan = make_plan(config, ds_for_bounds=financial_dataset)
    >>> validation_issues = plan.validate()
    >>> if not validation_issues:
    ...     print("Plan is valid")
    """
```

## expand_plan_coverage

**Module:** `src.pipeline.walk_forward.planning`

```python
def expand_plan_coverage(plan: SegmentPlan, 
                        target_start: Optional[np.datetime64] = None,
                        target_end: Optional[np.datetime64] = None,
                        config: Optional[SegmentConfig] = None) -> SegmentPlan:
    """
    Expand an existing plan to cover additional time periods.
    
    This function extends an existing segment plan to cover a wider time
    range, useful for expanding backtests or filling gaps in coverage.
    
    Parameters
    ----------
    plan : SegmentPlan
        Existing plan to expand
    target_start : np.datetime64, optional
        Target start time (expand backwards if before current start)
    target_end : np.datetime64, optional
        Target end time (expand forwards if after current end)
    config : SegmentConfig, optional
        Configuration for generating new segments (required if expanding)
        
    Returns
    -------
    SegmentPlan
        Expanded plan covering the target time range
        
    Raises
    ------
    ValueError
        If config is required but not provided
        
    Notes
    -----
    The expansion process:
    1. Determines current plan coverage
    2. Generates additional segments before/after existing coverage
    3. Merges new segments with existing plan
    4. Validates temporal consistency
    
    This is useful for iteratively building walk-forward plans or
    expanding existing plans when new data becomes available.
    """
```

## optimize_plan_for_dataset

**Module:** `src.pipeline.walk_forward.planning`

```python
def optimize_plan_for_dataset(plan: SegmentPlan, ds: xr.Dataset, 
                             min_train_samples: int = 100,
                             min_infer_samples: int = 10) -> SegmentPlan:
    """
    Optimize a segment plan based on actual data availability.
    
    This function analyzes data availability in each segment and removes
    or adjusts segments that don't have sufficient data for reliable
    model training and evaluation.
    
    Parameters
    ----------
    plan : SegmentPlan
        Original segment plan to optimize
    ds : xr.Dataset
        Dataset to analyze for data availability
    min_train_samples : int, default 100
        Minimum number of valid samples required in training period
    min_infer_samples : int, default 10
        Minimum number of valid samples required in inference period
        
    Returns
    -------
    SegmentPlan
        Optimized plan with segments having sufficient data
        
    Notes
    -----
    The optimization process:
    1. For each segment, slice the dataset to training and inference periods
    2. Count valid (non-NaN) samples in key variables
    3. Remove segments that don't meet minimum sample requirements
    4. Optionally adjust segment boundaries to maximize data usage
    
    This is particularly useful when working with datasets that have
    irregular coverage, holidays, or missing data periods.
    """
```

---

# Additional Processors

## FormulaEval

**Module:** `src.pipeline.data_handler.processors`

```python
@dataclass
class FormulaEval(Processor):
    """
    Formula evaluation processor using the AST system.
    
    This processor compiles and evaluates a set of YAML-defined formulas using the
    FormulaManager and AST system. It can optionally use JAX JIT compilation for
    performance optimization.
    
    Parameters
    ----------
    name : str
        Processor name identifier
    formula_configs : dict
        Dictionary mapping formula_name -> list of config dicts
        Example: {"sma": [{"window": 100}, {"window": 200}], "rsi": [{"window": 14}]}
    static_context : dict, optional
        Dictionary of constants/functions to provide to formulas
        Example: {"price": "close"} plus get_function_context()
    use_jit : bool, default True
        Whether to wrap evaluation in jax.jit for performance
    defs_dir : str, optional
        Optional directory to load custom formula YAML files
    assign_in_place : bool, default True
        If True, merge results into dataset; else results are namespaced with prefix
    prefix : str, optional
        Optional variable name prefix when assign_in_place=False
        
    Notes
    -----
    The compiled callable is cached on the instance (not serialized in state) for
    performance. The processor builds a single compiled function that:
    1) Prepares dataset for JIT compilation
    2) Calls FormulaManager.evaluate_bulk with context
    3) Returns an xr.Dataset of computed results
    
    Examples
    --------
    >>> formulas = {
    ...     "sma": [{"window": 100}, {"window": 200}],
    ...     "rsi": [{"window": 14}]
    ... }
    >>> processor = FormulaEval(
    ...     name="formulas",
    ...     formula_configs=formulas,
    ...     static_context={"price": "close"},
    ...     use_jit=True
    ... )
    """
```

### Methods

#### `fit(ds: xr.Dataset) -> None`

Stateless fit that ensures compiled function is ready.

This processor is stateless in terms of learned parameters, but it does
compile the formula evaluation function during fit for efficiency.

**Parameters:**
- `ds` (xr.Dataset): Input dataset (used to trigger compilation)

**Returns:**
- `None`

#### `transform(ds: xr.Dataset, state: Optional[ProcessorState] = None) -> xr.Dataset`

Apply compiled formula evaluation to dataset.

**Parameters:**
- `ds` (xr.Dataset): Input dataset to evaluate formulas on
- `state` (ProcessorState, optional): Unused for this stateless processor

**Returns:**
- `xr.Dataset`: Dataset with formula results merged or returned separately

---

**End of API Reference**

For conceptual documentation and usage examples, see: [full_docs.md](full_docs.md)


For conceptual documentation and tutorials, see the main documentation file: [full_docs.md](full_docs.md)
