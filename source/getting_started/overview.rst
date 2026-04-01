Overview
========

Hindsight is a Python library for multi-asset, time-series quantitative research.
It provides well-defined layers for loading data, computing features, transforming
datasets across temporal segments, and integrating machine learning models without
leaking future information into training or inference.

What This Guide Covers
----------------------

- How the library is structured around data transformations ("how") and temporal
  segmentation ("when").
- The xarray panel layout that every subsystem shares.
- The two execution paths: programmatic Python API and YAML-driven pipeline spec.
- A concise end-to-end example using the Python API.

The Two Ways to Use Hindsight
------------------------------

**Python API (direct)**
    Instantiate ``DataHandler``, ``SegmentConfig``, ``ModelRunner``, and
    related objects directly in Python. This is the right path when you want
    interactive exploration, custom loaders not yet expressed in YAML, or
    tight integration with an existing codebase.

**YAML pipeline spec (declarative)**
    Describe the full workflow in a ``PipelineSpec`` YAML file and run it
    through ``PipelineExecutor``. This path is content-addressably cached at
    each stage (L1 raw → L2 postprocessed → L3 features → L4 preprocessed
    → L5 model), so intermediate results survive across runs. See
    :doc:`yaml_pipeline` for authoring guidance.

Both paths use the same underlying subsystems. The YAML executor is
orchestration, not a separate implementation.

Architectural Separation
------------------------

**Data transformations (the "how")**
    Managed by ``src.pipeline.data_handler``. A ``DataHandler`` applies
    processors in three ordered stages:

    - ``shared`` processors run once on the entire dataset before any temporal
      slicing. They are appropriate for stateless operations (forward-fill,
      formula evaluation) or expensive transforms you want cached across
      segments.
    - ``learn`` processors fit on a training window and store a compact state
      dataset. That state is reapplied when transforming the corresponding
      inference window, which is what prevents lookahead bias in normalization
      or similar statistics.
    - ``infer`` processors run transform-only operations on inference slices,
      after learn-stage states have been applied. Portfolio construction steps
      (cross-sectional sorts, portfolio returns) typically live here.

    Processors live in ``src.pipeline.data_handler.processors`` and operate on
    xarray objects so they broadcast cleanly across ``asset`` and the
    hierarchical time dimensions.

**Temporal segmentation (the "when")**
    Exposed through ``src.pipeline.walk_forward``. Key abstractions:

    - ``Segment``: one walk-forward step, carrying ``train_start``,
      ``train_end``, ``infer_start``, and ``infer_end`` timestamps.
    - ``SegmentPlan``: an ordered list of segments.
    - ``SegmentConfig``: declarative description of a rolling or expanding
      schedule. ``make_plan`` converts it into a ``SegmentPlan`` while
      clipping to dataset boundaries and inserting optional gaps.

    ``WalkForwardRunner`` applies a ``DataHandler`` over each segment in a
    plan: shared view once, then per-segment learn/infer application. It
    aggregates processed inference panels back into a global dataset using a
    configurable overlap policy.

**Model integration**
    Resides in ``src.pipeline.model``. ``ModelRunner`` mirrors
    ``WalkForwardRunner`` but adds per-segment model fitting and prediction.
    ``ModelAdapter`` objects—``SklearnAdapter`` is the shipped implementation—
    translate between xarray slices and model-specific APIs. A factory
    callable instantiates a fresh adapter per segment to prevent cross-segment
    parameter leakage.

**Formula engine**
    The ``src.data.ast`` package provides a declarative formula language for
    feature engineering. ``FormulaEval`` (a ``DataHandler`` processor) wraps
    this engine, evaluating YAML-defined formulas and merging outputs into the
    dataset. The engine supports dependency resolution, static context aliases,
    and optional JAX JIT compilation.

Core Data Structures
--------------------

The framework uses ``xarray.Dataset`` and ``xarray.DataArray`` objects with a
hierarchical calendar:

- ``year``, ``month``, ``day`` (and optionally ``hour``) dims form a rectangular grid.
- ``asset`` dimension indexes securities or instruments.
- ``time`` coordinate holds ``datetime64`` values aligned to the stacked calendar.
- ``time_flat`` (when present) is a 1-D index corresponding to the stacked view
  used internally by runners for efficient integer slicing.

Every processor, planner, and adapter assumes this layout, which is what
enables vectorized cross-sectional and time-series operations without
reshaping data on every call.

End-to-End Flow (Python API)
-----------------------------

1. Load or construct an xarray dataset. The ``DataManager`` class under
   ``src.data.managers`` handles YAML-configured loading from registered
   providers; for custom sources, build the dataset directly.
2. Configure a ``HandlerConfig`` with the processors you need. Instantiate
   ``DataHandler`` with the base dataset.
3. Describe your walk-forward schedule with ``SegmentConfig`` and call
   ``make_plan``.
4. Choose a runner:

   - ``WalkForwardRunner`` for processed datasets without model predictions.
   - ``ModelRunner`` for walk-forward training and prediction aggregation.

5. Inspect results: both runners return objects containing the aggregated
   output dataset, per-segment metadata, and learned states.

Concise Example
---------------

.. code-block:: python

   import numpy as np
   import xarray as xr
   from src.pipeline import (
       DataHandler, HandlerConfig, PipelineMode,
       PerAssetFFill, CSZScore, make_plan, SegmentConfig,
   )
   from src.pipeline.model import ModelRunner, SklearnAdapter
   from sklearn.linear_model import Ridge

   # raw_ds is an xarray.Dataset with (year, month, day, asset) dims.

   config = HandlerConfig(
       shared=[PerAssetFFill(name="ffill")],
       learn=[CSZScore(name="norm", vars=["close"])],
       mode=PipelineMode.INDEPENDENT,
       feature_cols=["close_norm"],
       label_cols=["fwd_return"]
   )
   handler = DataHandler(base=raw_ds, config=config)

   seg_cfg = SegmentConfig(
       start=np.datetime64("2020-01-01"),
       end=np.datetime64("2023-01-01"),
       train_span=np.timedelta64(365, "D"),
       infer_span=np.timedelta64(30, "D"),
       step=np.timedelta64(30, "D"),
       gap=np.timedelta64(1, "D"),
   )
   plan = make_plan(seg_cfg, ds_for_bounds=raw_ds)

   def make_adapter():
       return SklearnAdapter(
           model=Ridge(),
           handler=handler,
           output_var="pred",
       )

   runner = ModelRunner(
       handler=handler,
       plan=plan,
       model_factory=make_adapter,
       feature_cols=config.feature_cols,
       label_col=config.label_cols[0],
   )
   result = runner.run()
   predictions = result.pred_ds["pred"]

Where to Go Next
----------------

- :doc:`data_loading` describes dataset construction and the ``DataManager`` API.
- :doc:`yaml_pipeline` explains the YAML spec format and ``PipelineExecutor``.
- :doc:`data_handler` covers processor design and the three-stage pipeline in depth.
- :doc:`feature_engineering` documents the built-in processors and the formula engine.
- :doc:`walk_forward` covers segment planning and execution internals.
- :doc:`model_integration` explains adapters and the ``ModelRunner`` workflow.
