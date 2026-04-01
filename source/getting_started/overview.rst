Hindsight Pipeline Framework Overview
=====================================

The Hindsight Pipeline Framework is a Python library for multi-asset, time-series research. It provides well-defined layers for transforming large xarray-based datasets, planning walk-forward segments, and integrating machine learning models without leaking future information.

What This Guide Covers
----------------------

- How the pipeline is structured around "how" (data transformations) and "when" (temporal segmentation).
- Core modules that form the public API.
- Data flow from raw datasets to aggregated predictions.
- A short example showing the typical control flow.

Architectural Separation
------------------------

**Data transformations (the "how")**
    Managed by ``src.pipeline.data_handler``. A ``DataHandler`` applies processors in three stages:

    - ``shared`` processors run once on the entire dataset. They are stateless or cache their output for both training and inference paths.
    - ``learn`` processors fit on training data and produce state objects (typically xarray Datasets) that are later reused when transforming inference slices.
    - ``infer`` processors run transform-only operations after the learn stage to finish inference-specific adjustments.

    Typical processors live in ``src.pipeline.data_handler.processors``. They operate on xarray objects so they can broadcast across dimensions such as ``asset`` and the flattened calendar index.

**Temporal segmentation (the "when")**
    Exposed through ``src.pipeline.walk_forward``. Key abstractions:

    - ``Segment`` defines train/infer windows for a single iteration.
    - ``SegmentPlan`` is an ordered list of segments.
    - ``SegmentConfig`` lets you describe a rolling or expanding schedule; ``make_plan`` converts that configuration into a ``SegmentPlan`` while clipping to available data and optionally inserting gaps.

    ``WalkForwardRunner`` takes a ``DataHandler`` and a ``SegmentPlan``. It applies the handler to each segment, respecting the train/infer boundaries, capturing learned states per segment, and aggregating processed inference panels back into a global Dataset.

**Model integration**
    Resides in ``src.pipeline.model``. ``ModelRunner`` mirrors ``WalkForwardRunner`` but adds per-segment model fitting and prediction. ``ModelAdapter`` objects (for example, ``SklearnAdapter``) act as bridges between xarray slices and model-specific APIs. A factory function instantiates a fresh adapter per segment to avoid cross-segment leakage.

Core Data Structures
--------------------

The framework consistently uses ``xarray.Dataset`` and ``xarray.DataArray`` objects that share a hierarchical calendar:

- ``year``, ``month``, ``day`` (and optionally ``hour``) dims form a rectangular grid.
- ``asset`` dimension indexes securities.
- ``time`` coordinate holds datetime64 values; ``time_flat`` (if present) is a flattened index aligned with stacked views.

Processors, planners, and adapters all rely on this layout to perform vectorized operations without copying large arrays.

End-to-End Flow
---------------

1. Load or construct an xarray dataset (see ``DataManager`` utilities under ``src.data``).
2. Configure a ``HandlerConfig`` with the processors you need. Instantiate ``DataHandler`` with the raw dataset.
3. Describe your walk-forward schedule using ``SegmentConfig`` and call ``make_plan``.
4. Choose an execution path:

   - ``WalkForwardRunner`` if you only need processed datasets per segment.
   - ``ModelRunner`` if you also want model predictions aggregated back into a Dataset.

5. Inspect results: both runners return objects with the aggregated output, per-segment metadata, and optional learned states.

Concise Example
---------------

.. code-block:: python

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

Where to Go Next
----------------

- ``getting_started/data_loading`` describes dataset requirements.
- ``getting_started/data_handler`` and ``getting_started/feature_engineering`` dive into processor design.
- ``getting_started/walk_forward`` documents segment planning and execution internals.
- ``getting_started/model_integration`` covers adapters and model runners in more depth.
