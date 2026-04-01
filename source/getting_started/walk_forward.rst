Walk-Forward Analysis and Temporal Segmentation
==============================================

Hindsight separates temporal logic from data processing. The ``walk_forward`` package wires this up by describing how training and inference windows advance through time. This section explains the core abstractions and how they interact with ``DataHandler``.

Core Abstractions
-----------------

``Segment``
    Immutable descriptor of one walk-forward step. Each segment carries ``train_start``, ``train_end``, ``infer_start``, and ``infer_end`` timestamps. Train boundaries are inclusive; inference end is exclusive to avoid overlap between consecutive segments.

``SegmentPlan``
    Ordered collection of segments. It offers validation helpers, coverage summaries, and iteration semantics.

``SegmentConfig``
    Declarative specification used to generate a ``SegmentPlan``. It defines the global start/end range, training window length, inference window length, step size, optional gap, and whether to clip to dataset boundaries.

Generating Plans
----------------

``make_plan`` builds a plan from a ``SegmentConfig``. It can optionally inspect a dataset for boundary clipping, ensuring segments do not extend beyond data availability.

.. code-block:: python

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

Execution Flow
--------------

``WalkForwardRunner`` orchestrates data processing over a ``SegmentPlan``. Given a ``DataHandler`` and a plan, it:

1. Computes the shared view once (using the handler’s shared processors).
2. Stacks the shared dataset for efficient integer slicing.
3. For each segment:
   - Identifies train and infer slices by integer bounds (avoiding expensive datetime indexing).
   - Applies learn processors in fit/transform mode on the train slice, capturing states.
   - Applies learn processors in transform-only mode plus infer processors on the inference slice.
   - Adds inference outputs to a global buffer according to the overlap policy ("last" or "first").
4. Unstacks the buffers back into the original calendar structure and returns a ``WalkForwardResult`` with the aggregated dataset and per-segment metadata.

State Handling
--------------

Each learn-stage processor produces an xarray ``Dataset`` containing the fitted parameters (means, variances, etc.). ``WalkForwardRunner`` stores these states alongside the segment metadata so you can inspect how parameters evolve over time. This also ensures inference slices reuse the exact statistics learned on the corresponding training window.

Overlap Policies
----------------

Segments may overlap on inference windows (e.g., rolling windows). ``WalkForwardRunner`` supports two strategies:

- ``last`` (default): Later segments overwrite earlier predictions where they overlap.
- ``first``: First segment to produce a value keeps it; later segments only fill holes.

Advanced Utilities
------------------

``expand_plan_coverage``
    Extend an existing plan to cover earlier or later periods using the same configuration.

``optimize_plan_for_dataset``
    Remove segments that do not meet minimum sample requirements for training or inference based on actual data coverage.

``run_segments``
    Execute the pipeline segment-by-segment and return the individual ``SegmentResult`` objects instead of the aggregated dataset.

Integration with ModelRunner
----------------------------

``ModelRunner`` reuses the same segmentation logic but introduces model fitting/prediction on top of the processed slices. Understanding the base ``WalkForwardRunner`` is essential because ``ModelRunner`` piggybacks on its shared view and state management routines.
