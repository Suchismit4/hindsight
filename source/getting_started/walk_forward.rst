Walk-Forward Analysis and Temporal Segmentation
================================================

Hindsight separates temporal logic from data processing. The ``walk_forward``
package describes how training and inference windows advance through time, and
wires that temporal structure into the data pipeline. This page explains the
core abstractions and how they interact with ``DataHandler``.

Core Abstractions
-----------------

``Segment``
    Immutable descriptor of one walk-forward step. Each segment carries
    ``train_start``, ``train_end``, ``infer_start``, and ``infer_end``
    timestamps. Train boundaries are inclusive; inference end is exclusive to
    prevent overlap between consecutive inference windows.

``SegmentPlan``
    An ordered collection of segments. It provides validation helpers,
    coverage summaries, and iteration semantics.

``SegmentConfig``
    Declarative specification used to generate a ``SegmentPlan``. It defines
    the global start/end range, training window length, inference window
    length, step size, optional gap between train end and infer start, and
    whether to clip segments to dataset boundaries.

Generating Plans
----------------

``make_plan`` builds a ``SegmentPlan`` from a ``SegmentConfig``. Passing
``ds_for_bounds`` clips generated segments to the actual data range, so you
do not have to compute boundaries manually.

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
       clip_to_data=True,
   )

   plan = make_plan(cfg, ds_for_bounds=dataset)

   issues = plan.validate()
   if issues:
       raise ValueError(issues)

A ``gap`` of one day ensures the last training observation is at least one day
before the first inference observation. This prevents any same-bar leakage at
the boundary.

Execution Flow
--------------

``WalkForwardRunner`` orchestrates data processing over a ``SegmentPlan``.
Given a ``DataHandler`` and a plan, it:

1. Computes the shared view once by running shared processors over the full
   base dataset.
2. Pre-stacks the shared dataset so each segment can be sliced by integer
   bounds rather than timestamp lookup (faster on large datasets).
3. For each segment:

   - Identifies train and infer slices by pre-computed integer indices.
   - Applies learn processors in ``fit_transform`` mode on the train slice,
     capturing states.
   - Applies learn processors in transform-only mode plus infer processors
     on the inference slice, using the stored states.
   - Adds inference outputs to a global buffer according to the overlap policy.

4. Unstacks the buffer back into the original calendar structure and returns a
   ``WalkForwardResult`` with the aggregated dataset and per-segment metadata.

.. code-block:: python

   from src.pipeline import WalkForwardRunner

   runner = WalkForwardRunner(handler=handler, plan=plan)
   result = runner.run()

   processed_ds    = result.processed_ds    # aggregated inference dataset
   segment_states  = result.segment_states  # per-segment metadata and states

State Handling
--------------

Each learn-stage processor produces an xarray ``Dataset`` containing fitted
parameters (means, variances, bin edges, etc.). ``WalkForwardRunner`` stores
these states alongside segment metadata so you can inspect how learned
parameters drift over time. This also guarantees that inference slices are
transformed using only the statistics that were available at training time.

Overlap Policies
----------------

When consecutive inference windows overlap (common with rolling step sizes),
``WalkForwardRunner`` resolves conflicts using one of two strategies:

- ``"last"`` (default): later segments overwrite earlier predictions where
  they overlap. The most recent training window's statistics take precedence.
- ``"first"``: the first segment to produce a value keeps it; later segments
  only fill positions that are still ``NaN``.

Advanced Utilities
------------------

``expand_plan_coverage``
    Extends an existing plan to cover earlier or later periods using the same
    ``SegmentConfig`` parameters. Useful when you need to prepend a warm-up
    period or extend a plan forward without regenerating from scratch.

``optimize_plan_for_dataset``
    Removes segments that do not meet minimum sample requirements for training
    or inference based on actual data coverage. Segments with too few
    non-``NaN`` rows are dropped before execution to avoid fitting models on
    insufficient data.

Integration with ModelRunner
----------------------------

``ModelRunner`` reuses the same segmentation logic but adds model fitting and
prediction on top of the processed slices. It pre-computes the same integer
slice boundaries and applies the same overlap policy. Understanding the base
walk-forward execution model first makes the ``ModelRunner`` behavior
straightforward: it is the same loop with model fit/predict inserted between
the learn and scatter steps.

Where to Go Next
----------------

- :doc:`model_integration` shows the complete ``ModelRunner`` workflow, including
  the adapter interface and the factory pattern for per-segment model isolation.
- :doc:`execution_analysis` covers result inspection and output analysis patterns.
- :doc:`../api/walk_forward` is the API reference for all segment planning and
  execution classes.
