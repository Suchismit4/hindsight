Walk-Forward API Reference
===========================

.. currentmodule:: src.pipeline.walk_forward

The ``walk_forward`` package manages the "when" side of the pipeline:
how training and inference windows are defined, planned, and executed across
time. It is independent of any particular model or data transformation, which
is what allows ``WalkForwardRunner`` to be used as the data-only execution
path and ``ModelRunner`` to build on top of it.

Segment Abstractions
--------------------

``Segment`` is the atomic unit: one train/infer window pair. ``SegmentPlan``
is the ordered sequence of segments that the runners iterate over.
``SegmentConfig`` is the declarative description of the schedule.

.. autosummary::
   :toctree: generated/

   Segment
   SegmentPlan
   SegmentConfig

Planning Functions
------------------

``make_plan`` is the primary factory for creating plans from a ``SegmentConfig``.
The utility functions handle edge cases and plan manipulation.

.. autosummary::
   :toctree: generated/

   make_plan
   expand_plan_coverage
   optimize_plan_for_dataset

Execution
---------

``WalkForwardRunner`` applies a ``DataHandler`` across a ``SegmentPlan``,
aggregating inference outputs into a global dataset. Results carry per-segment
metadata and learned processor states.

.. autosummary::
   :toctree: generated/

   WalkForwardRunner
   SegmentResult
   WalkForwardResult

Detailed Module Documentation
-------------------------------

.. automodule:: src.pipeline.walk_forward.segments
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.pipeline.walk_forward.planning
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.pipeline.walk_forward.execution
   :members:
   :undoc-members:
   :show-inheritance:
