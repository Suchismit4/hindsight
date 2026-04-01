Pipeline API Reference
======================

.. currentmodule:: src.pipeline

The ``src.pipeline`` package is the main entry point for the Hindsight
framework. It re-exports the core classes from ``data_handler``,
``walk_forward``, and ``model`` so you can import everything from a single
namespace.

Data Handler
------------

``DataHandler`` applies processors across shared, learn, and infer stages.
``HandlerConfig`` is the declarative description of those stages. ``View``
and ``PipelineMode`` are the enums that control access patterns and branching
behavior.

.. autosummary::
   :toctree: generated/

   DataHandler
   HandlerConfig
   View
   PipelineMode

Processors
----------

Built-in processors implement the ``ProcessorContract`` interface. All accept
xarray ``Dataset`` inputs and return xarray ``Dataset`` outputs.

.. autosummary::
   :toctree: generated/

   ProcessorContract
   Processor
   PerAssetFFill
   CSZScore
   FormulaEval

Walk-Forward Planning
---------------------

``SegmentConfig`` describes a rolling or expanding schedule. ``make_plan``
converts it into an ordered ``SegmentPlan``. The utility functions handle
edge cases like plan extension and dataset-constrained filtering.

.. autosummary::
   :toctree: generated/

   Segment
   SegmentPlan
   SegmentConfig
   make_plan
   expand_plan_coverage
   optimize_plan_for_dataset

Walk-Forward Execution
----------------------

``WalkForwardRunner`` applies a ``DataHandler`` over a ``SegmentPlan`` without
any model predictions. Use it for pure data processing or factor construction
pipelines.

.. autosummary::
   :toctree: generated/

   WalkForwardRunner
   SegmentResult
   WalkForwardResult

Detailed Module Documentation
-------------------------------

.. automodule:: src.pipeline
   :members:
   :undoc-members:
   :show-inheritance:
