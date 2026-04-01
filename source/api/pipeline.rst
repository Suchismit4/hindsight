Pipeline API Reference
=====================

.. currentmodule:: src.pipeline

This module provides the main entry point for the Hindsight Pipeline Framework, offering convenient access to all core components through a unified interface.

Core Classes and Functions
---------------------------

.. autosummary::
   :toctree: generated/

   DataHandler
   HandlerConfig
   View
   PipelineMode
   Segment
   SegmentPlan
   SegmentConfig
   make_plan
   expand_plan_coverage
   optimize_plan_for_dataset

Data Processing Components
--------------------------

.. autosummary::
   :toctree: generated/

   ProcessorContract
   Processor
   CSZScore
   PerAssetFFill
   FormulaEval

Walk-Forward Execution
----------------------

.. autosummary::
   :toctree: generated/

   WalkForwardRunner
   SegmentResult
   WalkForwardResult

Detailed Documentation
----------------------

.. automodule:: src.pipeline
   :members:
   :undoc-members:
   :show-inheritance:
