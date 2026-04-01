Walk-Forward Analysis API Reference
=====================================

.. currentmodule:: src.pipeline.walk_forward

This module provides comprehensive walk-forward analysis capabilities for temporal segmentation and execution.

Main Classes
------------

.. autosummary::
   :toctree: generated/

   Segment
   SegmentPlan
   SegmentConfig
   WalkForwardRunner
   SegmentResult
   WalkForwardResult

Planning Functions
------------------

.. autosummary::
   :toctree: generated/

   make_plan
   expand_plan_coverage
   optimize_plan_for_dataset

Segments Module
---------------

.. automodule:: src.pipeline.walk_forward.segments
   :members:
   :undoc-members:
   :show-inheritance:

Planning Module
---------------

.. automodule:: src.pipeline.walk_forward.planning
   :members:
   :undoc-members:
   :show-inheritance:

Execution Module
----------------

.. automodule:: src.pipeline.walk_forward.execution
   :members:
   :undoc-members:
   :show-inheritance:
