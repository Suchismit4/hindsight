Data Handler API Reference
===========================

.. currentmodule:: src.pipeline.data_handler

The ``data_handler`` package manages the "how" side of the pipeline: what
transformations are applied to the dataset, across which stages, and how
state is managed between training and inference. It is designed to be
reusable—the same ``DataHandler`` instance can be passed to multiple runners
or segment plans without mutation of the base dataset.

Main Classes
------------

``DataHandler`` orchestrates stage execution. ``HandlerConfig`` is its
declarative input.

.. autosummary::
   :toctree: generated/

   DataHandler
   HandlerConfig

Enums and Contracts
--------------------

``View`` selects which stage's output to access. ``PipelineMode`` controls
how the learn and infer branches receive input. ``ProcessorContract`` is the
abstract interface all processors must implement.

.. autosummary::
   :toctree: generated/

   View
   PipelineMode
   ProcessorContract

Processors
----------

Built-in processors cover the most common financial data preparation tasks.
All operate on xarray objects and preserve coordinates.

.. autosummary::
   :toctree: generated/

   Processor
   PerAssetFFill
   CSZScore
   FormulaEval

Detailed Module Documentation
-------------------------------

.. automodule:: src.pipeline.data_handler.core
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.pipeline.data_handler.handler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.pipeline.data_handler.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.pipeline.data_handler.processors
   :members:
   :undoc-members:
   :show-inheritance:
