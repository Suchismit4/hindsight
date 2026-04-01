Model API Reference
====================

.. currentmodule:: src.pipeline.model

The ``model`` package extends walk-forward execution with model fitting and
prediction. The adapter pattern decouples ``ModelRunner`` from any specific
ML library: the runner calls the ``ModelAdapter`` interface, and the adapter
translates to whatever library is underneath.

``SklearnAdapter`` is the shipped implementation. It wraps any scikit-learn
estimator exposing ``fit`` / ``predict``. Integrating a different library
requires implementing ``ModelAdapter`` (see :doc:`../getting_started/model_integration`).

Main Classes
------------

.. autosummary::
   :toctree: generated/

   ModelAdapter
   SklearnAdapter
   ModelRunner
   ModelRunnerResult

Detailed Module Documentation
-------------------------------

.. automodule:: src.pipeline.model.adapter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.pipeline.model.runner
   :members:
   :undoc-members:
   :show-inheritance:
