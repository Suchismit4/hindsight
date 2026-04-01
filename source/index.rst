.. Hindsight documentation master file.

Hindsight
=========

Hindsight is a YAML-first quantitative research library built around ``xarray``
datasets and JAX-backed numerical workflows. It is designed for researchers and
engineers who want a consistent path from raw market data through feature
engineering, preprocessing, walk-forward model execution, and portfolio-style
analysis without rewriting pipeline glue for each project.

The core system model is::

    provider → loader → xr.Dataset → formulas/processors → walk-forward/model

Most datasets are normalized into a panel layout over::

    (year, month, day, hour, asset)

That shape contract is what lets the same library support multi-frequency data
loading, formula-driven feature computation, cross-sectional processors, and
walk-forward evaluation without repeatedly reinterpreting the time axis.

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/overview
   getting_started/data_loading
   getting_started/yaml_pipeline
   getting_started/data_handler
   getting_started/feature_engineering
   getting_started/walk_forward
   getting_started/model_integration
   getting_started/execution_analysis

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/complete_workflow

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/pipeline
   api/data_handler
   api/walk_forward
   api/model

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
