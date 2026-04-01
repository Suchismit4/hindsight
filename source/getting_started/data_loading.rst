Data Loading
============

Every pipeline run begins with an xarray ``Dataset``. Hindsight provides a
``DataManager`` class (``src.data.managers.DataManager``) that loads from YAML
configuration files and returns ``xr.Dataset`` objects in the canonical panel
layout. For custom or ad-hoc sources, you can also construct the dataset
directly as long as the shape contract is respected.

Dataset Shape
-------------

Hindsight expects an explicit calendar grid rather than a flat datetime index:

- ``year``, ``month``, ``day`` (and optionally ``hour``) define the time grid.
- ``asset`` indexes instruments (tickers, permno codes, exchange pairs, etc.).
- ``time`` coordinate provides ``datetime64`` values aligned to the stacked grid.
- ``time_flat`` (when present) is a 1-D stacked index used internally by runners.

All variable arrays should have dimensions that match this grid. For example,
``close`` is stored as a ``DataArray`` with shape ``(year, month, day, asset)``.
Calendar slots with no market activity (weekends, holidays) appear as ``NaN``
in numeric variables; downstream processors handle them via forward-fill or masking.

Loading via DataManager
-----------------------

``DataManager`` supports three entry points depending on how you want to
specify the load configuration.

**From a YAML configuration file**

.. code-block:: python

   from src.data.managers import DataManager

   dm = DataManager()
   datasets = dm.load_from_config("configs/equity_analysis.yaml")

   equity_ds = datasets["equity_prices"]  # xr.Dataset

The YAML configuration describes one or more named sources with provider,
dataset, frequency, optional filters, column selections, and source-level
transforms. A minimal example:

.. code-block:: yaml

   data:
     name: "equity-analysis"
     start_date: "2020-01-01"
     end_date: "2024-01-01"

     sources:
       equity_prices:
         provider: "wrds"
         dataset: "crsp"
         frequency: "daily"

         processors:
           filters:
             share_classes: [10, 11]
             exchanges: [1, 2, 3]

           transforms:
             - type: "set_coordinates"
               coord_type: "permno"

``load_from_config`` returns a dictionary mapping source name strings to
``xr.Dataset`` objects.

**From a built DataConfig object**

If you are constructing the configuration programmatically:

.. code-block:: python

   from src.data.managers import DataManager
   from src.data.managers.config_schema import ConfigLoader

   dm = DataManager()
   config = ConfigLoader.load("configs/equity_analysis.yaml")
   datasets = dm.load_from_built_config(config)

**From a built-in configuration**

For datasets registered as built-ins in the provider registry:

.. code-block:: python

   from src.data.managers import DataManager

   dm = DataManager()
   datasets = dm.load_builtin(
       "crypto_standard",
       start_date="2020-01-01",
       end_date="2023-12-31",
   )

Dataset Validation
------------------

Before handing a dataset to ``DataHandler``, confirm these properties:

- **Dimensions**: ``raw_ds.sizes`` should reflect the expected
  ``year`` / ``month`` / ``day`` / ``asset`` counts.
- **Coordinates**: ``raw_ds.coords["time"]`` should contain monotonically
  increasing ``datetime64`` values; ``asset`` coordinates should be unique.
- **Dtypes**: Numeric features should be floating point. Integer arrays are
  not automatically upcast by processors and can cause unexpected behavior.
- **Missing values**: Leave ``NaN`` entries in place. Processors such as
  ``PerAssetFFill`` and ``CSZScore`` are designed to handle them correctly.

Merging Multiple Sources
------------------------

It is common to combine datasets from different sources—prices from one
provider and fundamentals from another. Programmatically:

.. code-block:: python

   combined = price_ds.merge(factor_ds, join="outer")

When merging, both datasets must share the same calendar. If they do not,
use ``xr.align`` or ``reindex`` to resolve coordinate mismatches before
merging.

For datasets with different update frequencies (e.g., daily prices and
quarterly fundamentals), use the ``DatasetMerger`` from
``src.pipeline.data_handler.merge``, which handles point-in-time alignment.
See :doc:`../api/data_handler` for the merger's configuration surface.

Custom Loaders
--------------

If your source is not covered by an existing provider, implement a loader
that follows the conventions in ``src.data.loaders``:

1. Parse raw files into pandas ``DataFrame`` objects.
2. Pivot so that instrument identifiers become the ``asset`` coordinate.
3. Convert to xarray via ``to_xarray()`` and reshape into the canonical
   calendar dims (``year``, ``month``, ``day``, ``asset``).
4. Attach metadata via ``Dataset.attrs`` if needed.

Registering the loader with the provider registry makes it available via
``DataManager`` and the YAML pipeline spec. See the data README
(``src/data/README.md``) for the registration pattern.

Where to Go Next
----------------

- :doc:`yaml_pipeline` shows how data loading fits into the full YAML pipeline spec.
- :doc:`data_handler` explains what happens to the dataset after loading.
- :doc:`feature_engineering` covers formula evaluation and processor chaining.
