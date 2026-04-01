Data Loading with DataManager
=============================

Every pipeline run begins with an xarray ``Dataset``. Hindsight ships with loaders in ``src.data`` that convert raw sources (e.g., WRDS extracts, Parquet archives, cached NetCDF files) into the canonical layout expected by the processing stack. If you are wiring in your own data, focus on matching that layout.

Dataset Shape
-------------

Hindsight expects a rectangular calendar with explicit calendar dimensions:

- ``year``, ``month``, ``day`` (and optionally ``hour``) define the calendar grid.
- ``asset`` indexes instruments (tickers, pairs, etc.).
- ``time`` coordinate provides datetime64 values, aligned with the stacked calendar.
- ``time_flat`` (if present) is a 1-D index that matches the stacked calendar order.

All variable arrays should align with these dimensions. Numeric series such as ``close`` or ``volume`` are stored as ``DataArray`` objects with shape ``(year, month, day, asset, ...)``.

Loading via DataManager
-----------------------

``DataManager`` (``src.data.DataManager``) exposes helper methods for loading built-in datasets or constructing pipelines from configuration files. A common workflow looks like this:

.. code-block:: python

   from src.data import DataManager

   dm = DataManager()
   raw_ds = dm.load(
       source="wrds_eod",    # Loader key configured in data YAML
       table="daily_equities",
       start="2020-01-01",
       end="2023-01-01",
       assets=["AAPL", "MSFT"],
       cache=True             # Reuse prepared dataset if available
   )

``raw_ds`` is now an xarray ``Dataset`` adhering to the structure above. If the loader cannot populate certain calendar slots (weekends, market holidays), those coordinates appear as NaT entries in ``time``; downstream processors handle them, typically via forward-fill or masking.

Validation Checklist
--------------------

Before handing the dataset to ``DataHandler``:

- **Inspect dimensions**: ``raw_ds.sizes`` should show expected ``year``/``month``/``day``/``asset`` counts.
- **Confirm coordinates**: ``raw_ds.time`` should contain monotonic datetime64 values; ``asset`` coordinate should be unique identifiers.
- **Check dtype**: Ensure numeric features are floating point; convert integers if necessary to avoid unintended casting.
- **Handle missing data**: Leave NaNs in place; processors such as ``PerAssetFFill`` or ``CSZScore`` are designed to address them.

Merging Multiple Sources
------------------------

It is common to stitch together multiple datasets (e.g., prices + fundamentals). Because everything is xarray-based, merging retains aligned coordinates:

.. code-block:: python

   price_ds = dm.load(source="prices", table="crypto_hourly")
   factor_ds = dm.load(source="factors", table="crypto_factors")
   combined = price_ds.merge(factor_ds, join="outer")

When merging, ensure both datasets share the same calendar. Use ``reindex`` or ``align`` to resolve mismatches before merging.

Custom Loaders
--------------

If you have a proprietary data source, implement a loader that follows the patterns under ``src.data.loaders``:

1. Parse raw files into pandas ``DataFrame`` objects.
2. Pivot to bring symbol identifiers to columns.
3. Convert to xarray via ``to_xarray()`` and reshape into the canonical calendar dims.
4. Attach metadata (e.g., exchange, frequency) via Dataset attributes if needed.

By respecting these conventions, your datasets seamlessly plug into ``DataHandler`` and the rest of the pipeline.
