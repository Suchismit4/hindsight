YAML Pipeline Authoring
========================

Hindsight provides a declarative pipeline spec format that lets you express a
complete research workflow—data loading, feature engineering, preprocessing,
and walk-forward model execution—in a single YAML file. ``PipelineExecutor``
parses the spec and runs each stage, with content-addressable caching at every
boundary so intermediate results survive across runs.

This page explains the spec format, how to run a spec, and the caching
behavior you can rely on during development.

When To Use the YAML Path
--------------------------

Use the YAML spec when:

- You want automatic stage-level caching with no manual cache management.
- You want to express the full pipeline as configuration so it can be version-
  controlled and reproduced reliably.
- You want to share a pipeline definition with collaborators without sharing
  the Python glue code.

For interactive exploration or when you need a custom loader not yet expressible
in YAML, the Python API (see :doc:`overview`) is more appropriate.

Running a Spec
--------------

.. code-block:: python

   from pathlib import Path
   from src.data.managers.data_manager import DataManager
   from src.pipeline.cache import GlobalCacheManager
   from src.pipeline.spec import PipelineExecutor, SpecParser

   cache_manager = GlobalCacheManager(cache_root="~/data/hindsight_cache")
   data_manager  = DataManager()
   executor      = PipelineExecutor(
       cache_manager=cache_manager,
       data_manager=data_manager,
   )

   spec   = SpecParser.load_from_yaml("examples/pipeline_specs/crypto_momentum_baseline.yaml")
   result = executor.execute(spec)

   # Per-stage outputs
   raw_data      = result.data               # dict of source name → xr.Dataset
   features_data = result.features_data      # merged dataset after feature stage
   preprocessed  = result.preprocessed_data  # dataset after preprocessing stage
   predictions   = result.model_predictions  # prediction dataset (if model block present)

On the second run with the same spec and unchanged upstream data, every stage
that has a cached result is skipped. The ``result.cache_keys`` dict maps stage
names to the content-addressable keys that were used.

Top-Level Spec Structure
------------------------

.. code-block:: yaml

   spec_version: "1.0"
   name: "pipeline_name"
   version: "1.0"

   time_range:
     start: "YYYY-MM-DD"
     end: "YYYY-MM-DD"

   data: {}
   merge_base: "optional_source_name"
   merges: []
   features: {}
   preprocessing: {}
   model: {}
   metadata: {}

Not every section is required for every workflow. A factor-construction pipeline
can stop after infer-stage processors with no ``model`` block. A modeled pipeline
uses all sections from ``data`` through ``model``.

Data Sources
------------

Each entry under ``data`` is loaded independently. Keys become source names
referenced elsewhere in the spec.

.. code-block:: yaml

   data:
     crypto_prices:
       provider: "crypto"
       dataset: "spot/binance"
       frequency: "H"
       filters: {}
       processors: []

Source fields:

- ``provider``: registered provider identifier (e.g., ``"crypto"``, ``"wrds"``).
- ``dataset``: provider-specific dataset path (e.g., ``"spot/binance"``, ``"equity/crsp"``).
- ``frequency``: data frequency string (``"H"`` for hourly, ``"D"`` for daily, ``"M"`` for monthly).
- ``filters``: dict of filter predicates applied during loading.
- ``columns``: optional list of columns to include (defaults to all).
- ``external_tables``: optional list of secondary tables to join or merge at
  load time (used in WRDS workflows for delisting returns, name lookups, etc.).
- ``processors.transforms``: list of source-level transforms applied after
  loading but before the pipeline spec's feature stage.

Multi-Source Merges
-------------------

When you have sources at different frequencies or with different publication
lags (prices and fundamentals, for example), use ``merge_base`` and ``merges``
to combine them after loading. The ``DatasetMerger`` handles point-in-time
alignment.

.. code-block:: yaml

   merge_base: "crsp"
   merges:
     - right_name: "compustat"
       on: "asset"
       time_alignment: "as_of"
       time_offset_months: 6
       variables: ["seq", "txditc", "ps"]

- ``merge_base``: the left dataset all others are merged onto.
- ``right_name``: source name to merge in.
- ``time_alignment``: ``"as_of"`` uses the most recent value available at each
  left timestamp; ``"ffill"`` forward-fills the right dataset onto the left
  calendar.
- ``time_offset_months``: shift the right dataset forward by this many months
  before merging. Used to enforce data availability lag (e.g., a 6-month
  lag for annual Compustat data to avoid look-ahead).
- ``variables``: which variables from the right source to include.

Feature Engineering
-------------------

Feature operations are grouped into named blocks under ``features.operations``.
Each block evaluates a ``formulas`` dict and may declare dependencies on
earlier blocks.

.. code-block:: yaml

   features:
     operations:
       - name: "momentum_indicators"
         formulas:
           sma:
             - {window: 20}
             - {window: 50}
           ema:
             - {window: 12}
             - {window: 26}
           rsi:
             - {window: 14}

       - name: "derived_signals"
         depends_on: ["momentum_indicators"]
         formulas:
           macd:
             expression: "$ema_ww12 - $ema_ww26"

Formula names and parameters follow the conventions in the formula registry
(``src/data/ast/definitions/``). Named formulas like ``sma``, ``rsi``, and
``ema`` are evaluated via the ``FormulaManager``; ``expression`` entries
use the inline formula DSL.

``depends_on`` forces the executor to evaluate a block only after the named
blocks have completed. This is what keeps feature stages composable and
individually cacheable.

Preprocessing
-------------

The ``preprocessing`` section maps directly onto a ``HandlerConfig``:

.. code-block:: yaml

   preprocessing:
     mode: "independent"

     shared:
       - type: "per_asset_ffill"
         name: "forward_fill"
         vars: ["close", "volume"]

     learn:
       - type: "cs_zscore"
         name: "normalizer"
         vars: ["close", "volume", "sma_ww20", "sma_ww50", "ema_ww12", "ema_ww26"]
         out_suffix: "_norm"

     infer: []

``mode`` maps to ``PipelineMode``: ``"independent"`` or ``"append"``.

Supported processor types under the processor registry:

.. list-table::
   :header-rows: 1

   * - YAML type
     - Behavior
   * - ``per_asset_ffill``
     - Forward-fill selected variables per asset
   * - ``cs_zscore``
     - Cross-sectional z-score normalization (learn-stage stateful)
   * - ``formula_eval``
     - Evaluate formula expressions during preprocessing
   * - ``cross_sectional_sort`` / ``sort``
     - Assign portfolio bucket labels from a signal
   * - ``portfolio_returns`` / ``port_ret``
     - Aggregate returns by portfolio group
   * - ``factor_spread``
     - Construct long-short spread from portfolio return groups

Processors in the ``infer`` stage run on the inference slice only, which is
where portfolio construction steps (sorts, return aggregation, spread
construction) belong in factor workflows.

Model Block
-----------

The ``model`` block is optional. When present, the executor builds a
``SegmentPlan`` from the ``walk_forward`` sub-block and runs ``ModelRunner``
with a fresh ``SklearnAdapter`` per segment.

.. code-block:: yaml

   model:
     adapter: "sklearn"
     type: "LinearRegression"
     params:
       fit_intercept: true
     features:
       - "close_norm"
       - "volume_norm"
       - "sma_ww20_norm"
     target: "close"
     walk_forward:
       train_span_hours: 120
       infer_span_hours: 24
       step_hours: 24
       gap_hours: 0
       start: "2020-01-22 18:00:00"
       end: "2023-12-01 00:00:00"
     adapter_params:
       output_var: "close_pred"
       use_proba: false
     runner_params:
       overlap_policy: "last"

- ``adapter``: only ``"sklearn"`` is supported by the current executor.
  Non-sklearn values raise a not-yet-implemented error.
- ``type``: the sklearn estimator class name (looked up from ``sklearn.*``
  submodules automatically).
- ``params``: dict of constructor kwargs passed to the estimator.
- ``walk_forward``: supports ``train_span_hours`` / ``infer_span_hours`` /
  ``step_hours`` / ``gap_hours`` for hourly data, and analogous ``_days``
  variants for daily data.

Cache Stages and Keys
----------------------

``PipelineExecutor`` assigns a content-addressable key at each stage boundary:

.. list-table::
   :header-rows: 1

   * - Stage
     - Cache level
     - Key includes
   * - Data loading
     - L2_POSTPROCESSED
     - Provider, dataset, date range, filters, columns, source processors
   * - Feature operations
     - L3_FEATURES
     - L2 key + operation config
   * - Preprocessing
     - L4_PREPROCESSED
     - L3 key + HandlerConfig
   * - Model
     - L5_MODEL
     - L4 key + model spec and walk-forward config

A cache hit at any stage skips everything downstream from it. Changing a
processor parameter or a formula only invalidates keys from that stage forward,
not earlier stages. This means loading and feature computation are almost
always served from cache during iterative model development.

Common Pitfalls
---------------

- **Formula output naming**: formula outputs follow ``{formula}_{param}_{value}``
  (e.g., ``sma_ww20``). Reference them by this name in ``preprocessing.learn.vars``
  and ``model.features``. Typos here cause silent ``NaN`` columns rather than
  errors, because the variable just does not exist in the dataset.
- **Processor type aliases**: ``sort`` is an alias for ``cross_sectional_sort``,
  and ``port_ret`` is an alias for ``portfolio_returns``. The bundled FF3 example
  uses the short aliases; the registry accepts both.
- **Non-sklearn adapters**: the executor currently raises an error for any
  ``adapter`` value other than ``"sklearn"``. The schema is worded broadly for
  future extensibility, but the execution path is sklearn-only today.
- **Multi-source frequency mismatch**: if you merge monthly fundamentals onto
  a daily price dataset without ``time_offset_months``, the fundamentals will
  be aligned to the matching month boundary rather than to the actual
  publication date. Use ``time_offset_months`` to enforce the data lag.

Where to Go Next
----------------

- :doc:`data_handler` explains the ``HandlerConfig`` the preprocessing block
  maps to, including processor stages and pipeline modes.
- :doc:`feature_engineering` covers the formula engine and how to add new
  formulas or processors.
- :doc:`execution_analysis` covers output inspection for both the Python API
  and the ``ExecutionResult`` returned by ``PipelineExecutor``.
