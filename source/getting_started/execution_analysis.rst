Pipeline Execution and Results Analysis
========================================

After configuring processors, segment plans, and (optionally) a model, you
run the pipeline and interpret the outputs. This page covers both runners,
the structure of their return objects, and common patterns for inspecting
and working with results.

Running Without a Model: WalkForwardRunner
------------------------------------------

Use ``WalkForwardRunner`` when you want processed inference panels for
analysis or factor construction without any model predictions.

.. code-block:: python

   from src.pipeline import WalkForwardRunner

   runner = WalkForwardRunner(handler=handler, plan=plan)
   result = runner.run()

``result`` is a ``WalkForwardResult`` with:

- ``processed_ds``: aggregated inference ``Dataset`` containing transformed
  variables. This is the union of all inference-window outputs across segments,
  resolved by the overlap policy.
- ``segment_states``: list of dicts capturing per-segment metadata, including
  processor states, sample counts, and duration information.
- ``attrs``: run-level metadata—overlap policy, total segment count, and
  a timestamp.

Running With a Model: ModelRunner
----------------------------------

``ModelRunner`` builds on the same segmentation infrastructure but adds
model training and prediction.

.. code-block:: python

   from src.pipeline.model import ModelRunner

   runner = ModelRunner(
       handler=handler,
       plan=plan,
       model_factory=make_adapter,
       feature_cols=config.feature_cols,
       label_col=config.label_cols[0],
       overlap_policy="last",
   )
   result = runner.run()

``result`` is a ``ModelRunnerResult`` with:

- ``pred_ds``: prediction ``Dataset`` on the same calendar as the input
  dataset. Contains the ``output_var`` configured in the adapter.
- ``segment_states``: list of per-segment state dicts, including learn-stage
  processor states and optional adapter state summaries.

Analyzing Outputs
-----------------

**Confirm coverage**

.. code-block:: python

   print("Prediction span:", result.pred_ds.coords["time"].values[[0, -1]])
   print("Assets covered: ", result.pred_ds.coords["asset"].values)

**Merge predictions with raw or processed data**

.. code-block:: python

   merged = raw_ds.merge(result.pred_ds)

This is the usual pattern for computing metrics: merge predictions with the
forward return variable, then compute correlation, hit rate, or similar.

**Compute evaluation metrics**

.. code-block:: python

   import xarray as xr

   # Cross-sectional IC (information coefficient) averaged over time.
   corr = xr.corr(
       merged["pred"],
       merged["fwd_return"],
       dim="asset",
   ).mean("time_flat").item()
   print(f"Mean IC: {corr:.4f}")

**Per-segment diagnostics**

.. code-block:: python

   for i, seg_state in enumerate(result.segment_states):
       print(f"Segment {i}: {seg_state}")

Iterating ``segment_states`` lets you examine sample counts, duration, and
processor states per segment. This is useful for spotting segments that had
insufficient non-NaN rows, which can indicate data sparsity or overly strict
NaN dropping in the adapter.

**Select a single asset**

The prediction dataset is multi-dimensional. Slice to a single asset for
plotting or quick diagnostics:

.. code-block:: python

   btc = merged.sel(asset="BTCUSDT")

**Persist results**

.. code-block:: python

   result.pred_ds.to_netcdf("predictions.nc")

Use ``to_netcdf`` for reproducibility. The file can be reloaded with
``xr.open_dataset`` and merged with any new data.

Working with YAML Pipeline Results
------------------------------------

If you ran the pipeline via ``PipelineExecutor`` instead of the Python API,
the return object is an ``ExecutionResult`` with:

- ``data``: dict mapping source names to loaded ``xr.Dataset`` objects.
- ``features_data``: merged dataset after feature operations.
- ``preprocessed_data``: dataset after the preprocessing stage.
- ``model_predictions``: prediction dataset from the model stage (when present).
- ``model_runner_result``: the ``ModelRunnerResult`` object from the model runner.
- ``cache_keys``: per-stage content-addressable cache keys for debugging.

.. code-block:: python

   from src.pipeline.spec import PipelineExecutor

   executor = PipelineExecutor.from_yaml("examples/pipeline_specs/crypto_momentum_baseline.yaml")
   exec_result = executor.execute()

   pred_ds    = exec_result.model_predictions
   runner_res = exec_result.model_runner_result

Common Pitfalls
---------------

- **Prediction timestamps are sparse**: predictions only cover inference
  windows. Calendar slots outside any inference window remain ``NaN`` in
  ``pred_ds``. This is expected.
- **Float32 precision**: if the base dataset uses float32, predictions will
  also be float32. Convert with ``pred_ds.astype(float)`` if you need
  float64 precision for metrics.
- **Large in-memory result**: for very long histories and many assets, the
  prediction buffer can be large. Use ``debug_asset`` on ``ModelRunner``
  during development to limit scope.
- **Segment state structure**: ``segment_states`` is a list of dicts, not
  keyed by processor name. The order corresponds to the processor list in
  ``HandlerConfig``.

Where to Go Next
----------------

- :doc:`../examples/complete_workflow` is a full script tying together all
  these components into one runnable example.
- :doc:`yaml_pipeline` covers the YAML spec path and the caching behavior of
  ``PipelineExecutor``.
- :doc:`../api/pipeline` and :doc:`../api/model` are the API references.
