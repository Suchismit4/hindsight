Pipeline Execution and Results Analysis
=====================================

After configuring processors and segments, the final step is to execute the pipeline and analyze outputs. The runners handle execution; your job is to interpret the resulting datasets.

WalkForwardRunner Usage
-----------------------

``WalkForwardRunner`` applies the data pipeline across a ``SegmentPlan`` without involving any models. Use it when you want processed inference panels for analysis or as a precursor to custom evaluation code.

.. code-block:: python

   from src.pipeline import WalkForwardRunner

   wf_runner = WalkForwardRunner(handler=handler, plan=plan)
   wf_result = wf_runner.run()

``wf_result`` is a ``WalkForwardResult`` with:

- ``processed_ds``: Aggregated inference Dataset containing transformed variables.
- ``segment_states``: List of dicts capturing per-segment metadata (state counts, durations, overlap decisions).
- ``attrs``: Run-level metadata (overlap policy, segment count, timestamp).

ModelRunner Usage
-----------------

``ModelRunner`` builds on ``WalkForwardRunner`` to include model training and prediction.

.. code-block:: python

   from src.pipeline.model import ModelRunner

   model_runner = ModelRunner(
       handler=handler,
       plan=plan,
       model_factory=make_adapter,
       feature_cols=handler_config.feature_cols,
       label_col=handler_config.label_cols[0],
       overlap_policy="last"
   )

   model_result = model_runner.run()

``model_result.pred_ds`` mirrors the calendar of the input dataset and contains the prediction variable configured in the adapter. ``segment_states`` includes both learn-stage processor states and optional adapter state summaries if the adapter populates them.

Analyzing Outputs
-----------------

**Inspect coverage**
    Confirm the prediction span: ``model_result.pred_ds.time.min()`` / ``max()`` and asset coverage via ``pred_ds.asset``.

**Merge with raw or processed data**
    Combine predictions with other variables:

    .. code-block:: python

       merged = raw_ds.merge(model_result.pred_ds)

**Compute metrics**
    Use xarray-friendly metrics (correlations, mean returns) or convert to pandas for custom evaluation.

**Per-segment diagnostics**
    Iterate ``model_result.segment_states`` to examine how many samples or states were learned per segment. Useful for spotting segments that had insufficient data.

Presentation Tips
-----------------

- The datasets are n-dimensional; slice to a smaller subset (e.g., ``sel(asset="BTCUSDT")``) when plotting.
- Preserve metadata by using xarray operations (``merge``, ``assign_coords``) rather than converting to numpy prematurely.
- Persist outputs with ``to_netcdf`` for reproducibility.
