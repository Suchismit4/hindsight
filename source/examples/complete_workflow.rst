Complete Workflow Example
=========================

This example stitches together all major components using the Python API:
loading data, configuring processors, planning walk-forward segments, fitting
a model, and analyzing predictions. The code closely follows ``examples/example.py``
in the repository.

Full Script
-----------

.. code-block:: python

   import numpy as np
   import xarray as xr

   from src.data.managers import DataManager
   from src.pipeline import (
       DataHandler, HandlerConfig, PipelineMode,
       PerAssetFFill, FormulaEval, CSZScore,
       SegmentConfig, make_plan,
   )
   from src.pipeline.model import ModelRunner, SklearnAdapter
   from sklearn.ensemble import RandomForestRegressor

   # ── 1. Load data ──────────────────────────────────────────────────────────
   dm = DataManager()
   datasets = dm.load_builtin("crypto_standard")
   raw_ds = datasets["crypto_prices"]

   # ── 2. Configure processors ───────────────────────────────────────────────
   formulas = {
       "sma": [
           {"window": 100, "wilder_weights": True},
           {"window": 200, "wilder_weights": True},
       ],
       "rsi": [{"window": 14}],
       "price_ret_var": [{"p": 1}, {"p": 3}, {"p": 6}],
   }

   config = HandlerConfig(
       shared=[
           PerAssetFFill(name="ffill", vars=["close", "volume"]),
           FormulaEval(
               name="formulas_core",
               formula_configs=formulas,
               static_context={"price": "close", "prc": "close"},
               use_jit=False,
           ),
       ],
       learn=[
           CSZScore(
               name="norm",
               vars=["sma_ww100", "sma_ww200", "rsi", "price_ret_var_p1"],
           ),
       ],
       infer=[],
       mode=PipelineMode.INDEPENDENT,
       feature_cols=["sma_ww100_norm", "sma_ww200_norm", "rsi_norm", "price_ret_var_p1_norm"],
       label_cols=["fwd_return_p5"],
   )

   handler = DataHandler(base=raw_ds, config=config)

   # ── 3. Plan walk-forward segments ─────────────────────────────────────────
   seg_cfg = SegmentConfig(
       start=np.datetime64("2020-01-22T18:00:00"),
       end=np.datetime64("2023-12-01T00:00:00"),
       train_span=np.timedelta64(24 * 5, "h"),
       infer_span=np.timedelta64(24 * 1, "h"),
       step=np.timedelta64(24 * 1, "h"),
       gap=np.timedelta64(0, "h"),
       clip_to_data=True,
   )
   plan = make_plan(seg_cfg, ds_for_bounds=raw_ds)

   # ── 4. Define adapter factory ─────────────────────────────────────────────
   def make_adapter():
       model = RandomForestRegressor(
           n_estimators=50,
           max_depth=14,
           min_samples_leaf=10,
           bootstrap=True,
           max_samples=0.5,
           n_jobs=-1,
           random_state=0,
       )
       return SklearnAdapter(
           model=model,
           handler=handler,
           output_var="score",
           use_proba=False,
       )

   # ── 5. Execute ────────────────────────────────────────────────────────────
   runner = ModelRunner(
       handler=handler,
       plan=plan,
       model_factory=make_adapter,
       feature_cols=config.feature_cols,
       label_col=config.label_cols[0],
       overlap_policy="last",
   )
   result = runner.run()

   # ── 6. Analyze ────────────────────────────────────────────────────────────
   pred_ds = result.pred_ds
   merged  = raw_ds.merge(pred_ds)
   btc     = merged.sel(asset="BTCUSDT")

   ic = xr.corr(btc["score"], btc["fwd_return_p5"], dim="time_flat").item()
   print(f"BTC prediction IC: {ic:.4f}")

Step-by-Step Explanation
------------------------

1. **Load**: ``DataManager.load_builtin`` returns a dict of ``xr.Dataset``
   objects keyed by source name. The result has the canonical
   ``(year, month, day, asset)`` panel structure.

2. **Processors**: The shared stage forward-fills gaps and evaluates formula-
   based indicators (SMA, RSI, return variance). The learn stage normalizes a
   subset of those indicators cross-sectionally, fitting per-timestamp
   statistics on the training window.

3. **Segmentation**: ``SegmentConfig`` describes a rolling hourly schedule
   with a 5-day training window and 1-day inference window. ``make_plan``
   clips the plan to actual data availability.

4. **Adapter factory**: A callable is passed rather than an instance so each
   segment receives a clean, untrained model. Passing an instance would share
   model state across segments.

5. **Execution**: ``ModelRunner`` runs the gather-scatter loop: shared view
   once, then per-segment learn, infer, fit, predict, scatter. The result is
   a global prediction buffer unstacked into the original calendar.

6. **Analysis**: Merge predictions with raw data and compute the information
   coefficient (IC), which is a standard per-asset correlation metric for
   signal quality.

Adapting This Example
---------------------

- **Different indicators**: swap the ``formula_configs`` dict in ``FormulaEval``.
  Formula definitions live under ``src/data/ast/definitions/``.
- **Different model**: replace ``RandomForestRegressor`` with any sklearn-
  compatible estimator (``Ridge``, ``GradientBoostingRegressor``, etc.) in
  ``make_adapter``.
- **Different schedule**: adjust ``train_span``, ``infer_span``, and ``step``
  in ``SegmentConfig``. Daily or monthly schedules follow the same pattern
  with ``np.timedelta64(..., "D")`` or similar.
- **YAML-based pipeline**: the equivalent workflow can be expressed in a
  ``PipelineSpec`` YAML file and run via ``PipelineExecutor``. See
  ``examples/pipeline_specs/crypto_momentum_baseline.yaml`` and
  :doc:`../getting_started/yaml_pipeline` for that path.

Where to Go Next
----------------

- :doc:`../getting_started/yaml_pipeline` shows how to express this same
  workflow in a YAML pipeline spec with automatic caching.
- :doc:`../getting_started/execution_analysis` covers output inspection patterns.
- :doc:`../api/model` is the API reference for ``ModelRunner``, ``SklearnAdapter``,
  and ``ModelRunnerResult``.
