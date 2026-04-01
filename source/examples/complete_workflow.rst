Complete Workflow Example
=========================

This example stitches together all major components: loading data, configuring processors, planning walk-forward segments, fitting a model, and analyzing predictions.

Full Script
-----------

.. code-block:: python

   import numpy as np
   import xarray as xr

   from src.data import DataManager
   from src.pipeline import (
       DataHandler, HandlerConfig, PipelineMode,
       PerAssetFFill, FormulaEval, CSZScore,
       SegmentConfig, make_plan
   )
   from src.pipeline.model import ModelRunner, SklearnAdapter
   from sklearn.ensemble import RandomForestRegressor

   # 1. Load data (hourly cryptocurrency prices in this example)
   dm = DataManager()
   raw_ds = dm.load(source="crypto_standard", table="crypto_prices")

   # 2. Configure processors
   formulas = {
       "sma": [{"window": 100, "wilder_weights": True}, {"window": 200, "wilder_weights": True}],
       "rsi": [{"window": 14}],
       "price_ret_var": [{"p": 1}, {"p": 3}, {"p": 6}]
   }

   handler_config = HandlerConfig(
       shared=[
           PerAssetFFill(name="ffill", vars=["close", "volume"]),
           FormulaEval(
               name="formulas_core",
               formula_configs=formulas,
               static_context={"price": "close", "prc": "close"},
               use_jit=False
           )
       ],
       learn=[CSZScore(name="norm", vars=["sma_ww100", "sma_ww200", "rsi", "price_ret_var_p1"] )],
       infer=[],
       mode=PipelineMode.INDEPENDENT,
       feature_cols=[
           "sma_ww100_norm",
           "sma_ww200_norm",
           "rsi_norm",
           "price_ret_var_p1_norm"
       ],
       label_cols=["fwd_return_p5"]
   )

   handler = DataHandler(base=raw_ds, config=handler_config)

   # 3. Plan walk-forward segments
   segment_cfg = SegmentConfig(
       start=np.datetime64("2020-01-22T18:00:00"),
       end=np.datetime64("2023-12-01T00:00:00"),
       train_span=np.timedelta64(24 * 5, "h"),
       infer_span=np.timedelta64(24 * 1, "h"),
       step=np.timedelta64(24 * 1, "h"),
       gap=np.timedelta64(0, "h"),
       clip_to_data=True
   )
   plan = make_plan(segment_cfg, ds_for_bounds=raw_ds)

   # 4. Define adapter factory
   def make_adapter():
       model = RandomForestRegressor(
           n_estimators=50,
           max_depth=14,
           min_samples_leaf=10,
           bootstrap=True,
           max_samples=0.5,
           n_jobs=-1,
           random_state=0
       )
       return SklearnAdapter(
           model=model,
           handler=handler,
           output_var="score",
           use_proba=False
       )

   # 5. Execute model runner
   runner = ModelRunner(
       handler=handler,
       plan=plan,
       model_factory=make_adapter,
       feature_cols=handler_config.feature_cols,
       label_col=handler_config.label_cols[0],
       overlap_policy="last"
   )
   results = runner.run()

   # 6. Analyze predictions
   pred_ds = results.pred_ds
   merged = raw_ds.merge(pred_ds)
   btc_panel = merged.sel(asset="BTCUSDT")

   corr = xr.corr(btc_panel["score"], btc_panel["fwd_return_p5"], dim="time_flat").item()
   print(f"Prediction correlation: {corr:.4f}")

Explanation
-----------

1. **Load**: ``DataManager`` returns an xarray ``Dataset`` with the canonical calendar layout.
2. **Processors**: Shared stage fills gaps and computes formulas; learn stage normalizes a subset of indicators.
3. **Segmentation**: ``SegmentConfig`` describes a rolling window; ``make_plan`` clips it to data availability.
4. **Adapter**: ``SklearnAdapter`` wraps a random forest and handles xarray-to-numpy conversions.
5. **Execution**: ``ModelRunner`` processes each segment, fits the model, aggregates predictions, and records per-segment states.
6. **Analysis**: Merge predictions with raw data and calculate diagnostic metrics (e.g., correlation between predicted score and forward returns).

This workflow can be adapted by plugging in different processors, segment schedules, or adapters depending on your research goals.
