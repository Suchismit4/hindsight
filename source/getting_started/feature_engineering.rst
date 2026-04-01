Feature Engineering with Processors
===================================

Hindsight’s feature engineering layer is built on ``Processor`` objects. A processor is a reusable transformation that consumes an xarray ``Dataset`` and returns a modified dataset while optionally producing a compact state (for learn-stage processors). Processors live in ``src.pipeline.data_handler.processors`` and implement ``ProcessorContract`` defined in ``core.py``.

Processor Stages
----------------

The DataHandler pipeline invokes processors in three ordered stages:

- **Shared**: Runs once on the entire dataset before any temporal slicing. Use this for operations that should be cached (formula evaluation, per-asset fills).
- **Learn**: Runs on segment-specific training windows. Each processor fits on the training slice, returns a state ``Dataset``, and then transforms the slice (and eventually the inference slice) using that state.
- **Infer**: Runs transform-only operations on the inference slice after learn-stage transformations.

Built-in Processors
-------------------

The framework ships with several processors tuned for common financial tasks:

``PerAssetFFill``
    Stateless processor that forward-fills NaNs independently per asset along the flattened time index.

``CSZScore``
    Cross-sectional z-score normalization. During ``fit`` it calculates per-timestamp mean and standard deviation across assets, storing the stats in a state ``Dataset``. ``transform`` then normalizes each asset using those stats.

``FormulaEval``
    Wraps the AST formula engine in ``src.data.ast``. It evaluates declarative formulas (e.g., RSI, moving averages) and merges the outputs into the dataset. Can optionally JIT compile evaluations with JAX.

Configuring Processors
----------------------

You attach processors to stages via ``HandlerConfig``. Example:

.. code-block:: python

   from src.pipeline import HandlerConfig, PipelineMode
   from src.pipeline.data_handler.processors import PerAssetFFill, FormulaEval, CSZScore

   formulas = {
       "rsi": [{"window": 14}, {"window": 21}],
       "sma": [{"window": 20}]
   }

   config = HandlerConfig(
       shared=[
           PerAssetFFill(vars=["close", "volume"]),
           FormulaEval(
               formula_configs={
                   "rsi": [{"window": 14}, {"window": 21}],
                   "sma": [{"window": 20}]
               },
               static_context={"price": "close"}
           )
       ],
       learn=[CSZScore(vars=["rsi_w14", "rsi_w21", "sma_w20"])]
   )

``FormulaEval`` Details
-----------------------

- **Formula definitions** live under ``src/data/ast/definitions`` (YAML). Each entry describes parameters, dependencies, and output naming conventions.
- **Static context** provides aliases and constants accessible to formulas (e.g., mapping ``price`` to ``close``).
- **Outputs** merge into the dataset (e.g., ``rsi_w14``). Use ``assign_in_place=False`` plus ``prefix`` if you want namespaced outputs instead.
- **Performance**: ``use_jit=True`` compiles evaluation with JAX; best used when the dataset is large and formulas are reused.

Custom Processor Development
----------------------------

When you need bespoke logic, extend ``Processor``:

.. code-block:: python

   from dataclasses import dataclass
   import xarray as xr
   from src.pipeline.data_handler.processors import Processor

   @dataclass
   class DemeanByAsset(Processor):
       name: str
       var: str

       def fit(self, ds: xr.Dataset) -> xr.Dataset:
           means = ds[self.var].mean(dim="asset", skipna=True)
           return xr.Dataset({f"{self.name}_mean": means})

       def transform(self, ds: xr.Dataset, state: xr.Dataset = None) -> xr.Dataset:
           if state is None:
               state = self.fit(ds)
           demeaned = ds[self.var] - state[f"{self.name}_mean"]
           return ds.assign({f"{self.var}_{self.name}": demeaned})

Place the new processor in a stage depending on whether it needs to be fitted. If ``transform`` does not rely on a fitted state, the processor is effectively stateless and can run in the shared or infer stage.

Best Practices
--------------

- **Coordinate safety**: Always rely on xarray broadcasting. When stacking/unstacking, preserve coordinates so downstream processors continue to align correctly.
- **State compactness**: Store only necessary statistics in state Datasets. Large states increase memory usage during walk-forward execution.
- **Naming**: Adopt predictable output naming (e.g., suffix-based) so configurations and downstream consumers can reference generated variables easily.
- **Statistical validity**: If a transformation requires fitting (e.g., normalization, PCA), it belongs in the learn stage to prevent leakage.
