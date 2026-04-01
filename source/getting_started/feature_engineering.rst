Feature Engineering with Processors
=====================================

Hindsight's feature engineering layer is built on ``Processor`` objects. A
processor consumes an xarray ``Dataset`` and returns a modified dataset, while
optionally producing a compact state dataset for learn-stage processors.
Processors live in ``src.pipeline.data_handler.processors`` and implement the
``ProcessorContract`` defined in ``core.py``.

Where processors live in the pipeline:

- **Shared stage**: applied once to the full dataset before any temporal
  slicing. Appropriate for fills, formula evaluation, or expensive transforms
  that should not run per-segment.
- **Learn stage**: fit on segment-specific training windows. Each processor
  stores a state dataset that is later used to transform inference slices,
  which is what prevents leakage of future statistics.
- **Infer stage**: transform-only, applied to the inference slice after learn
  states are applied. Cross-sectional sorts and portfolio construction steps
  typically live here.

Built-in Processors
-------------------

``PerAssetFFill``
    Stateless. Forward-fills ``NaN`` values independently per asset along the
    flattened time index. Commonly the first step in the shared stage to
    prepare raw data before formula evaluation.

``CSZScore``
    Stateful. During ``fit``, calculates per-timestamp mean and standard
    deviation across assets, storing those statistics in a state ``Dataset``.
    During ``transform``, normalizes each asset using the stored statistics.
    Belongs in the learn stage so inference data is normalized with
    training-period statistics.

``FormulaEval``
    Wraps the AST formula engine in ``src.data.ast``. Evaluates declarative
    formulas (RSI, SMA, EMA, return variance, etc.) and merges the outputs
    into the dataset under predictable variable names. Can optionally JIT
    compile evaluations with JAX for large datasets.

Example Configuration
---------------------

.. code-block:: python

   from src.pipeline import HandlerConfig, PipelineMode
   from src.pipeline.data_handler.processors import PerAssetFFill, FormulaEval, CSZScore

   config = HandlerConfig(
       shared=[
           PerAssetFFill(name="ffill", vars=["close", "volume"]),
           FormulaEval(
               name="indicators",
               formula_configs={
                   "rsi": [{"window": 14}, {"window": 21}],
                   "sma": [{"window": 20}],
               },
               static_context={"price": "close"},
           ),
       ],
       learn=[
           CSZScore(name="norm", vars=["rsi_w14", "rsi_w21", "sma_w20"]),
       ],
   )

Formula System Details
----------------------

The formula engine (``src.data.ast``) is a self-contained subsystem with
its own parser, AST node definitions, function registry, and dependency
resolver.

**Formula definitions**
    YAML files under ``src/data/ast/definitions/`` describe each formula:
    its parameters, dependencies, and output naming conventions. The
    ``FormulaManager`` loads and validates these at startup.

**Dependency resolution**
    The manager builds a dependency graph over formula definitions and uses
    topological sort to determine evaluation order. This supports both
    functional dependence (formula A calls formula B as a function) and
    time-series dependence (formula A uses the output DataArray of formula B).

**Static context**
    The ``static_context`` dict passed to ``FormulaEval`` provides name
    aliases accessible inside formula expressions. For example,
    ``{"price": "close"}`` allows formulas to refer to ``$price`` and have
    it resolve to the ``close`` variable in the dataset.

**Output naming**
    By default, formula outputs merge into the dataset using the convention
    ``{formula}_{param}_{value}`` (e.g., ``rsi_w14`` for RSI with window 14).
    Use a ``prefix`` or ``assign_in_place=False`` if you need namespaced
    outputs.

**JAX JIT compilation**
    ``use_jit=True`` compiles the evaluation function with JAX. This is most
    beneficial when the dataset is large and formulas are evaluated repeatedly
    across many segments. JAX compilation has a cold-start cost on the first
    call; subsequent calls on the same shape are fast.

**Registering custom functions**
    The ``@register_function`` decorator in ``src.data.ast.functions`` adds
    new callables to the formula language:

    .. code-block:: python

       from src.data.ast.functions import register_function
       import xarray as xr

       @register_function("zscore_ts")
       def zscore_ts(arr: xr.DataArray, window: int) -> xr.DataArray:
           """Rolling z-score along the time axis."""
           rolling = arr.rolling(time_flat=window, min_periods=window // 2)
           mean = rolling.mean()
           std  = rolling.std()
           return (arr - mean) / std.where(std > 0)

    Once registered, ``zscore_ts`` can be referenced in YAML formula
    configurations or used in formula expression strings.

Custom Processor Development
-----------------------------

When you need bespoke logic not expressible as a formula, extend ``Processor``:

.. code-block:: python

   from dataclasses import dataclass
   import xarray as xr
   from src.pipeline.data_handler.processors import Processor

   @dataclass
   class VolatilityNorm(Processor):
       name: str
       var: str
       window: int = 21

       def fit(self, ds: xr.Dataset) -> xr.Dataset:
           vol = (
               ds[self.var]
               .rolling(time_flat=self.window, min_periods=self.window // 2)
               .std()
               .mean(dim="asset", skipna=True)
           )
           return xr.Dataset({f"{self.name}_vol": vol})

       def transform(self, ds: xr.Dataset, state: xr.Dataset = None) -> xr.Dataset:
           if state is None:
               state = self.fit(ds)
           normalized = ds[self.var] / state[f"{self.name}_vol"].where(
               state[f"{self.name}_vol"] > 0
           )
           return ds.assign({f"{self.var}_volnorm": normalized})

If ``transform`` does not rely on a fitted state, the processor is stateless
and can live in the shared or infer stage.

Best Practices
--------------

- **Coordinate safety**: Rely on xarray broadcasting rather than manual
  indexing. When stacking or unstacking, preserve coordinates so downstream
  processors align correctly.
- **State compactness**: Store only the necessary statistics in learn-stage
  state datasets. Per-asset per-timestamp arrays can be large; when possible,
  store per-timestamp cross-sectional summaries (mean, std) rather than
  per-observation values.
- **Naming conventions**: Adopt predictable output variable names. The
  convention ``{base_var}_{processor_name}`` makes it easy for downstream
  configuration (``feature_cols``, YAML specs) to reference generated
  variables without looking up implementation details.
- **Statistical validity**: If a transformation requires fitting (normalization,
  PCA, quantile binning), it belongs in the learn stage. Processors in the
  shared stage should be safe to apply to the full dataset without creating
  lookahead bias.

Where to Go Next
----------------

- :doc:`data_handler` explains ``HandlerConfig``, pipeline modes, and how
  processors are invoked by the runner.
- :doc:`walk_forward` covers how learn-stage states are captured and reused
  across segments.
- :doc:`model_integration` shows how processed datasets feed into model adapters.
