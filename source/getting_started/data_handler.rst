Data Handler and Processing Pipeline
=====================================

``DataHandler`` (``src.pipeline.data_handler.handler``) orchestrates the
transformation side of the framework. It accepts an xarray ``Dataset`` and a
``HandlerConfig``, then applies processors in three ordered stages: shared,
learn, and infer. This three-stage design is what keeps training-time statistics
from leaking into inference slices during walk-forward execution.

HandlerConfig
-------------

``HandlerConfig`` (``src.pipeline.data_handler.config``) describes the
full processing pipeline with these fields:

- ``shared``: processors applied once to the entire dataset before any
  temporal splitting. Use this for stateless operations (``PerAssetFFill``)
  or computationally expensive transforms you want cached across segments.
- ``learn``: processors that fit on a training slice and store a compact state
  dataset. During inference, the stored state is reused to transform the
  corresponding inference slice. ``CSZScore`` is the canonical learn-stage
  processor.
- ``infer``: transform-only processors applied on the inference slice after
  learn states have been applied. Cross-sectional sorting, portfolio return
  aggregation, and similar operations typically belong here.
- ``mode``: controls how learn and infer branches receive input.

  - ``PipelineMode.INDEPENDENT`` (default): both branches receive the shared
    view as input. The infer branch does not see learn outputs.
  - ``PipelineMode.APPEND``: the infer branch output is fed into the learn
    branch. Use this when inference-time features need to feed into the
    learning objective.

- ``feature_cols`` / ``label_cols``: optional lists naming semantic groups of
  variables. ``handler.fetch`` uses these to assemble feature and label datasets
  for downstream consumers.

Example Configuration
---------------------

.. code-block:: python

   from src.pipeline import HandlerConfig, PipelineMode
   from src.pipeline.data_handler.processors import PerAssetFFill, FormulaEval, CSZScore

   config = HandlerConfig(
       shared=[
           PerAssetFFill(name="ffill", vars=["close", "volume"]),
           FormulaEval(
               name="formulas",
               formula_configs={
                   "rsi": [{"window": 14}],
                   "sma": [{"window": 20}],
               },
               static_context={"price": "close"},
           ),
       ],
       learn=[
           CSZScore(name="norm", vars=["rsi_w14", "sma_w20"]),
       ],
       infer=[],
       mode=PipelineMode.INDEPENDENT,
       feature_cols=["rsi_w14_norm", "sma_w20_norm"],
       label_cols=["fwd_return"],
   )

Pipeline Execution Model
------------------------

``DataHandler(base=raw_ds, config=config)`` builds lazily. Computation is
triggered when you call ``handler.view`` or ``handler.fetch`` for the first
time. The execution sequence is:

1. **Shared stage**: each processor's ``transform`` method runs on the full
   base dataset. The result is cached as the shared view.

2. **Branching** based on ``mode``:

   - ``INDEPENDENT``: the shared view flows independently into both the learn
     and infer branches.
   - ``APPEND``: the shared view passes through infer processors first; that
     result then enters the learn branch.

3. **Learn stage**: for each processor, ``fit_transform`` is called on the
   appropriate training slice. The returned state datasets are kept in
   ``handler.learn_states`` indexed by processor name.

4. **Infer stage**: transform-only operations run on the inference slice,
   reusing stored learn states where needed.

Accessing Views
---------------

``handler.view`` accepts a ``View`` enum value:

- ``View.RAW``: the original unmodified base dataset.
- ``View.LEARN``: shared + learn stages applied. States are created on demand
  if they have not been computed yet.
- ``View.INFER``: shared + infer stages applied, reusing any learn states that
  infer processors need.

To retrieve semantic column groups, use ``handler.fetch``:

.. code-block:: python

   from src.pipeline.data_handler import View

   features = handler.fetch(View.LEARN, ["features"])
   labels   = handler.fetch(View.LEARN, ["labels"])

State Management
----------------

Learn-stage processors return compact state datasets: typically a few arrays
holding summary statistics (means, standard deviations, bin edges, etc.).
``DataHandler`` preserves processor order so processors and their states can
be zipped. Walk-forward runners reuse these states when transforming inference
slices, which is the mechanism that enforces temporal isolation.

Pipeline Modes in Practice
---------------------------

- **INDEPENDENT** is appropriate for the common case where you want
  normalization statistics learned on the training window applied to inference
  data, with no bleedthrough between branches.
- **APPEND** is useful in less common workflows where an inference-stage
  transformation (for example, a cross-sectional ranking) needs to act as
  input to a learning procedure.

Custom Processors
-----------------

Extend ``Processor`` (``src.pipeline.data_handler.processors``) to add new
transformations:

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
           return ds.assign({f"{self.var}_demeaned": demeaned})

Processors that do not require fitting (i.e., ``fit`` is trivial and
``transform`` is stateless) belong in the shared or infer stage. Processors
that compute statistics to be reused belong in the learn stage.

Integration with Walk-Forward Runners
--------------------------------------

``WalkForwardRunner`` and ``ModelRunner`` call ``DataHandler`` internally for
each segment. They never mutate the base dataset, which means a single
``DataHandler`` instance can safely be shared across multiple segment plans or
runner invocations as long as the underlying dataset is constant.

Where to Go Next
----------------

- :doc:`feature_engineering` covers built-in processors and the formula engine.
- :doc:`walk_forward` explains how runners apply the handler across segments.
- :doc:`model_integration` shows how ``DataHandler`` integrates with ``ModelRunner``.
