Data Handler and Processing Pipeline
====================================

``DataHandler`` orchestrates the "how" side of the framework. It accepts an xarray ``Dataset`` plus a ``HandlerConfig`` describing how to transform the dataset across three stages: shared, learn, and infer. These stages mirror qlib’s design but are tailored for multi-dimensional xarray workflows.

HandlerConfig Parameters
------------------------

``HandlerConfig`` (``src.pipeline.data_handler.config``) exposes the following fields:

- ``shared``: List of processors applied once to the full dataset. Use this for stateless operations (e.g., ``PerAssetFFill``) or expensive transforms you want cached.
- ``learn``: Processors that fit on training data and output state objects. During inference, the stored state is reused to transform new slices. ``CSZScore`` is a typical learn-stage processor.
- ``infer``: Transform-only processors applied after the learn stage on inference slices. Use this for inference-specific post-processing.
- ``mode``: ``PipelineMode.INDEPENDENT`` (default) keeps learn and infer branches separate; ``PipelineMode.APPEND`` feeds the infer branch output into the learn branch.
- ``feature_cols`` / ``label_cols``: Optional lists that describe semantic groups. ``handler.fetch`` relies on these names to assemble feature/label datasets.

Example Configuration
---------------------

.. code-block:: python

   from src.pipeline import HandlerConfig, PipelineMode
   from src.pipeline.data_handler.processors import PerAssetFFill, FormulaEval, CSZScore

   handler_config = HandlerConfig(
       shared=[
           PerAssetFFill(name="ffill", vars=["close", "volume"]),
           FormulaEval(
               name="formulas",
               formula_configs={"rsi": [{"window": 14}], "sma": [{"window": 20}]},
               static_context={"price": "close"}
           )
       ],
       learn=[CSZScore(name="norm", vars=["rsi", "sma"])],
       infer=[],
       mode=PipelineMode.INDEPENDENT,
       feature_cols=["rsi_norm", "sma_norm"],
       label_cols=["target_return"]
   )

Pipeline Execution Model
------------------------

When you instantiate ``DataHandler(base=raw_ds, config=handler_config)``, it builds lazily. The first call to ``handler.view`` triggers the following steps (cached afterwards):

1. **Shared stage**: Each processor’s ``transform`` method runs on the full dataset. Outputs are stored in the handler cache under ``shared_view``.

2. **Branch** depending on ``mode``:

   - ``INDEPENDENT``: Shared output feeds two separate pipelines (learn, infer).
   - ``APPEND``: Shared output is passed through infer processors first; the result is then fed into learn processors.

3. **Learn stage** (if applicable): For each processor, call ``fit_transform`` on the appropriate training slice. The returned states (xarray Datasets) are kept in ``handler.learn_states``.

4. **Infer stage**: Run transform-only operations using the cached states when needed.

Accessing Views
---------------

``handler.view`` accepts ``View.RAW``, ``View.LEARN``, or ``View.INFER``:

- ``RAW``: Original dataset (no transformations).
- ``LEARN``: Shared + learn stages applied. If learn states have not been computed yet, they’re created on demand.
- ``INFER``: Shared + infer stages applied; may reuse learn states depending on mode.

To pull semantic column groups, use ``handler.fetch``:

.. code-block:: python

   from src.pipeline.data_handler import View

   feature_ds = handler.fetch(View.LEARN, ["features"])
   label_ds = handler.fetch(View.LEARN, ["labels"])

State Management
----------------

Learn-stage processors must return compact state datasets (usually a few arrays with summary statistics). ``DataHandler`` preserves state order so you can zip processors with their states. Walk-forward runners reuse these states when transforming inference slices to guarantee temporal isolation.

Pipeline Modes
--------------

- ``PipelineMode.INDEPENDENT`` (default): Shared output branches into learn and infer pipelines separately. Inference never sees learn-stage outputs.
- ``PipelineMode.APPEND``: Shared output flows through infer processors first; the result is then passed into the learn pipeline. Choose this if you need inference transformations to feed into learning.

Custom Processors
-----------------

To create new transformations, subclass ``src.pipeline.data_handler.processors.Processor`` (which itself implements ``ProcessorContract``):

- Implement ``fit`` to compute state (return an xarray ``Dataset``).
- Implement ``transform`` to apply the transformation using the state.
- Optionally override ``fit_transform`` for efficiency.

Remember to operate on xarray objects and preserve coordinates. For example, to produce a demeaned series per asset, align dimensions carefully so broadcasting works as expected.

Integrations
------------

``WalkForwardRunner`` and ``ModelRunner`` call ``handler._apply_pipeline`` internally for each segment. They never mutate the base dataset. This makes ``DataHandler`` reusable across multiple segment plans or models as long as the underlying dataset remains constant.
