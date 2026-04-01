Model Integration and Execution
================================

The ``model`` package extends walk-forward execution by fitting predictive models on each segment and aggregating forecasts. Rather than binding the pipeline to a specific ML library, Hindsight uses adapters that translate between xarray slices and model APIs.

ModelAdapter Interface
----------------------

``ModelAdapter`` (``src.pipeline.model.adapter``) defines the minimal contract:

- ``fit(ds, features, label=None, sample_weight=None)`` trains the underlying model using variables extracted from ``ds``.
- ``predict(ds, features)`` returns an xarray ``DataArray`` aligned with the segment’s inference slice.
- ``partial_fit`` (optional) enables incremental training.
- ``get_state`` / ``load_state`` (optional) provide persistence hooks.

Adapters are responsible for converting the xarray slice into the format expected by the model (typically flattening ``(time, asset)`` into ``(n_samples, n_features)`` and dropping NaNs) and mapping predictions back to the original coordinates.

SklearnAdapter
--------------

``SklearnAdapter`` wraps any scikit-learn estimator exposing ``fit``/``predict``. It uses the handler’s ``to_arrays_for_model`` helper to extract feature and label arrays. Example:

.. code-block:: python

   from sklearn.ensemble import RandomForestRegressor
   from src.pipeline.model import SklearnAdapter

   adapter = SklearnAdapter(
       model=RandomForestRegressor(n_estimators=100, random_state=42),
       handler=handler,
       output_var="score",
       drop_nan_rows=True
   )

During ``fit``, the adapter:

1. Converts the training dataset to ``X`` (``T x N x J``) and ``y`` arrays.
2. Flattens to ``(T*N, J)`` while removing rows with NaNs (if ``drop_nan_rows`` is True).
3. Calls the estimator’s ``fit`` method (with optional sample weights).

``predict`` performs the reverse process, reshaping predictions back to the full time/asset grid while preserving NaNs where the model could not produce values.

ModelRunner Workflow
--------------------

``ModelRunner`` coordinates data processing, segmentation, and model execution:

1. Compute shared view once.
2. Pre-stack the dataset and pre-compute train/infer integer slices for all segments.
3. For each segment:
   - Slice train and infer windows from the shared view.
   - Fit learn processors on the train slice and produce states.
   - Transform the inference slice using the stored states and infer processors.
   - Instantiate a fresh adapter via ``model_factory`` and fit/predict on the processed slices.
   - Scatter predictions into a global buffer using the configured overlap policy.
4. Unstack the global buffer to produce an aggregated ``Dataset`` with the specified ``output_var``.

The result is a ``ModelRunnerResult`` containing ``pred_ds`` (predictions), ``segment_states`` (learn-stage states), and run metadata.

Overlap Policies
----------------

Identical to ``WalkForwardRunner``: ``last`` overwrites earlier predictions with later ones where they overlap; ``first`` keeps the earliest prediction and only fills gaps with subsequent segments.

Debugging Controls
------------------

``ModelRunner`` accepts optional parameters to narrow execution during development:

- ``debug_start`` / ``debug_end``: Limit the inference window in the global buffer.
- ``debug_asset``: Restrict processing to a single asset for quicker iteration.

These flags ensure you can inspect a slice of the workflow without running the full backtest.

Custom Adapters
---------------

To support other libraries (PyTorch, TensorFlow, statsmodels), subclass ``ModelAdapter``:

.. code-block:: python

   from src.pipeline.model import ModelAdapter
   import torch

   class TorchAdapter(ModelAdapter):
       def __init__(self, network, handler):
           self.network = network
           self.handler = handler

       def fit(self, ds, features, label=None, sample_weight=None):
           X, y = self.handler.to_arrays_for_model(ds, features, [label] if label else None)
           # Convert to tensors, train network...
           return self

       def predict(self, ds, features):
           X, _, idx = self.handler.to_arrays_for_model(ds, features, None, return_indexer=True)
           # Run inference, map back using idx
           # Return xarray.DataArray with proper coordinates

Factory Functions
-----------------

``model_factory`` can be either an adapter instance or a callable returning a fresh adapter. Passing a callable is recommended so each segment gets a clean model instance, preventing leakage of learned parameters.

Putting It Together
-------------------

The following snippet assumes you have already prepared ``handler``, ``plan``, and ``handler_config`` as described in earlier sections (for example, by loading a dataset with ``DataManager`` and building a ``SegmentPlan``).

.. code-block:: python

   from src.pipeline.model import ModelRunner

   def make_adapter():
       return SklearnAdapter(
           model=RandomForestRegressor(n_estimators=50, random_state=0),
           handler=handler,
           output_var="score"
       )

   runner = ModelRunner(
       handler=handler,
       plan=plan,
       model_factory=make_adapter,
       feature_cols=handler_config.feature_cols,
       label_col=handler_config.label_cols[0],
       overlap_policy="last"
   )

   result = runner.run()
   pred_ds = result.pred_ds

From here, you can merge ``pred_ds`` with the original dataset, compute metrics, or feed predictions into downstream systems.
