Model Integration and Execution
================================

The ``model`` package extends walk-forward execution by fitting predictive
models on each segment and aggregating forecasts. Rather than binding the
pipeline to a specific ML library, Hindsight uses an adapter interface that
translates between xarray slices and model-specific APIs.

ModelAdapter Interface
----------------------

``ModelAdapter`` (``src.pipeline.model.adapter``) defines the minimal contract
that any model integration must implement:

- ``fit(ds, features, label=None, sample_weight=None)``: train the underlying
  model using variables extracted from the dataset.
- ``predict(ds, features)``: return an ``xr.DataArray`` aligned with the
  segment's inference slice.
- ``partial_fit`` (optional): enables incremental training.
- ``get_state`` / ``load_state`` (optional): persistence hooks for checkpoint
  and restore workflows.

Adapters convert the xarray slice into the format the model expects—typically
flattening ``(time, asset)`` into ``(n_samples, n_features)`` while dropping
``NaN`` rows—and map predictions back to the original coordinates.

SklearnAdapter
--------------

``SklearnAdapter`` is the shipped implementation. It wraps any scikit-learn
estimator that exposes ``fit`` / ``predict``, and handles all xarray-to-numpy
conversion internally.

.. code-block:: python

   from sklearn.ensemble import GradientBoostingRegressor
   from src.pipeline.model import SklearnAdapter

   adapter = SklearnAdapter(
       model=GradientBoostingRegressor(n_estimators=100, random_state=42),
       handler=handler,
       output_var="score",
       drop_nan_rows=True,
   )

During ``fit``, the adapter:

1. Calls ``handler.to_arrays_for_model`` to extract ``X`` (shape
   ``T × N × J``) and ``y`` arrays from the processed dataset.
2. Flattens to ``(T*N, J)`` and drops rows with ``NaN`` entries when
   ``drop_nan_rows=True``.
3. Calls the estimator's ``fit`` method, passing optional sample weights if
   provided.

``predict`` performs the reverse: the model produces a 1-D prediction array,
which is reshaped back to the time/asset grid and returned as an
``xr.DataArray`` with the coordinates from the inference slice.

ModelRunner Workflow
--------------------

``ModelRunner`` coordinates segmentation, data processing, and model execution.
It uses a gather-scatter pattern for efficiency:

**Setup**
    Compute the shared view once and pre-stack the dataset. Pre-allocate a
    global ``NaN``-filled prediction buffer. Pre-compute integer-based slice
    boundaries for all segments in the plan.

**Loop (gather + scatter)**
    For each ``Segment``:

    1. Slice train and infer windows from the stacked shared view using
       pre-computed integer indices.
    2. Apply learn processors on the train slice, capturing states.
    3. Transform the inference slice using the stored states and any infer
       processors.
    4. Instantiate a fresh model via ``model_factory()``—this is critical for
       segment isolation; each segment gets a clean, untrained model.
    5. Fit the model on the processed training slice, predict on the processed
       inference slice.
    6. Scatter predictions into the global buffer using the configured
       overlap policy.

**Finalize**
    Unstack the global prediction buffer into the original multi-dimensional
    calendar format.

.. code-block:: python

   from src.pipeline.model import ModelRunner, SklearnAdapter
   from sklearn.linear_model import Ridge

   def make_adapter():
       return SklearnAdapter(
           model=Ridge(alpha=1.0),
           handler=handler,
           output_var="score",
       )

   runner = ModelRunner(
       handler=handler,
       plan=plan,
       model_factory=make_adapter,
       feature_cols=config.feature_cols,
       label_col=config.label_cols[0],
       overlap_policy="last",
   )

   result = runner.run()
   pred_ds = result.pred_ds

The ``model_factory`` should always be a callable, not a pre-instantiated
adapter, so each segment receives an independent model object.

Overlap Policies
----------------

Identical to ``WalkForwardRunner``:

- ``"last"`` (default): later-segment predictions overwrite earlier ones where
  inference windows overlap.
- ``"first"``: first-segment predictions are preserved; later segments fill
  remaining ``NaN`` positions.

Debugging Controls
------------------

``ModelRunner`` accepts optional parameters to restrict execution during
development:

- ``debug_start`` / ``debug_end``: limit the inference window written to the
  global buffer.
- ``debug_asset``: restrict processing to a single asset for faster iteration.

These do not change the model training behavior—training still uses the full
train slice—but they let you inspect a subset of predictions without a full
run.

ModelRunnerResult
-----------------

``runner.run()`` returns a ``ModelRunnerResult`` with:

- ``pred_ds``: xarray ``Dataset`` on the same calendar as the input, containing
  the prediction variable (``output_var`` from the adapter).
- ``segment_states``: list of per-segment state dicts, including learn-stage
  processor states.

Implementing Custom Adapters
-----------------------------

``SklearnAdapter`` is the only shipped implementation. If you need to integrate
a different library, subclass ``ModelAdapter`` and implement ``fit`` and
``predict``. The contract is minimal: receive an xarray ``Dataset`` plus
column name lists, return an xarray ``DataArray`` aligned to the segment slice.

.. code-block:: python

   from dataclasses import dataclass
   from typing import List, Optional
   import xarray as xr
   from src.pipeline.model import ModelAdapter

   @dataclass
   class StatsmodelsOLSAdapter(ModelAdapter):
       handler: object
       output_var: str = "pred"
       _model = None

       def fit(
           self,
           ds: xr.Dataset,
           features: List[str],
           label: Optional[str] = None,
           sample_weight=None,
       ):
           import statsmodels.api as sm
           X, y, _ = self.handler.to_arrays_for_model(ds, features, [label])
           # flatten and drop NaN rows
           X_flat = X.reshape(-1, X.shape[-1])
           y_flat = y.reshape(-1)
           mask   = ~np.isnan(X_flat).any(axis=1) & ~np.isnan(y_flat)
           self._model = sm.OLS(y_flat[mask], sm.add_constant(X_flat[mask])).fit()
           return self

       def predict(self, ds: xr.Dataset, features: List[str]) -> xr.DataArray:
           import statsmodels.api as sm
           X, _, idx = self.handler.to_arrays_for_model(ds, features, None, return_indexer=True)
           X_flat = X.reshape(-1, X.shape[-1])
           preds  = self._model.predict(sm.add_constant(X_flat))
           # map back to xr.DataArray with original coordinates...

The adapter pattern means the rest of the runner pipeline—segmentation, state
management, buffer scatter—stays unchanged regardless of which model library
you plug in.

Where to Go Next
----------------

- :doc:`execution_analysis` covers result inspection and how to merge predictions
  with the original dataset.
- :doc:`../api/model` is the API reference for ``ModelAdapter``, ``SklearnAdapter``,
  and ``ModelRunner``.
