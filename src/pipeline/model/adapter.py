from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import numpy as np
import xarray as xr

from src.pipeline.data_handler import DataHandler

class ModelAdapter:
    """
    Thin wrapper around any model to present a stable interface to the pipeline.
    The adapter is responsible only for converting xr.Dataset slices to numpy
    arrays, calling the underlying model, and mapping predictions back to xr.

    Conventions:
      - fit returns self so callers can write adapter.fit(...).predict(...)
      - predict returns an xr.DataArray named by output_var with dims aligned
        to the input dataset time/asset coords.
      - Missing values policy is intentionally minimal. Use processors upstream
        for real imputations. Here we support dropping invalid rows safely.
    """

    def fit(
        self,
        ds: xr.Dataset,
        features: Sequence[str],
        label: Optional[str] = None,
        sample_weight: Optional[xr.DataArray] = None,
    ) -> "ModelAdapter":
        raise NotImplementedError

    def partial_fit(
        self,
        ds: xr.Dataset,
        features: Sequence[str],
        label: Optional[str] = None,
        sample_weight: Optional[xr.DataArray] = None,
    ) -> "ModelAdapter":
        """
        Optional incremental training. Default falls back to fit.
        """
        return self.fit(ds, features, label, sample_weight)

    def predict(self, ds: xr.Dataset, features: Sequence[str]) -> xr.DataArray:
        raise NotImplementedError

    def get_state(self) -> Optional[Dict[str, Any]]:
        """
        Optional persistence hook. Return a JSON-serializable dict of model params.
        """
        return None

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Optional persistence hook to load params created by get_state.
        """
        return None


@dataclass
class SklearnAdapter(ModelAdapter):
    """
    Wrap a sklearn-like estimator exposing .fit(X, y) and .predict(X) or .predict_proba(X).

    Notes on shapes:
      - DataHandler.to_arrays_for_model emits X with shape (T, N, J) and optional y (T, N, K).
      - We flatten to 2D for sklearn: X2 = X.reshape(T*N, J) and y2 accordingly.
      - We drop rows where any feature is non-finite or y is non-finite if supervised.
      - Predictions are mapped back to a full stacked calendar and then unstacked.

    Parameters
    ----------
    model : Any
        The sklearn-like estimator.
    handler : DataHandler
        Used for consistent stacking rules via to_arrays_for_model.
    output_var : str
        Name of the output variable in xr.
    use_proba : bool
        If True, use predict_proba and select proba_index column.
    proba_index : int
        Column index to take from predict_proba.
    drop_nan_rows : bool
        If True, drop rows with non-finite features before calling the model.
    """
    model: Any
    handler: DataHandler
    output_var: str = "score"
    use_proba: bool = False
    proba_index: int = 1
    drop_nan_rows: bool = True

    # internal flag to remember if supervised
    _supervised: bool = field(default=False, init=False, repr=False)

    def _stack_for_alignment(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Stack to a (time_index, asset, ...) view. This MUST match the stacking
        convention used in DataHandler.to_arrays_for_model to keep alignment.
        """
        time_dims = tuple(d for d in ("year", "month", "day", "hour") if d in ds.dims)
        return ds.stack(time_index=time_dims).transpose("time_index", "asset", ...)

    def _to_2d(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        sw: Optional[np.ndarray],
        drop_nan_rows: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Flatten X from (T, N, J) to (M, J), apply a finite mask, and flatten y/sw accordingly.
        Returns (X2, y2, sw2, mask_used) where mask_used selects rows retained from the flattening.
        """
        T, N = X.shape[0], X.shape[1]
        X2 = X.reshape(T * N, X.shape[-1])
        mask_finite = np.all(np.isfinite(X2), axis=1)
        mask_used = mask_finite if drop_nan_rows else np.ones_like(mask_finite, dtype=bool)

        y2 = None
        if y is not None:
            y_flat = y.reshape(T * N, y.shape[-1])
            y2 = y_flat[mask_used]
            if y2.ndim == 2 and y2.shape[1] == 1:
                y2 = y2[:, 0]

        sw2 = None
        if sw is not None:
            sw_flat = sw.reshape(T * N)
            sw2 = sw_flat[mask_used]

        X2 = X2[mask_used]
        return X2, y2, sw2, mask_used

    def fit(
        self,
        ds: xr.Dataset,
        features: Sequence[str],
        label: Optional[str] = None,
        sample_weight: Optional[xr.DataArray] = None,
    ) -> "SklearnAdapter":
        # Build feature/label arrays from the same handler pathway the pipeline uses.
        X, y = self.handler.to_arrays_for_model(
            ds, feature_vars=features, label_vars=[label] if label else None, drop_invalid_time=True, drop_all_nan_rows=False
        )

        self._supervised = label is not None

        # Optional sample weights, aligned to ds, then stacked like features.
        sw_np = None
        if sample_weight is not None:
            st = self._stack_for_alignment(sample_weight.to_dataset(name="sw"))
            sw_np = st["sw"].values  # (T, N)

        X2, y2, sw2, _ = self._to_2d(X, y, sw_np, drop_nan_rows=True)

        if self._supervised and y2 is None:
            raise ValueError("Supervised fit requires a label, but y2 is None after preprocessing")

        # Fill in NaNs with 0s
        X2 = np.nan_to_num(X2, copy=False)
        y2 = np.nan_to_num(y2, copy=False)

        if self._supervised:
            if sw2 is not None:
                sw2 = np.nan_to_num(sw2, copy=False)
                self.model.fit(X2, y2, sample_weight=sw2)
            else:
                self.model.fit(X2, y2)
        else:
            # Unsupervised models have various APIs. Try .fit(X) first.
            if hasattr(self.model, "fit"):
                self.model.fit(X2)

        return self

    def predict(self, ds: xr.Dataset, features: Sequence[str]) -> xr.DataArray:
        # auth of coords
        stacked_full = self._stack_for_alignment(ds)
        T_Full = stacked_full.sizes.get("time_index", 0)
        N_Full = stacked_full.sizes.get("asset", 0)
        if T_Full == 0 or N_Full == 0:
            raise ValueError("Stacked dataset is null")

        # Use the same to_arrays conversion to preserve semantics
        X, _, valid_idx = self.handler.to_arrays_for_model(ds, 
                                                           feature_vars=features, 
                                                           label_vars=None, 
                                                           drop_invalid_time=True, 
                                                           drop_all_nan_rows=False, 
                                                           return_indexer=True)
        # Shapes now:
        #   X: (T_valid, N_full, J)
        #   valid_idx: (T_valid,) selecting rows from T_full
        T_valid = X.shape[0]

        # Flatten to 2D and build a row mask
        X2, _, _, used_flat_mask = self._to_2d(X, None, None, drop_nan_rows=True)
        # Shape now: (M, J)
        
        # Model call
        if self.use_proba:
            if not hasattr(self.model, "predict_proba"):
                raise AttributeError("use_proba=True but model has no predict_proba")
            raw = self.model.predict_proba(X2)
            pred_flat_valid = raw[:, self.proba_index] if raw.ndim == 2 else raw
        else:
            raw = self.model.predict(X2)
            pred_flat_valid = raw  # shape (M,) or (M,)
            
        pred_flat_valid = np.asarray(pred_flat_valid, dtype=np.float64)

        # Allocate a full flat array and fill valid rows, NaN elsewhere
        # inflate once (M,) to (T_valid, N_Full)
        pred_flat = np.full(used_flat_mask.shape[0], np.nan, dtype=np.float64)
        pred_flat[used_flat_mask] = pred_flat_valid
        pred_flat_tn = pred_flat.reshape(T_valid, N_Full) # like (24, 265)
        # print("From predict flat tn:",pred_flat_tn[:, idx])

        # Inflate again for (T_valid, N_full) -> (T_full, N_full)
        pred_full_tn = np.full((T_Full, N_Full), np.nan, dtype=np.float64)
        pred_full_tn[valid_idx] = pred_flat_tn

        # Build stacked DA aligned to stacked calendar, then unstack to original dims.
        out_stacked = xr.DataArray(
            data=pred_full_tn,
            dims=("time_index", "asset"),
            coords={
                "time_index": stacked_full.indexes["time_index"],  # MultiIndex
                "asset": stacked_full["asset"].values,
            },
            name=self.output_var,
        )
        
        # carry datetime coord
        out_stacked = out_stacked.assign_coords(time=("time_index", stacked_full["time"].values))
        if "time_flat" in ds.dims and ds.sizes.get("time_flat", -1) == T_Full:
            # bind the 1-D time_flat vector to time_index; unstack() will reshape it to
            # (year, month, day[, hour]) and keep it as a coord on the returned DataArray
            out_stacked = out_stacked.assign_coords(time_flat=("time_index", stacked_full["time_flat"].values))

        return out_stacked
