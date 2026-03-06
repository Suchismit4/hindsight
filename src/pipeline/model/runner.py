from __future__ import annotations

from threadpoolctl import threadpool_limits
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import time
import numpy as np
import xarray as xr
from tqdm import tqdm

from src.pipeline.data_handler import DataHandler
from src.pipeline.walk_forward import Segment, SegmentPlan
from src.pipeline.model.adapter import ModelAdapter

# Prevent runaway threading defaults if env is not set
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


@dataclass
class ModelRunnerResult:
    """
    Final result of a modeling run.
    """
    pred_ds: xr.Dataset
    segment_states: List[Dict[str, Any]] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRunner:
    """
    Controller that mirrors WalkForwardRunner logic and plugs in a ModelAdapter.

    Strategy:
      1) Compute shared view once.
      2) Pre-stack shared for fast integer slicing and to precompute segment bounds.
      3) For each segment:
           a) Slice train and infer windows by integer search over the stacked time vector.
           b) Fit learn processors on train slice; transform infer slice using learn states.
           c) Apply infer processors transform-only on infer slice.
           d) Fit the model on processed train slice; predict on processed infer slice.
           e) Aggregate predictions into a single stacked array using the chosen policy.
      4) Unstack final prediction panel back to (year, month, day, hour, asset).
    """

    handler: DataHandler
    plan: SegmentPlan
    model_factory: Union[ModelAdapter, Callable[[], ModelAdapter]]
    feature_cols: Sequence[str]
    label_col: Optional[str] = None
    overlap_policy: str = "last"  # "last" or "first"
    output_var: str = "score"
    return_model_states: bool = True
    
    # debug controls (optional)
    debug_asset: Optional[str] = None
    debug_start: Optional[np.datetime64] = None
    debug_end: Optional[np.datetime64] = None

    # internal cache for the shared view
    _shared: Optional[xr.Dataset] = field(default=None, init=False, repr=False)

    def _ensure_shared(self) -> xr.Dataset:
        """
        Compute the shared-transformed dataset once, mirroring WalkForwardRunner._ensure_shared.
        We do not rely on a DataHandler.shared_view() helper to avoid modifying DataHandler.
        """
        if self._shared is not None:
            return self._shared

        # features: either cached by handler or just base
        if "features" in self.handler.cache:
            ds_features = self.handler.cache["features"]
        else:
            self.handler.cache["features"] = self.handler.base
            ds_features = self.handler.base

        # run shared processors transform-only
        shared_res = self.handler._apply_pipeline(
            ds_in=ds_features,
            pipeline=self.handler.config.shared,
            fit=False,
            states=None,
        )
        self._shared = shared_res.ds
        
        return self._shared

    def _make_model(self) -> ModelAdapter:
        # Accept either a prebuilt instance or a factory
        if isinstance(self.model_factory, ModelAdapter):
            return self.model_factory
        return self.model_factory()

    def _compute_bounds(self, stacked: xr.Dataset) -> Tuple[List[slice], List[slice]]:
        """
        Compute integer slices for all segments against the stacked 'time' vector.
        Uses np.searchsorted (fast) rather than .sel (slower).
        """
        t = stacked["time"].values
        T = t.shape[0]
        unit = t.dtype.name.split("[")[-1][:-1] if t.dtype.kind == "M" else "ns"
        valid = ~np.isnat(t)  
        t_valid = t[valid]                             # strictly datetime64, no NaT
        idx_map = np.flatnonzero(valid)                # map from valid view -> original indices

        def bounds(seg, use_infer):
            if use_infer:
                s = np.datetime64(seg.infer_start, unit)
                e = np.datetime64(seg.infer_end, unit)
            else:
                s = np.datetime64(seg.train_start, unit)
                e = np.datetime64(seg.train_end, unit)

            # Binary search on the NaT-free, sorted view
            i0v = int(np.searchsorted(t_valid, s, side="left"))
            i1v = int(np.searchsorted(t_valid, e, side="right"))

            # If nothing falls in-range among valid times, return empty slice
            if i0v >= i1v or i0v == len(t_valid):
                return slice(0, 0)

            # Map back to original indices; include all rows between first/last valid hit
            i0 = idx_map[i0v]
            i1 = idx_map[i1v - 1] + 1                  # +1 to make it python-slice end
            # clamp (defensive)
            i0 = max(0, min(T, i0))
            i1 = max(0, min(T, i1))
            return slice(i0, i1)

        infer_slices = [bounds(seg, True) for seg in self.plan]
        train_slices = [bounds(seg, False) for seg in self.plan]
        return train_slices, infer_slices

    def _masked_scatter_rows_inplace(
        self,
        buf: xr.DataArray,
        rows_global: np.ndarray,          # 1D integer indexer into buf.time_index
        src_stacked: xr.DataArray,        # dims ('time_index','asset') for the *segment*
        policy: str,
    ) -> None:
        """
        In-place masked write into arbitrary rows of buf.values using integer indexing.
        rows_global maps the segment's local time_index rows to buf's global time_index rows.

        policy == 'last'  -> overwrite where src is non-NaN
        policy == 'first' -> write only into currently-NaN cells where src is non-NaN
        """
        if rows_global.size == 0:
            return

        # Align assets if needed (cheap check; avoids per-segment reindex in the common aligned case)
        if not np.array_equal(src_stacked["asset"].values, buf["asset"].values):
            src_stacked = src_stacked.reindex(asset=buf["asset"].values)

        src = src_stacked.values                    # (T_local, N)
        buf_vals = buf.values                       # (T_global, N)

        # Advanced indexing assignment (one shot)
        tgt_slice = buf_vals[rows_global, :]        # copy view (read)
        src_valid = ~np.isnan(src)

        if policy == "last":
            # overwrite only where new segment has values
            updated = np.where(src_valid, src, tgt_slice)
            buf_vals[rows_global, :] = updated
        elif policy == "first":
            write_mask = np.isnan(tgt_slice) & src_valid
            tgt_slice[write_mask] = src[write_mask]
            buf_vals[rows_global, :] = tgt_slice
            

    def run(self) -> ModelRunnerResult:
        ds_shared = self._ensure_shared()
        
        # Stack the whole shared dataset once
        time_dims = tuple(d for d in ("year", "month", "day", "hour") if d in ds_shared.dims)
        stacked_shared = ds_shared.stack(time_index=time_dims).transpose("time_index", "asset", ...)
        T = stacked_shared.sizes.get("time_index", 0)
        N = stacked_shared.sizes.get("asset", 0)
        
        # Prepare final stacked target array filled with NaN
        final_stacked = xr.DataArray(
            np.full((T, N), np.nan, dtype=np.float64),
            dims=("time_index", "asset"),
            coords={
                "time_index": stacked_shared.indexes["time_index"], 
                "asset": stacked_shared["asset"].values, 
                "time": ("time_index", stacked_shared["time"].values)
            },
            name=self.output_var,
        )
        
        # Precompute integer slices for all segments
        train_slices, infer_slices = self._compute_bounds(stacked_shared)
        segment_states: List[Dict[str, Any]] = []
        
        segment_iter = zip(self.plan, train_slices, infer_slices)
        progress = tqdm(
            segment_iter,
            desc="Walk-forward segments",
            total=len(self.plan),
            unit="segment",
            dynamic_ncols=True,
        )
        
        for i, (seg, tr_s, inf_s) in enumerate(progress):
            
            # Slice train and infer from pre-stacked shared, then unstack
            # Note: Unstack may introduce rectangular expansions with NaNs for partial slices,
            # which will be handled later
            train_stacked_sliced = stacked_shared.isel(time_index=tr_s)
            train_ds = train_stacked_sliced.unstack("time_index")
            
            infer_stacked_sliced = stacked_shared.isel(time_index=inf_s)
            infer_ds = infer_stacked_sliced.unstack("time_index")
            
            # Empty-guard via sizes
            if train_ds.sizes.get("asset", 0) == 0 or infer_ds.sizes.get("asset", 0) == 0:
                continue
            if train_ds.sizes.get("year", 0) == 0 or infer_ds.sizes.get("year", 0) == 0:
                continue
            
            # Learn processors: fit on train
            learn_res = self.handler._apply_pipeline(
                ds_in=train_ds,
                pipeline=self.handler.config.learn,
                fit=True,
                states=None,
            )
            learn_states = learn_res.states
            
            # Apply learned states to infer slice (transform-only)
            ds_infer_applied = infer_ds
            for proc, st in zip(self.handler.config.learn, learn_states):
                ds_infer_applied = proc.transform(ds_infer_applied, st)
            
            # Apply infer processors transform-only
            infer_final = self.handler._apply_pipeline(
                ds_in=ds_infer_applied,
                pipeline=self.handler.config.infer,
                fit=False,
                states=None,
            ).ds
            
            # Build the model for this segment
            model = self._make_model()

            # Limit BLAS threads during model.fit to avoid oversubscription with joblib workers
            with threadpool_limits(limits=1):
                model.fit(learn_res.ds, features=self.feature_cols, label=self.label_col)
            with threadpool_limits(limits=1):
                pred_stacked = model.predict(infer_final, features=self.feature_cols)
                if pred_stacked.name != self.output_var:
                    pred_stacked = pred_stacked.rename(self.output_var)

            # Scatter back!
            
            # In-place masked scatter into the global buffer using integers
            global_mi = final_stacked.indexes["time_index"]
            local_mi  = pred_stacked.indexes["time_index"]
            rows_global = global_mi.get_indexer(local_mi)         # shape (T_local,), -1 for misses
            
            # paranoid check
            keep = rows_global >= 0
            if not keep.all():
                # Drop any local rows that don't exist in the global calendar (rare)
                rows_global = rows_global[keep]
                pred_stacked = pred_stacked.isel(time_index=keep)
            
            in_window = (rows_global >= inf_s.start) & (rows_global < inf_s.stop)
            if not in_window.any():
                segment_states.append({"segment": i, "infer_rows": np.array([], dtype=int)})
                continue

            rows_kept = rows_global[in_window]
            pred_kept = pred_stacked.isel(time_index=in_window)
            progress.set_postfix({'segment': i, 'rows': int(in_window.sum())})

            self._masked_scatter_rows_inplace(
                buf=final_stacked,
                rows_global=rows_kept,
                src_stacked=pred_kept.transpose("time_index", "asset"),
                policy=self.overlap_policy,
            )

        progress.close()

        pred_da = final_stacked.unstack("time_index")  # -> (year, month, day[, hour], asset)
        if "time" in final_stacked.coords:
            pred_da = pred_da.assign_coords(time=final_stacked["time"].unstack("time_index"))
        if "time_flat" in final_stacked.coords:
            pred_da = pred_da.assign_coords(time_flat=final_stacked["time_flat"].unstack("time_index"))
            
        time_dims = [d for d in ("year", "month", "day", "hour") if d in pred_da.dims]
        pred_da = pred_da.transpose(*time_dims, "asset")

        # Wrap as a dataset with the configured output var name
        pred_ds = pred_da.to_dataset(name=self.output_var)
        pred_ds = pred_ds.assign_coords(asset=stacked_shared["asset"].values)
        

        attrs = {
            "overlap_policy": self.overlap_policy,
            "segments": len(self.plan),
            "created_at_unix": time.time(),
        }
        pred_ds = pred_ds.assign_attrs(attrs)

        return ModelRunnerResult(pred_ds=pred_ds, segment_states=segment_states, attrs=attrs)
