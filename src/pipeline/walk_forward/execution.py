"""
Walk-forward execution engine and result management.

This module contains the execution engine for walk-forward analysis,
orchestrating the application of data processing pipelines across
temporal segments. It follows qlib's pattern of separating "when"
(temporal logic) from "how" (data processing logic).

The execution system handles segment-by-segment processing with proper
state management and efficient caching for high-performance backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import xarray as xr
from tqdm import tqdm

from src.pipeline.data_handler import DataHandler
from src.pipeline.data_handler.core import ProcessorState
from .segments import Segment, SegmentPlan


@dataclass
class SegmentResult:
    """
    Results from processing a single walk-forward segment.
    
    This class encapsulates the complete output from processing one segment,
    including the processed inference data and any learned processor states.
    It provides a clean interface for collecting and analyzing segment-level
    results in walk-forward workflows.
    
    Parameters
    ----------
    segment : Segment
        The segment that was processed to generate these results
    ds_infer : xr.Dataset
        Fully processed inference dataset after all pipeline stages:
        - Shared processors applied
        - Learn processors fitted on training data and applied to inference
        - Infer processors applied transform-only
    learn_states : List[ProcessorState]
        States learned from fitting processors on the training data.
        These are opaque objects whose structure depends on each processor.
        
    Notes
    -----
    The processing pipeline for each segment follows this sequence:
    1. Slice training and inference data from shared-processed dataset
    2. Fit learn processors on training slice -> learn_states
    3. Apply learn_states to inference slice (consistent transformation)
    4. Apply infer processors transform-only to inference slice -> ds_infer
    
    This design ensures that inference data is processed using only
    information available at training time, preventing lookahead bias
    while maintaining consistent feature engineering across time periods.
    
    Examples
    --------
    Analyzing results from a segment:
    
    >>> result = segment_results[0]
    >>> print(f"Segment period: {result.segment.infer_start} to {result.segment.infer_end}")
    >>> print(f"Processed variables: {list(result.ds_infer.data_vars)}")
    >>> print(f"Learn states count: {len(result.learn_states)}")
    >>> 
    >>> # Access processed inference data
    >>> features = result.ds_infer[['feature1', 'feature2']]
    >>> 
    >>> # Analyze learned normalization stats
    >>> if result.learn_states:
    ...     first_state = result.learn_states[0]
    ...     print(f"State type: {type(first_state).__name__}")
    """
    segment: Segment
    ds_infer: xr.Dataset
    learn_states: List[ProcessorState]

    @property
    def inference_period(self) -> tuple[np.datetime64, np.datetime64]:
        """
        Get the inference period for this result.
        
        Returns
        -------
        tuple[np.datetime64, np.datetime64]
            Tuple of (infer_start, infer_end) from the segment
        """
        return self.segment.infer_start, self.segment.infer_end
    
    @property 
    def training_period(self) -> tuple[np.datetime64, np.datetime64]:
        """
        Get the training period used to generate this result.
        
        Returns
        -------
        tuple[np.datetime64, np.datetime64]
            Tuple of (train_start, train_end) from the segment
        """
        return self.segment.train_start, self.segment.train_end

    def get_state_summary(self) -> dict:
        """
        Get a summary of learned states for analysis.
        
        Returns
        -------
        dict
            Summary information about learned processor states
        """
        summary = {
            "num_states": len(self.learn_states),
            "state_types": [],
            "dataset_details": [],
        }

        for state in self.learn_states:
            summary["state_types"].append(type(state).__name__)
            if isinstance(state, xr.Dataset):
                summary["dataset_details"].append(
                    {
                        "vars": list(state.data_vars),
                        "coords": list(state.coords),
                        "dims": dict(state.dims),
                    }
                )
            else:
                summary["dataset_details"].append(None)

        return summary


@dataclass
class WalkForwardResult:
    """
    Final result of a walk-forward analysis run.
    
    This class encapsulates the aggregated output from walk-forward analysis,
    following the same pattern as ModelRunnerResult but for data processing
    rather than model predictions.
    
    Parameters
    ----------
    processed_ds : xr.Dataset
        Aggregated processed dataset containing all inference periods
    segment_states : List[Dict[str, Any]]
        Summary information about each processed segment
    attrs : Dict[str, Any]
        Additional metadata about the walk-forward run
    """
    processed_ds: xr.Dataset
    segment_states: List[Dict[str, Any]] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardRunner:
    """
    Orchestrates walk-forward evaluation using an existing DataHandler.
    
    This class manages the complete walk-forward execution workflow,
    applying data processing pipelines across temporal segments with
    proper state management and efficient caching. It separates "when"
    (temporal segmentation) from "how" (data processing) following qlib's
    architectural principles.
    
    Parameters
    ----------
    handler : DataHandler
        Configured data handler with processing pipelines
    plan : SegmentPlan
        Walk-forward segment plan defining temporal structure
        
    Attributes
    ----------
    _shared : xr.Dataset, optional
        Cached shared processing view to avoid recomputation
        
    Notes
    -----
    The runner executes this strategy for each segment:
    1. Run shared processors once on full dataset (transform-only)
    2. For each segment:
       a. Fit learn processors on training slice to get states  
       b. Apply learned states to inference slice (transform-only)
       c. Apply infer processors to inference slice (transform-only)
    3. Collect per-segment outputs as SegmentResult objects
    
    This design ensures:
    - No model integration (focuses purely on "when" logic)
    - Reuses DataHandler processors and semantics for consistency
    - Efficient caching of shared transformations
    - Proper temporal isolation preventing lookahead bias
    
    The runner is designed to be model-agnostic, focusing on data
    processing and leaving model-specific logic to higher-level
    orchestrators like ModelRunner.
    
    Examples
    --------
    Basic walk-forward execution:
    
    >>> from src.pipeline.data_handler import DataHandler, HandlerConfig
    >>> from src.pipeline.walk_forward import make_plan, SegmentConfig
    >>> 
    >>> # Configure data processing
    >>> config = HandlerConfig(
    ...     shared=[PerAssetFFill(name="ffill")],
    ...     learn=[CSZScore(name="norm")],
    ...     infer=[]
    ... )
    >>> handler = DataHandler(base=dataset, config=config)
    >>> 
    >>> # Configure temporal segments
    >>> seg_config = SegmentConfig(
    ...     start=np.datetime64('2020-01-01'),
    ...     end=np.datetime64('2023-12-31'),
    ...     train_span=np.timedelta64(365, 'D'),
    ...     infer_span=np.timedelta64(30, 'D'),
    ...     step=np.timedelta64(30, 'D')
    ... )
    >>> plan = make_plan(seg_config)
    >>> 
    >>> # Execute walk-forward analysis
    >>> runner = WalkForwardRunner(handler=handler, plan=plan)
    >>> results = runner.run()
    >>> 
    >>> print(f"Processed {len(results)} segments")
    >>> for i, result in enumerate(results):
    ...     print(f"Segment {i}: {result.inference_period}")
    
    Advanced usage with result analysis:
    
    >>> results = runner.run()
    >>> 
    >>> # Analyze processing consistency
    >>> for result in results:
    ...     summary = result.get_state_summary()
    ...     print(f"Segment learned {summary['num_states']} processor states")
    ... 
    >>> # Extract all inference datasets
    >>> inference_datasets = [r.ds_infer for r in results]
    >>> combined = xr.concat(inference_datasets, dim='time_flat')
    """
    handler: DataHandler
    plan: SegmentPlan

    # cached shared view to avoid recomputing per segment
    _shared: Optional[xr.Dataset] = field(default=None, init=False, repr=False)
    
    # Configuration for gather>scatter pattern
    overlap_policy: str = "last"  # "last" or "first" for overlapping segments
    return_segments: bool = False  # Whether to return individual segment results

    def _ensure_shared(self) -> xr.Dataset:
        """
        Compute shared transforms once using handler.config.shared.
        
        This method applies shared processors to the base dataset and caches
        the result to avoid recomputation for each segment. Shared processors
        are applied in transform-only mode since they typically don't require
        fitting (e.g., data cleaning, feature engineering).
        
        Returns
        -------
        xr.Dataset
            Dataset after shared processor application
            
        Notes
        -----
        The shared view computation mirrors the shared part of handler.build()
        but maintains full control over segmentation. This ensures consistency
        with the DataHandler while enabling efficient segment-by-segment
        processing.
        
        Shared processors typically include:
        - Data cleaning and validation
        - Basic feature engineering
        - Stateless transformations
        - Formula evaluation (if not segment-specific)
        """
        if self._shared is not None:
            return self._shared

        # This mirrors the shared part of handler.build but keeps full control over segmentation.
        if "features" in self.handler.cache:
            ds_features = self.handler.cache["features"]
        else:
            self.handler.cache["features"] = self.handler.base
            ds_features = self.handler.base

        shared_res = self.handler._apply_pipeline(
            ds_in=ds_features,
            pipeline=self.handler.config.shared,
            fit=False,
            states=None,
        )
        self._shared = shared_res.ds
        return self._shared

    def _compute_bounds(self, stacked: xr.Dataset) -> Tuple[List[slice], List[slice]]:
        """
        Compute integer slices for all segments against the stacked 'time' vector.
        Uses np.searchsorted (fast) rather than .sel (slower).
        
        This mirrors the ModelRunner implementation for consistency.
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

        This exactly mirrors ModelRunner._masked_scatter_rows_inplace to handle
        rectangular expansion and NaN overlaps properly.

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

    def run(self, show_progress: bool = True) -> WalkForwardResult:
        """
        Execute the complete walk-forward plan and return aggregated results.
        
        This method exactly mirrors ModelRunner's gather>scatter pattern to properly
        handle rectangular expansion and NaN overlaps that occur when unstacking
        after slicing between irregular times.
        
        Strategy (mirroring ModelRunner exactly):
        1. Compute shared view once
        2. Pre-stack shared for fast integer slicing and precompute segment bounds  
        3. Create separate stacked buffers for each variable (like ModelRunner does for predictions)
        4. For each segment:
           a. Slice train and infer windows by integer search over stacked time vector
           b. Fit learn processors on train slice; transform infer slice using states
           c. Apply infer processors transform-only on infer slice  
           d. Stack segment result and scatter into global buffers using precise indexing
        5. Unstack all buffers at the end and return aggregated dataset
        """
        ds_shared = self._ensure_shared()
        
        # Stack the whole shared dataset once (exactly like ModelRunner)
        time_dims = tuple(d for d in ("year", "month", "day", "hour") if d in ds_shared.dims)
        stacked_shared = ds_shared.stack(time_index=time_dims).transpose("time_index", "asset", ...)
        T = stacked_shared.sizes.get("time_index", 0)
        N = stacked_shared.sizes.get("asset", 0)
        
        # Create separate stacked buffers for each variable (like ModelRunner's final_stacked)
        # Each buffer is a single DataArray, not a full Dataset
        final_buffers = {}
        for var_name in ds_shared.data_vars:
            # Copy all relevant coordinates from stacked_shared (like ModelRunner does)
            buffer_coords = {
                "time_index": stacked_shared.indexes["time_index"], 
                "asset": stacked_shared["asset"].values, 
                "time": ("time_index", stacked_shared["time"].values)
            }
            # Add time_flat if present (like ModelRunner)
            if "time_flat" in stacked_shared.coords:
                buffer_coords["time_flat"] = ("time_index", stacked_shared["time_flat"].values)
            
            final_buffers[var_name] = xr.DataArray(
                np.full((T, N), np.nan, dtype=np.float64),
                dims=("time_index", "asset"),
                coords=buffer_coords,
                name=var_name,
            )
        
        # Precompute integer slices for all segments using ModelRunner's method
        train_slices, infer_slices = self._compute_bounds(stacked_shared)
        segment_states: List[Dict[str, Any]] = []
        
        # Setup progress bar
        iterator = zip(self.plan, train_slices, infer_slices)
        if show_progress:
            iterator = tqdm(iterator, total=len(self.plan), desc="Walk-forward analysis")
        
        for i, (seg, tr_s, inf_s) in enumerate(iterator):
            # Slice train and infer from pre-stacked shared, then unstack
            # Note: Unstack may introduce rectangular expansions with NaNs for partial slices,
            # which will be handled by the precise indexing below
            train_stacked_sliced = stacked_shared.isel(time_index=tr_s)
            train_ds = train_stacked_sliced.unstack("time_index")
            
            infer_stacked_sliced = stacked_shared.isel(time_index=inf_s)
            infer_ds = infer_stacked_sliced.unstack("time_index")
            
            # Empty-guard via sizes (exactly like ModelRunner)
            if train_ds.sizes.get("asset", 0) == 0 or infer_ds.sizes.get("asset", 0) == 0:
                segment_states.append({"segment": i, "infer_rows": np.array([], dtype=int)})
                continue
            if train_ds.sizes.get("year", 0) == 0 or infer_ds.sizes.get("year", 0) == 0:
                segment_states.append({"segment": i, "infer_rows": np.array([], dtype=int)})
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
            
            # Stack the processed result for scattering (exactly like ModelRunner does for predictions)
            infer_stacked = infer_final.stack(time_index=time_dims).transpose("time_index", "asset", ...)
            
            # Scatter back! (exactly mimicking ModelRunner's approach)
            # For each variable, do the precise indexing and scatter
            for var_name in infer_final.data_vars:
                if var_name not in final_buffers:
                    continue
                
                var_stacked = infer_stacked[var_name]
                final_buffer = final_buffers[var_name]
                
                # In-place masked scatter into the global buffer using integers (like ModelRunner)
                global_mi = final_buffer.indexes["time_index"]
                local_mi = var_stacked.indexes["time_index"]
                rows_global = global_mi.get_indexer(local_mi)         # shape (T_local,), -1 for misses
                
                # paranoid check (exactly like ModelRunner)
                keep = rows_global >= 0
                if not keep.all():
                    # Drop any local rows that don't exist in the global calendar (rare)
                    rows_global = rows_global[keep]
                    var_stacked = var_stacked.isel(time_index=keep)
                
                in_window = (rows_global >= inf_s.start) & (rows_global < inf_s.stop)
                if not in_window.any():
                    continue

                rows_kept = rows_global[in_window]
                var_kept = var_stacked.isel(time_index=in_window)

                self._masked_scatter_rows_inplace(
                    buf=final_buffer,
                    rows_global=rows_kept,
                    src_stacked=var_kept.transpose("time_index", "asset"),
                    policy=self.overlap_policy,
                )
            
            # Record segment metadata (like ModelRunner)
            segment_states.append({
                "segment": i,
                "infer_start": seg.infer_start,
                "infer_end": seg.infer_end,
                "infer_rows": inf_s.stop - inf_s.start,
                "num_learn_states": len(learn_states),
                "status": "processed"
            })
        
        # Unstack all buffers at the end (exactly like ModelRunner)
        final_vars = {}
        for var_name, buffer in final_buffers.items():
            var_da = buffer.unstack("time_index")  # -> (year, month, day[, hour], asset)
            if "time" in buffer.coords:
                var_da = var_da.assign_coords(time=buffer["time"].unstack("time_index"))
            if "time_flat" in buffer.coords:
                var_da = var_da.assign_coords(time_flat=buffer["time_flat"].unstack("time_index"))
                
            time_dims_sorted = [d for d in ("year", "month", "day", "hour") if d in var_da.dims]
            var_da = var_da.transpose(*time_dims_sorted, "asset")
            final_vars[var_name] = var_da
        
        # Create final dataset
        final_ds = xr.Dataset(final_vars)
        final_ds = final_ds.assign_coords(asset=stacked_shared["asset"].values)
        
        # Add metadata attributes (like ModelRunner)
        attrs = {
            "overlap_policy": self.overlap_policy,
            "segments": len(self.plan),
            "created_at_unix": time.time(),
            "processing_type": "walk_forward_data_processing"
        }
        final_ds = final_ds.assign_attrs(attrs)
        
        return WalkForwardResult(
            processed_ds=final_ds,
            segment_states=segment_states,
            attrs=attrs
        )

    def run_segments(self, show_progress: bool = True) -> List[SegmentResult]:
        """
        Execute walk-forward analysis and return individual segment results.
        
        This method preserves the original behavior for backward compatibility
        while the main run() method now implements the gather>scatter pattern.
            
        Returns
        -------
        List[SegmentResult]
            Individual results from each processed segment
        """
        ds_shared = self._ensure_shared()
        
        # Use the more efficient bounds computation
        time_dims = tuple(d for d in ("year", "month", "day", "hour") if d in ds_shared.dims)
        stacked_shared = ds_shared.stack(time_index=time_dims).transpose("time_index", "asset", ...)
        
        # Precompute bounds using the efficient method
        train_slices, infer_slices = self._compute_bounds(stacked_shared)
        results: List[SegmentResult] = []
        
        # Setup progress bar
        iterator = zip(self.plan, train_slices, infer_slices)
        if show_progress:
            iterator = tqdm(iterator, total=len(self.plan), desc="Walk-forward segments")
        
        for seg, tr_s, inf_s in iterator:
            # Slice train and infer from pre-stacked shared, then unstack
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

            results.append(SegmentResult(
                segment=seg, 
                ds_infer=infer_final, 
                learn_states=learn_states
            ))

        return results

    def run_single_segment(self, segment: Segment) -> SegmentResult:
        """
        Execute processing for a single segment.
        
        This method provides fine-grained control for processing individual
        segments, useful for debugging, testing, or custom workflows.
        
        Parameters
        ----------
        segment : Segment
            Single segment to process
            
        Returns
        -------
        SegmentResult
            Result from processing the segment
            
        Notes
        -----
        This method follows the same processing logic as run() but for
        a single segment. It's useful for:
        - Debugging specific time periods
        - Custom segment processing workflows
        - Testing processing logic on smaller datasets
        - Interactive analysis and development
        """
        # Create a single-segment plan and delegate to run()
        single_plan = SegmentPlan(segments=[segment])
        temp_runner = WalkForwardRunner(handler=self.handler, plan=single_plan)
        results = temp_runner.run(show_progress=False)
        return results[0] if results else None

    def get_execution_summary(self) -> dict:
        """
        Get summary information about the execution plan.
        
        Returns
        -------
        dict
            Summary statistics about the walk-forward execution
        """
        if not self.plan.segments:
            return {
                'num_segments': 0,
                'total_training_period': None,
                'total_inference_period': None,
                'avg_segment_gap': None
            }
        
        train_start, train_end = self.plan.total_training_period
        infer_start, infer_end = self.plan.total_inference_period
        
        # Calculate average gap between segments
        gaps = []
        for i in range(1, len(self.plan.segments)):
            prev_seg = self.plan.segments[i-1]
            curr_seg = self.plan.segments[i]
            gap = curr_seg.infer_start - prev_seg.infer_end
            gaps.append(gap)
        
        avg_gap = np.mean(gaps) if gaps else np.timedelta64(0)
        
        return {
            'num_segments': len(self.plan),
            'total_training_period': (train_start, train_end),
            'total_inference_period': (infer_start, infer_end),
            'avg_segment_gap': avg_gap,
            'avg_train_duration': np.mean([s.train_duration for s in self.plan.segments]),
            'avg_infer_duration': np.mean([s.infer_duration for s in self.plan.segments])
        }
