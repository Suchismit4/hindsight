"""
Main DataHandler implementation for the data processing pipeline.

This module contains the DataHandler class which orchestrates the complete
data processing pipeline, managing views, caching, and processor execution.
It follows qlib's design pattern of separating data processing logic from
temporal segmentation concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from .core import View, PipelineMode, ProcessorState
from .config import HandlerConfig
from .processors import Processor


@dataclass
class DataHandler:
    """
    Central orchestrator for data processing pipelines.
    
    The DataHandler manages the complete data processing workflow, including
    caching of intermediate results, pipeline execution, and view management.
    It separates "how data is processed" (processors) from "when" (segments),
    following qlib's architectural principles.
    
    Parameters
    ----------
    base : xr.Dataset
        The raw input dataset that serves as the foundation for all processing
    config : HandlerConfig
        Configuration object defining the processing pipeline
        
    Attributes
    ----------
    cache : dict
        Cache for storing intermediate processing results
    learn_states : list of ProcessorState
        Learned states from the learn processor pipeline
    infer_states : list of ProcessorState
        Learned states from the infer processor pipeline
        
    Notes
    -----
    The DataHandler supports three main views:
    - RAW: Original data after feature graph construction
    - LEARN: Data processed through shared + learn pipelines (for training)
    - INFER: Data processed through shared + infer pipelines (for inference)
    
    The handler uses lazy evaluation and caching to avoid recomputing expensive
    transformations. Views are built on-demand and cached for subsequent access.
    
    Processing Pipeline Structure:
    1. Shared processors: Run once on full dataset (transform-only)
    2. Learn processors: Fit on training segments, transform on both train/infer
    3. Infer processors: Transform-only on inference segments
    
    Examples
    --------
    >>> from src.pipeline.data_handler import DataHandler, HandlerConfig
    >>> from src.pipeline.data_handler.processors import CSZScore, PerAssetFFill
    >>> 
    >>> config = HandlerConfig(
    ...     shared=[PerAssetFFill(name="ffill")],
    ...     learn=[CSZScore(name="norm", vars=["close", "volume"])],
    ...     feature_cols=["close_csz", "volume_csz"]
    ... )
    >>> handler = DataHandler(base=dataset, config=config)
    >>> learn_view = handler.view(View.LEARN)
    >>> features = handler.fetch(View.LEARN, ["features"])
    """
    base: xr.Dataset
    config: HandlerConfig
    
    cache: Dict[str, xr.Dataset] = field(default_factory=dict)
    learn_states: List[ProcessorState] = field(default_factory=list)
    infer_states: List[ProcessorState] = field(default_factory=list)

    def build(self) -> None:
        """
        Build all pipeline views and cache the results.
        
        This method executes the complete processing pipeline according to the
        configured mode and caches all intermediate results. It should be called
        before accessing views if not using lazy evaluation.
        
        The build process follows these steps:
        1. Initialize features cache with base dataset
        2. Apply shared processors (transform-only)
        3. Branch execution based on pipeline mode:
           - INDEPENDENT: Parallel learn and infer pipelines
           - APPEND: Sequential infer -> learn pipeline
        4. Cache final views and learned states
        """
        if "features" not in self.cache:
            self.cache["features"] = self.base
            
        ds_features = self.cache["features"]

        # Run shared processors transform-only (stateless or state baked in later)
        ds_shared = self._apply_pipeline(ds_features, self.config.shared, fit=False, states=None).ds

        # Branch or append for learn/infer
        if self.config.mode == PipelineMode.INDEPENDENT:
            learn_res = self._apply_pipeline(ds_shared, self.config.learn, fit=True, states=None)
            infer_res = self._apply_pipeline(ds_shared, self.config.infer, fit=True, states=None)
        else:
            infer_res = self._apply_pipeline(ds_shared, self.config.infer, fit=True, states=None)
            learn_res = self._apply_pipeline(infer_res.ds, self.config.learn, fit=True, states=None)

        self.cache["learn_view"] = learn_res.ds
        self.cache["infer_view"] = infer_res.ds
        self.cache["shared_view"] = ds_shared
        self.learn_states = learn_res.states
        self.infer_states = infer_res.states
        
    def shared_view(self) -> xr.Dataset:
        """
        Get the shared processing view.
        
        Returns the dataset after shared processor application but before
        learn/infer branching. Useful for debugging and analysis.
        """
        return self.cache["shared_view"]

    def view(self, which: View) -> xr.Dataset:
        """
        Get a specific view of the processed data.
        
        Returns the requested data view, building the pipeline if necessary.
        Views are cached after first access for performance.
        
        Parameters
        ----------
        which : View
            The data view to retrieve (RAW, LEARN, or INFER)
            
        Returns
        -------
        xr.Dataset
            The requested data view
            
        Notes
        -----
        - RAW view returns the original base dataset
        - LEARN view returns data processed through shared + learn pipelines
        - INFER view returns data processed through shared + infer pipelines
        """
        if which == View.RAW:
            return self.base
        if "learn_view" not in self.cache or "infer_view" not in self.cache:
            self.build()
        return self.cache["learn_view"] if which == View.LEARN else self.cache["infer_view"]

    def fetch(self, which: View, col_set: Optional[Sequence[str]] = None) -> xr.Dataset:
        """
        Fetch specific columns from a data view.
        
        This method provides a convenient way to extract subsets of variables
        from processed views, with support for semantic column groups.
        
        Parameters
        ----------
        which : View
            The data view to fetch from
        col_set : Sequence[str], optional
            Column specifications. Can include:
            - Specific variable names
            - "feature"/"features" to get feature_cols
            - "label"/"labels" to get label_cols
            
        Returns
        -------
        xr.Dataset
            Dataset containing only the requested columns
            
        Notes
        -----
        If col_set contains "feature"/"features", the method will include all
        variables listed in config.feature_cols. Similarly for "label"/"labels"
        and config.label_cols.
        """
        ds = self.view(which)
        if not col_set:
            return ds
            
        cols: List[str] = []
        if self.config.feature_cols and any(c in ("feature", "features") for c in col_set):
            cols.extend(self.config.feature_cols)
        if self.config.label_cols and any(c in ("label", "labels") for c in col_set):
            cols.extend(self.config.label_cols)
        cols = list(dict.fromkeys(cols))  # Remove duplicates while preserving order
        return ds if not cols else ds[cols]

    # DEPRECATED: Use ModelRunner.slice_time instead for now.
    def slice_time(self, ds: xr.Dataset, start: np.datetime64, end: np.datetime64) -> xr.Dataset:
        """
        Fast time-based slicing with business day handling.
        
        Efficiently slice a dataset by time range, handling the complex time
        coordinate structure with proper business day alignment.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset to slice with time coordinates
        start : np.datetime64
            Start time (inclusive)
        end : np.datetime64
            End time (inclusive)
            
        Returns
        -------
        xr.Dataset
            Sliced dataset containing only data within the time range
            
        Notes
        -----
        This method handles datasets with unstacked layouts using dims 
        (year, month, day, hour, asset) and parallel time coordinates.
        It uses binary search for efficient time-based indexing.
        
        The method maintains alignment with time_flat dimensions when present
        and handles invalid calendar entries (NaT) properly.
        """
        # required dims and coord
        time_dims = tuple(d for d in ("year", "month", "day", "hour") if d in ds.dims)
        if len(time_dims) == 0:
            raise ValueError("slice_time expects dims to include some of: year, month, day, hour")
        if "time" not in ds.coords:
            raise ValueError("slice_time expects a 'time' coordinate aligned with (year, month, day, hour)")
        
        cache = getattr(self, "_slice_cache", None)
        ds_id = id(ds)
        if (cache is None or cache.get("ds_id") != ds_id or cache.get("time_dims") != time_dims):
            # (re)build the cache
            stacked = ds.stack(time_index=time_dims)
            t = stacked["time"].values
            cache = {
                "ds_id": ds_id,
                "time_dims": time_dims,
                "t": t,
                "has_time_flat": ("time_flat" in ds.dims and ds.sizes.get("time_flat") == t.shape[0]),
                "T": t.shape[0],
            }
        else:
            stacked = cache["stacked"]
            t = cache["t"]

        # empty guard
        if t.size == 0:
            # empty dataset
            if "time_flat" in ds.dims:
                return ds.isel(time_flat=slice(0, 0))
            return ds

        # integer bounds via binary search
        start64 = np.datetime64(start, t.dtype.name.split('[')[-1][:-1]) if t.dtype.kind == 'M' else np.datetime64(start)
        end64   = np.datetime64(end,   t.dtype.name.split('[')[-1][:-1]) if t.dtype.kind == 'M' else np.datetime64(end)
    
        i0 = int(np.searchsorted(t, start64, side="left"))
        i1 = int(np.searchsorted(t, end64,   side="right"))

        # clamp to [0, T]
        T = t.shape[0]
        i0 = 0 if i0 < 0 else (T if i0 > T else i0)
        i1 = 0 if i1 < 0 else (T if i1 > T else i1)

        if i0 >= i1:
            # return an empty slice with correct structure
            empty_stacked = stacked.isel(time_index=slice(0, 0))
            # also empty time_flat if present
            if "time_flat" in ds.dims and ds.sizes.get("time_flat") == T:
                empty_stacked = empty_stacked.isel(time_flat=slice(0, 0))
            return empty_stacked.unstack("time_index")

        s = slice(i0, i1)

        # slice time_index
        sliced = stacked.isel(time_index=s)

        # keep mask/mask_indices aligned by slicing time_flat when it is parallel to time_index
        if "time_flat" in ds.dims and ds.sizes.get("time_flat") == T:
            sliced = sliced.isel(time_flat=s)

        # unstack back
        return sliced.unstack("time_index")

    def to_arrays_for_model(
        self,
        ds: xr.Dataset,
        feature_vars: Sequence[str],
        label_vars: Optional[Sequence[str]] = None,
        drop_invalid_time: bool = True,
        drop_all_nan_rows: bool = False,
        return_indexer: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]] | Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Convert dataset to dense numpy arrays for model training/inference.
        
        This method converts an unstacked dataset slice to the (T, N, J) format
        commonly used by machine learning models, with proper handling of invalid
        time entries and missing data.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to convert
        feature_vars : Sequence[str]
            Variable names to use as features
        label_vars : Sequence[str], optional
            Variable names to use as labels/targets
        drop_invalid_time : bool, default True
            Whether to drop rows where 'time' coordinate is NaT
        drop_all_nan_rows : bool, default False
            Whether to drop rows where all features across assets are NaN
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray or None]
            Tuple of (X, y) where:
            - X: Features array with shape (T, N, J)
            - y: Labels array with shape (T, N, K) or None if no label_vars
            - If return_indexer=True:            (X, y, valid_idx)
                where valid_idx selects the kept time_index rows from the *full* stacked calendar.

            
        Notes
        -----
        The conversion process:
        1. Stack time dimensions to create time_index
        2. Transpose to (time_index, asset, ...) order
        3. [Optionally] drop invalid times (NaT entries)
        4. Stack feature variables along last dimension
        5. [Optionally] drop rows with all-NaN features
        
        Invalid time entries typically result from unstacking operations that
        create rectangular grids from full day calendars span.
        
        We only drop invalid times (NaT) here (if requested). We do NOT drop rows for
        "all-NaN across assets/features" by default in inference; that per-row filtering
        is handled downstream in the adapter to preserve the calendar geometry.
        """
        # Stack time dimensions
        time_dims = [d for d in ("year", "month", "day", "hour") if d in ds.dims]
        stacked = ds.stack(time_index=tuple(time_dims)).transpose("time_index", "asset", ...)

        # drop invalid times (NaT rows)
        T_Full = stacked.sizes["time_index"]
        valid_idx = np.arange(T_Full)
        if "time" in stacked.coords and np.issubdtype(stacked["time"].dtype, np.datetime64) and drop_invalid_time:
            t = stacked["time"].values  # (T,)
            keep = ~np.isnat(t)
            if not keep.all():
                valid_idx = valid_idx[keep]
                stacked = stacked.isel(time_index=valid_idx)
                
        # build X (and y)
        X = np.stack([stacked[v].values for v in feature_vars], axis=-1)  # (T, N, J)
        y = None
        if label_vars:
            y = np.stack([stacked[v].values for v in label_vars], axis=-1)  # (T, N, K)

        # optionally drop rows where all features across assets are NaN
        if drop_all_nan_rows:
            # any finite value across assets or features?
            row_ok = np.any(np.isfinite(X), axis=(1, 2))
            X = X[row_ok]
            if y is not None:
                y = y[row_ok]

        return (X, y) if not return_indexer else (X, y, valid_idx)

    def shared_view(self) -> xr.Dataset:
        """
        Get the shared processing view.
        
        Returns the dataset after shared processor application but before
        learn/infer branching. Useful for debugging and analysis.
        
        Returns
        -------
        xr.Dataset
            Dataset after shared processing
        """
        if "features" not in self.cache:
            self.cache["features"] = self.base
        if "shared_view" not in self.cache:
            self.cache["shared_view"] = self._apply_pipeline(
                self.cache["features"], self.config.shared, fit=False, states=None
            ).ds
        return self.cache["shared_view"]

    def fit_learn(self, ds: xr.Dataset) -> Tuple[xr.Dataset, List[ProcessorState]]:
        """
        Fit the learn processor pipeline on a dataset.
        
        This method fits all learn processors sequentially and returns both
        the transformed dataset and the learned states.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to fit processors on
            
        Returns
        -------
        tuple[xr.Dataset, list[ProcessorState]]
            Tuple of (transformed_dataset, list_of_states)
        """
        cur = ds
        states: List[ProcessorState] = []
        for p in self.config.learn:
            cur, st = p.fit_transform(cur)
            states.append(st)
        return cur, states

    def transform_with_learn(self, ds: xr.Dataset, states: List[ProcessorState]) -> xr.Dataset:
        """
        Transform a dataset using pre-fitted learn processor states.
        
        This method applies learned transformations to new data using previously
        fitted processor states, enabling consistent processing across time segments.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to transform
        states : list[ProcessorState]
            List of learned states from fit_learn method
            
        Returns
        -------
        xr.Dataset
            Transformed dataset
        """
        cur = ds
        for p, st in zip(self.config.learn, states):
            cur = p.transform(cur, st)
        return cur

    def apply_infer(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply the infer processor pipeline to a dataset.
        
        This method applies infer processors in transform-only mode, typically
        used for post-processing after learn transformations.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to process
            
        Returns
        -------
        xr.Dataset
            Processed dataset
        """
        return self._apply_pipeline(ds, self.config.infer, fit=False, states=None).ds

    @dataclass
    class _PipeResult:
        """Internal result container for pipeline application."""
        ds: xr.Dataset
        states: List[ProcessorState]

    def _apply_pipeline(
        self,
        ds_in: xr.Dataset,
        pipeline: Sequence[Processor],
        fit: bool,
        states: Optional[Sequence[ProcessorState]],
    ) -> _PipeResult:
        """
        Apply a sequence of processors to a dataset.
        
        This internal method handles the execution of processor pipelines with
        proper state management and error handling.
        
        Parameters
        ----------
        ds_in : xr.Dataset
            Input dataset
        pipeline : Sequence[Processor]
            Processors to apply sequentially
        fit : bool
            Whether to fit processors (True) or just transform (False)
        states : Sequence[ProcessorState], optional
            Pre-fitted states for transform-only mode
            
        Returns
        -------
        _PipeResult
            Result containing transformed dataset and processor states
        """
        ds = ds_in
        out_states: List[ProcessorState] = []
        
        if fit:
            for p in pipeline:
                # If the processor can do both quickly, let it override fit_transform
                out, st = p.fit_transform(ds)
                ds = out
                out_states.append(st)
            return DataHandler._PipeResult(ds=ds, states=out_states)
        else:
            if states is not None and len(states) != len(pipeline):
                raise ValueError("Provided states length does not match pipeline length")
            for i, p in enumerate(pipeline):
                st = None if states is None else states[i]
                ds = p.transform(ds, st)
            return DataHandler._PipeResult(ds=ds, states=list(states) if states is not None else [])
