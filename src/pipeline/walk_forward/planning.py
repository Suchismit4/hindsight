"""
Walk-forward planning and segment generation utilities.

This module provides functions for generating walk-forward segment plans
from configuration parameters. It handles the complex logic of creating
temporally consistent segments while respecting data boundaries and
avoiding lookahead bias.

The planning system follows qlib's pattern of separating temporal logic
from data processing, enabling flexible backtesting workflows.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import xarray as xr

from .segments import Segment, SegmentPlan, SegmentConfig


def _time_min_max(ds: xr.Dataset) -> Tuple[np.datetime64, np.datetime64]:
    """
    Extract min/max valid datetime bounds from dataset time coordinate.
    
    This function finds the valid time range from a dataset's time coordinate,
    properly handling invalid calendar entries (NaT) that may result from
    rectangular grid unstacking operations.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'time' coordinate to analyze
        
    Returns
    -------
    Tuple[np.datetime64, np.datetime64]
        Tuple of (min_time, max_time)
        
    Raises
    ------
    ValueError
        If dataset has no 'time' coordinate, empty time coordinate,
        or all time entries are NaT
        
    Notes
    -----
    This function works with both 1D time coordinates and ND time grids
    that span (year, month, day, hour) dimensions. It flattens the coordinate
    and filters out invalid calendar combinations.
    
    Invalid time entries typically result from unstacking operations that
    create rectangular grids from business day calendars, where some
    combinations (like weekends) produce NaT values.
    """
    if "time" not in ds.coords:
        raise ValueError("Dataset has no 'time' coordinate")

    t = ds["time"].values
    t = t.reshape(-1)  # flatten

    if t.size == 0:
        raise ValueError("Empty time coordinate")

    # Filter out invalid calendar combos only (NaT). These are not real 'missing observations'.
    mask_valid = ~np.isnat(t)
    if not mask_valid.any():
        raise ValueError("All time entries are NaT; cannot determine bounds.")

    t_valid = t[mask_valid]
    lo = np.datetime64(t_valid.min(), "ns")
    hi = np.datetime64(t_valid.max(), "ns")
    return lo, hi


def _clip_timestamp(ts: np.datetime64, lo: np.datetime64, hi: np.datetime64) -> np.datetime64:
    """
    Clip a timestamp to be within specified bounds.
    
    Parameters
    ----------
    ts : np.datetime64
        Timestamp to clip
    lo : np.datetime64
        Lower bound (inclusive)
    hi : np.datetime64
        Upper bound (inclusive)
        
    Returns
    -------
    np.datetime64
        Clipped timestamp in nanosecond precision
    """
    # normalize units for consistent comparisons
    ts = np.datetime64(ts, "ns")
    lo = np.datetime64(lo, "ns")
    hi = np.datetime64(hi, "ns")
    return np.minimum(np.maximum(ts, lo), hi)


def make_plan(config: SegmentConfig, ds_for_bounds: Optional[xr.Dataset] = None) -> SegmentPlan:
    """
    Generate a complete walk-forward segment plan from configuration.
    
    This function creates a contiguous walk-forward schedule across the
    specified time range, with optional clipping to dataset boundaries
    to ensure segments don't extend beyond available data.
    
    Parameters
    ----------
    config : SegmentConfig
        Configuration parameters for segment generation
    ds_for_bounds : xr.Dataset, optional
        Dataset to use for boundary clipping if config.clip_to_data is True.
        If provided, segments will be clipped to the dataset's valid time domain.
        
    Returns
    -------
    SegmentPlan
        Generated plan containing all walk-forward segments
        
    Raises
    ------
    ValueError
        If step size is not positive
    FileNotFoundError
        If ds_for_bounds is required but not provided
        
    Notes
    -----
    The generation process follows these steps:
    1. Normalize all timestamps to nanosecond precision for consistency
    2. Extract dataset bounds if clipping is enabled
    3. Generate segments by advancing cursor in step increments:
       - Create training window: [cursor, cursor + train_span]
       - Add gap: training_end + gap
       - Create inference window: [train_end + gap, train_end + gap + infer_span]
       - Clip to dataset bounds if enabled
       - Skip segments with collapsed windows after clipping
    4. Continue until cursor exceeds end time
    
    The clipping process ensures no segment extends beyond actual data
    availability, which is crucial for robust backtesting with real datasets
    that may have irregular coverage or business day calendars.
    
    Examples
    --------
    Basic monthly walk-forward plan:
    
    >>> config = SegmentConfig(
    ...     start=np.datetime64('2020-01-01'),
    ...     end=np.datetime64('2020-06-30'),
    ...     train_span=np.timedelta64(90, 'D'),
    ...     infer_span=np.timedelta64(30, 'D'),
    ...     step=np.timedelta64(30, 'D'),
    ...     gap=np.timedelta64(1, 'D')
    ... )
    >>> plan = make_plan(config)
    >>> print(f"Generated {len(plan)} segments")
    
    Plan with dataset boundary clipping:
    
    >>> plan = make_plan(config, ds_for_bounds=financial_dataset)
    >>> validation_issues = plan.validate()
    >>> if not validation_issues:
    ...     print("Plan is valid")
    """
    # normalize to ns to avoid unit surprises
    cursor       = np.datetime64(config.start, "ns")
    end_ns       = np.datetime64(config.end, "ns")
    train_span   = config.train_span.astype("timedelta64[ns]")
    infer_span   = config.infer_span.astype("timedelta64[ns]")
    step         = config.step.astype("timedelta64[ns]")
    gap          = config.gap.astype("timedelta64[ns]")

    if step <= np.timedelta64(0, "ns"):
        raise ValueError("step must be positive")

    # Dataset bounds only gate clipping. They do not drive contiguity.
    lo = hi = None
    if ds_for_bounds is not None and config.clip_to_data:
        lo, hi = _time_min_max(ds_for_bounds)  # valid calendar min/max

    # Generate candidate train starts using numpy to cut Python loop overhead.
    # np.arange handles datetime64 arithmetic natively and yields the same
    # cursor progression semantics as the original while-loop.
    candidate_train_starts = np.arange(cursor, end_ns + step, step, dtype="datetime64[ns]")
    if candidate_train_starts.size == 0:
        return SegmentPlan(segments=[])

    train_starts = candidate_train_starts
    train_ends = train_starts + train_span
    infer_starts = train_ends + gap

    # Respect original stopping condition: break once either train_start or
    # infer_start exceeds the configured end bound.
    valid_mask = (train_starts <= end_ns) & (infer_starts <= end_ns)
    if not np.any(valid_mask):
        return SegmentPlan(segments=[])

    train_starts = train_starts[valid_mask]
    train_ends = train_ends[valid_mask]
    infer_starts = infer_starts[valid_mask]
    infer_ends = infer_starts + infer_span - np.timedelta64(1, "ns")  # exclusive end

    if lo is not None and hi is not None:
        train_starts = np.clip(train_starts, lo, hi)
        train_ends = np.clip(train_ends, lo, hi)
        infer_starts = np.clip(infer_starts, lo, hi)
        infer_ends = np.clip(infer_ends, lo, hi)

    valid_windows = (train_starts <= train_ends) & (infer_starts <= infer_ends)

    segments = [
        Segment(ts, te, us, ue)
        for ts, te, us, ue in zip(
            train_starts[valid_windows],
            train_ends[valid_windows],
            infer_starts[valid_windows],
            infer_ends[valid_windows],
        )
    ]

    return SegmentPlan(segments=segments)


# Pretty functions below.

def expand_plan_coverage(plan: SegmentPlan, 
                        target_start: Optional[np.datetime64] = None,
                        target_end: Optional[np.datetime64] = None,
                        config: Optional[SegmentConfig] = None) -> SegmentPlan:
    """
    Expand an existing plan to cover additional time periods.
    
    This function extends an existing segment plan to cover a wider time
    range, useful for expanding backtests or filling gaps in coverage.
    
    Parameters
    ----------
    plan : SegmentPlan
        Existing plan to expand
    target_start : np.datetime64, optional
        Target start time (expand backwards if before current start)
    target_end : np.datetime64, optional
        Target end time (expand forwards if after current end)
    config : SegmentConfig, optional
        Configuration for generating new segments (required if expanding)
        
    Returns
    -------
    SegmentPlan
        Expanded plan covering the target time range
        
    Raises
    ------
    ValueError
        If config is required but not provided
        
    Notes
    -----
    The expansion process:
    1. Determines current plan coverage
    2. Generates additional segments before/after existing coverage
    3. Merges new segments with existing plan
    4. Validates temporal consistency
    
    This is useful for iteratively building walk-forward plans or
    expanding existing plans when new data becomes available.
    """
    if not plan.segments:
        if config is None:
            raise ValueError("Config required to expand empty plan")
        if target_start is None or target_end is None:
            raise ValueError("Target start/end required for empty plan expansion")
        
        new_config = SegmentConfig(
            start=target_start,
            end=target_end,
            train_span=config.train_span,
            infer_span=config.infer_span,
            step=config.step,
            gap=config.gap,
            clip_to_data=config.clip_to_data
        )
        return make_plan(new_config)
    
    current_start, current_end = plan.total_inference_period
    new_segments = []
    
    # Expand backwards if needed
    if target_start is not None and target_start < current_start:
        if config is None:
            raise ValueError("Config required for plan expansion")
        
        # Generate segments before current start
        backward_config = SegmentConfig(
            start=target_start,
            end=current_start,
            train_span=config.train_span,
            infer_span=config.infer_span,
            step=config.step,
            gap=config.gap,
            clip_to_data=config.clip_to_data
        )
        backward_plan = make_plan(backward_config)
        new_segments.extend(backward_plan.segments)
    
    # Add existing segments
    new_segments.extend(plan.segments)
    
    # Expand forwards if needed
    if target_end is not None and target_end > current_end:
        if config is None:
            raise ValueError("Config required for plan expansion")
        
        # Generate segments after current end
        forward_config = SegmentConfig(
            start=current_end,
            end=target_end,
            train_span=config.train_span,
            infer_span=config.infer_span,
            step=config.step,
            gap=config.gap,
            clip_to_data=config.clip_to_data
        )
        forward_plan = make_plan(forward_config)
        new_segments.extend(forward_plan.segments)
    
    return SegmentPlan(segments=new_segments)


def optimize_plan_for_dataset(plan: SegmentPlan, ds: xr.Dataset, 
                             min_train_samples: int = 100,
                             min_infer_samples: int = 10) -> SegmentPlan:
    """
    Optimize a segment plan based on actual data availability.
    
    This function analyzes data availability in each segment and removes
    or adjusts segments that don't have sufficient data for reliable
    model training and evaluation.
    
    Parameters
    ----------
    plan : SegmentPlan
        Original segment plan to optimize
    ds : xr.Dataset
        Dataset to analyze for data availability
    min_train_samples : int, default 100
        Minimum number of valid samples required in training period
    min_infer_samples : int, default 10
        Minimum number of valid samples required in inference period
        
    Returns
    -------
    SegmentPlan
        Optimized plan with segments having sufficient data
        
    Notes
    -----
    The optimization process:
    1. For each segment, slice the dataset to training and inference periods
    2. Count valid (non-NaN) samples in key variables
    3. Remove segments that don't meet minimum sample requirements
    4. Optionally adjust segment boundaries to maximize data usage
    
    This is particularly useful when working with datasets that have
    irregular coverage, holidays, or missing data periods.
    """
    optimized_segments = []
    
    # Simple approach: filter segments with insufficient data
    # More sophisticated approaches could adjust boundaries
    
    for segment in plan.segments:
        # Quick data availability check using time coordinate
        time_vals = ds.time.values.flatten()
        valid_mask = ~np.isnat(time_vals)
        
        if not valid_mask.any():
            continue  # Skip if no valid times
        
        valid_times = time_vals[valid_mask]
        
        # Count samples in training period
        train_mask = ((valid_times >= segment.train_start) & 
                     (valid_times <= segment.train_end))
        train_samples = train_mask.sum()
        
        # Count samples in inference period  
        infer_mask = ((valid_times >= segment.infer_start) & 
                     (valid_times <= segment.infer_end))
        infer_samples = infer_mask.sum()
        
        # Keep segment if it meets minimum requirements
        if train_samples >= min_train_samples and infer_samples >= min_infer_samples:
            optimized_segments.append(segment)
    
    return SegmentPlan(segments=optimized_segments)
