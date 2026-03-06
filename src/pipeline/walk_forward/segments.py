"""
Segment definitions and management for walk-forward analysis.

This module defines the core data structures for representing time segments
in walk-forward backtesting. It follows qlib's pattern of separating temporal
concerns from data processing logic.

The segment system enables rigorous walk-forward validation by clearly defining
training and inference periods with proper temporal boundaries and gap handling
to prevent lookahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Segment:
    """
    A single walk-forward segment defining training and inference periods.

    This class represents a single step in walk-forward analysis, containing
    both the training period for model fitting and the inference period for
    out-of-sample evaluation. Training boundaries are inclusive, inference
    end is exclusive to prevent overlap between consecutive segments.

    Parameters
    ----------
    train_start : np.datetime64
        Start timestamp for training period (inclusive)
    train_end : np.datetime64
        End timestamp for training period (inclusive)
    infer_start : np.datetime64
        Start timestamp for inference period (inclusive)
    infer_end : np.datetime64
        End timestamp for inference period (exclusive)

    Notes
    -----
    The segment design ensures proper temporal separation:
    - Training period: [train_start, train_end] inclusive
    - Inference period: [infer_start, infer_end) exclusive at end
    - Typical pattern: train_end < infer_start (with optional gap)

    The inference end is exclusive to prevent overlap with the next segment's
    inference start when step equals infer_span and gap is zero.

    All timestamps should use consistent np.datetime64 units ('ns', 'ms', 's', etc.)
    for proper comparison and sorting operations.

    Examples
    --------
    >>> segment = Segment(
    ...     train_start=np.datetime64('2020-01-01'),
    ...     train_end=np.datetime64('2020-12-31'),
    ...     infer_start=np.datetime64('2021-01-01'),
    ...     infer_end=np.datetime64('2021-01-31')
    ... )
    >>> print(f"Training: {segment.train_start} to {segment.train_end}")
    >>> print(f"Inference: {segment.infer_start} to {segment.infer_end}")
    """
    train_start: np.datetime64
    train_end: np.datetime64
    infer_start: np.datetime64
    infer_end: np.datetime64

    def __post_init__(self):
        """
        Validate segment temporal consistency.
        
        Raises
        ------
        ValueError
            If temporal boundaries are inconsistent or invalid
        """
        if self.train_start > self.train_end:
            raise ValueError(f"Invalid training period: {self.train_start} > {self.train_end}")
        if self.infer_start > self.infer_end:
            raise ValueError(f"Invalid inference period: {self.infer_start} > {self.infer_end}")

    @property
    def train_duration(self) -> np.timedelta64:
        """
        Get the duration of the training period.
        
        Returns
        -------
        np.timedelta64
            Duration of training period
        """
        return self.train_end - self.train_start

    @property
    def infer_duration(self) -> np.timedelta64:
        """
        Get the duration of the inference period.
        
        Returns
        -------
        np.timedelta64
            Duration of inference period
        """
        return self.infer_end - self.infer_start

    @property
    def gap_duration(self) -> np.timedelta64:
        """
        Get the gap between training and inference periods.
        
        Returns
        -------
        np.timedelta64
            Gap duration (can be negative if periods overlap)
        """
        return self.infer_start - self.train_end

    def overlaps_with(self, other: "Segment") -> bool:
        """
        Check if this segment overlaps with another segment.
        
        Parameters
        ----------
        other : Segment
            Other segment to check overlap with
            
        Returns
        -------
        bool
            True if segments have any temporal overlap
        """
        # Check for any overlap in train or infer periods
        train_overlap = (self.train_start <= other.train_end and 
                        other.train_start <= self.train_end)
        infer_overlap = (self.infer_start <= other.infer_end and 
                        other.infer_start <= self.infer_end)
        cross_overlap = ((self.train_start <= other.infer_end and 
                         other.infer_start <= self.train_end) or
                        (self.infer_start <= other.train_end and 
                         other.train_start <= self.infer_end))
        
        return train_overlap or infer_overlap or cross_overlap


@dataclass
class SegmentPlan:
    """
    A collection of segments describing the complete walk-forward schedule.
    
    This class manages the complete sequence of walk-forward segments,
    providing iteration support and validation of the overall schedule.
    It ensures segments follow proper temporal ordering and provides
    utilities for schedule analysis.
    
    Parameters
    ----------
    segments : List[Segment], default empty
        List of segments in chronological order
        
    Notes
    -----
    The plan maintains segments in chronological order based on inference
    start times. This ensures proper temporal progression in walk-forward
    analysis and enables efficient validation of the overall schedule.
    
    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> 
    >>> segments = []
    >>> for i in range(3):
    ...     seg = Segment(
    ...         train_start=np.datetime64(f'2020-{i+1:02d}-01'),
    ...         train_end=np.datetime64(f'2020-{i+6:02d}-28'),
    ...         infer_start=np.datetime64(f'2020-{i+7:02d}-01'),
    ...         infer_end=np.datetime64(f'2020-{i+7:02d}-28')
    ...     )
    ...     segments.append(seg)
    >>> 
    >>> plan = SegmentPlan(segments)
    >>> print(f"Plan contains {len(plan)} segments")
    >>> for i, seg in enumerate(plan):
    ...     print(f"Segment {i}: {seg.infer_start} to {seg.infer_end}")
    """
    segments: List[Segment] = field(default_factory=list)

    def __len__(self) -> int:
        """
        Get the number of segments in the plan.
        
        Returns
        -------
        int
            Number of segments
        """
        return len(self.segments)

    def __iter__(self):
        """
        Iterate over segments in chronological order.
        
        Yields
        ------
        Segment
            Each segment in the plan
        """
        return iter(self.segments)

    def __getitem__(self, index):
        """
        Get segment by index.
        
        Parameters
        ----------
        index : int or slice
            Index or slice to access segments
            
        Returns
        -------
        Segment or List[Segment]
            Segment(s) at the specified index
        """
        return self.segments[index]

    def add_segment(self, segment: Segment) -> None:
        """
        Add a segment to the plan.
        
        Segments are automatically inserted in chronological order based
        on their inference start times.
        
        Parameters
        ----------
        segment : Segment
            Segment to add to the plan
        """
        # Insert in chronological order by inference start time
        insert_idx = 0
        for i, existing in enumerate(self.segments):
            if segment.infer_start < existing.infer_start:
                insert_idx = i
                break
            insert_idx = i + 1
        
        self.segments.insert(insert_idx, segment)

    def validate(self, allow_overlaps: bool = False) -> List[str]:
        """
        Validate the segment plan for temporal consistency.
        
        Parameters
        ----------
        allow_overlaps : bool, default False
            Whether to allow overlapping segments
            
        Returns
        -------
        List[str]
            List of validation warnings/errors (empty if valid)
        """
        issues = []
        
        if not self.segments:
            return issues
        
        # Check chronological ordering
        for i in range(1, len(self.segments)):
            prev_seg = self.segments[i-1]
            curr_seg = self.segments[i]
            
            if curr_seg.infer_start < prev_seg.infer_start:
                issues.append(f"Segment {i} starts before segment {i-1} "
                            f"({curr_seg.infer_start} < {prev_seg.infer_start})")
        
        # Check for overlaps if not allowed
        if not allow_overlaps:
            for i in range(len(self.segments)):
                for j in range(i+1, len(self.segments)):
                    if self.segments[i].overlaps_with(self.segments[j]):
                        issues.append(f"Segments {i} and {j} overlap")
        
        return issues

    @property
    def total_inference_period(self) -> Tuple[np.datetime64, np.datetime64]:
        """
        Get the total inference period covered by all segments.
        
        Returns
        -------
        Tuple[np.datetime64, np.datetime64]
            Tuple of (earliest_infer_start, latest_infer_end)
            
        Raises
        ------
        ValueError
            If plan contains no segments
        """
        if not self.segments:
            raise ValueError("Cannot compute total period for empty plan")
        
        start = min(seg.infer_start for seg in self.segments)
        end = max(seg.infer_end for seg in self.segments)
        return start, end

    @property
    def total_training_period(self) -> Tuple[np.datetime64, np.datetime64]:
        """
        Get the total training period covered by all segments.
        
        Returns
        -------
        Tuple[np.datetime64, np.datetime64]
            Tuple of (earliest_train_start, latest_train_end)
            
        Raises
        ------
        ValueError
            If plan contains no segments
        """
        if not self.segments:
            raise ValueError("Cannot compute total period for empty plan")
        
        start = min(seg.train_start for seg in self.segments)
        end = max(seg.train_end for seg in self.segments)
        return start, end


@dataclass
class SegmentConfig:
    """
    Configuration for generating rolling walk-forward schedules.
    
    This class defines the parameters needed to automatically generate
    a sequence of walk-forward segments with consistent spacing and sizing.
    It supports flexible gap handling and optional data clipping for
    robust backtesting workflows.
    
    Parameters
    ----------
    start : np.datetime64
        First timestamp to consider for scheduling
    end : np.datetime64
        Last timestamp to consider for scheduling  
    train_span : np.timedelta64
        Length of each training window
    infer_span : np.timedelta64
        Length of each inference window
    step : np.timedelta64
        Shift between consecutive segments (typically equal to infer_span)
    gap : np.timedelta64, default 0
        Optional gap between train_end and infer_start to avoid leakage
    clip_to_data : bool, default True
        Whether to clip each segment to the dataset's valid time domain
        
    Notes
    -----
    The configuration generates segments following this pattern:
    1. Start at the configured start time
    2. Create training window of train_span duration  
    3. Add gap between training and inference
    4. Create inference window of infer_span duration
    5. Advance by step amount for next segment
    6. Repeat until end time is reached
    
    The clip_to_data option ensures segments don't extend beyond actual
    data availability when working with real datasets that may have
    irregular time coverage.
    
    Examples
    --------
    Monthly walk-forward with 12-month training, 1-month inference:
    
    >>> config = SegmentConfig(
    ...     start=np.datetime64('2020-01-01'),
    ...     end=np.datetime64('2023-12-31'),
    ...     train_span=np.timedelta64(365, 'D'),  # 12 months
    ...     infer_span=np.timedelta64(30, 'D'),   # 1 month
    ...     step=np.timedelta64(30, 'D'),         # 1 month step
    ...     gap=np.timedelta64(1, 'D'),           # 1 day gap
    ...     clip_to_data=True
    ... )
    
    Weekly walk-forward with 52-week training, 1-week inference:
    
    >>> config = SegmentConfig(
    ...     start=np.datetime64('2020-01-01'),
    ...     end=np.datetime64('2023-12-31'),  
    ...     train_span=np.timedelta64(52*7, 'D'), # 52 weeks
    ...     infer_span=np.timedelta64(7, 'D'),    # 1 week
    ...     step=np.timedelta64(7, 'D'),          # 1 week step
    ...     gap=np.timedelta64(0, 'D')            # No gap
    ... )
    """
    start: np.datetime64
    end: np.datetime64
    train_span: np.timedelta64
    infer_span: np.timedelta64
    step: np.timedelta64
    gap: np.timedelta64 = np.timedelta64(0, "s")
    clip_to_data: bool = True

    def __post_init__(self):
        """
        Validate configuration parameters.
        
        Raises
        ------
        ValueError
            If configuration parameters are invalid
        """
        if self.start >= self.end:
            raise ValueError(f"Start time must be before end time: {self.start} >= {self.end}")
        if self.step <= np.timedelta64(0):
            raise ValueError("Step must be positive")
        if self.train_span <= np.timedelta64(0):
            raise ValueError("Training span must be positive")
        if self.infer_span <= np.timedelta64(0):
            raise ValueError("Inference span must be positive")

    @property
    def total_duration(self) -> np.timedelta64:
        """
        Get the total duration of the configuration period.
        
        Returns
        -------
        np.timedelta64
            Duration from start to end
        """
        return self.end - self.start

    def estimate_segment_count(self) -> int:
        """
        Estimate the number of segments this configuration will generate.
        
        Returns
        -------
        int
            Estimated number of segments (may vary due to clipping)
        """
        if self.step == np.timedelta64(0):
            return 1
        return max(0, int((self.end - self.start) // self.step))
