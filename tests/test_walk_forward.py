"""
Tests for walk-forward planning: Segment, SegmentConfig, make_plan(), SegmentPlan.
No data loading is needed — all tests operate on date arithmetic only.
"""

import numpy as np
import pytest

from src.pipeline.walk_forward.planning import make_plan
from src.pipeline.walk_forward.segments import Segment, SegmentConfig, SegmentPlan


# ---------------------------------------------------------------------------
# Segment creation
# ---------------------------------------------------------------------------

def test_segment_creation():
    seg = Segment(
        train_start=np.datetime64("2020-01-01"),
        train_end=np.datetime64("2020-12-31"),
        infer_start=np.datetime64("2021-01-01"),
        infer_end=np.datetime64("2021-03-31"),
    )
    assert seg.train_start < seg.train_end
    assert seg.infer_start <= seg.infer_end
    assert seg.infer_start >= seg.train_end


def test_segment_invalid_train_raises():
    with pytest.raises(ValueError):
        Segment(
            train_start=np.datetime64("2020-12-31"),
            train_end=np.datetime64("2020-01-01"),  # inverted
            infer_start=np.datetime64("2021-01-01"),
            infer_end=np.datetime64("2021-03-31"),
        )


# ---------------------------------------------------------------------------
# SegmentConfig
# ---------------------------------------------------------------------------

def test_segment_config_basic():
    cfg = SegmentConfig(
        start=np.datetime64("2020-01-01"),
        end=np.datetime64("2022-12-31"),
        train_span=np.timedelta64(365, "D"),
        infer_span=np.timedelta64(30, "D"),
        step=np.timedelta64(30, "D"),
    )
    assert cfg.total_duration > np.timedelta64(0)
    assert cfg.estimate_segment_count() > 0


def test_segment_config_invalid_start_raises():
    with pytest.raises(ValueError):
        SegmentConfig(
            start=np.datetime64("2022-12-31"),
            end=np.datetime64("2020-01-01"),  # end before start
            train_span=np.timedelta64(365, "D"),
            infer_span=np.timedelta64(30, "D"),
            step=np.timedelta64(30, "D"),
        )


# ---------------------------------------------------------------------------
# make_plan()
# ---------------------------------------------------------------------------

def _base_config():
    return SegmentConfig(
        start=np.datetime64("2020-01-01"),
        end=np.datetime64("2021-12-31"),
        train_span=np.timedelta64(180, "D"),
        infer_span=np.timedelta64(30, "D"),
        step=np.timedelta64(30, "D"),
        gap=np.timedelta64(0, "D"),
        clip_to_data=False,
    )


def test_make_plan_segment_count():
    plan = make_plan(_base_config())
    assert isinstance(plan, SegmentPlan)
    assert len(plan) > 0


def test_make_plan_temporal_ordering():
    """Each segment's infer_start must be >= its train_end."""
    plan = make_plan(_base_config())
    for seg in plan:
        assert seg.infer_start >= seg.train_end, (
            f"infer_start {seg.infer_start} < train_end {seg.train_end}"
        )


def test_make_plan_no_overlap():
    """Consecutive inference windows should not overlap."""
    plan = make_plan(_base_config())
    segs = list(plan)
    for i in range(len(segs) - 1):
        curr, nxt = segs[i], segs[i + 1]
        assert nxt.infer_start >= curr.infer_end, (
            f"Segment {i} infer_end {curr.infer_end} overlaps "
            f"segment {i+1} infer_start {nxt.infer_start}"
        )


def test_make_plan_with_gap():
    """Non-zero gap should be reflected in segment boundaries."""
    gap = np.timedelta64(7, "D")
    cfg = SegmentConfig(
        start=np.datetime64("2020-01-01"),
        end=np.datetime64("2021-12-31"),
        train_span=np.timedelta64(180, "D"),
        infer_span=np.timedelta64(30, "D"),
        step=np.timedelta64(30, "D"),
        gap=gap,
        clip_to_data=False,
    )
    plan = make_plan(cfg)
    for seg in plan:
        actual_gap = seg.infer_start - seg.train_end
        assert actual_gap >= np.timedelta64(0), "Gap should be non-negative"


# 
# ff
# RD Agents
# lseg dataset
# LLM concurentlly interacting w hindsight and rl agnets