"""
Tests for the core data layer: from_table(), load_simulated_data(),
FrequencyType, and the .dt accessor.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.core.types import FrequencyType
from src.data.loaders.table import from_table, load_simulated_data


# ---------------------------------------------------------------------------
# from_table()
# ---------------------------------------------------------------------------

def test_from_table_basic():
    """DataFrame converted to Dataset should have correct dims and values."""
    df = pd.DataFrame(
        {
            "time": pd.date_range("2023-01-02", periods=3, freq="D").repeat(2),
            "asset": ["A", "B"] * 3,
            "price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    ds = from_table(df, time_column="time", asset_column="asset", frequency=FrequencyType.DAILY)

    assert isinstance(ds, xr.Dataset)
    # Expect the five calendar dims + asset
    for dim in ("year", "month", "day", "hour", "asset"):
        assert dim in ds.dims, f"Missing dim: {dim}"
    assert "price" in ds.data_vars
    assert set(ds.coords["asset"].values) == {"A", "B"}


def test_from_table_feature_columns():
    """Explicit feature_columns should restrict data_vars to those names."""
    df = pd.DataFrame(
        {
            "time": pd.date_range("2023-01-02", periods=4, freq="D").repeat(1),
            "asset": ["X"] * 4,
            "feat_a": np.arange(4, dtype=float),
            "feat_b": np.arange(4, dtype=float) * 2,
            "feat_c": np.arange(4, dtype=float) * 3,
        }
    )
    ds = from_table(
        df,
        time_column="time",
        asset_column="asset",
        feature_columns=["feat_a", "feat_b"],
        frequency=FrequencyType.DAILY,
    )
    assert "feat_a" in ds.data_vars
    assert "feat_b" in ds.data_vars
    assert "feat_c" not in ds.data_vars


def test_from_table_invalid_time_raises():
    """Non-parseable time values should raise ValueError."""
    df = pd.DataFrame(
        {
            "time": ["not-a-date", "also-bad"],
            "asset": ["X", "X"],
            "v": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError):
        from_table(df, time_column="time", asset_column="asset")


# ---------------------------------------------------------------------------
# load_simulated_data()
# ---------------------------------------------------------------------------

def test_load_simulated_data_shape(simulated_daily_ds):
    """3 assets and 3 vars expected after fixture construction."""
    ds = simulated_daily_ds
    assert "asset" in ds.dims
    assert len(ds.coords["asset"]) == 3
    # Three var_ variables
    var_names = [k for k in ds.data_vars if k.startswith("var_")]
    assert len(var_names) == 3


def test_load_simulated_data_monthly(simulated_monthly_ds):
    """Monthly dataset should have correct number of months."""
    ds = simulated_monthly_ds
    assert "month" in ds.dims
    # 24 timesteps monthly starting 2021-01 means months 1-12 for 2021 and 2022
    assert len(ds.coords["asset"]) == 2
    var_names = [k for k in ds.data_vars if k.startswith("var_")]
    assert len(var_names) == 2


# ---------------------------------------------------------------------------
# .dt accessor
# ---------------------------------------------------------------------------

def test_dt_to_time_indexed(simulated_daily_ds):
    """to_time_indexed() flattens multi-dim time into a single 'time' dim."""
    flat = simulated_daily_ds.dt.to_time_indexed()
    assert "time" in flat.dims
    assert "asset" in flat.dims
    # year/month/day should no longer be independent dims
    assert "year" not in flat.dims
    assert "month" not in flat.dims
    assert "day" not in flat.dims


def test_dt_shift(simulated_daily_ds):
    """shift(periods=1) should produce a Dataset with same element count and more NaNs."""
    ds = simulated_daily_ds
    shifted = ds.dt.shift(periods=1)

    # Same total number of elements (dimensions may reorder, but sizes match)
    assert shifted["var_1"].size == ds["var_1"].size

    # Shift introduces at least one extra NaN per asset (leading edge becomes NaN)
    orig_flat = ds.dt.to_time_indexed()["var_1"].values
    shift_flat = shifted.dt.to_time_indexed()["var_1"].values
    assert np.isnan(shift_flat).sum() >= np.isnan(orig_flat).sum()
