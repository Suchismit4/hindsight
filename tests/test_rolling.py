"""
Tests for rolling window operations via the .dt.rolling() accessor.
Rolling is done on Datasets (not bare DataArrays) since the DataArray
accessor requires the parent Dataset for mask computation.
to_time_indexed() is also called on Datasets then variables are extracted.
"""

import numpy as np
import pytest
import xarray as xr

from src.data.core.operations import mean as core_mean, ema as core_ema, rma as core_rma, sum_func as core_sum
from src.data.loaders.table import from_table
from src.data.core.types import FrequencyType


# ---------------------------------------------------------------------------
# SMA shape — rolling on Dataset; result var should have same size as input
# ---------------------------------------------------------------------------

def test_rolling_sma_shape(ohlcv_ds):
    ds = ohlcv_ds
    result = ds.dt.rolling(dim="time", window=5).reduce(core_mean)
    assert result["close"].size == ds["close"].size


# ---------------------------------------------------------------------------
# SMA values — small known dataset
# ---------------------------------------------------------------------------

def test_rolling_sma_values():
    """Manually verify rolling mean on a tiny 1-asset dataset."""
    import pandas as pd

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    dates = pd.date_range("2023-01-02", periods=6, freq="D")
    df = pd.DataFrame({"time": dates, "asset": "X", "v": values})
    ds = from_table(df, time_column="time", asset_column="asset", frequency=FrequencyType.DAILY)

    result = ds.dt.rolling(dim="time", window=3).reduce(core_mean)
    # Use Dataset.dt.to_time_indexed() and then select variable
    flat = result.dt.to_time_indexed()["v"].sel(asset="X").values

    finite_vals = flat[np.isfinite(flat)]
    assert len(finite_vals) > 0, "Expected some finite values from rolling SMA"
    assert float(finite_vals[0]) == pytest.approx(2.0, rel=1e-5)


# ---------------------------------------------------------------------------
# EMA runs and produces finite values
# ---------------------------------------------------------------------------

def test_rolling_ema_runs(ohlcv_ds):
    result = ohlcv_ds.dt.rolling(dim="time", window=10).reduce(core_ema)
    flat = result.dt.to_time_indexed()["close"].values
    assert np.isfinite(flat[~np.isnan(flat)]).all()


# ---------------------------------------------------------------------------
# Dataset rolling consistency — close and high should differ
# ---------------------------------------------------------------------------

def test_rolling_dataset_vs_dataarray(ohlcv_ds):
    """Rolling on Dataset should produce consistent per-variable results."""
    ds = ohlcv_ds
    window = 5
    result = ds.dt.rolling(dim="time", window=window).reduce(core_mean)
    flat = result.dt.to_time_indexed()

    close_vals = flat["close"].values
    high_vals = flat["high"].values

    assert close_vals.shape == high_vals.shape
    finite_mask = np.isfinite(close_vals) & np.isfinite(high_vals)
    if finite_mask.any():
        assert not np.allclose(close_vals[finite_mask], high_vals[finite_mask]), (
            "close and high rolling means should differ"
        )


# ---------------------------------------------------------------------------
# Multiple operations produce same-size output
# ---------------------------------------------------------------------------

def test_rolling_multiple_ops(ohlcv_ds):
    ds = ohlcv_ds
    expected_size = ds["close"].size

    for fn, name in [(core_ema, "ema"), (core_rma, "rma"), (core_sum, "rolling_sum")]:
        result = ds.dt.rolling(dim="time", window=10).reduce(fn)
        assert result["close"].size == expected_size, f"{name} size mismatch"
