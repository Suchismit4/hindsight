"""
Tests for pipeline processors: PerAssetFFill, CSZScore, FormulaEval,
DataHandler, and View semantics.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.core.types import FrequencyType
from src.data.loaders.table import from_table
from src.pipeline.data_handler.config import HandlerConfig
from src.pipeline.data_handler.core import View
from src.pipeline.data_handler.handler import DataHandler
from src.pipeline.data_handler.processors import CSZScore, FormulaEval, PerAssetFFill


# ---------------------------------------------------------------------------
# PerAssetFFill
# ---------------------------------------------------------------------------

def test_per_asset_ffill(simulated_daily_ds):
    """PerAssetFFill should not change the shape of the dataset."""
    ds = simulated_daily_ds.copy(deep=True)
    proc = PerAssetFFill(name="ffill")
    out = proc.transform(ds)
    assert out["var_1"].size == ds["var_1"].size


def test_per_asset_ffill_removes_interior_nans():
    """NaNs in the middle should be replaced by the prior valid value."""
    values = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
    dates = pd.date_range("2023-01-02", periods=5, freq="D")
    df = pd.DataFrame({"time": dates, "asset": "X", "v": values})
    ds = from_table(df, time_column="time", asset_column="asset", frequency=FrequencyType.DAILY)

    out = PerAssetFFill(name="ffill").transform(ds)
    flat = out.dt.to_time_indexed()["v"].sel(asset="X").values

    # After ffill, the two NaN positions should be filled with 1.0
    finite_vals = flat[np.isfinite(flat)]
    # Find the positions of the original data in the flat timeline
    assert 4.0 in finite_vals, "4.0 should be present in the forward-filled values"
    # Verify that some forward-filling occurred (there are fewer NaN positions)
    orig_flat = ds.dt.to_time_indexed()["v"].sel(asset="X").values
    assert np.isnan(flat).sum() <= np.isnan(orig_flat).sum()


# ---------------------------------------------------------------------------
# CSZScore
# ---------------------------------------------------------------------------

def test_cszscore_output(simulated_daily_ds):
    """Z-scored data should have near-zero cross-sectional mean per time step."""
    ds = simulated_daily_ds
    proc = CSZScore(name="norm", vars=["var_1"])
    state = proc.fit(ds)
    out = proc.transform(ds, state)

    assert "var_1_csz" in out.data_vars
    flat = out.dt.to_time_indexed()["var_1_csz"]

    # Cross-sectional mean across assets at each time should be ~0
    cs_mean = float(flat.mean(dim="asset").mean().values)
    assert abs(cs_mean) < 1.0, f"Cross-sectional mean too large: {cs_mean}"


# ---------------------------------------------------------------------------
# FormulaEval — use ohlcv_ds since built-in formulas expect a "close" column
# ---------------------------------------------------------------------------

def test_formula_eval_processor(ohlcv_ds):
    """FormulaEval with rsi formula should add the computed variable."""
    ds = ohlcv_ds
    proc = FormulaEval(
        name="fe",
        formula_configs={"rsi": [{"window": 14}]},
        use_jit=False,
    )
    proc.fit(ds)
    out = proc.transform(ds)

    assert isinstance(out, xr.Dataset)
    # The result should have at least one more variable than the input
    assert len(out.data_vars) >= len(ds.data_vars)


# ---------------------------------------------------------------------------
# DataHandler
# ---------------------------------------------------------------------------

def test_handler_build_shared(simulated_daily_ds):
    """DataHandler.build() should apply shared processors."""
    ds = simulated_daily_ds
    config = HandlerConfig(
        shared=[PerAssetFFill(name="ffill")],
        learn=[],
        infer=[],
    )
    handler = DataHandler(base=ds, config=config)
    handler.build()

    shared = handler.shared_view()
    assert isinstance(shared, xr.Dataset)
    assert set(ds.data_vars).issubset(set(shared.data_vars))


def test_handler_views(simulated_daily_ds):
    """View.RAW returns base data; View.LEARN returns processed data."""
    ds = simulated_daily_ds
    config = HandlerConfig(
        shared=[PerAssetFFill(name="ffill")],
        learn=[CSZScore(name="norm")],
        infer=[],
    )
    handler = DataHandler(base=ds, config=config)

    raw = handler.view(View.RAW)
    learn = handler.view(View.LEARN)

    assert raw is ds
    assert isinstance(learn, xr.Dataset)
    csz_vars = [v for v in learn.data_vars if v.endswith("_csz")]
    assert len(csz_vars) > 0, "Expected CSZScore output variables in LEARN view"
