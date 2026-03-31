"""
Tests for the backtesting engine: order structs, metrics, and engine execution.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.backtester.core import BacktestEngine, EventBasedStrategy
from src.backtester.metrics.standard import MaxDrawdown, SharpeRatio
from src.backtester.struct import BacktestState, MarketOrder, Order, OrderDirection


# ---------------------------------------------------------------------------
# Order struct tests (no engine required)
# ---------------------------------------------------------------------------

def test_market_order_creation():
    order = MarketOrder(
        asset="AAPL",
        quantity=10.0,
        direction=OrderDirection.BUY,
        timestamp=0,
    )
    assert order.asset == "AAPL"
    assert order.quantity == pytest.approx(10.0)
    assert order.direction == OrderDirection.BUY
    assert order.price == -1  # market orders start with sentinel price
    assert order.timestamp == 0


def test_order_direction_enum():
    assert OrderDirection.BUY is not None
    assert OrderDirection.SELL is not None
    assert OrderDirection.BUY != OrderDirection.SELL


# ---------------------------------------------------------------------------
# Metric tests — pass synthetic DataArrays directly
# ---------------------------------------------------------------------------

def _make_synthetic_portfolio(n=50, initial=100_000.0, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, size=n)
    values = initial * np.cumprod(1 + returns)
    times = pd.date_range("2022-01-03", periods=n, freq="D")
    pv = xr.DataArray(values, dims=["time"], coords={"time": times.values})
    return pv


def test_engine_metrics_sharpe():
    pv = _make_synthetic_portfolio(n=60)
    assets = np.array(["X"])
    positions = xr.DataArray(
        np.ones((60, 1)),
        dims=["time_flat", "asset"],
        coords={"time_flat": np.arange(60), "asset": assets},
    )
    cum_ret = xr.DataArray(
        np.log(pv.values / pv.values[0]),
        dims=["time"],
        coords={"time": pv.time.values},
    )
    dummy_state = BacktestState(
        positions=xr.DataArray(np.zeros(1), dims=["asset"]),
        cash=0.0,
        total_portfolio_value=float(pv[-1]),
        timestamp_idx=60,
    )
    sharpe = SharpeRatio().calculate(positions, pv, cum_ret, dummy_state)
    assert np.isfinite(sharpe), f"SharpeRatio is not finite: {sharpe}"


def test_engine_metrics_max_drawdown():
    pv = _make_synthetic_portfolio(n=60)
    assets = np.array(["X"])
    positions = xr.DataArray(
        np.ones((60, 1)),
        dims=["time_flat", "asset"],
        coords={"time_flat": np.arange(60), "asset": assets},
    )
    cum_ret = xr.DataArray(
        np.log(pv.values / pv.values[0]),
        dims=["time"],
        coords={"time": pv.time.values},
    )
    dummy_state = BacktestState(
        positions=xr.DataArray(np.zeros(1), dims=["asset"]),
        cash=0.0,
        total_portfolio_value=float(pv[-1]),
        timestamp_idx=60,
    )
    mdd = MaxDrawdown().calculate(positions, pv, cum_ret, dummy_state)
    assert np.isfinite(mdd), f"MaxDrawdown is not finite: {mdd}"
    assert mdd <= 0.0, "MaxDrawdown should be <= 0"


# ---------------------------------------------------------------------------
# Engine no-trades test
# ---------------------------------------------------------------------------

class _NoTradeStrategy(EventBasedStrategy):
    """Strategy that never issues orders."""

    def next(
        self,
        market_data: xr.Dataset,
        characteristics: xr.Dataset,
        state: BacktestState,
    ) -> Tuple[List[Order], List[str]]:
        return [], []


def test_engine_no_trades(ohlcv_ds):
    """Strategy with no orders should leave cash unchanged."""
    ds = ohlcv_ds
    initial_cash = 50_000.0

    # The start_date must be present in the time coordinate.
    # Pick the first valid date in the dataset.
    times_flat = ds.coords["time"].values.reshape(-1)
    valid_times = times_flat[~np.isnat(times_flat)]
    start_date = str(pd.Timestamp(valid_times.min()).date())
    end_date = str(pd.Timestamp(valid_times.max()).date())

    engine = BacktestEngine(
        initial_cash=initial_cash,
        date=(start_date, end_date),
        verbose=0,
    )
    engine.add_data(market_data=ds, characteristics=ds)
    strategy = _NoTradeStrategy(name="no_trade", window_size=1)
    engine.add_strategy(strategy)

    final_state, cum_returns, metrics = engine.run()

    # With no trades, all portfolio value should equal initial cash
    # (no price exposure since positions are zero)
    assert final_state.cash == pytest.approx(initial_cash, rel=1e-6)
