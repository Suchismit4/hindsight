"""
Methodology tests for the standard June-rebalanced FF3 pipeline.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.ast.functions import eq, month_coord, nan_const, shift, where
from src.data.core.types import FrequencyType
from src.data.loaders.table import from_table
from src.pipeline.data_handler.config import HandlerConfig
from src.pipeline.data_handler.core import PipelineMode, View
from src.pipeline.data_handler.handler import DataHandler
from src.pipeline.data_handler.processors import (
    CrossSectionalSort,
    FactorSpread,
    PerAssetFFill,
    PortfolioReturns,
)
from src.pipeline.spec.executor import PipelineExecutor


def _build_ff3_synthetic_dataset(n_assets=100, n_months=24, seed=2025):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    assets = list(range(n_assets))

    rows = []
    for timestamp in dates:
        for asset in assets:
            rows.append(
                {
                    "time": timestamp,
                    "asset": asset,
                    "ret": rng.normal(0.01, 0.05),
                    "me": rng.lognormal(5.0, 1.5),
                    "seq": rng.uniform(50, 500),
                    "txditc": rng.uniform(0, 30),
                    "ps": rng.uniform(0, 15),
                    "exchcd": 1.0 if asset < 60 else 2.0,
                }
            )

    return from_table(
        pd.DataFrame(rows),
        time_column="time",
        asset_column="asset",
        frequency=FrequencyType.MONTHLY,
    )


def _stack_time(da: xr.DataArray) -> xr.DataArray:
    time_dims = tuple(dim for dim in ("year", "month", "day", "hour") if dim in da.dims)
    stacked = da.stack(time_index=time_dims)
    if "asset" in da.dims:
        return stacked.transpose("time_index", "asset")
    return stacked.transpose("time_index")


def _time_months(ds: xr.Dataset) -> np.ndarray:
    time_index = ds["time"].stack(time_index=("year", "month", "day", "hour")).values
    return pd.DatetimeIndex(time_index).month.to_numpy()


def _compute_standard_ff3_features(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    ds["is_nyse"] = eq(ds["exchcd"], 1)
    ds["be"] = ds["seq"] + ds["txditc"] - ds["ps"]
    ds["be_lag"] = shift(ds["be"], periods=6)
    ds["me_dec"] = shift(ds["me"], periods=6)
    ds["beme"] = ds["be_lag"] / ds["me_dec"]
    ds["me_june"] = where(eq(month_coord(ds["me"]), 6), ds["me"], nan_const())
    ds["beme_june"] = where(eq(month_coord(ds["beme"]), 6), ds["beme"], nan_const())
    ds["me_lag1"] = shift(ds["me"], periods=1)
    return ds


def _make_standard_handler_config(include_factor_spreads=True) -> HandlerConfig:
    infer = [
        CrossSectionalSort(signal="me_june", n_bins=2, scope="is_nyse", name="size_sort"),
        CrossSectionalSort(
            signal="beme_june",
            n_bins=3,
            quantiles=[0.3, 0.7],
            scope="is_nyse",
            name="bm_sort",
        ),
        PerAssetFFill(name="ffill_portfolios", vars=["me_june_port", "beme_june_port"]),
    ]

    if include_factor_spreads:
        infer.extend(
            [
                PortfolioReturns(
                    groupby=["me_june_port", "beme_june_port"],
                    returns_var="ret",
                    weights_var="me_lag1",
                    name="ff_port_ret",
                ),
                FactorSpread(
                    source="port_ret_me_june_port_beme_june_port",
                    factors={
                        "SMB": {
                            "long": {"me_june_port_dim": 0},
                            "short": {"me_june_port_dim": 1},
                            "average_over": "beme_june_port_dim",
                        },
                        "HML": {
                            "long": {"beme_june_port_dim": 2},
                            "short": {"beme_june_port_dim": 0},
                            "average_over": "me_june_port_dim",
                        },
                    },
                    name="ff_factors",
                ),
            ]
        )

    return HandlerConfig(shared=[], learn=[], infer=infer, mode=PipelineMode.INDEPENDENT)


def _run_handler(ds: xr.Dataset, *, include_factor_spreads=True) -> xr.Dataset:
    handler = DataHandler(base=ds, config=_make_standard_handler_config(include_factor_spreads))
    handler.build()
    return handler.view(View.INFER)


def test_month_coord_function():
    ds = _build_ff3_synthetic_dataset()

    month_values = month_coord(ds)
    month_stacked = _stack_time(month_values).values
    expected_months = _time_months(ds)

    np.testing.assert_array_equal(month_stacked[:, 0], expected_months)
    np.testing.assert_array_equal(month_stacked, np.broadcast_to(expected_months[:, None], month_stacked.shape))


def test_june_mask():
    ds = _compute_standard_ff3_features(_build_ff3_synthetic_dataset())

    me_june = _stack_time(ds["me_june"]).values
    me = _stack_time(ds["me"]).values
    months = _time_months(ds)
    is_june = months == 6

    assert np.isnan(me_june[~is_june]).all()
    np.testing.assert_allclose(me_june[is_june], me[is_june], rtol=1e-10, atol=1e-10)


def test_sort_with_scope():
    ds = _compute_standard_ff3_features(_build_ff3_synthetic_dataset())
    out = CrossSectionalSort(signal="me_june", n_bins=2, scope="is_nyse", name="size_sort").transform(ds)

    months = _time_months(ds)
    june_idx = int(np.where(months == 6)[0][0])
    signal = _stack_time(ds["me_june"]).values[june_idx]
    scope = _stack_time(ds["is_nyse"]).values[june_idx].astype(bool)
    actual = _stack_time(out["me_june_port"]).values[june_idx]

    valid_mask = np.isfinite(signal)
    calc_data = signal[valid_mask & scope]
    breakpoints = np.nanquantile(calc_data, np.linspace(0.0, 1.0, 3))
    breakpoints[0] -= 1e-9
    breakpoints[-1] += 1e-9

    expected = np.full_like(signal, np.nan, dtype=np.float64)
    expected[valid_mask] = np.clip(np.digitize(signal[valid_mask], breakpoints) - 1, 0, 1)

    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_ffill_portfolio_ids():
    ds = _compute_standard_ff3_features(_build_ff3_synthetic_dataset())
    out = _run_handler(ds, include_factor_spreads=False)

    ports = _stack_time(out["me_june_port"]).values
    months = _time_months(ds)
    june_rows = np.where(months == 6)[0]
    first_june, second_june = int(june_rows[0]), int(june_rows[1])

    june_assignment = ports[first_june]
    for row_idx in range(first_june + 1, second_june):
        np.testing.assert_allclose(ports[row_idx], june_assignment, equal_nan=True)


def test_full_standard_ff3_chain():
    out = _run_handler(_compute_standard_ff3_features(_build_ff3_synthetic_dataset()))

    assert "SMB" in out.data_vars
    assert "HML" in out.data_vars
    assert set(out["SMB"].dims) == {"year", "month", "day", "hour"}
    assert set(out["HML"].dims) == {"year", "month", "day", "hour"}
    assert not np.isnan(out["SMB"].values).all()
    assert not np.isnan(out["HML"].values).all()


def test_annual_rebalance():
    ds = _compute_standard_ff3_features(_build_ff3_synthetic_dataset())
    out = _run_handler(ds, include_factor_spreads=False)

    ports = _stack_time(out["me_june_port"]).values
    months = _time_months(ds)

    for row_idx in range(1, len(months)):
        if months[row_idx] == 6:
            continue
        np.testing.assert_allclose(ports[row_idx], ports[row_idx - 1], equal_nan=True)


def test_pipeline_executor_inline_expression_support():
    ds = _build_ff3_synthetic_dataset()
    executor = PipelineExecutor()

    features = executor._compute_formulas(
        {
            "be": [{"expression": "$seq + $txditc - $ps"}],
            "be_lag": [{"expression": "shift($be, 6)"}],
            "me_june": [{"expression": "where(eq(month_coord($me), 6), $me, nan_const())"}],
        },
        ds,
    )

    assert "be" in features.data_vars
    assert "be_lag" in features.data_vars
    assert "me_june" in features.data_vars
    assert np.isnan(_stack_time(features["be_lag"]).values[:6]).all()
