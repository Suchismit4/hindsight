"""
Integration tests for the end-to-end FF3 preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.ast.functions import shift
from src.data.core.types import FrequencyType
from src.data.loaders.table import from_table
from src.pipeline.data_handler.config import HandlerConfig
from src.pipeline.data_handler.core import PipelineMode, View
from src.pipeline.data_handler.handler import DataHandler
from src.pipeline.data_handler.processors import (
    CrossSectionalSort,
    FactorSpread,
    PortfolioReturns,
)
from src.pipeline.spec.processor_registry import ProcessorRegistry


def _build_ff3_synthetic_dataset(n_assets=100, n_months=24, seed=2024):
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

    df = pd.DataFrame(rows)
    return from_table(
        df,
        time_column="time",
        asset_column="asset",
        frequency=FrequencyType.MONTHLY,
    )


def _compute_ff3_features(ds):
    be = ds["seq"] + ds["txditc"] - ds["ps"]
    ds = ds.copy()
    ds["be_lag"] = shift(be, periods=6)
    ds["me_dec"] = shift(ds["me"], periods=6)
    ds["beme"] = ds["be_lag"] / ds["me_dec"]
    return ds


def _make_handler_config():
    return HandlerConfig(
        shared=[],
        learn=[],
        infer=[
            CrossSectionalSort(signal="me", n_bins=2, name="sz_sort"),
            CrossSectionalSort(
                signal="beme",
                n_bins=3,
                quantiles=[0.3, 0.7],
                name="bm_sort",
            ),
            PortfolioReturns(
                groupby=["me_port", "beme_port"],
                returns_var="ret",
                weights_var="me",
                name="ff_port_ret",
            ),
            FactorSpread(
                source="port_ret_me_port_beme_port",
                factors={
                    "SMB": {
                        "long": {"me_port_dim": 0},
                        "short": {"me_port_dim": 1},
                        "average_over": "beme_port_dim",
                    },
                    "HML": {
                        "long": {"beme_port_dim": 2},
                        "short": {"beme_port_dim": 0},
                        "average_over": "me_port_dim",
                    },
                },
                name="ff_factors",
            ),
        ],
        mode=PipelineMode.INDEPENDENT,
    )


def _stack_time(da: xr.DataArray) -> xr.DataArray:
    time_dims = tuple(dim for dim in ("year", "month", "day", "hour") if dim in da.dims)
    stacked = da.stack(time_index=time_dims)
    if "asset" in da.dims:
        return stacked.transpose("time_index", "asset")
    return stacked.transpose("time_index")


def _run_handler(ds):
    handler = DataHandler(base=ds, config=_make_handler_config())
    handler.build()
    return handler.view(View.INFER)


def test_synthetic_data_shape():
    ds = _build_ff3_synthetic_dataset()

    assert ds.sizes["year"] == 2
    assert ds.sizes["month"] == 12
    assert ds.sizes["day"] == 1
    assert ds.sizes["hour"] == 1
    assert ds.sizes["asset"] == 100

    for var_name in ("ret", "me", "seq", "txditc", "ps", "exchcd"):
        assert var_name in ds.data_vars
        assert not np.isnan(ds[var_name].values).all()


def test_shift_on_monthly_data():
    ds = _build_ff3_synthetic_dataset()

    shifted = shift(ds["me"], periods=6)
    shifted_stacked = _stack_time(shifted).values

    assert set(shifted.dims) == set(ds["me"].dims)
    for dim in ds["me"].dims:
        assert shifted.sizes[dim] == ds["me"].sizes[dim]
    assert np.isnan(shifted_stacked[:6]).all()
    assert np.isfinite(shifted_stacked[6:]).all()


def test_feature_be_lag():
    ds = _build_ff3_synthetic_dataset()

    be = ds["seq"] + ds["txditc"] - ds["ps"]
    be_lag = shift(be, periods=6)

    be_stacked = _stack_time(be).values
    be_lag_stacked = _stack_time(be_lag).values

    assert np.isnan(be_lag_stacked[:6]).all()
    np.testing.assert_allclose(be_lag_stacked[6:], be_stacked[:-6], rtol=1e-10, atol=1e-10)


def test_feature_beme():
    ds = _compute_ff3_features(_build_ff3_synthetic_dataset())

    valid_mask = np.isfinite(ds["be_lag"].values) & np.isfinite(ds["me_dec"].values)
    assert valid_mask.any()
    assert np.isfinite(ds["beme"].values[valid_mask]).all()
    assert (ds["beme"].values[valid_mask] > 0).all()


def test_datahandler_produces_smb_hml():
    out = _run_handler(_compute_ff3_features(_build_ff3_synthetic_dataset()))

    assert "SMB" in out.data_vars
    assert "HML" in out.data_vars
    assert set(out["SMB"].dims) == {"year", "month", "day", "hour"}
    assert set(out["HML"].dims) == {"year", "month", "day", "hour"}
    assert not np.isnan(out["SMB"].values).all()
    assert not np.isnan(out["HML"].values).all()


def test_smb_hml_reasonable_values():
    out = _run_handler(_compute_ff3_features(_build_ff3_synthetic_dataset()))

    port_ret = out["port_ret_me_port_beme_port"]

    assert port_ret.sizes["me_port_dim"] == 2
    assert port_ret.sizes["beme_port_dim"] == 3
    assert np.nanmax(np.abs(out["SMB"].values)) < 0.5
    assert np.nanmax(np.abs(out["HML"].values)) < 0.5
    assert float(np.nanstd(out["SMB"].values)) > 0.0
    assert float(np.nanstd(out["HML"].values)) > 0.0


def test_smb_matches_manual():
    out = _run_handler(_compute_ff3_features(_build_ff3_synthetic_dataset()))

    port_ret = out["port_ret_me_port_beme_port"]
    small_avg = port_ret.sel(me_port_dim=0).mean(dim="beme_port_dim", skipna=True)
    big_avg = port_ret.sel(me_port_dim=1).mean(dim="beme_port_dim", skipna=True)
    expected_smb = small_avg - big_avg

    np.testing.assert_allclose(out["SMB"].values, expected_smb.values, rtol=1e-10, atol=1e-10)


def test_processor_registry_roundtrip():
    ds = _compute_ff3_features(_build_ff3_synthetic_dataset())
    processors = [
        ProcessorRegistry.create_processor(
            {"type": "sort", "signal": "me", "n_bins": 2, "name": "sz_sort"}
        ),
        ProcessorRegistry.create_processor(
            {
                "type": "sort",
                "signal": "beme",
                "n_bins": 3,
                "quantiles": [0.3, 0.7],
                "name": "bm_sort",
            }
        ),
        ProcessorRegistry.create_processor(
            {
                "type": "port_ret",
                "groupby": ["me_port", "beme_port"],
                "returns_var": "ret",
                "weights_var": "me",
                "name": "ff_port_ret",
            }
        ),
        ProcessorRegistry.create_processor(
            {
                "type": "factor_spread",
                "source": "port_ret_me_port_beme_port",
                "factors": {
                    "SMB": {
                        "long": {"me_port_dim": 0},
                        "short": {"me_port_dim": 1},
                        "average_over": "beme_port_dim",
                    },
                    "HML": {
                        "long": {"beme_port_dim": 2},
                        "short": {"beme_port_dim": 0},
                        "average_over": "me_port_dim",
                    },
                },
                "name": "ff_factors",
            }
        ),
    ]

    for processor in processors:
        ds, _ = processor.fit_transform(ds)

    assert "SMB" in ds.data_vars
    assert "HML" in ds.data_vars
